from __future__ import print_function
import argparse
import os
import shutil
import time
import glob
from PIL import Image
import grpc
import requests
import numpy as np
from google.protobuf.json_format import MessageToDict

from tensorflow_serving.apis import predict_pb2
from tensorflow_serving.apis import prediction_service_pb2_grpc

# import tensorflow as tf
# from tensorflow.contrib.util import make_tensor_proto
from tensorflow.core.framework import tensor_pb2  
from tensorflow.core.framework import tensor_shape_pb2  
from tensorflow.core.framework import types_pb2
import voc as voc_trt
import mAP as mAP_trt

VOC_CLASSES = voc_trt.VOC_CLASSES_LIST
COCO_LABELS = voc_trt.COCO_CLASSES_LIST

def dtypes_as_dtype(dtype):
    if dtype == 'float32':
    	return types_pb2.DT_FLOAT
    if dtype == 'uint8':
		return types_pb2.DT_UINT8
    raise Exception('dtype %s is not supported' % dtype)


def make_tensor_proto(data):
    # replace tf.make_tensor_proto(data, shape=data.shape) to optimize client request
    shape = data.shape
    dims = [tensor_shape_pb2.TensorShapeProto.Dim(size=i) for i in shape]
    proto_shape= tensor_shape_pb2.TensorShapeProto(dim=dims)

    proto_dtype = dtypes_as_dtype(data.dtype)

    tensor_proto = tensor_pb2.TensorProto(
        dtype=proto_dtype,
        tensor_shape=proto_shape)
    tensor_proto.tensor_content = data.tostring()

    return tensor_proto

def load_img(image_path):
	img = Image.open(image_path)
	(im_width, im_height) = img.size
	img_np = np.array(img).reshape((im_width, im_height, 3)).astype(np.uint8)
	return img_np

def load_imgs(image_paths):
	numpy_array = np.zeros((len(image_paths),) + (300, 300, 3))
	for idx, image_path in enumerate(image_paths):
		img_np = load_img(image_path)
		numpy_array[idx] = img_np
	return numpy_array.astype(np.uint8)

def send_request(server_url, model_name, signature_name, input_name, image_paths):
	start = time.time()
	imgs = load_imgs(image_paths)
	load_time = time.time()
	print("Time taken for loading {} images: {} ms\n".format(len(image_paths), int(round((load_time - start) * 1000))))
	# create the gRPC stub
	max_message_length = 500 * 1024 * 1024
	options = [('grpc.max_message_length', max_message_length),
		('grpc.send_message_length', max_message_length),
		('grpc.max_receive_message_length', max_message_length)]
	channel = grpc.insecure_channel(server_url, options = options)
	stub = prediction_service_pb2_grpc.PredictionServiceStub(channel)

	# create the request object and set the name and signature_name params
	request = predict_pb2.PredictRequest()
	request.model_spec.name = model_name
	request.model_spec.signature_name = signature_name

	# fill in the request object with the necessary data
	tensor_proto = make_tensor_proto(imgs)
	proto_time = time.time()
	print("Time taken for make_tensor_proto(): {} ms\n".format(int(round((proto_time - load_time) * 1000))))

	request.inputs[input_name].CopyFrom(tensor_proto)
	result = stub.Predict(request, 200000.)
	request_time = time.time()
	print("Time taken for send, inference and receive: {} ms\n".format(int(round((request_time - proto_time) * 1000))))

	# output_dict = MessageToDict(result, preserving_proto_field_name = True)
	# dict_time = time.time()
	# print("Time taken for MessageToDict(): {} ms\n".format(int(round((dict_time - request_time) * 1000))))

	return result

def preprocess_voc(voc_root):
    """Resizes all VOC images to 300x300 and saves them into .ppm files."""
    voc_jpegs = glob.glob(os.path.join(voc_root, 'JPEGImages', '*.jpg'))
    voc_ppms = glob.glob(os.path.join(voc_root, 'PPMImages', '*.ppm'))
    voc_ppm_img_path = os.path.join(voc_root, 'PPMImages', '{}.ppm')
    # Check if preprocessing is needed by comparing
    # image names between JPEGImages and PPMImages
    voc_jpegs_basenames = \
        [os.path.splitext(os.path.basename(p))[0] for p in voc_jpegs]
    voc_ppms_basenames = \
        [os.path.splitext(os.path.basename(p))[0] for p in voc_ppms]
    # If lists are not the same, preprocessing is needed
    if sorted(voc_jpegs_basenames) != sorted(voc_ppms_basenames):
        print("Preprocessing VOC dataset. It may take few minutes.")
        # Make PPMImages directory if it doesn't exist
        voc_ppms_path = voc_ppm_img_path
        if not os.path.exists(os.path.dirname(voc_ppms_path)):
            os.makedirs(os.path.dirname(voc_ppms_path))
        # For each .jpg file, make a resized
        # .ppm copy to fit model input expectations
        for voc_jpeg_path in voc_jpegs:
            voc_jpeg_basename = os.path.basename(voc_jpeg_path)
            voc_ppm_path = voc_ppms_path.format(
                os.path.splitext(voc_jpeg_basename)[0])
            if not os.path.exists(voc_ppm_path):
                img_pil = Image.open(voc_jpeg_path)
                img_pil = img_pil.resize(
                    size=(300,300),
                    resample=Image.BILINEAR
                )
                img_pil.save(voc_ppm_path)

class Detection(object):
	def __init__(self, image_number, confidence, xmin, ymin, xmax, ymax):
		self.image_number = image_number
		self.confidence = confidence
		self.xmin = xmin
		self.ymin = ymin
		self.xmax = xmax
		self.ymax = ymax
	
	def __repr__(self):
		return "{} {:.3f} {:.1f} {:.1f} {:.1f} {:.1f}\n".format(self.image_number, self.confidence, self.xmin, self.ymin, self.xmax, self.ymax)

	def write_to_file(self, f):
		f.write(self.__repr__())

if __name__ == '__main__':
	parser = argparse.ArgumentParser(description='Run object detection evaluation on VOC2007 dataset using TensorFlow:Serving.')
	parser.add_argument('-s', '--server_url', type=str, default='localhost:9000', help='server url where the model launched')
	parser.add_argument('-mn', '--model_name', type=str, default='mrcnn', help='model name will request')
	parser.add_argument('-sn', '--signature_name', type=str, default='serving_default', help='signature name defined in .pb file during export')
	parser.add_argument('-in', '--input_name', type=str, default='inputs', help='input name')
	parser.add_argument('-voc', '--voc_dir', type=str, default='./VOCdevkit/VOC2007', help='VOC2007 root directory')
	parser.add_argument('-b', '--batch_size', type=int, default=512, help='batch size in one request')
	parser.add_argument('-o', '--output_size', type=int, default=100, help='output size defined in .pb file during export')

	# Parse arguments passed
	args = parser.parse_args()
	server_url = args.server_url
	model_name = args.model_name
	signature_name = args.signature_name
	input_name =args.input_name
	voc_dir = args.voc_dir
	batch_size = args.batch_size
	output_size = args.output_size

	voc_image_set_path = os.path.join(voc_dir, 'ImageSets', 'Main', 'test.txt')
	voc_jpg_img_path = os.path.join(voc_dir, 'JPEGImages', '{}.jpg')
	voc_ppm_img_path = os.path.join(voc_dir, 'PPMImages', '{}.ppm')
	preprocess_voc(voc_dir)

	# Create the result files to save detections
	results_dir = os.path.join(voc_dir, 'results')
	if os.path.exists(results_dir):
		shutil.rmtree(results_dir)
	os.makedirs(results_dir)
	# Each detection test file corresponds to one class
	detection_files = {}
	for voc_class in VOC_CLASSES:
            detection_files[voc_class] = open(
                os.path.join(
                    results_dir, 'det_test_{}.txt'.format(voc_class)
                ), 'w'
            )

	# Fetch image list and input .ppm files path
	with open(voc_image_set_path, 'r') as f:
		voc_image_numbers = f.readlines()
		image_numbers = [line.strip() for line in voc_image_numbers]
	image_path = voc_ppm_img_path
	total_imgs = len(image_numbers)

	# Start measuring time
	inference_start_time = time.time()
	# Inference with batching
	for idx in range(0, len(image_numbers), batch_size):
		print("Infering image {}/{}".format(idx+1, total_imgs))
		imgs = image_numbers[idx:idx+batch_size]
		image_paths = [image_path.format(img) for img in imgs]
		result = send_request(server_url, model_name, signature_name, input_name, image_paths)
		real_batch_size = len(image_paths)
		# Reshape the output_dicts(which was received as one-dim)
		keep_count = np.asarray(result.outputs['num_detections'].float_val).astype(np.int32)
		labels = np.asarray(result.outputs['detection_classes'].float_val).astype(np.uint8).reshape(real_batch_size, output_size)
		scores = np.asarray(result.outputs['detection_scores'].float_val).reshape(real_batch_size, output_size)
		boxes = np.asarray(result.outputs['detection_boxes'].float_val).reshape(real_batch_size, output_size, 4)
	
		# Save results to detection_files
		for img_idx, img_number in enumerate(imgs):
			for det in range(int(keep_count[img_idx])):
				label = labels[img_idx][det]
				score = scores[img_idx][det]
				box = boxes[img_idx][det]
				ymin, xmin, ymax, xmax = box
				# Received bounding boxes are in range [0, 1]
				ymin = float(ymin) * 300
				xmin = float(xmin) * 300
				ymax = float(ymax) * 300
				xmax = float(xmax) * 300
				if score > 0.0:
					label_name = voc_trt.coco_label_to_voc_label(COCO_LABELS[label])
					if label_name:
						detection_file = detection_files[label_name]
						detection = Detection(img_number, score, xmin, ymin, xmax, ymax)
						detection.write_to_file(detection_file)
		print("Time taken for write result: {} ms\n".format(int(round((time.time() - start) * 1000))))

	# Output total [img load + inference] time
	print("Total time taken for inference: {} ms\n".format(int(round((time.time() - inference_start_time) * 1000))))

	# Compute mAP 
	for key in detection_files:
		detection_files[key].flush()

	mAP_trt.do_python_eval(results_dir, voc_dir)

	for key in detection_files:
		detection_files[key].close()

