from __future__ import print_function
import argparse
import os
import shutil
import time
import glob
from PIL import Image
import tensorflow as tf
import numpy as np
import voc as voc_trt
import mAP as mAP_trt

VOC_CLASSES = voc_trt.VOC_CLASSES_LIST
COCO_LABELS = voc_trt.COCO_CLASSES_LIST


class Inference(object):
    def __init__(self, pb_model_path, frozen_model = True):
        self.model_path = pb_model_path
        # Load a frozen model
        if frozen_model:
            self.detection_graph = tf.Graph()
            with self.detection_graph.as_default():
                od_graph_def = tf.GraphDef()
                with tf.gfile.GFile(pb_model_path, 'rb') as fid:
                    serialized_graph = fid.read()
                    od_graph_def.ParseFromString(serialized_graph)
                    tf.import_graph_def(od_graph_def, name='')
        # Load a saved model
        else:
            with tf.Session(graph = tf.Graph()) as sess:
                tf.saved_model.loader.load(sess, ['serve'], self.model_path)
                self.detection_graph = tf.get_default_graph()
        self.sess = tf.Session(graph=self.detection_graph)

    def _load_img(self, image_path):
        img = Image.open(image_path)
        (im_width, im_height) = img.size
        img_np = np.array(img).reshape((im_width, im_height, 3)).astype(np.uint8)
        return img_np

    def _load_imgs(self, image_paths):
        numpy_array = np.zeros((len(image_paths),) + (300, 300, 3))
        for idx, image_path in enumerate(image_paths):
            img_np = self._load_img(image_path)
            numpy_array[idx] = img_np
        return numpy_array.astype(np.uint8)

    def doInference(self, image_paths):
        image_input = self._load_imgs(image_paths)
        ops = self.detection_graph.get_operations()
        all_tensor_names = {output.name for op in ops for output in op.outputs}
        tensor_dict = {}
        for key in [
            'num_detections', 'detection_boxes',
            'detection_scores', 'detection_classes'
        ]:
            tensor_name = key + ':0'
            if tensor_name in all_tensor_names:
                tensor_dict[key] = self.detection_graph.get_tensor_by_name(
                    tensor_name)

        image_tensor = self.detection_graph.get_tensor_by_name('image_tensor:0')
        output_dict = self.sess.run(tensor_dict,
            feed_dict={image_tensor: image_input})

        # All outputs are float32 numpy arrays, so convert types as appropriate
        output_dict['num_detections'] = output_dict['num_detections'].astype(np.int32)
        output_dict['detection_classes'] = output_dict[
            'detection_classes'].astype(np.uint8)

        return output_dict

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
    parser = argparse.ArgumentParser(description='Run object detection evaluation on VOC2007 dataset using TensorFlow.')
    parser.add_argument('-mp', '--pb_model_path', type = str, required=True)
    parser.add_argument('-b', '--batch_size', type = int, default=32)
    parser.add_argument('-voc', '--voc_dir', type = str, default='./VOCdevkit/VOC2007')

    args = parser.parse_args()
    pb_model_path = args.pb_model_path
    batch_size = args.batch_size
    voc_dir = args.voc_dir

    voc_image_set_path = os.path.join(voc_dir, 'ImageSets', 'Main', 'test.txt')
    voc_jpg_img_path = os.path.join(voc_dir, 'JPEGImages', '{}.jpg')
    voc_ppm_img_path = os.path.join(voc_dir, 'PPMImages', '{}.ppm')
    preprocess_voc(voc_dir)

    results_dir = os.path.join(voc_dir, 'results')
    if os.path.exists(results_dir):
        shutil.rmtree(results_dir)
    os.makedirs(results_dir)

    detection_files = {}
    for voc_class in VOC_CLASSES:
            detection_files[voc_class] = open(
                os.path.join(
                    results_dir, 'det_test_{}.txt'.format(voc_class)
                ), 'w'
            )

    with open(voc_image_set_path, 'r') as f:
        voc_image_numbers = f.readlines()
        image_numbers = [line.strip() for line in voc_image_numbers]
    image_path = voc_ppm_img_path
    total_imgs = len(image_numbers)

    # Start measuring time
    inference_start_time = time.time()
    # Initialize inference model
    tf_inference = Inference(pb_model_path, frozen_model = False)
    # Inference with batching
    for idx in range(0, len(image_numbers), batch_size):
        print("Infering image {}/{}".format(idx+1, total_imgs))
        imgs = image_numbers[idx:idx+batch_size]
        image_paths = [image_path.format(img) for img in imgs]
        output_dict = tf_inference.doInference(image_paths)

        keep_count = output_dict['num_detections']
        for img_idx, img_number in enumerate(imgs):
            for det in range(int(keep_count[img_idx])):
                label = output_dict['detection_classes'][img_idx][det]
                confidence = output_dict['detection_scores'][img_idx][det]
                bbox = output_dict['detection_boxes'][img_idx][det]

                ymin, xmin, ymax, xmax = bbox
                xmin = float(xmin) * 300
                ymin = float(ymin) * 300
                xmax = float(xmax) * 300
                ymax = float(ymax) * 300

                # Detection is saved only if confidence is bigger than zero
                if confidence > 0.0:
                    # Model was trained on COCO, so we need to convert label to VOC one
                    label_name = voc_trt.coco_label_to_voc_label(COCO_LABELS[label])
                    if label_name: # Checks for label_name correctness
                        det_file = detection_files[label_name]
                        detection = Detection(
                            img_number,
                            confidence,
                            xmin,
                            ymin,
                            xmax,
                            ymax,
                        )
                        detection.write_to_file(det_file)  

    # Output total [img load + inference] time
    print("Total time taken for inference: {} ms\n".format(int(round((time.time() - inference_start_time) * 1000))))

    # Compute mAP 
    for key in detection_files:
        detection_files[key].flush()

    mAP_trt.do_python_eval(results_dir, voc_dir)

    for key in detection_files:
        detection_files[key].close()
