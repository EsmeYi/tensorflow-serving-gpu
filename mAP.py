# VOC mAP computation, based on https://github.com/amdegroot/ssd.pytorch
import os
import sys
import yaml
import numpy as np

if sys.version_info[0] == 2:
    import xml.etree.cElementTree as ET
else:
    import xml.etree.ElementTree as ET

import voc as voc_utils

class PATHS(object):
    def __init__(self, voc_dir):
        self.voc_image_set_path = os.path.join(voc_dir, 'ImageSets', 'Main', 'test.txt')
        self.voc_annotation_path = os.path.join(voc_dir, 'results', 'Annotations')
        self.voc_annotation_xml_path = os.path.join(voc_dir, 'Annotations', '{}.xml')
        # self.voc_segmentation_object_path = os.path.join(voc_dir, 'SegmentationObject', '{}.png')
        # self.voc_segmentation_class_path = os.path.join(voc_dir, 'SegmentationClass', '{}.png')
    

def parse_voc_annotation_xml(voc_annotiotion_xml):
    """Parse VOC annotation XML file.

    VOC image annotations are described in XML files
    shipped with VOC dataset, with one XML file per each image.
    This function reads relevant object detection data from given
    file and saves it to Python data structures.

    Args:
        voc_annotation_xml (str): VOC annotation XML file path

    Returns:
        Python list of object detections metadata.
    """
    tree = ET.parse(voc_annotiotion_xml)
    size = tree.find('size')
    objects = []
    for obj in tree.findall('object'):
        obj_struct = {}
        obj_struct['image_width'] = size.find('width').text
        obj_struct['image_height'] = size.find('height').text
        obj_struct['name'] = obj.find('name').text
        obj_struct['pose'] = obj.find('pose').text
        obj_struct['truncated'] = int(obj.find('truncated').text)
        obj_struct['difficult'] = int(obj.find('difficult').text)
        bbox = obj.find('bndbox')
        # Coordinates in VOC XMLs are in [1, 256] format, but we use [0, 255]
        obj_struct['bbox'] = [int(bbox.find('xmin').text) - 1,
                              int(bbox.find('ymin').text) - 1,
                              int(bbox.find('xmax').text) - 1,
                              int(bbox.find('ymax').text) - 1]
        objects.append(obj_struct)
    return objects

def get_voc_results_file_template(cls, results_dir):
    """Fetches inference detection result file path for given class.

    During TensorRT/Tensorflow inference, we save class detections into
    separate files, for later mAP computation. This function fetches
    paths of these files.

    Args:
        cls (str): VOC class label
        results_dir (str): path of directory containing detection results

    Returns:
        str: Detection results path for given class.
    """
    # VOCdevkit/VOC2007/results/det_test_aeroplane.txt
    filename = 'det_test_{}.txt'.format(cls)
    if not os.path.exists(results_dir):
        os.makedirs(results_dir)
    path = os.path.join(results_dir, filename)
    return path

def do_python_eval(results_dir, voc_dir):
    aps = []
    for i, cls in enumerate(voc_utils.VOC_CLASSES_LIST):
        filename = get_voc_results_file_template(cls, results_dir)
        rec, prec, ap = voc_eval(filename, voc_dir, cls, ovthresh=0.5)
        aps += [ap]
        print('AP for {} = {:.4f}'.format(cls, ap))
    print('Mean AP = {:.4f}'.format(np.mean(aps)))

def voc_ap(rec, prec):
    ap = 0.
    for t in np.arange(0., 1.1, 0.1):
        if np.sum(rec >= t) == 0:
            p = 0
        else:
            p = np.max(prec[rec >= t])
        ap = ap + p / 11.
    return ap

def read_voc_annotations(annotations_dir, annotations_xml_dir, image_numbers):
    if not os.path.isdir(annotations_dir):
        os.makedirs(annotations_dir)
    annotations_file = os.path.join(annotations_dir, 'annots.pkl')
    if not os.path.isfile(annotations_file):
        # If annotations were not present, compute them
        detections = {}
        for i, image_num in enumerate(image_numbers):
            detections[image_num] = parse_voc_annotation_xml(
                annotations_xml_dir.format(image_num))
            if i % 100 == 0:
                print('Reading annotation for {:d}/{:d}'.format(
                   i + 1, len(image_numbers)))
        # Save
        print('Saving cached annotations to {:s}'.format(annotations_file))
        with open(annotations_file, 'wb') as f:
            yaml.dump(detections, f)
    else:
        # If annotations were present, load them
        with open(annotations_file, 'rb') as f:
            detections = yaml.load(f)
    return detections

def extract_class_detetions(voc_detections, classname, image_numbers):
    class_detections = {}
    for image_num in image_numbers:
        R = [obj for obj in voc_detections[image_num] if obj['name'] == classname]
        image_bboxes = [x['bbox'] for x in R]

        # Transform VOC bboxes to make them describe pre-resized 300x300 images
        for idx, bbox in enumerate(image_bboxes):
            bbox = np.array(bbox).astype(np.float32)
            width = float(R[0]['image_width'])
            height = float(R[0]['image_height'])
            bbox[0] *= (300.0 / width)
            bbox[2] *= (300.0 / width)
            bbox[1] *= (300.0 / height)
            bbox[3] *= (300.0 / height)
            image_bboxes[idx] = bbox
        image_bboxes = np.array(image_bboxes)
        difficult = np.array([x['difficult'] for x in R]).astype(np.bool)
        det = [False] * len(R)
        class_detections[image_num] = {
            'bbox': image_bboxes,
            'difficult': difficult,
            'det': det
        }

    return class_detections

def voc_eval(detpath, voc_dir, classname, ovthresh=0.5):
    imagesetfile = PATHS(voc_dir).voc_image_set_path
    annotations_dir = PATHS(voc_dir).voc_annotation_path
    annotations_xml_dir = PATHS(voc_dir).voc_annotation_xml_path
    with open(imagesetfile, 'r') as f:
        lines = f.readlines()
    image_numbers = [x.strip() for x in lines]

    voc_detections = read_voc_annotations(annotations_dir, annotations_xml_dir, image_numbers)
    class_detections = extract_class_detetions(voc_detections, classname,
        image_numbers)

    is_detection_difficult = np.concatenate(
        [class_detections[image_num]['difficult'] for image_num in image_numbers]
    )
    not_difficult_count = sum(~is_detection_difficult)

    # Read detections outputed by model
    detfile = detpath.format(classname)
    with open(detfile, 'r') as f:
        lines = f.readlines()

    if any(lines):
        splitlines = [x.strip().split(' ') for x in lines]
        image_ids = [x[0] for x in splitlines]
        confidence = np.array([float(x[1]) for x in splitlines])
        bboxes = np.array([[float(z) for z in x[2:]] for x in splitlines])

        # sort by confidence
        sorted_ind = np.argsort(-confidence)
        sorted_scores = np.sort(-confidence)
        bboxes = bboxes[sorted_ind, :]
        image_ids = [image_ids[x] for x in sorted_ind]

        # Go down dets and mark TPs and FPs
        num_detections = len(image_ids)
        tp = np.zeros(num_detections)
        fp = np.zeros(num_detections)
        for detection in range(num_detections):
            R = class_detections[image_ids[detection]]
            bbox = bboxes[detection, :].astype(float)
            ovmax = -np.inf
            bbox_gt = R['bbox'].astype(float)
            if bbox_gt.size > 0:
                # compute overlaps
                # intersection
                ixmin = np.maximum(bbox_gt[:, 0], bbox[0])
                iymin = np.maximum(bbox_gt[:, 1], bbox[1])
                ixmax = np.minimum(bbox_gt[:, 2], bbox[2])
                iymax = np.minimum(bbox_gt[:, 3], bbox[3])
                iw = np.maximum(ixmax - ixmin, 0.)
                ih = np.maximum(iymax - iymin, 0.)
                inters = iw * ih
                uni = ((bbox[2] - bbox[0]) * (bbox[3] - bbox[1]) +
                       (bbox_gt[:, 2] - bbox_gt[:, 0]) *
                       (bbox_gt[:, 3] - bbox_gt[:, 1]) - inters)
                overlaps = inters / uni
                ovmax = np.max(overlaps)
                jmax = np.argmax(overlaps)

            if ovmax > ovthresh:
                if not R['difficult'][jmax]:
                    if not R['det'][jmax]:
                        tp[detection] = 1.
                        R['det'][jmax] = 1
                    else:
                        fp[detection] = 1.
            else:
                fp[detection] = 1.

        # Compute precision and recall
        fp = np.cumsum(fp)
        tp = np.cumsum(tp)
        rec = tp / float(not_difficult_count)
        # avoid divide by zero in case the first detection matches a difficult
        # ground truth
        prec = tp / np.maximum(tp + fp, np.finfo(np.float64).eps)
        ap = voc_ap(rec, prec)
    else:
        rec = -1.
        prec = -1.
        ap = -1.

    return rec, prec, ap
