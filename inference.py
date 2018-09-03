import numpy as np
import os
import six.moves.urllib as urllib
import sys
import tarfile
import tensorflow as tf
from collections import defaultdict
from io import StringIO
from PIL import Image
from object_detection.utils import label_map_util
from object_detection.utils import visualization_utils as vis_util
import json
import cv2
import datetime
import argparse 

parser = argparse.ArgumentParser(description='Detect boxes on images from a specific folder')
parser.add_argument('--source', help='Source images', required=True)
parser.add_argument('--destination', help='Destination folder for markup images', required=True)
parser.add_argument('--graph', help='Path to pb file', required=True)
parser.add_argument('--classes', help='Path to txt file with all classes', required=True)
parser.add_argument('--score_threshold', help='Default score threshold on displayed images', default=0.5)

args = parser.parse_args()

OUTPUT_DIRECTORY = args.destination
PATH_TO_TEST_IMAGES_DIR = args.source
PATH_TO_CKPT = args.graph
PATH_TO_LABELS = args.classes
SCORE_THRESHOLD = float(args.score_threshold)

detection_graph = tf.Graph()
with detection_graph.as_default():
    od_graph_def = tf.GraphDef()
    with tf.gfile.GFile(PATH_TO_CKPT, 'rb') as fid:
        serialized_graph = fid.read()
        od_graph_def.ParseFromString(serialized_graph)
        tf.import_graph_def(od_graph_def, name='')


def get_categories(path_to_labels):
    result = []
    with open(path_to_labels, "rb") as cls:
        lines = [l.strip() for l in cls.readlines()]
        for ind,label in enumerate(lines):
            if len(label) > 0:
                if label[0] != "'":
                    label = "'" + label
                if label[len(label) - 1] != "'":
                    label = label + "'"
                result.append({"id": ind + 1, "name": label})
    return result

categories = get_categories(PATH_TO_LABELS)
category_index = label_map_util.create_category_index(categories)

if not os.path.exists(OUTPUT_DIRECTORY):
    os.makedirs(OUTPUT_DIRECTORY)

def load_image_into_numpy_array(image):
  (im_width, im_height) = image.size
  return np.array(image.getdata()).reshape(
      (im_height, im_width, 3)).astype(np.float32)


TEST_IMAGE_PATHS = os.listdir(PATH_TO_TEST_IMAGES_DIR)

# Size, in inches, of the output images.
IMAGE_SIZE = (12, 8)

config = tf.ConfigProto(allow_soft_placement = True)

with detection_graph.as_default():
    with tf.Session(graph=detection_graph) as sess:
        for image_path in [os.path.join(PATH_TO_TEST_IMAGES_DIR ,img_name) for img_name in TEST_IMAGE_PATHS]:
            image = Image.open(image_path)
            # the array based representation of the image will be used later in order to prepare the
            # result image with boxes and labels on it.
            image_np = load_image_into_numpy_array(image)
            # imgae_np = tf.(image_np, perm=[2, 0, 1])

            # image = tf.cast(image, tf.float32)

            # Expand dimensions since the model expects images to have shape: [1, None, None, 3]
            image_np_expanded = np.expand_dims(image_np, axis=0)
            start = datetime.datetime.now()
            image_tensor = detection_graph.get_tensor_by_name('image_tensor:0')
            # Each box represents a part of the image where a particular object was detected.
            boxes = detection_graph.get_tensor_by_name('detection_boxes:0')
            # Each score represent how level of confidence for each of the objects.
            # Score is shown on the result image, together with the class label.
            scores = detection_graph.get_tensor_by_name('detection_scores:0')
            classes = detection_graph.get_tensor_by_name('detection_classes:0')
            num_detections = detection_graph.get_tensor_by_name('num_detections:0')
            # Actual detection.
            (boxes, scores, classes, num_detections) = sess.run(
            [boxes, scores, classes, num_detections],
            feed_dict={image_tensor: image_np_expanded})
            print("The detection took %s s" % (datetime.datetime.now() - start))
            # Visualization of the results of a detection.
            vis_util.visualize_boxes_and_labels_on_image_array(
                image_np,
                np.squeeze(boxes),
                np.squeeze(classes).astype(np.int32),
                np.squeeze(scores),
                category_index,
                use_normalized_coordinates=True,
                line_thickness=8,
                max_boxes_to_draw=300,
                min_score_thresh=SCORE_THRESHOLD)
            #cv2.imwrite(os.path.join(OUTPUT_DIRECTORY, os.path.basename(image_path)) ,image_np)        
            result = Image.fromarray(image_np.astype(np.uint8))
            result.save(os.path.join(OUTPUT_DIRECTORY, os.path.basename(image_path)))
