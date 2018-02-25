# coding: utf-8
import numpy as np
import os
import six.moves.urllib as urllib
import tarfile
import tensorflow as tf

from utils import label_map_util
from utils import visualization_utils as vis_util
from object_detection.protos import string_int_label_map_pb2
from optparse import OptionParser

import cv2

CAMERA_ID = 0

# What model to download.
MODEL_NAME = 'ssd_mobilenet_v1_coco_11_06_2017'
MODEL_FILE = MODEL_NAME + '.tar.gz'
DOWNLOAD_BASE = 'http://download.tensorflow.org/models/object_detection/'

# Path to frozen detection graph. This is the actual model that is used for the object detection.
PATH_TO_CKPT = MODEL_NAME + '/frozen_inference_graph.pb'

# List of the strings that is used to add correct label for each box.
PATH_TO_LABELS = os.path.join('data', 'mscoco_label_map.pbtxt')

NUM_CLASSES = 90

def capture_callback(storagedir, np_image):
    #print(storagedir, np_image)
    pass

def load_image_into_numpy_array(image):
    (im_width, im_height) = image.size
    return np.array(image.getdata()).reshape(
        (im_height, im_width, 3)).astype(np.uint8)

def classify_live_image(callback, savedir=None, freq=10, displayoff=False):
    cap = cv2.VideoCapture(CAMERA_ID)

    # ## Download Model
    if not os.path.isfile(MODEL_FILE):
        print('download model')
        opener = urllib.request.URLopener()
        opener.retrieve(DOWNLOAD_BASE + MODEL_FILE, MODEL_FILE)
        tar_file = tarfile.open(MODEL_FILE)
        for file in tar_file.getmembers():
            file_name = os.path.basename(file.name)
            if 'frozen_inference_graph.pb' in file_name:
                tar_file.extract(file, os.getcwd())
    else:
        print('using cached model')
    # download_model()

    # ## Load a (frozen) Tensorflow model into memory.
    print('loading model')
    detection_graph = tf.Graph()
    with detection_graph.as_default():
        od_graph_def = tf.GraphDef()
        with tf.gfile.GFile(PATH_TO_CKPT, 'rb') as fid:
            serialized_graph = fid.read()
            od_graph_def.ParseFromString(serialized_graph)
            print('importing graph')
            tf.import_graph_def(od_graph_def, name='')
            print('done')

    # ## Loading label map
    # Label maps map indices to category names, so that when our convolution network predicts `5`, we know that this corresponds to `airplane`.  Here we use internal utility functions, but anything that returns a dictionary mapping integers to appropriate string labels would be fine

    label_map = label_map_util.load_labelmap(PATH_TO_LABELS)
    categories = label_map_util.convert_label_map_to_categories(label_map, max_num_classes=NUM_CLASSES,
                                                                use_display_name=True)
    category_index = label_map_util.create_category_index(categories)
    print(category_index)

    frame = 0
    count = 0
    with detection_graph.as_default():
        with tf.Session(graph=detection_graph) as sess:
            while True:
                ret, image_np = cap.read()

                # Expand dimensions since the model expects images to have shape: [1, None, None, 3]
                image_np_expanded = np.expand_dims(image_np, axis=0)
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
                # Visualization of the results of a detection.
                vis_util.visualize_boxes_and_labels_on_image_array(
                    image_np,
                    np.squeeze(boxes),
                    np.squeeze(classes).astype(np.int32),
                    np.squeeze(scores),
                    category_index,
                    use_normalized_coordinates=True,
                    line_thickness=8)
                min_score_thresh = .5
                labels = []
                s = np.squeeze(scores)
                b = np.squeeze(boxes)
                c = np.squeeze(classes).astype(np.int32)
                for i in range(b.shape[0]):
                    if s is None or s[i] > min_score_thresh:
                        labels.append(category_index[c[i]]['name'])
                print('{} : {}'.format(' '.join(labels), frame))
                frame += 1

                if savedir != None:
                    if count == freq:
                        count = 0
                        callback(savedir, image_np)
                        print('save image')
                    else:
                        count += 1

                # mag = 1 # for small display and 6 for large
                mag = 6
                if not displayoff:
                    cv2.imshow('R2K9', cv2.resize(image_np, (160 * mag, 90 * mag)))
                if cv2.waitKey(25) & 0xFF == ord('q'):
                    break
        cv2.destroyAllWindows()

if __name__ == '__main__':
    parser = OptionParser()
    parser.add_option("-d", "--displayoff",
                      action="store_false", dest="displayoff", default=False,
                      help="turns off the display for headless operation")
    parser.add_option("-s", "--savedir", dest="savedir", type="string",
                      help="directory for saved training data")
    parser.add_option("-f", "--frequency", dest="frequency", type="int",
                      default=10, help="frequency of image capture")
    (options, args) = parser.parse_args()
    classify_live_image(capture_callback, savedir=options.savedir, freq=options.frequency, displayoff=options.displayoff)
