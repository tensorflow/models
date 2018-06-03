import numpy as np
import os
import six.moves.urllib as urllib
import tarfile
import sys
import glob
import tensorflow as tf
import time
from matplotlib import pyplot as plt
from PIL import Image

sys.path.append("..")

from object_detection.utils import ops as utils_ops

if tf.__version__ < '1.4.0':
    raise ImportError('Please upgrade your tensorflow installation to v1.4.* or later!')

from utils import label_map_util
from utils import visualization_utils as vis_util

# Path to frozen detection graph. This is the actual model that is used for the object detection.
PATH_TO_CKPT = '/home/deepdot/Dataset/check_point/model.ckpt'

# List of the strings that is used to add correct label for each box.
PATH_TO_LABELS = "/home/deepdot/Dataset/Budweiser/label_map.pbtxt"

NUM_CLASSES = 30


detection_graph = tf.Graph()

print("Successfully loaded the model.")

label_map = label_map_util.load_labelmap(PATH_TO_LABELS)
categories = label_map_util.convert_label_map_to_categories(label_map, max_num_classes=NUM_CLASSES, use_display_name=True)
category_index = label_map_util.create_category_index(categories)

print("Successfully loaded the label map.")

def load_image_into_numpy_array(image):
    (im_width, im_height) = image.size
    print("Image size: %d x %d" % (im_width, im_height))
    return np.array(image.getdata()).reshape((im_height, im_width, 3)).astype(np.uint8)

PATH_TO_TEST_IMAGES_DIR = 'test_images'
TEST_IMAGE_PATHS = [ os.path.join(PATH_TO_TEST_IMAGES_DIR, 'image{}.jpg'.format(i)) for i in range(1, 3) ]
PATH_TO_TEST_IMAGES_DIR = '/home/deepdot/Dataset/Budweiser/top1000/images/'
TEST_IMAGE_PATHS = glob.glob(PATH_TO_TEST_IMAGES_DIR + '/02*.jpg')

# Size, in inches, of the output images.
IMAGE_SIZE = (12, 8)

def run_inference_for_single_image(image, graph):
    with graph.as_default():
        with tf.Session() as sess:
            saver = tf.train.import_meta_graph(PATH_TO_CKPT + '.meta')
            saver.restore(sess, PATH_TO_CKPT)
            # Get handles to input and output tensors
            ops = tf.get_default_graph().get_operations()
            all_tensor_names = {output.name for op in ops for output in op.outputs}
            tensor_dict = {}
            for key in [ 'num_detections', 'detection_boxes', 'detection_scores', 'detection_classes', 'detection_masks' ]:
                tensor_name = key + ':0'
                if tensor_name in all_tensor_names:
                    tensor_dict[key] = tf.get_default_graph().get_tensor_by_name(tensor_name)
            image_tensor = tf.get_default_graph().get_tensor_by_name('image_tensor:0')
            # Run inference
            output_dict = sess.run(tensor_dict, feed_dict={image_tensor: np.expand_dims(image, 0)})

            output_dict['num_detections'] = int(output_dict['num_detections'][0])
            output_dict['detection_classes'] = output_dict['detection_classes'][0].astype(np.uint8)
            output_dict['detection_boxes'] = output_dict['detection_boxes'][0]
            output_dict['detection_scores'] = output_dict['detection_scores'][0]
    return output_dict

for image_path in TEST_IMAGE_PATHS:
    image = Image.open(image_path)
    start = time.time()
    image_np = load_image_into_numpy_array(image)

    print image_np.shape
    print "First pixel"
    print "0: %d" % image_np[0][0][0]
    print "1: %d" % image_np[0][0][1]
    print "2: %d" % image_np[0][0][2]
    # Expand dimensions since the model expects images to have shape: [1, None, None, 3]
    image_np_expanded = np.expand_dims(image_np, axis=0)
    # Actual detection.
    output_dict = run_inference_for_single_image(image_np, detection_graph)
    end = time.time()
    print("Finished inference in %f seconds." % (end - start))
    print("Number of detections: %d" % output_dict['num_detections'])
    print("detection_boxes")
    print output_dict['detection_boxes'].shape
    print("detection_classes")
    print output_dict['detection_classes'].shape
    print("detection_scores")
    print output_dict['detection_scores'].shape
    print "First detection:"
    box = output_dict['detection_boxes'][0, :]
    print "box %f, %f, %f, %f" % (box[0], box[1], box[2], box[3])
    print "class %d" % (output_dict['detection_classes'][0])
    print "score %f" % (output_dict['detection_scores'][0])
    # Visualization of the results of a detection.
    vis_util.visualize_boxes_and_labels_on_image_array(
        image_np,
        output_dict['detection_boxes'],
        output_dict['detection_classes'],
        output_dict['detection_scores'],
        category_index,
        instance_masks=output_dict.get('detection_masks'),
        use_normalized_coordinates=True,
        max_boxes_to_draw=50,
        min_score_thresh=.05,
          line_thickness=10)
    plt.figure(figsize=IMAGE_SIZE)
    plt.imshow(image_np)
    plt.show()

