import numpy as np
import os
import six.moves.urllib as urllib
import sys
import tarfile
import tensorflow as tf
import zipfile
import cv2
import pandas as pd

from distutils.version import StrictVersion
from collections import defaultdict
from io import StringIO
from matplotlib import pyplot as plt
from PIL import Image, ImageDraw

# This is needed since the notebook is stored in the object_detection folder.
from object_detection.utils import ops as utils_ops
#sys.path.append("..")
if StrictVersion(tf.__version__) < StrictVersion('1.9.0'):
  raise ImportError('Please upgrade your TensorFlow installation to v1.9.* or later!')


# This is needed to display the images.
#%matplotlib

from utils import label_map_util
# path= r'C:\Users\alexs\OneDrive\Documents\GitHub\models\research\object_detection'
# os.chdir(path)
from utils import visualization_utils as vis_util

corgiClassLabels = pd.read_csv('corgiClasslabels.csv', header=None)


# Path to frozen detection graph. This is the actual model that is used for the object detection.
frozenDir='inference_graph'
PATH_TO_FROZEN_GRAPH = os.path.join(frozenDir, 'frozen_inference_graph.pb')
#PATH_TO_FROZEN_GRAPH=r'C:\Users\alexs\OneDrive\Documents\GitHub\models\research\object_detection\inference_graph\frozen_inference_graph.pb'

# List of the strings that is used to add correct label for each box.
#PATH_TO_LABELS = os.path.join('data', 'mscoco_label_map.pbtxt')
PATH_TO_LABELS= r'C:\Users\alexs\OneDrive\Documents\GitHub\models\research\object_detection\training\labelmap.pbtxt'

#Loading in frozen model
detection_graph = tf.Graph()
with detection_graph.as_default():
  od_graph_def = tf.GraphDef()
  with tf.gfile.GFile(PATH_TO_FROZEN_GRAPH, 'rb') as fid:
    serialized_graph = fid.read()
    od_graph_def.ParseFromString(serialized_graph)
    tf.import_graph_def(od_graph_def, name='')

category_index = label_map_util.create_category_index_from_labelmap(PATH_TO_LABELS, use_display_name=True)

def load_image_into_numpy_array(image):
  (im_width, im_height) = image.size
  return np.array(image.getdata()).reshape(
      (im_height, im_width, 3)).astype(np.uint8)


# If you want to test the code with your images, just add path to the images to the TEST_IMAGE_PATHS.
PATH_TO_TEST_IMAGES_DIR = 'test_images'
TEST_IMAGE_PATHS = [ os.path.join(PATH_TO_TEST_IMAGES_DIR, 'corgi{}.jpg'.format(i)) for i in range(1, len(corgiClassLabels)+1) ]

# Size, in inches, of the output images.
IMAGE_SIZE = (12, 8)
occupied=0
empty=0
count=1
correct=0
def run_inference_for_single_image(image, graph):
  with graph.as_default():
    with tf.Session() as sess:
      # Get handles to input and output tensors
      ops = tf.get_default_graph().get_operations()
      all_tensor_names = {output.name for op in ops for output in op.outputs}
      tensor_dict = {}
      for key in [
          'num_detections', 'detection_boxes', 'detection_scores',
          'detection_classes', 'detection_masks'
      ]:
        tensor_name = key + ':0'
        if tensor_name in all_tensor_names:
          tensor_dict[key] = tf.get_default_graph().get_tensor_by_name(
              tensor_name)
      if 'detection_masks' in tensor_dict:
        # The following processing is only for single image
        detection_boxes = tf.squeeze(tensor_dict['detection_boxes'], [0])
        detection_masks = tf.squeeze(tensor_dict['detection_masks'], [0])
        # Reframe is required to translate mask from box coordinates to image coordinates and fit the image size.
        real_num_detection = tf.cast(tensor_dict['num_detections'][0], tf.int32)
        detection_boxes = tf.slice(detection_boxes, [0, 0], [real_num_detection, -1])
        detection_masks = tf.slice(detection_masks, [0, 0, 0], [real_num_detection, -1, -1])
        detection_masks_reframed = utils_ops.reframe_box_masks_to_image_masks(
            detection_masks, detection_boxes, image.shape[0], image.shape[1])
        detection_masks_reframed = tf.cast(
            tf.greater(detection_masks_reframed, 0.5), tf.uint8)
        # Follow the convention by adding back the batch dimension
        tensor_dict['detection_masks'] = tf.expand_dims(
            detection_masks_reframed, 0)
      image_tensor = tf.get_default_graph().get_tensor_by_name('image_tensor:0')

      # Run inference
      output_dict = sess.run(tensor_dict,
                             feed_dict={image_tensor: np.expand_dims(image, 0)})

      # all outputs are float32 numpy arrays, so convert types as appropriate
      output_dict['num_detections'] = int(output_dict['num_detections'][0])
      output_dict['detection_classes'] = output_dict[
          'detection_classes'][0].astype(np.uint8)
      output_dict['detection_boxes'] = output_dict['detection_boxes'][0]
      output_dict['detection_scores'] = output_dict['detection_scores'][0]
      if 'detection_masks' in output_dict:
        output_dict['detection_masks'] = output_dict['detection_masks'][0]
  return output_dict

for image_path in TEST_IMAGE_PATHS:
  image= cv2.imread(image_path)
  
  # Expand dimensions since the model expects images to have shape: [1, None, None, 3]
  image_np_expanded = np.expand_dims(image, axis=0)
  # Actual detection.
  output_dict = run_inference_for_single_image(image, detection_graph)

  

  # Visualization of the results of a detection.
  vis_util.visualize_boxes_and_labels_on_image_array(
      image,
      output_dict['detection_boxes'],
      output_dict['detection_classes'],
      output_dict['detection_scores'],
      category_index,
      instance_masks=output_dict.get('detection_masks'),
      use_normalized_coordinates=True,
      line_thickness=8)



  #Testing/output purposes
  predictedBreed='none'
  actualBreed= corgiClassLabels[0][count-1]
  if(output_dict['detection_scores'][0]>0.80 and output_dict['detection_classes'][0]==1):
    predictedBreed= 'cardigan'
  elif(output_dict['detection_scores'][0]>0.80 and output_dict['detection_classes'][0]==2):
    predictedBreed= 'pembroke'
  
  if(actualBreed==predictedBreed):
    correct+=1

  
  print("Corgi Image:",count)
  print("Predicted Breed:",predictedBreed," Actual Breed: ",actualBreed)
  count+=1

  cv2.imshow('object detector', image)
  cv2.waitKey(0)
  cv2.destroyAllWindows()

print("Number of predictions correct was: ", correct,"/",len(corgiClassLabels)," Percent Correct: ", correct/len(corgiClassLabels)*100)
