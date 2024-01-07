import os
import tensorflow as tf
from PIL import Image, ImageDraw, ImageFont
from six import BytesIO
import numpy as np
from object_detection.utils import label_map_util

def download_ckpt():
    os.system("rm /worker/BboxSuggestion/efficientdet_d0_coco17_tpu-32.tar.gz*")
    os.system("rm -r /worker/BboxSuggestion/research/object_detection/test_data/*")
    os.system("wget http://download.tensorflow.org/models/object_detection/tf2/20200711/efficientdet_d0_coco17_tpu-32.tar.gz")
    os.system("tar -xf efficientdet_d0_coco17_tpu-32.tar.gz")
    os.system("mv  efficientdet_d0_coco17_tpu-32/ /worker/BboxSuggestion/research/object_detection/test_data/")

def load_labelmap(configs, ):
    label_map_path = configs['eval_input_config'].label_map_path
    label_map = label_map_util.load_labelmap(label_map_path)
    categories = label_map_util.convert_label_map_to_categories(
        label_map,
        max_num_classes=label_map_util.get_max_label_map_index(label_map),
        use_display_name=True)
    category_index = label_map_util.create_category_index(categories)
    label_map_dict = label_map_util.get_label_map_dict(label_map, use_display_name=True)
    return category_index, label_map_dict

def replace(fpath, from_str, to_str):
  with open(fpath) as f:
    data_lines = f.read()
  data_lines = data_lines.replace(from_str, to_str)
  with open(fpath, mode="w") as f:
    f.write(data_lines)

def load_image_into_numpy_array(path):
  """Load an image from file into a numpy array.

  Puts image into numpy array to feed into tensorflow graph.
  Note that by convention we put it into a numpy array with shape
  (height, width, channels), where channels=3 for RGB.

  Args:
    path: the file path to the image

  Returns:
    uint8 numpy array with shape (img_height, img_width, 3)
  """
  img_data = tf.io.gfile.GFile(path, 'rb').read()
  image = Image.open(BytesIO(img_data))
  (im_width, im_height) = image.size
  return np.array(image.getdata()).reshape(
      (im_height, im_width, 3)).astype(np.uint8)

def get_keypoint_tuples(eval_config):
  """Return a tuple list of keypoint edges from the eval config.
  
  Args:
    eval_config: an eval config containing the keypoint edges
  
  Returns:
    a list of edge tuples, each in the format (start, end)
  """
  tuple_list = []
  kp_list = eval_config.keypoint_edge
  for edge in kp_list:
    tuple_list.append((edge.start, edge.end))
  return tuple_list