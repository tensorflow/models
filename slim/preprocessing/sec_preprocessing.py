"""Secondary preprocessing."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import tensorflow as tf
import numpy as np

slim = tf.contrib.slim


def preprocess_image(image, output_height, output_width, is_training):
  """Preprocesses the given image.

  Args:
    image: A `Tensor` representing an image of arbitrary size.
    output_height: The height of the image after preprocessing.
    output_width: The width of the image after preprocessing.
    is_training: `True` if we're preprocessing the image for training and
      `False` otherwise.

  Returns:
    A secondary preprocessed images (an image list).
  """
# Transform the image to floats.
  image = tf.to_float(image)
  
# Rotation
  angle = [np.pi/2,np.pi,1.5*np.pi]    # rotate to 3 other orientations (4 ori in total)
  image_list = [image]    # initialize image_list; save orig image into image_list
  for ang in angle:
      image_single = tf.contrib.image.rotate(image,angle)
      image_list.append(image_single)
  return image_list
