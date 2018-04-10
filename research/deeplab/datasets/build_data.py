# Copyright 2018 The TensorFlow Authors All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================

"""Contains common utility functions and classes for building dataset.

This script contains utility functions and classes to converts dataset to
TFRecord file format with Example protos.

The Example proto contains the following fields:

  image/encoded: encoded image content.
  image/filename: image filename.
  image/format: image file format.
  image/height: image height.
  image/width: image width.
  image/channels: image channels.
  image/segmentation/class/encoded: encoded semantic segmentation content.
  image/segmentation/class/format: semantic segmentation file format.
"""
import collections
import six
import tensorflow as tf

FLAGS = tf.app.flags.FLAGS

tf.app.flags.DEFINE_enum('image_format', 'png', ['jpg', 'jpeg', 'png'],
                         'Image format.')

tf.app.flags.DEFINE_enum('label_format', 'png', ['png'],
                         'Segmentation label format.')

# A map from image format to expected data format.
_IMAGE_FORMAT_MAP = {
    'jpg': 'jpeg',
    'jpeg': 'jpeg',
    'png': 'png',
}


class ImageReader(object):
  """Helper class that provides TensorFlow image coding utilities."""

  def __init__(self, image_format='jpeg', channels=3):
    """Class constructor.

    Args:
      image_format: Image format. Only 'jpeg', 'jpg', or 'png' are supported.
      channels: Image channels.
    """
    with tf.Graph().as_default():
      self._decode_data = tf.placeholder(dtype=tf.string)
      self._image_format = image_format
      self._session = tf.Session()
      if self._image_format in ('jpeg', 'jpg'):
        self._decode = tf.image.decode_jpeg(self._decode_data,
                                            channels=channels)
      elif self._image_format == 'png':
        self._decode = tf.image.decode_png(self._decode_data,
                                           channels=channels)

  def read_image_dims(self, image_data):
    """Reads the image dimensions.

    Args:
      image_data: string of image data.

    Returns:
      image_height and image_width.
    """
    image = self.decode_image(image_data)
    return image.shape[:2]

  def decode_image(self, image_data):
    """Decodes the image data string.

    Args:
      image_data: string of image data.

    Returns:
      Decoded image data.

    Raises:
      ValueError: Value of image channels not supported.
    """
    image = self._session.run(self._decode,
                              feed_dict={self._decode_data: image_data})
    if len(image.shape) != 3 or image.shape[2] not in (1, 3):
      raise ValueError('The image channels not supported.')

    return image


def _int64_list_feature(values):
  """Returns a TF-Feature of int64_list.

  Args:
    values: A scalar or list of values.

  Returns:
    A TF-Feature.
  """
  if not isinstance(values, collections.Iterable):
    values = [values]

  return tf.train.Feature(int64_list=tf.train.Int64List(value=values))


def _bytes_list_feature(values):
  """Returns a TF-Feature of bytes.

  Args:
    values: A string.

  Returns:
    A TF-Feature.
  """
  def norm2bytes(value):
      return value.encode() if isinstance(value, str) and six.PY3 else value

  return tf.train.Feature(bytes_list=tf.train.BytesList(value=[norm2bytes(values)]))


def image_seg_to_tfexample(image_data, filename, height, width, seg_data):
  """Converts one image/segmentation pair to tf example.

  Args:
    image_data: string of image data.
    filename: image filename.
    height: image height.
    width: image width.
    seg_data: string of semantic segmentation data.

  Returns:
    tf example of one image/segmentation pair.
  """
  return tf.train.Example(features=tf.train.Features(feature={
      'image/encoded': _bytes_list_feature(image_data),
      'image/filename': _bytes_list_feature(filename),
      'image/format': _bytes_list_feature(
          _IMAGE_FORMAT_MAP[FLAGS.image_format]),
      'image/height': _int64_list_feature(height),
      'image/width': _int64_list_feature(width),
      'image/channels': _int64_list_feature(3),
      'image/segmentation/class/encoded': (
          _bytes_list_feature(seg_data)),
      'image/segmentation/class/format': _bytes_list_feature(
          FLAGS.label_format),
  }))
