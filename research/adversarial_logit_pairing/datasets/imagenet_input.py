# Copyright 2018 Google Inc. All Rights Reserved.
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

"""Imagenet input."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os
from absl import flags
import tensorflow as tf

FLAGS = flags.FLAGS


flags.DEFINE_string('imagenet_data_dir', None,
                    'Directory with Imagenet dataset in TFRecord format.')


def _decode_and_random_crop(image_buffer, bbox, image_size):
  """Randomly crops image and then scales to target size."""
  with tf.name_scope('distorted_bounding_box_crop',
                     values=[image_buffer, bbox]):
    sample_distorted_bounding_box = tf.image.sample_distorted_bounding_box(
        tf.image.extract_jpeg_shape(image_buffer),
        bounding_boxes=bbox,
        min_object_covered=0.1,
        aspect_ratio_range=[0.75, 1.33],
        area_range=[0.08, 1.0],
        max_attempts=10,
        use_image_if_no_bounding_boxes=True)
    bbox_begin, bbox_size, _ = sample_distorted_bounding_box

    # Crop the image to the specified bounding box.
    offset_y, offset_x, _ = tf.unstack(bbox_begin)
    target_height, target_width, _ = tf.unstack(bbox_size)
    crop_window = tf.stack([offset_y, offset_x, target_height, target_width])
    image = tf.image.decode_and_crop_jpeg(image_buffer, crop_window, channels=3)
    image = tf.image.convert_image_dtype(
        image, dtype=tf.float32)

    image = tf.image.resize_bicubic([image],
                                    [image_size, image_size])[0]

    return image


def _decode_and_center_crop(image_buffer, image_size):
  """Crops to center of image with padding then scales to target size."""
  shape = tf.image.extract_jpeg_shape(image_buffer)
  image_height = shape[0]
  image_width = shape[1]

  padded_center_crop_size = tf.cast(
      0.875 * tf.cast(tf.minimum(image_height, image_width), tf.float32),
      tf.int32)

  offset_height = ((image_height - padded_center_crop_size) + 1) // 2
  offset_width = ((image_width - padded_center_crop_size) + 1) // 2
  crop_window = tf.stack([offset_height, offset_width,
                          padded_center_crop_size, padded_center_crop_size])
  image = tf.image.decode_and_crop_jpeg(image_buffer, crop_window, channels=3)
  image = tf.image.convert_image_dtype(
      image, dtype=tf.float32)

  image = tf.image.resize_bicubic([image],
                                  [image_size, image_size])[0]

  return image


def _normalize(image):
  """Rescale image to [-1, 1] range."""
  return tf.multiply(tf.subtract(image, 0.5), 2.0)


def image_preprocessing(image_buffer, bbox, image_size, is_training):
  """Does image decoding and preprocessing.

  Args:
    image_buffer: string tensor with encoded image.
    bbox: bounding box of the object at the image.
    image_size: image size.
    is_training: whether to do training or eval preprocessing.

  Returns:
    Tensor with the image.
  """
  if is_training:
    image = _decode_and_random_crop(image_buffer, bbox, image_size)
    image = _normalize(image)
    image = tf.image.random_flip_left_right(image)
  else:
    image = _decode_and_center_crop(image_buffer, image_size)
    image = _normalize(image)
  image = tf.reshape(image, [image_size, image_size, 3])
  return image


def imagenet_parser(value, image_size, is_training):
  """Parse an ImageNet record from a serialized string Tensor.

  Args:
    value: encoded example.
    image_size: size of the output image.
    is_training: if True then do training preprocessing,
      otherwise do eval preprocessing.

  Returns:
    image: tensor with the image.
    label: true label of the image.
  """
  keys_to_features = {
      'image/encoded':
          tf.FixedLenFeature((), tf.string, ''),
      'image/format':
          tf.FixedLenFeature((), tf.string, 'jpeg'),
      'image/class/label':
          tf.FixedLenFeature([], tf.int64, -1),
      'image/class/text':
          tf.FixedLenFeature([], tf.string, ''),
      'image/object/bbox/xmin':
          tf.VarLenFeature(dtype=tf.float32),
      'image/object/bbox/ymin':
          tf.VarLenFeature(dtype=tf.float32),
      'image/object/bbox/xmax':
          tf.VarLenFeature(dtype=tf.float32),
      'image/object/bbox/ymax':
          tf.VarLenFeature(dtype=tf.float32),
      'image/object/class/label':
          tf.VarLenFeature(dtype=tf.int64),
  }

  parsed = tf.parse_single_example(value, keys_to_features)

  image_buffer = tf.reshape(parsed['image/encoded'], shape=[])

  xmin = tf.expand_dims(parsed['image/object/bbox/xmin'].values, 0)
  ymin = tf.expand_dims(parsed['image/object/bbox/ymin'].values, 0)
  xmax = tf.expand_dims(parsed['image/object/bbox/xmax'].values, 0)
  ymax = tf.expand_dims(parsed['image/object/bbox/ymax'].values, 0)
  # Note that ordering is (y, x)
  bbox = tf.concat([ymin, xmin, ymax, xmax], 0)
  # Force the variable number of bounding boxes into the shape
  # [1, num_boxes, coords].
  bbox = tf.expand_dims(bbox, 0)
  bbox = tf.transpose(bbox, [0, 2, 1])

  image = image_preprocessing(
      image_buffer=image_buffer,
      bbox=bbox,
      image_size=image_size,
      is_training=is_training
  )

  # Labels are in [1, 1000] range
  label = tf.cast(
      tf.reshape(parsed['image/class/label'], shape=[]), dtype=tf.int32)

  return image, label


def imagenet_input(split, batch_size, image_size, is_training):
  """Returns ImageNet dataset.

  Args:
    split: name of the split, "train" or "validation".
    batch_size: size of the minibatch.
    image_size: size of the one side of the image. Output images will be
      resized to square shape image_size*image_size.
    is_training: if True then training preprocessing is done, otherwise eval
      preprocessing is done.

  Raises:
    ValueError: if name of the split is incorrect.

  Returns:
    Instance of tf.data.Dataset with the dataset.
  """
  if split.lower().startswith('train'):
    file_pattern = os.path.join(FLAGS.imagenet_data_dir, 'train-*')
  elif split.lower().startswith('validation'):
    file_pattern = os.path.join(FLAGS.imagenet_data_dir, 'validation-*')
  else:
    raise ValueError('Invalid split: %s' % split)

  dataset = tf.data.Dataset.list_files(file_pattern, shuffle=is_training)

  if is_training:
    dataset = dataset.repeat()

  def fetch_dataset(filename):
    return tf.data.TFRecordDataset(filename, buffer_size=8*1024*1024)

  # Read the data from disk in parallel
  dataset = dataset.apply(
      tf.contrib.data.parallel_interleave(
          fetch_dataset, cycle_length=4, sloppy=True))
  dataset = dataset.shuffle(1024)

  # Parse, preprocess, and batch the data in parallel
  dataset = dataset.apply(
      tf.contrib.data.map_and_batch(
          lambda value: imagenet_parser(value, image_size, is_training),
          batch_size=batch_size,
          num_parallel_batches=4,
          drop_remainder=True))

  def set_shapes(images, labels):
    """Statically set the batch_size dimension."""
    images.set_shape(images.get_shape().merge_with(
        tf.TensorShape([batch_size, None, None, None])))
    labels.set_shape(labels.get_shape().merge_with(
        tf.TensorShape([batch_size])))
    return images, labels

  # Assign static batch size dimension
  dataset = dataset.map(set_shapes)

  # Prefetch overlaps in-feed with training
  dataset = dataset.prefetch(tf.contrib.data.AUTOTUNE)
  return dataset


def num_examples_per_epoch(split):
  """Returns the number of examples in the data set.

  Args:
    split: name of the split, "train" or "validation".

  Raises:
    ValueError: if split name is incorrect.

  Returns:
    Number of example in the split.
  """
  if split.lower().startswith('train'):
    return 1281167
  elif split.lower().startswith('validation'):
    return 50000
  else:
    raise ValueError('Invalid split: %s' % split)
