# Copyright 2017 The TensorFlow Authors All Rights Reserved.
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

"""Functions to read, decode and pre-process input data for the Model.
"""
import collections
import functools
import tensorflow as tf
from tensorflow.contrib import slim

import inception_preprocessing

# Tuple to store input data endpoints for the Model.
# It has following fields (tensors):
#    images: input images,
#      shape [batch_size x H x W x 3];
#    labels: ground truth label ids,
#      shape=[batch_size x seq_length];
#    labels_one_hot: labels in one-hot encoding,
#      shape [batch_size x seq_length x num_char_classes];
InputEndpoints = collections.namedtuple(
    'InputEndpoints', ['images', 'images_orig', 'labels', 'labels_one_hot'])

# A namedtuple to define a configuration for shuffled batch fetching.
#   num_batching_threads: A number of parallel threads to fetch data.
#   queue_capacity: a max number of elements in the batch shuffling queue.
#   min_after_dequeue: a min number elements in the queue after a dequeue, used
#     to ensure a level of mixing of elements.
ShuffleBatchConfig = collections.namedtuple('ShuffleBatchConfig', [
    'num_batching_threads', 'queue_capacity', 'min_after_dequeue'
])

DEFAULT_SHUFFLE_CONFIG = ShuffleBatchConfig(
    num_batching_threads=8, queue_capacity=3000, min_after_dequeue=1000)


def augment_image(image):
  """Augmentation the image with a random modification.

  Args:
    image: input Tensor image of rank 3, with the last dimension
           of size 3.

  Returns:
    Distorted Tensor image of the same shape.
  """
  with tf.variable_scope('AugmentImage'):
    height = image.get_shape().dims[0].value
    width = image.get_shape().dims[1].value

    # Random crop cut from the street sign image, resized to the same size.
    # Assures that the crop is covers at least 0.8 area of the input image.
    bbox_begin, bbox_size, _ = tf.image.sample_distorted_bounding_box(
        tf.shape(image),
        bounding_boxes=tf.zeros([0, 0, 4]),
        min_object_covered=0.8,
        aspect_ratio_range=[0.8, 1.2],
        area_range=[0.8, 1.0],
        use_image_if_no_bounding_boxes=True)
    distorted_image = tf.slice(image, bbox_begin, bbox_size)

    # Randomly chooses one of the 4 interpolation methods
    distorted_image = inception_preprocessing.apply_with_random_selector(
        distorted_image,
        lambda x, method: tf.image.resize_images(x, [height, width], method),
        num_cases=4)
    distorted_image.set_shape([height, width, 3])

    # Color distortion
    distorted_image = inception_preprocessing.apply_with_random_selector(
        distorted_image,
        functools.partial(
            inception_preprocessing.distort_color, fast_mode=False),
        num_cases=4)
    distorted_image = tf.clip_by_value(distorted_image, -1.5, 1.5)

  return distorted_image


def central_crop(image, crop_size):
  """Returns a central crop for the specified size of an image.

  Args:
    image: A tensor with shape [height, width, channels]
    crop_size: A tuple (crop_width, crop_height)

  Returns:
    A tensor of shape [crop_height, crop_width, channels].
  """
  with tf.variable_scope('CentralCrop'):
    target_width, target_height = crop_size
    image_height, image_width = tf.shape(image)[0], tf.shape(image)[1]
    assert_op1 = tf.Assert(
        tf.greater_equal(image_height, target_height),
        ['image_height < target_height', image_height, target_height])
    assert_op2 = tf.Assert(
        tf.greater_equal(image_width, target_width),
        ['image_width < target_width', image_width, target_width])
    with tf.control_dependencies([assert_op1, assert_op2]):
      offset_width = (image_width - target_width) / 2
      offset_height = (image_height - target_height) / 2
      return tf.image.crop_to_bounding_box(image, offset_height, offset_width,
                                           target_height, target_width)


def preprocess_image(image, augment=False, central_crop_size=None,
                     num_towers=4):
  """Normalizes image to have values in a narrow range around zero.

  Args:
    image: a [H x W x 3] uint8 tensor.
    augment: optional, if True do random image distortion.
    central_crop_size: A tuple (crop_width, crop_height).
    num_towers: optional, number of shots of the same image in the input image.

  Returns:
    A float32 tensor of shape [H x W x 3] with RGB values in the required
    range.
  """
  with tf.variable_scope('PreprocessImage'):
    image = tf.image.convert_image_dtype(image, dtype=tf.float32)
    if augment or central_crop_size:
      if num_towers == 1:
        images = [image]
      else:
        images = tf.split(value=image, num_or_size_splits=num_towers, axis=1)
      if central_crop_size:
        view_crop_size = (central_crop_size[0] / num_towers,
                          central_crop_size[1])
        images = [central_crop(img, view_crop_size) for img in images]
      if augment:
        images = [augment_image(img) for img in images]
      image = tf.concat(images, 1)

    image = tf.subtract(image, 0.5)
    image = tf.multiply(image, 2.5)

  return image


def get_data(dataset,
             batch_size,
             augment=False,
             central_crop_size=None,
             shuffle_config=None,
             shuffle=True):
  """Wraps calls to DatasetDataProviders and shuffle_batch.

  For more details about supported Dataset objects refer to datasets/fsns.py.

  Args:
    dataset: a slim.data.dataset.Dataset object.
    batch_size: number of samples per batch.
    augment: optional, if True does random image distortion.
    central_crop_size: A CharLogittuple (crop_width, crop_height).
    shuffle_config: A namedtuple ShuffleBatchConfig.
    shuffle: if True use data shuffling.

  Returns:

  """
  if not shuffle_config:
    shuffle_config = DEFAULT_SHUFFLE_CONFIG

  provider = slim.dataset_data_provider.DatasetDataProvider(
      dataset,
      shuffle=shuffle,
      common_queue_capacity=2 * batch_size,
      common_queue_min=batch_size)
  image_orig, label = provider.get(['image', 'label'])

  image = preprocess_image(
      image_orig, augment, central_crop_size, num_towers=dataset.num_of_views)
  label_one_hot = slim.one_hot_encoding(label, dataset.num_char_classes)

  images, images_orig, labels, labels_one_hot = (tf.train.shuffle_batch(
      [image, image_orig, label, label_one_hot],
      batch_size=batch_size,
      num_threads=shuffle_config.num_batching_threads,
      capacity=shuffle_config.queue_capacity,
      min_after_dequeue=shuffle_config.min_after_dequeue))

  return InputEndpoints(
      images=images,
      images_orig=images_orig,
      labels=labels,
      labels_one_hot=labels_one_hot)
