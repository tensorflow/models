# Copyright 2015 The TensorFlow Authors. All Rights Reserved.
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

"""Routine for decoding the CIFAR-10 binary file format."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os

from six.moves import xrange  # pylint: disable=redefined-builtin
import tensorflow as tf


# Global constants describing the CIFAR-10 data set.
NUM_CLASSES = 10


INPUT_IMAGE_HEIGHT = 32
INPUT_IMAGE_WIDTH = 32
RESIZED_IMAGE_HEIGHT = 24
RESIZED_IMAGE_WIDTH = 24

def read_cifar10(filenames, mini_batch_size, do_data_augmentation, drop_remainder):
    label_bytes = 1  # 2 for CIFAR-100
    depth = 3
    image_bytes = INPUT_IMAGE_HEIGHT * INPUT_IMAGE_WIDTH * depth
    # Every record consists of a label followed by the image, with a
    # fixed number of bytes for each.
    record_bytes = label_bytes + image_bytes
    record_dataset = tf.data.FixedLengthRecordDataset(filenames, record_bytes).batch(mini_batch_size,drop_remainder=drop_remainder).shuffle(buffer_size=10000)
    #label = tf.strided_slice(record_bytes, [0], [label_bytes])
    iter = record_dataset.make_one_shot_iterator()
    next = iter.get_next()
    record_bytes = tf.cast(tf.decode_raw(next, tf.uint8), tf.float32)
    batch_size = tf.shape(record_bytes)[0]
    batched_labels = tf.reshape(tf.cast(
            tf.strided_slice(record_bytes, [0, 0], [batch_size, label_bytes]), tf.int32), [batch_size])
    depth_major = tf.reshape(
        tf.strided_slice(record_bytes, [0, label_bytes],
                        [batch_size, label_bytes + image_bytes]),
        [batch_size, depth, INPUT_IMAGE_HEIGHT, INPUT_IMAGE_WIDTH])
    # Convert from [depth, height, width] to [height, width, depth].
    batched_original_images = tf.transpose(depth_major, [0, 2, 3, 1])
    if do_data_augmentation:
      batched_original_images = tf.map_fn(lambda frame: tf.random_crop(frame, [RESIZED_IMAGE_HEIGHT, RESIZED_IMAGE_WIDTH, 3]), batched_original_images)
      batched_original_images = tf.map_fn(lambda frame: tf.image.random_flip_left_right(frame), batched_original_images)
      batched_original_images = tf.map_fn(lambda frame: tf.image.random_brightness(frame, max_delta=63), batched_original_images)
      batched_original_images = tf.map_fn(lambda frame: tf.image.random_contrast(frame, lower=0.2, upper=1.8), batched_original_images)
    else:
      batched_original_images = tf.image.resize_image_with_crop_or_pad(batched_original_images, RESIZED_IMAGE_HEIGHT, RESIZED_IMAGE_WIDTH)

    batched_images =  tf.map_fn(lambda frame: tf.image.per_image_standardization(frame), batched_original_images)

    return batched_images, batched_labels
