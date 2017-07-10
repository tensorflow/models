# Copyright 2017 The TensorFlow Authors. All Rights Reserved.
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
"""CIFAR-10 data set.

See http://www.cs.toronto.edu/~kriz/cifar.html.
"""
import cPickle
import os

import numpy as np
from six.moves import xrange  # pylint: disable=redefined-builtin

import tensorflow as tf

HEIGHT = 32
WIDTH = 32
DEPTH = 3

class Cifar10DataSet(object):
  """Cifar10 data set.

  Described by http://www.cs.toronto.edu/~kriz/cifar.html.
  """

  def __init__(self, data_dir, subset='train', use_distortion=True):
    self.data_dir = data_dir
    self.subset = subset
    self.use_distortion = use_distortion
 
  def get_filenames(self):
    if self.subset == 'train':
      return [
          os.path.join(self.data_dir, 'data_batch_%d.bin' % i)
          for i in xrange(1, 5)
      ]
    elif self.subset == 'validation':
      return [os.path.join(self.data_dir, 'data_batch_5.bin')]
    elif self.subset == 'eval':
      return [os.path.join(self.data_dir, 'test_batch.bin')]
    else:
      raise ValueError('Invalid data subset "%s"' % self.subset)

  def make_batch(self, batch_size):
    """Read the images and labels from 'filenames'."""
    filenames = self.get_filenames()
    record_bytes = (32 * 32 * 3) + 1
    # Repeat infinitely.
    dataset = tf.contrib.data.FixedLengthRecordDataset(filenames,
                                                       record_bytes).repeat()
    # Parse records.
    dataset = dataset.map(self.parser, num_threads=batch_size,
      output_buffer_size=2 * batch_size)
    # Potentially shuffle records.
    if self.subset == 'train':
      min_queue_examples = int(
          Cifar10DataSet.num_examples_per_epoch(self.subset) * 0.4)
      # Ensure that the capacity is sufficiently large to provide good random
      # shuffling.
      dataset = dataset.shuffle(buffer_size=min_queue_examples + 3 * batch_size)
    # Batch it up.
    dataset = dataset.batch(batch_size)
    iterator = dataset.make_one_shot_iterator()
    image_batch, label_batch = iterator.get_next()
    return image_batch, label_batch

  def parser(self, value):
    """Parse a Cifar10 record from value.

    Output images are in [height, width, depth] layout.
    """
    # Dimensions of the images in the CIFAR-10 dataset.
    # See http://www.cs.toronto.edu/~kriz/cifar.html for a description of the
    # input format.
    label_bytes = 1
    image_bytes = HEIGHT * WIDTH * DEPTH
    # Every record consists of a label followed by the image, with a
    # fixed number of bytes for each.
    record_bytes = label_bytes + image_bytes

    # Convert from a string to a vector of uint8 that is record_bytes long.
    record_as_bytes = tf.decode_raw(value, tf.uint8)

    # The first bytes represent the label, which we convert from
    # uint8->int32.
    label = tf.cast(
        tf.strided_slice(record_as_bytes, [0], [label_bytes]), tf.int32)

    label.set_shape([1])

    # The remaining bytes after the label represent the image, which
    # we reshape from [depth * height * width] to [depth, height, width].
    depth_major = tf.reshape(
        tf.strided_slice(record_as_bytes, [label_bytes], [record_bytes]),
        [3, 32, 32])
    # Convert from [depth, height, width] to [height, width, depth].
    # This puts data in a compatible layout with TF image preprocessing APIs.
    image = tf.transpose(depth_major, [1, 2, 0])

    # Do custom preprocessing here.
    image = self.preprocess(image)

    return image, label

  def preprocess(self, image):
    """Preprocess a single image in [height, width, depth] layout."""
    if self.subset == 'train' and self.use_distortion:
      # Pad 4 pixels on each dimension of feature map, done in mini-batch
      image = tf.image.resize_image_with_crop_or_pad(image, 40, 40)
      image = tf.random_crop(image, [HEIGHT, WIDTH, DEPTH])
      image = tf.image.random_flip_left_right(image)
    return image

  @staticmethod
  def num_examples_per_epoch(subset='train'):
    if subset == 'train':
      return 45000
    elif subset == 'validation':
      return 5000
    elif subset == 'eval':
      return 10000
    else:
      raise ValueError('Invalid data subset "%s"' % subset)
