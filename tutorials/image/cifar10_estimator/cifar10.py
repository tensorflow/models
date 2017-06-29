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
NUM_CLASSES = 10


class Cifar10DataSet(object):
  """Cifar10 data set.

  Described by http://www.cs.toronto.edu/~kriz/cifar.html.
  """

  def __init__(self, data_dir):
    self.data_dir = data_dir

  def read_all_data(self, subset='train'):
    """Reads from data file and return images and labels in a numpy array."""
    if subset == 'train':
      filenames = [
          os.path.join(self.data_dir, 'data_batch_%d' % i)
          for i in xrange(1, 5)
      ]
    elif subset == 'validation':
      filenames = [os.path.join(self.data_dir, 'data_batch_5')]
    elif subset == 'eval':
      filenames = [os.path.join(self.data_dir, 'test_batch')]
    else:
      raise ValueError('Invalid data subset "%s"' % subset)

    inputs = []
    for filename in filenames:
      with tf.gfile.Open(filename, 'r') as f:
        inputs.append(cPickle.load(f))
    all_images = np.concatenate([each_input['data']
                                 for each_input in inputs]).astype(np.float32)
    all_labels = np.concatenate([each_input['labels'] for each_input in inputs])
    return all_images, all_labels

  @staticmethod
  def preprocess(image, is_training, distortion):
    with tf.name_scope('preprocess'):
      # Read image layout as flattened CHW.
      image = tf.reshape(image, [DEPTH, HEIGHT, WIDTH])
      # Convert to NHWC layout, compatible with TF image preprocessing APIs
      image = tf.transpose(image, [1, 2, 0])
      if is_training and distortion:
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
