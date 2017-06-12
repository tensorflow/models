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

"""
Contains code for loading the MNIST data.

"""

import os
import gzip
import tensorflow as tf
import numpy as np
from six.moves import urllib
from tensorflow.python.platform import gfile
from base import config

IMAGE_SIZE = 28
NUM_CHANNELS = 1
PIXEL_DEPTH = 255
NUM_CLASSES = 10
NUM_TRAINING_EXAMPLES = 60000
NUM_TEST_EXAMPLES = 10000

def GetMnistConfig():
  mnist_config = config.LearningParams()
  mnist_config.SetValue('number_of_classes', NUM_CLASSES)
  mnist_config.SetValue('number_of_examples', NUM_TRAINING_EXAMPLES)
  mnist_config.SetValue('number_of_test_examples', NUM_TEST_EXAMPLES)
  mnist_config.SetValue('base_data_dir', 'mnist/datasets/')
  return mnist_config

SOURCE_URL = 'http://yann.lecun.com/exdb/mnist/'

def maybe_download(work_directory, filename):
  """Download the data from source url, unless it's already here.

  Args:
      filename: string, name of the file in the directory.
      work_directory: string, path to working directory.
      source_url: url to download from if file doesn't exist.

  Returns:
      Path to resulting file.
  """
  if not gfile.Exists(work_directory):
    gfile.MakeDirs(work_directory)
  filepath = os.path.join(work_directory, filename)
  if not gfile.Exists(filepath):
    temp_file_name, _ = urllib.request.urlretrieve(SOURCE_URL + filename, None)
    gfile.Copy(temp_file_name, filepath)
    with gfile.GFile(filepath) as f:
      size = f.size()
    print('Successfully downloaded', filename, size, 'bytes.')
  return filepath


def extract_data(filename, num_images):
  """Extract the images into a 4D tensor [image index, y, x, channels].

  Values are rescaled from [0, 255] down to [-0.5, 0.5].
  """
  print('Extracting', filename)
  with gzip.open(filename) as bytestream:
    bytestream.read(16)
    buf = bytestream.read(IMAGE_SIZE * IMAGE_SIZE * num_images * NUM_CHANNELS)
    data = np.frombuffer(buf, dtype=np.uint8).astype(np.float32)
    data = (data - (PIXEL_DEPTH / 2.0)) / PIXEL_DEPTH
    data = data.reshape(num_images, IMAGE_SIZE, IMAGE_SIZE, NUM_CHANNELS)
    return data

def extract_labels(filename, num_images):
  """Extract the labels into a vector of int64 label IDs."""
  print('Extracting', filename)
  with gzip.open(filename) as bytestream:
    bytestream.read(8)
    buf = bytestream.read(1 * num_images)
    labels = np.frombuffer(buf, dtype=np.uint8).astype(np.int64)
  return labels

class MNIST_Input(object):

  def __init__(self, config, split_name):
    # Get the data.
    if split_name == 'train':
      data_filename = maybe_download(config.base_data_dir,
                                     'train-images-idx3-ubyte.gz')
      labels_filename = maybe_download(config.base_data_dir,
                                       'train-labels-idx1-ubyte.gz')
      self.size = config.GetValue('number_of_examples', NUM_TRAINING_EXAMPLES)
    elif split_name == 'test':
      data_filename = maybe_download(config.base_data_dir,
                                     't10k-images-idx3-ubyte.gz')
      labels_filename = maybe_download(config.base_data_dir,
                                       't10k-labels-idx1-ubyte.gz')
      self.size = config.GetValue('number_of_test_examples', NUM_TEST_EXAMPLES)
    else:
      raise ValueError('split %s not recognized', split_name)

    # Extract it into numpy arrays.
    self.data = extract_data(data_filename, self.size)
    self.data /= np.absolute(self.data).max()   # make between -1 and 1
    self.labels = extract_labels(labels_filename, self.size)
    self.num_classes = config.GetValue('number_of_classes', NUM_CLASSES)


  def NormalizeData(self, input, dim):
    """ Assuming each entry lies in [-1, 1], normalize so they lie on a unit
    circle using the mapping: f(x) = (cos(x * pi/2), sin(x * pi/2)) """
    input_pi_by_2 = input * np.pi/2
    return tf.concat([tf.cos(input_pi_by_2), tf.sin(input_pi_by_2)], dim)

  def Shuffle(self, indices, current_index):
    indices_update = tf.assign(indices, tf.random_shuffle(indices))
    current_index_update = tf.assign(current_index, 0)
    return current_index_update, indices_update

  def ProvideData(self, batch_size):
    """Provides batches of MNIST digits.

    Args:
      batch_size: the number of images in each batch.

    Returns:
      the images, a tensor of size [batch_size, 28, 28, 1]
      and labels, a tensor of size [batch_size, NUM_CLASSES]

    """
    all_examples = tf.constant(self.data)
    all_labels = tf.constant(self.labels)
    all_size = tf.constant(self.size)
    current_index = tf.Variable(self.size, dtype=tf.int32)
    indices = tf.Variable(tf.constant(np.arange(self.size)),
                          dtype=tf.int64)

    start, ind = tf.cond(current_index + batch_size < all_size,
                         lambda: (current_index, indices),
                         lambda: self.Shuffle(indices, current_index))

    batch = tf.slice(ind, [start], [batch_size])

    with tf.control_dependencies([batch]):
      current_index_next = current_index.assign_add(batch_size)
      with tf.control_dependencies([current_index_next]):
        images = tf.gather(all_examples, batch)
        labels = tf.gather(all_labels, batch)

    return images, self.NormalizeData(images, 3), tf.one_hot(labels, self.num_classes)
