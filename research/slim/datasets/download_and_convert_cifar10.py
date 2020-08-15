# Copyright 2016 The TensorFlow Authors. All Rights Reserved.
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
r"""Downloads and converts cifar10 data to TFRecords of TF-Example protos.

This module downloads the cifar10 data, uncompresses it, reads the files
that make up the cifar10 data and creates two TFRecord datasets: one for train
and one for test. Each TFRecord dataset is comprised of a set of TF-Example
protocol buffers, each of which contain a single image and label.

The script should take several minutes to run.

"""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os
import sys
import tarfile

import numpy as np
from six.moves import cPickle
from six.moves import urllib
import tensorflow.compat.v1 as tf

from datasets import dataset_utils

# The URL where the CIFAR data can be downloaded.
_DATA_URL = 'https://www.cs.toronto.edu/~kriz/cifar-10-python.tar.gz'

# The number of training files.
_NUM_TRAIN_FILES = 5

# The height and width of each image.
_IMAGE_SIZE = 32

# The names of the classes.
_CLASS_NAMES = [
    'airplane',
    'automobile',
    'bird',
    'cat',
    'deer',
    'dog',
    'frog',
    'horse',
    'ship',
    'truck',
]


def _add_to_tfrecord(filename, tfrecord_writer, offset=0):
  """Loads data from the cifar10 pickle files and writes files to a TFRecord.

  Args:
    filename: The filename of the cifar10 pickle file.
    tfrecord_writer: The TFRecord writer to use for writing.
    offset: An offset into the absolute number of images previously written.

  Returns:
    The new offset.
  """
  with tf.gfile.Open(filename, 'rb') as f:
    if sys.version_info < (3,):
      data = cPickle.load(f)
    else:
      data = cPickle.load(f, encoding='bytes')

  images = data[b'data']
  num_images = images.shape[0]

  images = images.reshape((num_images, 3, 32, 32))
  labels = data[b'labels']

  with tf.Graph().as_default():
    image_placeholder = tf.placeholder(dtype=tf.uint8)
    encoded_image = tf.image.encode_png(image_placeholder)

    with tf.Session('') as sess:

      for j in range(num_images):
        sys.stdout.write('\r>> Reading file [%s] image %d/%d' % (
            filename, offset + j + 1, offset + num_images))
        sys.stdout.flush()

        image = np.squeeze(images[j]).transpose((1, 2, 0))
        label = labels[j]

        png_string = sess.run(encoded_image,
                              feed_dict={image_placeholder: image})

        example = dataset_utils.image_to_tfexample(
            png_string, b'png', _IMAGE_SIZE, _IMAGE_SIZE, label)
        tfrecord_writer.write(example.SerializeToString())

  return offset + num_images


def _get_output_filename(dataset_dir, split_name):
  """Creates the output filename.

  Args:
    dataset_dir: The dataset directory where the dataset is stored.
    split_name: The name of the train/test split.

  Returns:
    An absolute file path.
  """
  return '%s/cifar10_%s.tfrecord' % (dataset_dir, split_name)


def _download_and_uncompress_dataset(dataset_dir):
  """Downloads cifar10 and uncompresses it locally.

  Args:
    dataset_dir: The directory where the temporary files are stored.
  """
  filename = _DATA_URL.split('/')[-1]
  filepath = os.path.join(dataset_dir, filename)

  if not os.path.exists(filepath):
    def _progress(count, block_size, total_size):
      sys.stdout.write('\r>> Downloading %s %.1f%%' % (
          filename, float(count * block_size) / float(total_size) * 100.0))
      sys.stdout.flush()
    filepath, _ = urllib.request.urlretrieve(_DATA_URL, filepath, _progress)
    print()
    statinfo = os.stat(filepath)
    print('Successfully downloaded', filename, statinfo.st_size, 'bytes.')
    tarfile.open(filepath, 'r:gz').extractall(dataset_dir)


def _clean_up_temporary_files(dataset_dir):
  """Removes temporary files used to create the dataset.

  Args:
    dataset_dir: The directory where the temporary files are stored.
  """
  filename = _DATA_URL.split('/')[-1]
  filepath = os.path.join(dataset_dir, filename)
  tf.gfile.Remove(filepath)

  tmp_dir = os.path.join(dataset_dir, 'cifar-10-batches-py')
  tf.gfile.DeleteRecursively(tmp_dir)


def run(dataset_dir):
  """Runs the download and conversion operation.

  Args:
    dataset_dir: The dataset directory where the dataset is stored.
  """
  if not tf.gfile.Exists(dataset_dir):
    tf.gfile.MakeDirs(dataset_dir)

  training_filename = _get_output_filename(dataset_dir, 'train')
  testing_filename = _get_output_filename(dataset_dir, 'test')

  if tf.gfile.Exists(training_filename) and tf.gfile.Exists(testing_filename):
    print('Dataset files already exist. Exiting without re-creating them.')
    return

  dataset_utils.download_and_uncompress_tarball(_DATA_URL, dataset_dir)

  # First, process the training data:
  with tf.python_io.TFRecordWriter(training_filename) as tfrecord_writer:
    offset = 0
    for i in range(_NUM_TRAIN_FILES):
      filename = os.path.join(dataset_dir,
                              'cifar-10-batches-py',
                              'data_batch_%d' % (i + 1))  # 1-indexed.
      offset = _add_to_tfrecord(filename, tfrecord_writer, offset)

  # Next, process the testing data:
  with tf.python_io.TFRecordWriter(testing_filename) as tfrecord_writer:
    filename = os.path.join(dataset_dir,
                            'cifar-10-batches-py',
                            'test_batch')
    _add_to_tfrecord(filename, tfrecord_writer)

  # Finally, write the labels file:
  labels_to_class_names = dict(zip(range(len(_CLASS_NAMES)), _CLASS_NAMES))
  dataset_utils.write_label_file(labels_to_class_names, dataset_dir)

  _clean_up_temporary_files(dataset_dir)
  print('\nFinished converting the Cifar10 dataset!')
