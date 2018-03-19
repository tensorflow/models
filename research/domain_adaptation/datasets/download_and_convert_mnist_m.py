# Copyright 2017 Google Inc.
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

r"""Downloads and converts MNIST-M data to TFRecords of TF-Example protos.

This module downloads the MNIST-M data, uncompresses it, reads the files
that make up the MNIST-M data and creates two TFRecord datasets: one for train
and one for test. Each TFRecord dataset is comprised of a set of TF-Example
protocol buffers, each of which contain a single image and label.

The script should take about a minute to run.

"""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os
import random
import sys

# Dependency imports
import numpy as np
from six.moves import urllib
import tensorflow as tf

from slim.datasets import dataset_utils

tf.app.flags.DEFINE_string(
    'dataset_dir', None,
    'The directory where the output TFRecords and temporary files are saved.')

FLAGS = tf.app.flags.FLAGS

_IMAGE_SIZE = 32
_NUM_CHANNELS = 3

# The number of images in the training set.
_NUM_TRAIN_SAMPLES = 59001

# The number of images to be kept from the training set for the validation set.
_NUM_VALIDATION = 1000

# The number of images in the test set.
_NUM_TEST_SAMPLES = 9001

# Seed for repeatability.
_RANDOM_SEED = 0

# The names of the classes.
_CLASS_NAMES = [
    'zero',
    'one',
    'two',
    'three',
    'four',
    'five',
    'size',
    'seven',
    'eight',
    'nine',
]


class ImageReader(object):
  """Helper class that provides TensorFlow image coding utilities."""

  def __init__(self):
    # Initializes function that decodes RGB PNG data.
    self._decode_png_data = tf.placeholder(dtype=tf.string)
    self._decode_png = tf.image.decode_png(self._decode_png_data, channels=3)

  def read_image_dims(self, sess, image_data):
    image = self.decode_png(sess, image_data)
    return image.shape[0], image.shape[1]

  def decode_png(self, sess, image_data):
    image = sess.run(
        self._decode_png, feed_dict={self._decode_png_data: image_data})
    assert len(image.shape) == 3
    assert image.shape[2] == 3
    return image


def _convert_dataset(split_name, filenames, filename_to_class_id, dataset_dir):
  """Converts the given filenames to a TFRecord dataset.

  Args:
    split_name: The name of the dataset, either 'train' or 'valid'.
    filenames: A list of absolute paths to png images.
    filename_to_class_id: A dictionary from filenames (strings) to class ids
      (integers).
    dataset_dir: The directory where the converted datasets are stored.
  """
  print('Converting the {} split.'.format(split_name))
  # Train and validation splits are both in the train directory.
  if split_name in ['train', 'valid']:
    png_directory = os.path.join(dataset_dir, 'mnist_m', 'mnist_m_train')
  elif split_name == 'test':
    png_directory = os.path.join(dataset_dir, 'mnist_m', 'mnist_m_test')

  with tf.Graph().as_default():
    image_reader = ImageReader()

    with tf.Session('') as sess:
      output_filename = _get_output_filename(dataset_dir, split_name)

      with tf.python_io.TFRecordWriter(output_filename) as tfrecord_writer:
        for filename in filenames:
          # Read the filename:
          image_data = tf.gfile.FastGFile(
              os.path.join(png_directory, filename), 'r').read()
          height, width = image_reader.read_image_dims(sess, image_data)

          class_id = filename_to_class_id[filename]
          example = dataset_utils.image_to_tfexample(image_data, 'png', height,
                                                     width, class_id)
          tfrecord_writer.write(example.SerializeToString())

  sys.stdout.write('\n')
  sys.stdout.flush()


def _extract_labels(label_filename):
  """Extract the labels into a dict of filenames to int labels.

  Args:
    labels_filename: The filename of the MNIST-M labels.

  Returns:
    A dictionary of filenames to int labels.
  """
  print('Extracting labels from: ', label_filename)
  label_file = tf.gfile.FastGFile(label_filename, 'r').readlines()
  label_lines = [line.rstrip('\n').split() for line in label_file]
  labels = {}
  for line in label_lines:
    assert len(line) == 2
    labels[line[0]] = int(line[1])
  return labels


def _get_output_filename(dataset_dir, split_name):
  """Creates the output filename.

  Args:
    dataset_dir: The directory where the temporary files are stored.
    split_name: The name of the train/test split.

  Returns:
    An absolute file path.
  """
  return '%s/mnist_m_%s.tfrecord' % (dataset_dir, split_name)


def _get_filenames(dataset_dir):
  """Returns a list of filenames and inferred class names.

  Args:
    dataset_dir: A directory containing a set PNG encoded MNIST-M images.

  Returns:
    A list of image file paths, relative to `dataset_dir`.
  """
  photo_filenames = []
  for filename in os.listdir(dataset_dir):
    photo_filenames.append(filename)
  return photo_filenames


def run(dataset_dir):
  """Runs the download and conversion operation.

  Args:
    dataset_dir: The dataset directory where the dataset is stored.
  """
  if not tf.gfile.Exists(dataset_dir):
    tf.gfile.MakeDirs(dataset_dir)

  train_filename = _get_output_filename(dataset_dir, 'train')
  testing_filename = _get_output_filename(dataset_dir, 'test')

  if tf.gfile.Exists(train_filename) and tf.gfile.Exists(testing_filename):
    print('Dataset files already exist. Exiting without re-creating them.')
    return

  # TODO(konstantinos): Add download and cleanup functionality

  train_validation_filenames = _get_filenames(
      os.path.join(dataset_dir, 'mnist_m', 'mnist_m_train'))
  test_filenames = _get_filenames(
      os.path.join(dataset_dir, 'mnist_m', 'mnist_m_test'))

  # Divide into train and validation:
  random.seed(_RANDOM_SEED)
  random.shuffle(train_validation_filenames)
  train_filenames = train_validation_filenames[_NUM_VALIDATION:]
  validation_filenames = train_validation_filenames[:_NUM_VALIDATION]

  train_validation_filenames_to_class_ids = _extract_labels(
      os.path.join(dataset_dir, 'mnist_m', 'mnist_m_train_labels.txt'))
  test_filenames_to_class_ids = _extract_labels(
      os.path.join(dataset_dir, 'mnist_m', 'mnist_m_test_labels.txt'))

  # Convert the train, validation, and test sets.
  _convert_dataset('train', train_filenames,
                   train_validation_filenames_to_class_ids, dataset_dir)
  _convert_dataset('valid', validation_filenames,
                   train_validation_filenames_to_class_ids, dataset_dir)
  _convert_dataset('test', test_filenames, test_filenames_to_class_ids,
                   dataset_dir)

  # Finally, write the labels file:
  labels_to_class_names = dict(zip(range(len(_CLASS_NAMES)), _CLASS_NAMES))
  dataset_utils.write_label_file(labels_to_class_names, dataset_dir)

  print('\nFinished converting the MNIST-M dataset!')


def main(_):
  assert FLAGS.dataset_dir
  run(FLAGS.dataset_dir)


if __name__ == '__main__':
  tf.app.run()
