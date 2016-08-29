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
r"""Downloads and converts MNIST data to TFRecords of TF-Example protos.

This module downloads the MNIST data, uncompresses it, reads the files
that make up the MNIST data and creates two TFRecord datasets: one for train
and one for test. Each TFRecord dataset is comprised of a set of TF-Example
protocol buffers, each of which contain a single image and label.

The script should take about a minute to run.

"""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import gzip
import os
import sys

import numpy as np
from six.moves import urllib
import tensorflow as tf

<<<<<<< HEAD
from datasets import dataset_utils
=======
tf.app.flags.DEFINE_string(
    'dataset_dir',
    None,
    'The directory where the output TFRecords and temporary files are saved.')

FLAGS = tf.app.flags.FLAGS
>>>>>>> 0af5999e5e6e3147cea5a5d136ff7546a9957939

# The URLs where the MNIST data can be downloaded.
_DATA_URL = 'http://yann.lecun.com/exdb/mnist/'
_TRAIN_DATA_FILENAME = 'train-images-idx3-ubyte.gz'
_TRAIN_LABELS_FILENAME = 'train-labels-idx1-ubyte.gz'
_TEST_DATA_FILENAME = 't10k-images-idx3-ubyte.gz'
_TEST_LABELS_FILENAME = 't10k-labels-idx1-ubyte.gz'

_IMAGE_SIZE = 28
_NUM_CHANNELS = 1

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


def _int64_feature(values):
  """Returns a TF-Feature of int64s.

  Args:
    values: A scalar or list of values.

  Returns:
    a TF-Feature.
  """
  if not isinstance(values, (tuple, list)):
    values = [values]
  return tf.train.Feature(int64_list=tf.train.Int64List(value=values))


def _bytes_feature(values):
  """Returns a TF-Feature of bytes.

  Args:
    values: A string.

  Returns:
    a TF-Feature.
  """
  return tf.train.Feature(bytes_list=tf.train.BytesList(value=[values]))


def _image_to_tfexample(image_data, image_format, height, width, class_id):
  return tf.train.Example(features=tf.train.Features(feature={
      'image/encoded': _bytes_feature(image_data),
      'image/format': _bytes_feature(image_format),
      'image/class/label': _int64_feature(class_id),
      'image/height': _int64_feature(height),
      'image/width': _int64_feature(width),
  }))


def _extract_images(filename, num_images):
  """Extract the images into a numpy array.

  Args:
    filename: The path to an MNIST images file.
    num_images: The number of images in the file.

  Returns:
    A numpy array of shape [number_of_images, height, width, channels].
  """
  print('Extracting images from: ', filename)
  with gzip.open(filename) as bytestream:
    bytestream.read(16)
    buf = bytestream.read(
        _IMAGE_SIZE * _IMAGE_SIZE * num_images * _NUM_CHANNELS)
    data = np.frombuffer(buf, dtype=np.uint8)
    data = data.reshape(num_images, _IMAGE_SIZE, _IMAGE_SIZE, _NUM_CHANNELS)
  return data


def _write_label_file(labels_to_class_names, dataset_dir,
                      filename='labels.txt'):
  """Writes a file with the list of class names.

  Args:
    labels_to_class_names: A map of (integer) labels to class names.
    dataset_dir: The directory in which the labels file should be written.
    filename: The filename where the class names are written.
  """
  labels_filename = os.path.join(dataset_dir, filename)
  with tf.gfile.Open(labels_filename, 'w') as f:
    for label in labels_to_class_names:
      class_name = labels_to_class_names[label]
      f.write('%d:%s\n' % (label, class_name))


def _extract_labels(filename, num_labels):
  """Extract the labels into a vector of int64 label IDs.

  Args:
    filename: The path to an MNIST labels file.
    num_labels: The number of labels in the file.

  Returns:
    A numpy array of shape [number_of_labels]
  """
  print('Extracting labels from: ', filename)
  with gzip.open(filename) as bytestream:
    bytestream.read(8)
    buf = bytestream.read(1 * num_labels)
    labels = np.frombuffer(buf, dtype=np.uint8).astype(np.int64)
  return labels


def _add_to_tfrecord(data_filename, labels_filename, num_images,
                     tfrecord_writer):
  """Loads data from the binary MNIST files and writes files to a TFRecord.

  Args:
    data_filename: The filename of the MNIST images.
    labels_filename: The filename of the MNIST labels.
    num_images: The number of images in the dataset.
    tfrecord_writer: The TFRecord writer to use for writing.
  """
  images = _extract_images(data_filename, num_images)
  labels = _extract_labels(labels_filename, num_images)

  shape = (_IMAGE_SIZE, _IMAGE_SIZE, _NUM_CHANNELS)
  with tf.Graph().as_default():
    image = tf.placeholder(dtype=tf.uint8, shape=shape)
    encoded_png = tf.image.encode_png(image)

    with tf.Session('') as sess:
      for j in range(num_images):
        sys.stdout.write('\r>> Converting image %d/%d' % (j + 1, num_images))
        sys.stdout.flush()

        png_string = sess.run(encoded_png, feed_dict={image: images[j]})

        example = _image_to_tfexample(
            png_string, 'png', _IMAGE_SIZE, _IMAGE_SIZE, labels[j])
        tfrecord_writer.write(example.SerializeToString())


def _get_output_filename(dataset_dir, split_name):
  """Creates the output filename.

  Args:
    dataset_dir: The directory where the temporary files are stored.
    split_name: The name of the train/test split.

  Returns:
    An absolute file path.
  """
  return '%s/mnist_%s.tfrecord' % (dataset_dir, split_name)


def _download_dataset(dataset_dir):
  """Downloads MNIST locally.

  Args:
    dataset_dir: The directory where the temporary files are stored.
  """
  for filename in [_TRAIN_DATA_FILENAME,
                   _TRAIN_LABELS_FILENAME,
                   _TEST_DATA_FILENAME,
                   _TEST_LABELS_FILENAME]:
    filepath = os.path.join(dataset_dir, filename)

    if not os.path.exists(filepath):
      print('Downloading file %s...' % filename)
      def _progress(count, block_size, total_size):
        sys.stdout.write('\r>> Downloading %.1f%%' % (
            float(count * block_size) / float(total_size) * 100.0))
        sys.stdout.flush()
      filepath, _ = urllib.request.urlretrieve(_DATA_URL + filename,
                                               filepath,
                                               _progress)
      print()
      with tf.gfile.GFile(filepath) as f:
        size = f.Size()
      print('Successfully downloaded', filename, size, 'bytes.')


def _clean_up_temporary_files(dataset_dir):
  """Removes temporary files used to create the dataset.

  Args:
    dataset_dir: The directory where the temporary files are stored.
  """
  for filename in [_TRAIN_DATA_FILENAME,
                   _TRAIN_LABELS_FILENAME,
                   _TEST_DATA_FILENAME,
                   _TEST_LABELS_FILENAME]:
    filepath = os.path.join(dataset_dir, filename)
    tf.gfile.Remove(filepath)


def run(dataset_dir):
  """Runs the download and conversion operation.

  Args:
    dataset_dir: The dataset directory where the dataset is stored.
  """
  if not tf.gfile.Exists(dataset_dir):
    tf.gfile.MakeDirs(dataset_dir)

  training_filename = _get_output_filename(dataset_dir, 'train')
  testing_filename = _get_output_filename(dataset_dir, 'test')

<<<<<<< HEAD
=======
  training_filename = _get_output_filename('train')
  testing_filename = _get_output_filename('test')

>>>>>>> 0af5999e5e6e3147cea5a5d136ff7546a9957939
  if tf.gfile.Exists(training_filename) and tf.gfile.Exists(testing_filename):
    print('Dataset files already exist. Exiting without re-creating them.')
    return

<<<<<<< HEAD
  _download_dataset(dataset_dir)

  # First, process the training data:
  with tf.python_io.TFRecordWriter(training_filename) as tfrecord_writer:
    data_filename = os.path.join(dataset_dir, _TRAIN_DATA_FILENAME)
    labels_filename = os.path.join(dataset_dir, _TRAIN_LABELS_FILENAME)
=======
  _download_dataset(FLAGS.dataset_dir)

  # First, process the training data:
  with tf.python_io.TFRecordWriter(training_filename) as tfrecord_writer:
    data_filename = os.path.join(FLAGS.dataset_dir, _TRAIN_DATA_FILENAME)
    labels_filename = os.path.join(FLAGS.dataset_dir, _TRAIN_LABELS_FILENAME)
>>>>>>> 0af5999e5e6e3147cea5a5d136ff7546a9957939
    _add_to_tfrecord(data_filename, labels_filename, 60000, tfrecord_writer)

  # Next, process the testing data:
  with tf.python_io.TFRecordWriter(testing_filename) as tfrecord_writer:
<<<<<<< HEAD
    data_filename = os.path.join(dataset_dir, _TEST_DATA_FILENAME)
    labels_filename = os.path.join(dataset_dir, _TEST_LABELS_FILENAME)
=======
    data_filename = os.path.join(FLAGS.dataset_dir, _TEST_DATA_FILENAME)
    labels_filename = os.path.join(FLAGS.dataset_dir, _TEST_LABELS_FILENAME)
>>>>>>> 0af5999e5e6e3147cea5a5d136ff7546a9957939
    _add_to_tfrecord(data_filename, labels_filename, 10000, tfrecord_writer)

  # Finally, write the labels file:
  labels_to_class_names = dict(zip(range(len(_CLASS_NAMES)), _CLASS_NAMES))
<<<<<<< HEAD
  dataset_utils.write_label_file(labels_to_class_names, dataset_dir)
=======
  _write_label_file(labels_to_class_names, FLAGS.dataset_dir)
>>>>>>> 0af5999e5e6e3147cea5a5d136ff7546a9957939

  _clean_up_temporary_files(dataset_dir)
  print('\nFinished converting the MNIST dataset!')
