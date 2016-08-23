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

This script downloads the cifar10 data, uncompresses it, reads the files
that make up the cifar10 data and creates two TFRecord datasets: one for train
and one for test. Each TFRecord dataset is comprised of a set of TF-Example
protocol buffers, each of which contain a single image and label.

The script should take several minutes to run.

Usage:
$ bazel build slim:download_and_convert_cifar10
$ .bazel-bin/slim/download_and_convert_cifar10 --dataset_dir=[DIRECTORY]
"""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import cPickle
import os
import sys
import tarfile

import google3
import numpy as np
from six.moves import urllib
import tensorflow as tf

tf.app.flags.DEFINE_string(
    'dataset_dir',
    None,
    'The directory where the output TFRecords and temporary files are saved.')
tf.app.flags.MarkFlagAsRequired('dataset_dir')

FLAGS = tf.app.flags.FLAGS

# The URL where the CIFAR data can be downloaded.
_DATA_URL = 'https://www.cs.toronto.edu/~kriz/cifar-10-python.tar.gz'

# The number of training files.
_NUM_TRAIN_FILES = 5


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


def _encode_png_image(image):
  """Encodes the given image using PNG encoding.

  Args:
    image: A numpy image of size [height, width, 3].

  Returns:
    An encoding image string.
  """
  with tf.Graph().as_default():
    with tf.Session(''):
      return tf.image.encode_png(tf.constant(image)).eval()


def _add_to_tfrecord(filename, tfrecord_writer, offset=0):
  """Loads data from the cifar10 pickle files and writes files to a TFRecord.

  Args:
    filename: The filename of the cifar10 pickle file.
    tfrecord_writer: The TFRecord writer to use for writing.
    offset: An offset into the absolute number of images previously written.

  Returns:
    The new offset.
  """
  with tf.gfile.Open(filename, 'r') as f:
    data = cPickle.load(f)

  images = data['data']
  num_images = images.shape[0]

  images = images.reshape((num_images, 3, 32, 32))
  labels = data['labels']

  for j in range(num_images):
    if j % 100 == 0:
      print('Reading file [%s], image %d/%d' % (
          filename, offset + j + 1, offset + num_images))
    image = np.squeeze(images[j]).transpose((1, 2, 0))
    label = labels[j]

    png_string = _encode_png_image(image)

    example = tf.train.Example(features=tf.train.Features(feature={
        'image/encoded': _bytes_feature(png_string),
        'image/format': _bytes_feature('png'),
        'image/class/label': _int64_feature(label),
        'image/height': _int64_feature(32),
        'image/width': _int64_feature(32),
    }))

    tfrecord_writer.write(example.SerializeToString())

  return offset + num_images


def _get_output_filename(split_name):
  """Creates the output filename.

  Args:
    split_name: The name of the train/test split.

  Returns:
    An absolute file path.
  """
  return '%s/cifar10_%s.tfrecord' % (FLAGS.dataset_dir, split_name)


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


def main(_):
  if not tf.gfile.Exists(FLAGS.dataset_dir):
    tf.gfile.MakeDirs(FLAGS.dataset_dir)

  _download_and_uncompress_dataset(FLAGS.dataset_dir)

  # First, process the training data:
  output_file = _get_output_filename('train')
  with tf.python_io.TFRecordWriter(output_file) as tfrecord_writer:
    offset = 0
    for i in range(_NUM_TRAIN_FILES):
      filename = os.path.join(FLAGS.dataset_dir,
                              'cifar-10-batches-py',
                              'data_batch_%d' % (i + 1))  # 1-indexed.
      offset = _add_to_tfrecord(filename, tfrecord_writer, offset)

  # Next, process the testing data:
  output_file = _get_output_filename('test')
  with tf.python_io.TFRecordWriter(output_file) as tfrecord_writer:
    filename = os.path.join(FLAGS.dataset_dir,
                            'cifar-10-batches-py',
                            'test_batch')
    _add_to_tfrecord(filename, tfrecord_writer)

  _clean_up_temporary_files(FLAGS.dataset_dir)
  print('Finished converting the Cifar10 dataset!')

if __name__ == '__main__':
  tf.app.run()
