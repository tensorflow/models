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
"""Contains utilities for downloading and converting datasets."""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os
import sys
import tarfile
import zipfile

from six.moves import urllib
import tensorflow.compat.v1 as tf

LABELS_FILENAME = 'labels.txt'


def int64_feature(values):
  """Returns a TF-Feature of int64s.

  Args:
    values: A scalar or list of values.

  Returns:
    A TF-Feature.
  """
  if not isinstance(values, (tuple, list)):
    values = [values]
  return tf.train.Feature(int64_list=tf.train.Int64List(value=values))


def bytes_list_feature(values):
  """Returns a TF-Feature of list of bytes.

  Args:
    values: A string or list of strings.

  Returns:
    A TF-Feature.
  """
  return tf.train.Feature(bytes_list=tf.train.BytesList(value=values))


def float_list_feature(values):
  """Returns a TF-Feature of list of floats.

  Args:
    values: A float or list of floats.

  Returns:
    A TF-Feature.
  """
  return tf.train.Feature(float_list=tf.train.FloatList(value=values))


def bytes_feature(values):
  """Returns a TF-Feature of bytes.

  Args:
    values: A string.

  Returns:
    A TF-Feature.
  """
  return tf.train.Feature(bytes_list=tf.train.BytesList(value=[values]))


def float_feature(values):
  """Returns a TF-Feature of floats.

  Args:
    values: A scalar of list of values.

  Returns:
    A TF-Feature.
  """
  if not isinstance(values, (tuple, list)):
    values = [values]
  return tf.train.Feature(float_list=tf.train.FloatList(value=values))


def image_to_tfexample(image_data, image_format, height, width, class_id):
  return tf.train.Example(features=tf.train.Features(feature={
      'image/encoded': bytes_feature(image_data),
      'image/format': bytes_feature(image_format),
      'image/class/label': int64_feature(class_id),
      'image/height': int64_feature(height),
      'image/width': int64_feature(width),
  }))


def download_url(url, dataset_dir):
  """Downloads the tarball or zip file from url into filepath.

  Args:
    url: The URL of a tarball or zip file.
    dataset_dir: The directory where the temporary files are stored.

  Returns:
    filepath: path where the file is downloaded.
  """
  filename = url.split('/')[-1]
  filepath = os.path.join(dataset_dir, filename)

  def _progress(count, block_size, total_size):
    sys.stdout.write('\r>> Downloading %s %.1f%%' % (
        filename, float(count * block_size) / float(total_size) * 100.0))
    sys.stdout.flush()

  filepath, _ = urllib.request.urlretrieve(url, filepath, _progress)
  print()
  statinfo = os.stat(filepath)
  print('Successfully downloaded', filename, statinfo.st_size, 'bytes.')
  return filepath


def download_and_uncompress_tarball(tarball_url, dataset_dir):
  """Downloads the `tarball_url` and uncompresses it locally.

  Args:
    tarball_url: The URL of a tarball file.
    dataset_dir: The directory where the temporary files are stored.
  """
  filepath = download_url(tarball_url, dataset_dir)
  tarfile.open(filepath, 'r:gz').extractall(dataset_dir)


def download_and_uncompress_zipfile(zip_url, dataset_dir):
  """Downloads the `zip_url` and uncompresses it locally.

  Args:
    zip_url: The URL of a zip file.
    dataset_dir: The directory where the temporary files are stored.
  """
  filename = zip_url.split('/')[-1]
  filepath = os.path.join(dataset_dir, filename)

  if tf.gfile.Exists(filepath):
    print('File {filename} has been already downloaded at {filepath}. '
          'Unzipping it....'.format(filename=filename, filepath=filepath))
  else:
    filepath = download_url(zip_url, dataset_dir)

  with zipfile.ZipFile(filepath, 'r') as zip_file:
    for member in zip_file.namelist():
      memberpath = os.path.join(dataset_dir, member)
      # extract only if file doesn't exist
      if not (os.path.exists(memberpath) or os.path.isfile(memberpath)):
        zip_file.extract(member, dataset_dir)


def write_label_file(labels_to_class_names,
                     dataset_dir,
                     filename=LABELS_FILENAME):
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


def has_labels(dataset_dir, filename=LABELS_FILENAME):
  """Specifies whether or not the dataset directory contains a label map file.

  Args:
    dataset_dir: The directory in which the labels file is found.
    filename: The filename where the class names are written.

  Returns:
    `True` if the labels file exists and `False` otherwise.
  """
  return tf.gfile.Exists(os.path.join(dataset_dir, filename))


def read_label_file(dataset_dir, filename=LABELS_FILENAME):
  """Reads the labels file and returns a mapping from ID to class name.

  Args:
    dataset_dir: The directory in which the labels file is found.
    filename: The filename where the class names are written.

  Returns:
    A map from a label (integer) to class name.
  """
  labels_filename = os.path.join(dataset_dir, filename)
  with tf.gfile.Open(labels_filename, 'rb') as f:
    lines = f.read().decode()
  lines = lines.split('\n')
  lines = filter(None, lines)

  labels_to_class_names = {}
  for line in lines:
    index = line.index(':')
    labels_to_class_names[int(line[:index])] = line[index+1:]
  return labels_to_class_names


def open_sharded_output_tfrecords(exit_stack, base_path, num_shards):
  """Opens all TFRecord shards for writing and adds them to an exit stack.

  Args:
    exit_stack: A context2.ExitStack used to automatically closed the TFRecords
      opened in this function.
    base_path: The base path for all shards
    num_shards: The number of shards

  Returns:
    The list of opened TFRecords. Position k in the list corresponds to shard k.
  """
  tf_record_output_filenames = [
      '{}-{:05d}-of-{:05d}'.format(base_path, idx, num_shards)
      for idx in range(num_shards)
  ]

  tfrecords = [
      exit_stack.enter_context(tf.python_io.TFRecordWriter(file_name))
      for file_name in tf_record_output_filenames
  ]

  return tfrecords
