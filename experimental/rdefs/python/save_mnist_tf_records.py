# Copyright 2016 Google Inc. All Rights Reserved.
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
"""Save MNIST into tf.records format."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os
import urllib

import numpy as np
import tensorflow as tf
# from https://github.com/yburda/iwae/blob/master/datasets.py

DATASETS_DIR = '/tmp/BinarizedMNIST'
if not os.path.exists(DATASETS_DIR):
  os.makedirs(DATASETS_DIR)
subdatasets = ['train', 'valid', 'test']
for subdataset in subdatasets:
  filename = 'binarized_mnist_{}.amat'.format(subdataset)
  url = 'http://www.cs.toronto.edu/~larocheh/public/datasets/binarized_mnist/binarized_mnist_{}.amat'.format(subdataset)  # pylint: disable=line-too-long
  local_filename = os.path.join(DATASETS_DIR, filename)
  if not os.path.exists(local_filename):
    urllib.urlretrieve(url, local_filename)


def binarized_mnist_fixed_binarization():
  """parse .mat file and get numpy array of MNIST."""
  def lines_to_np_array(lines):
    return np.array([[int(i) for i in line.split()] for line in lines])
  with open(os.path.join(DATASETS_DIR, 'binarized_mnist_train.amat')) as f:
    lines = f.readlines()
  train_data = lines_to_np_array(lines).astype('float32')
  with open(os.path.join(DATASETS_DIR, 'binarized_mnist_valid.amat')) as f:
    lines = f.readlines()
  validation_data = lines_to_np_array(lines).astype('float32')
  with open(os.path.join(DATASETS_DIR, 'binarized_mnist_test.amat')) as f:
    lines = f.readlines()
  test_data = lines_to_np_array(lines).astype('float32')
  return train_data, validation_data, test_data

train, validation, test = binarized_mnist_fixed_binarization()

train_and_validation = np.vstack([train, validation])

data_dict = {'train': train, 'valid': validation, 'test': test,
             'train_and_valid': np.vstack([train, validation])}


def serialize_array_with_label(array, path):
  writer = tf.python_io.TFRecordWriter(path)
  indices = range(array.shape[0])
  # one MUST randomly shuffle data before putting it into one of these
  # formats. Without this, one cannot make use of tensorflow's great
  # out of core shuffling.
  np.random.shuffle(indices)
  # iterate over each example
  for example_idx in indices:
    features = array[example_idx]

    # construct the Example proto boject
    example = tf.train.Example(
        # Example contains a Features proto object
        features=tf.train.Features(
            # Features contains a map of string to Feature proto objects
            feature={
                # A Feature contains one of either a int64_list,
                # float_list, or bytes_list
                'image': tf.train.Feature(
                    float_list=tf.train.FloatList(
                        value=features.astype('float'))),
                'label': tf.train.Feature(
                    int64_list=tf.train.Int64List(value=[example_idx]))
            }
        )
    )
    # use the proto object to serialize the example to a string
    serialized = example.SerializeToString()
    # write the serialized object to disk
    writer.write(serialized)

subdatasets = ['train', 'valid', 'test', 'train_and_valid']

for subdataset in subdatasets:
  print 'serializing %s' % subdataset
  file_name = os.path.join(
      DATASETS_DIR, 'binarized_mnist_{}_labeled.tfrecords'.format(subdataset))
  if not os.path.exists(file_name):
    serialize_array_with_label(data_dict[subdataset], file_name)
