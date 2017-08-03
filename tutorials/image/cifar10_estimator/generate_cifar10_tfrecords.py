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
"""Read CIFAR-10 data from pickled numpy arrays and write TFExamples.

Generates TFRecord files from the python version of the CIFAR-10 dataset
downloaded from https://www.cs.toronto.edu/~kriz/cifar.html.
"""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import cPickle
import os
import tensorflow as tf

FLAGS = tf.flags.FLAGS

tf.flags.DEFINE_string('input_dir', '',
                       'Directory where CIFAR10 data is located.')

tf.flags.DEFINE_string('output_dir', '',
                       'Directory where TFRecords will be saved.'
                       'The TFRecords will have the same name as'
                       ' the CIFAR10 inputs + .tfrecords.')


def _int64_feature(value):
  return tf.train.Feature(int64_list=tf.train.Int64List(value=[value]))


def _bytes_feature(value):
  return tf.train.Feature(bytes_list=tf.train.BytesList(value=[str(value)]))


def _get_file_names():
  """Returns the file names expected to exist in the input_dir."""
  file_names = {}
  file_names['train'] = ['data_batch_%d' % i for i in xrange(1, 5)]
  file_names['validation'] = ['data_batch_5']
  file_names['eval'] = ['test_batch']
  return file_names


def read_pickle_from_file(filename):
  with open(filename, 'r') as f:
    data_dict = cPickle.load(f)
  return data_dict


def convert_to_tfrecord(input_files, output_file):
  """Converts a file to tfrecords."""
  print('Generating %s' % output_file)
  record_writer = tf.python_io.TFRecordWriter(output_file)

  for input_file in input_files:
    data_dict = read_pickle_from_file(input_file)
    data = data_dict['data']
    labels = data_dict['labels']

    num_entries_in_batch = len(labels)
    for i in range(num_entries_in_batch):
      example = tf.train.Example(
          features=tf.train.Features(feature={
              'image': _bytes_feature(data[i].tobytes()),
              'label': _int64_feature(labels[i])
          }))
      record_writer.write(example.SerializeToString())
  record_writer.close()


def main(unused_argv):
  file_names = _get_file_names()
  for mode, files in file_names.items():
    input_files = [
        os.path.join(FLAGS.input_dir, f) for f in files]
    output_file = os.path.join(FLAGS.output_dir, mode + '.tfrecords')
    # Convert to Examples and write the result to TFRecords.
    convert_to_tfrecord(input_files, output_file)
  print('Done!')


if __name__ == '__main__':
  tf.app.run(main)
