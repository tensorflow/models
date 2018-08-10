# Copyright 2018 The TensorFlow Authors All Rights Reserved.
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

"""Utility methods for accessing and operating on test data."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os
from absl import flags
import tensorflow as tf
from google.protobuf import text_format

import input as seq2species_input
from protos import seq2label_pb2

FLAGS = flags.FLAGS

# Target names included in the example inputs.
TEST_TARGETS = ['test_target_1', 'test_target_2']


def _as_bytes_feature(in_string):
  """Converts the given string to a tf.train.BytesList feature.

  Args:
    in_string: string to be converted to BytesList Feature.

  Returns:
    The TF BytesList Feature representing the given string.
  """
  return tf.train.Feature(bytes_list=tf.train.BytesList(value=[in_string]))


def create_tmp_train_file(num_examples,
                          read_len,
                          characters=seq2species_input.DNA_BASES,
                          name='test.tfrecord'):
  """Write a test TFRecord of input examples to temporary test directory.

  The generated input examples are test tf.train.Example protos, each comprised
  of a toy sequence of length read_len and non-meaningful labels for targets in
  TEST_TARGETS.

  Args:
    num_examples: int; number of examples to write to test input file.
    read_len: int; length of test read sequences.
    characters: string; set of characters from which to construct test reads.
      Defaults to canonical DNA bases.
    name: string; filename for the test input file.

  Returns:
    Full path to the generated temporary test input file.
  """
  tmp_path = os.path.join(FLAGS.test_tmpdir, name)
  with tf.python_io.TFRecordWriter(tmp_path) as writer:
    for i in xrange(num_examples):
      char = characters[i % len(characters)]
      features_dict = {'sequence': _as_bytes_feature(char * read_len)}
      for target_name in TEST_TARGETS:
        nonsense_label = _as_bytes_feature(str(i))
        features_dict[target_name] = nonsense_label
      tf_features = tf.train.Features(feature=features_dict)
      example = tf.train.Example(features=tf_features)
      writer.write(example.SerializeToString())
  return tmp_path


def create_tmp_metadata(num_examples, read_len):
  """Write a test Seq2LabelDatasetInfo test proto to temporary test directory.

  Args:
    num_examples: int; number of example labels to write into test metadata.
    read_len: int; length of test read sequences.

  Returns:
    Full path to the generated temporary test file containing the
    Seq2LabelDatasetInfo text proto.
  """
  dataset_info = seq2label_pb2.Seq2LabelDatasetInfo(
      read_length=read_len,
      num_examples=num_examples,
      read_stride=1,
      dataset_path='test.tfrecord')

  for target in TEST_TARGETS:
    dataset_info.labels.add(
        name=target, values=[str(i) for i in xrange(num_examples)])

  tmp_path = os.path.join(FLAGS.test_tmpdir, 'test.pbtxt')
  with tf.gfile.GFile(tmp_path, 'w') as f:
    f.write(text_format.MessageToString(dataset_info))
  return tmp_path
