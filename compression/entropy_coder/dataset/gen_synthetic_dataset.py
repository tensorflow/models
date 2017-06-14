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

"""Generate a synthetic dataset."""

import os

import numpy as np
import tensorflow as tf

import synthetic_model


FLAGS = tf.app.flags.FLAGS

tf.app.flags.DEFINE_string(
    'dataset_dir', None,
    """Directory where to write the dataset and the configs.""")
tf.app.flags.DEFINE_integer(
    'count', 1000,
    """Number of samples to generate.""")


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


def AddToTFRecord(code, tfrecord_writer):
  example = tf.train.Example(features=tf.train.Features(feature={
      'code_shape': int64_feature(code.shape),
      'code': float_feature(code.flatten().tolist()),
  }))
  tfrecord_writer.write(example.SerializeToString())


def GenerateDataset(filename, count, code_shape):
  with tf.python_io.TFRecordWriter(filename) as tfrecord_writer:
    for _ in xrange(count):
      code = synthetic_model.GenerateSingleCode(code_shape)
      # Convert {0,1} codes to {-1,+1} codes.
      code = 2.0 * code - 1.0
      AddToTFRecord(code, tfrecord_writer)


def main(argv=None):  # pylint: disable=unused-argument
  GenerateDataset(os.path.join(FLAGS.dataset_dir + '/synthetic_dataset'),
                  FLAGS.count,
                  [35, 48, 8])


if __name__ == '__main__':
  tf.app.run()
