# Copyright 2022 The TensorFlow Authors. All Rights Reserved.
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

"""Tests for tf_example_builder.

See `test_add_image_matrix_feature_with_fake_image` for the typical structure of
a unit test.
"""

from absl.testing import parameterized
import tensorflow as tf
from official.core import tf_example_builder


class TfExampleBuilderTest(tf.test.TestCase, parameterized.TestCase):

  def test_init_an_empty_example(self):
    example_builder = tf_example_builder.TfExampleBuilder()
    example = example_builder.example
    self.assertProtoEquals('', example)

  def test_init_an_empty_serialized_example(self):
    example_builder = tf_example_builder.TfExampleBuilder()
    example = example_builder.serialized_example
    self.assertProtoEquals('', example)

  def test_add_feature(self):
    example_builder = tf_example_builder.TfExampleBuilder()
    example_builder.add_feature(
        'foo',
        tf.train.Feature(
            bytes_list=tf.train.BytesList(value=[b'Hello World!'])))
    example = example_builder.example
    # Use proto text to show how the entire proto would look like.
    self.assertProtoEquals(
        """
        features: {
          feature: {
            key: "foo"
            value: {
              bytes_list: {
                value: "Hello World!"
              }
            }
          }
        }""", example)

  def test_add_feature_dict(self):
    example_builder = tf_example_builder.TfExampleBuilder()
    example_builder.add_feature_dict({
        'foo':
            tf.train.Feature(
                bytes_list=tf.train.BytesList(value=[b'Hello World!'])),
        'bar':
            tf.train.Feature(
                int64_list=tf.train.Int64List(value=[299, 792, 458]))
    })
    example = example_builder.example
    # Use proto text to show how the entire proto would look like.
    self.assertProtoEquals(
        """
        features: {
          feature: {
            key: "foo"
            value: {
              bytes_list: {
                value: "Hello World!"
              }
            }
          }
          feature: {
            key: "bar"
            value: {
              int64_list: {
                value: 299
                value: 792
                value: 458
              }
            }
          }
        }""", example)

  @parameterized.named_parameters(
      ('single_bytes', b'Hello World!', b'Hello World!'),
      ('single_string', 'Hello World!', b'Hello World!'))
  def test_add_single_byte_feature(self, value, expected_value):
    example_builder = tf_example_builder.TfExampleBuilder()
    example_builder.add_bytes_feature('foo', value)
    example = example_builder.example
    # Use constructor to easily work with test parameters.
    self.assertProtoEquals(
        tf.train.Example(
            features=tf.train.Features(
                feature={
                    'foo':
                        tf.train.Feature(
                            bytes_list=tf.train.BytesList(
                                value=[expected_value]))
                })), example)

  @parameterized.named_parameters(
      ('multiple_bytes', [b'Hello World!', b'Good Morning!'
                         ], [b'Hello World!', b'Good Morning!']),
      ('multiple_sring', ['Hello World!', 'Good Morning!'
                         ], [b'Hello World!', b'Good Morning!']))
  def test_add_multiple_bytes_feature(self, values, expected_values):
    example_builder = tf_example_builder.TfExampleBuilder()
    example_builder.add_bytes_feature('foo', values)
    example = example_builder.example
    self.assertProtoEquals(
        tf.train.Example(
            features=tf.train.Features(
                feature={
                    'foo':
                        tf.train.Feature(
                            bytes_list=tf.train.BytesList(
                                value=expected_values))
                })), example)

  @parameterized.named_parameters(
      ('single_integer', 123, [123]),
      ('multiple_integers', [123, 456, 789], [123, 456, 789]))
  def test_add_ints_feature(self, value, expected_value):
    example_builder = tf_example_builder.TfExampleBuilder()
    example_builder.add_ints_feature('bar', value)
    example = example_builder.example
    self.assertProtoEquals(
        tf.train.Example(
            features=tf.train.Features(
                feature={
                    'bar':
                        tf.train.Feature(
                            int64_list=tf.train.Int64List(value=expected_value))
                })), example)

  @parameterized.named_parameters(
      ('single_float', 3.14, [3.14]),
      ('multiple_floats', [3.14, 1.57, 6.28], [3.14, 1.57, 6.28]))
  def test_add_floats_feature(self, value, expected_value):
    example_builder = tf_example_builder.TfExampleBuilder()
    example_builder.add_floats_feature('baz', value)
    example = example_builder.example
    self.assertProtoEquals(
        tf.train.Example(
            features=tf.train.Features(
                feature={
                    'baz':
                        tf.train.Feature(
                            float_list=tf.train.FloatList(value=expected_value))
                })), example)


if __name__ == '__main__':
  tf.test.main()
