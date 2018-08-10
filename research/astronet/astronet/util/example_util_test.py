# Copyright 2018 The TensorFlow Authors.
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

"""Tests for example_util.py."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import numpy as np
import tensorflow as tf

from astronet.util import example_util


class ExampleUtilTest(tf.test.TestCase):

  def test_get_feature(self):
    # Create Example.
    bytes_list = tf.train.BytesList(
        value=[v.encode("latin-1") for v in ["a", "b", "c"]])
    float_list = tf.train.FloatList(value=[1.0, 2.0, 3.0])
    int64_list = tf.train.Int64List(value=[11, 22, 33])
    ex = tf.train.Example(
        features=tf.train.Features(
            feature={
                "a_bytes": tf.train.Feature(bytes_list=bytes_list),
                "b_float": tf.train.Feature(float_list=float_list),
                "c_int64": tf.train.Feature(int64_list=int64_list),
                "d_empty": tf.train.Feature(),
            }))

    # Get bytes feature.
    np.testing.assert_array_equal(
        example_util.get_feature(ex, "a_bytes").astype(str), ["a", "b", "c"])
    np.testing.assert_array_equal(
        example_util.get_feature(ex, "a_bytes", "bytes_list").astype(str),
        ["a", "b", "c"])
    np.testing.assert_array_equal(
        example_util.get_bytes_feature(ex, "a_bytes").astype(str),
        ["a", "b", "c"])
    with self.assertRaises(TypeError):
      example_util.get_feature(ex, "a_bytes", "float_list")
    with self.assertRaises(TypeError):
      example_util.get_float_feature(ex, "a_bytes")
    with self.assertRaises(TypeError):
      example_util.get_int64_feature(ex, "a_bytes")

    # Get float feature.
    np.testing.assert_array_almost_equal(
        example_util.get_feature(ex, "b_float"), [1.0, 2.0, 3.0])
    np.testing.assert_array_almost_equal(
        example_util.get_feature(ex, "b_float", "float_list"), [1.0, 2.0, 3.0])
    np.testing.assert_array_almost_equal(
        example_util.get_float_feature(ex, "b_float"), [1.0, 2.0, 3.0])
    with self.assertRaises(TypeError):
      example_util.get_feature(ex, "b_float", "int64_list")
    with self.assertRaises(TypeError):
      example_util.get_bytes_feature(ex, "b_float")
    with self.assertRaises(TypeError):
      example_util.get_int64_feature(ex, "b_float")

    # Get int64 feature.
    np.testing.assert_array_equal(
        example_util.get_feature(ex, "c_int64"), [11, 22, 33])
    np.testing.assert_array_equal(
        example_util.get_feature(ex, "c_int64", "int64_list"), [11, 22, 33])
    np.testing.assert_array_equal(
        example_util.get_int64_feature(ex, "c_int64"), [11, 22, 33])
    with self.assertRaises(TypeError):
      example_util.get_feature(ex, "c_int64", "bytes_list")
    with self.assertRaises(TypeError):
      example_util.get_bytes_feature(ex, "c_int64")
    with self.assertRaises(TypeError):
      example_util.get_float_feature(ex, "c_int64")

    # Get empty feature.
    np.testing.assert_array_equal(example_util.get_feature(ex, "d_empty"), [])
    np.testing.assert_array_equal(
        example_util.get_feature(ex, "d_empty", "float_list"), [])
    np.testing.assert_array_equal(
        example_util.get_bytes_feature(ex, "d_empty"), [])
    np.testing.assert_array_equal(
        example_util.get_float_feature(ex, "d_empty"), [])
    np.testing.assert_array_equal(
        example_util.get_int64_feature(ex, "d_empty"), [])

    # Get nonexistent feature.
    with self.assertRaises(KeyError):
      example_util.get_feature(ex, "nonexistent")
    with self.assertRaises(KeyError):
      example_util.get_feature(ex, "nonexistent", "bytes_list")
    with self.assertRaises(KeyError):
      example_util.get_bytes_feature(ex, "nonexistent")
    with self.assertRaises(KeyError):
      example_util.get_float_feature(ex, "nonexistent")
    with self.assertRaises(KeyError):
      example_util.get_int64_feature(ex, "nonexistent")
    np.testing.assert_array_equal(
        example_util.get_feature(ex, "nonexistent", strict=False), [])
    np.testing.assert_array_equal(
        example_util.get_bytes_feature(ex, "nonexistent", strict=False), [])
    np.testing.assert_array_equal(
        example_util.get_float_feature(ex, "nonexistent", strict=False), [])
    np.testing.assert_array_equal(
        example_util.get_int64_feature(ex, "nonexistent", strict=False), [])

  def test_set_feature(self):
    ex = tf.train.Example()

    # Set bytes features.
    example_util.set_feature(ex, "a1_bytes", ["a", "b"])
    example_util.set_feature(ex, "a2_bytes", ["A", "B"], kind="bytes_list")
    example_util.set_bytes_feature(ex, "a3_bytes", ["x", "y"])
    np.testing.assert_array_equal(
        np.array(ex.features.feature["a1_bytes"].bytes_list.value).astype(str),
        ["a", "b"])
    np.testing.assert_array_equal(
        np.array(ex.features.feature["a2_bytes"].bytes_list.value).astype(str),
        ["A", "B"])
    np.testing.assert_array_equal(
        np.array(ex.features.feature["a3_bytes"].bytes_list.value).astype(str),
        ["x", "y"])
    with self.assertRaises(ValueError):
      example_util.set_feature(ex, "a3_bytes", ["xxx"])  # Duplicate.

    # Set float features.
    example_util.set_feature(ex, "b1_float", [1.0, 2.0])
    example_util.set_feature(ex, "b2_float", [10.0, 20.0], kind="float_list")
    example_util.set_float_feature(ex, "b3_float", [88.0, 99.0])
    np.testing.assert_array_almost_equal(
        ex.features.feature["b1_float"].float_list.value, [1.0, 2.0])
    np.testing.assert_array_almost_equal(
        ex.features.feature["b2_float"].float_list.value, [10.0, 20.0])
    np.testing.assert_array_almost_equal(
        ex.features.feature["b3_float"].float_list.value, [88.0, 99.0])
    with self.assertRaises(ValueError):
      example_util.set_feature(ex, "b3_float", [1234.0])  # Duplicate.

    # Set int64 features.
    example_util.set_feature(ex, "c1_int64", [1, 2, 3])
    example_util.set_feature(ex, "c2_int64", [11, 22, 33], kind="int64_list")
    example_util.set_int64_feature(ex, "c3_int64", [88, 99])
    np.testing.assert_array_equal(
        ex.features.feature["c1_int64"].int64_list.value, [1, 2, 3])
    np.testing.assert_array_equal(
        ex.features.feature["c2_int64"].int64_list.value, [11, 22, 33])
    np.testing.assert_array_equal(
        ex.features.feature["c3_int64"].int64_list.value, [88, 99])
    with self.assertRaises(ValueError):
      example_util.set_feature(ex, "c3_int64", [1234])  # Duplicate.

    # Overwrite features.
    example_util.set_feature(ex, "a3_bytes", ["xxx"], allow_overwrite=True)
    np.testing.assert_array_equal(
        np.array(ex.features.feature["a3_bytes"].bytes_list.value).astype(str),
        ["xxx"])

    example_util.set_feature(ex, "b3_float", [1234.0], allow_overwrite=True)
    np.testing.assert_array_almost_equal(
        ex.features.feature["b3_float"].float_list.value, [1234.0])

    example_util.set_feature(ex, "c3_int64", [1234], allow_overwrite=True)
    np.testing.assert_array_equal(
        ex.features.feature["c3_int64"].int64_list.value, [1234])


if __name__ == "__main__":
  tf.test.main()
