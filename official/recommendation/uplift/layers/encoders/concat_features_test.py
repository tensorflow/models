# Copyright 2024 The TensorFlow Authors. All Rights Reserved.
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

"""Tests for concat_feature_encoder."""

from absl.testing import parameterized
import tensorflow as tf, tf_keras
from official.recommendation.uplift import keras_test_case
from official.recommendation.uplift.layers.encoders import concat_features


class ConcatFeaturesTest(keras_test_case.KerasTestCase, parameterized.TestCase):

  @parameterized.named_parameters(
      {
          "testcase_name": "single_dense",
          "feature_names": ["feature"],
          "inputs": {"feature": tf.ones((3, 1))},
          "expected_output": tf.ones((3, 1)),
      },
      {
          "testcase_name": "single_sparse",
          "feature_names": ["feature"],
          "inputs": {
              "feature": tf.sparse.SparseTensor(
                  indices=[[0, 0], [1, 2]], values=[1, 2], dense_shape=[3, 4]
              )
          },
          "expected_output": tf.constant(
              [[1, 0, 0, 0], [0, 0, 2, 0], [0, 0, 0, 0]]
          ),
      },
      {
          "testcase_name": "single_ragged",
          "feature_names": ["feature"],
          "inputs": {"feature": tf.ragged.constant([[5, 7], [0, 3, 1], [6]])},
          "expected_output": tf.constant([[5, 7, 0], [0, 3, 1], [6, 0, 0]]),
      },
      {
          "testcase_name": "excess_features",
          "feature_names": ["feature3"],
          "inputs": {
              "feature1": tf.ones((3, 1)),
              "feature2": 2 * tf.ones((3, 1)),
              "feature3": 3 * tf.ones((3, 1)),
          },
          "expected_output": 3 * tf.ones((3, 1)),
      },
      {
          "testcase_name": "one_dimensional_features",
          "feature_names": ["feature1", "feature2"],
          "inputs": {
              "feature1": tf.ones((1, 3)),
              "feature2": 2 * tf.ones((1, 2)),
          },
          "expected_output": tf.constant([1, 1, 1, 2, 2], shape=(1, 5)),
      },
      {
          "testcase_name": "mixed_features",
          "feature_names": ["dense", "sparse", "ragged"],
          "inputs": {
              "dense": tf.constant([-1.4, 2.0], shape=(2, 1)),
              "sparse": tf.sparse.SparseTensor(
                  indices=[[0, 1], [1, 0]],
                  values=[2.718, 3.14],
                  dense_shape=[2, 2],
              ),
              "ragged": tf.ragged.constant([[5, 7.77], [8]]),
              "other_feature": tf.ones((2, 5)),
          },
          "expected_output": tf.constant(
              [[-1.4, 0, 2.718, 5, 7.77], [2.0, 3.14, 0, 8, 0]]
          ),
      },
  )
  def test_layer_correctness(self, feature_names, inputs, expected_output):
    layer = concat_features.ConcatFeatures(feature_names=feature_names)
    self.assertAllClose(expected_output, layer(inputs))

  @parameterized.named_parameters(
      {
          "testcase_name": "none_dimensions",
          "inputs": {
              "x1": tf_keras.Input(shape=(2, None, 1, 3)),
              "x2": tf_keras.Input(shape=(2, None, 1, None)),
          },
          "expected_shape": [None, 2, None, 1, None],
      },
      {
          "testcase_name": "dense_sparse_ragged",
          "inputs": {
              "dense": tf_keras.Input(shape=(2, None, 1, 3), batch_size=10),
              "sparse": tf_keras.Input(shape=(2, None, 1, 3), sparse=True),
              "ragged": tf_keras.Input(shape=(2, None, None, 1), ragged=True),
          },
          "expected_shape": [10, 2, None, 1, None],
      },
  )
  def test_layer_correctness_keras_inputs(self, inputs, expected_shape):
    layer = concat_features.ConcatFeatures(feature_names=list(inputs.keys()))
    output = layer(inputs)

    KerasTensor = tf_keras.Input(shape=(1,)).__class__  # pylint: disable=invalid-name
    self.assertIsInstance(output, KerasTensor)
    self.assertEqual(tf.TensorShape(expected_shape), output.shape)

  def test_layer_stability(self):
    layer = concat_features.ConcatFeatures(
        feature_names=["dense", "sparse", "ragged"]
    )
    inputs = {
        "dense": tf.constant([-1.4, 2.0], shape=(2, 1)),
        "sparse": tf.sparse.SparseTensor(
            indices=[[0, 1], [1, 0]],
            values=[2.718, 3.14],
            dense_shape=[2, 2],
        ),
        "ragged": tf.ragged.constant([[5, 7.77], [8]]),
        "other_feature": tf.ones((2, 5)),
    }
    self.assertLayerStable(inputs=inputs, layer=layer)

  def test_layer_savable(self):
    layer = concat_features.ConcatFeatures(
        feature_names=["dense", "sparse", "ragged"]
    )
    inputs = {
        "dense": tf.constant([-1.4, 2.0], shape=(2, 1)),
        "sparse": tf.sparse.SparseTensor(
            indices=[[0, 1], [1, 0]],
            values=[2.718, 3.14],
            dense_shape=[2, 2],
        ),
        "ragged": tf.ragged.constant([[5, 7.77], [8]]),
        "other_feature": tf.ones((2, 5)),
    }
    self.assertLayerSavable(inputs=inputs, layer=layer)

  def test_missing_input_features(self):
    layer = concat_features.ConcatFeatures(feature_names=["feature"])

    with self.assertRaisesRegex(
        ValueError, "Layer inputs is missing features*"
    ):
      layer({"other_feature": tf.ones((3, 1))})

  def test_unsupported_tensor_type(self):
    class TestType(tf.experimental.ExtensionType):
      tensor: tf.Tensor

    layer = concat_features.ConcatFeatures(feature_names=["feature"])
    with self.assertRaisesRegex(TypeError, "Got unsupported tensor shape type"):
      layer({
          "feature": TestType(tensor=tf.ones((3, 1))),
          "other_feature": tf.ones((3, 1)),
      })

  def test_empty_feature_names_list(self):
    with self.assertRaisesRegex(
        ValueError, "feature_names must be a non-empty list"
    ):
      concat_features.ConcatFeatures(feature_names=[])

  def test_non_string_feature_name(self):
    with self.assertRaisesRegex(
        TypeError, "feature_names must be a list of strings"
    ):
      concat_features.ConcatFeatures(feature_names=["x", 1])

  @parameterized.named_parameters(
      {
          "testcase_name": "different_shapes_dense",
          "inputs": {
              "x1": tf.ones((2, 4)),
              "x2": tf.ones((1, 4)),
          },
      },
      {
          "testcase_name": "different_shapes_sparse",
          "inputs": {
              "x1": tf.ones((10, 4)),
              "x2": tf.sparse.SparseTensor(
                  indices=[[0, 0], [1, 2]], values=[1, 2], dense_shape=[3, 4]
              ),
          },
      },
      {
          "testcase_name": "different_shapes_ragged",
          "inputs": {
              "x1": tf.ones((2, 2, 2)),
              "x2": tf.ragged.constant([[5, 7], [0, 3, 1], [6]]),
          },
      },
      {
          "testcase_name": "keras_input_batch_size",
          "inputs": {
              "x1": tf_keras.Input(shape=(2, 3), batch_size=10),
              "x2": tf_keras.Input(shape=(2, 3), batch_size=4),
          },
      },
  )
  def test_shape_mismatch(self, inputs):
    layer = concat_features.ConcatFeatures(feature_names=list(inputs.keys()))
    with self.assertRaisesRegex(
        ValueError,
        (
            "All features from the feature_names set must be tensors with the"
            " same shape except for the last dimension"
        ),
    ):
      layer(inputs)

  @parameterized.named_parameters(
      {
          "testcase_name": "different_ranks_dense",
          "inputs": {
              "x1": tf.ones((2, 4)),
              "x2": tf.ones((2, 4, 6)),
          },
      },
      {
          "testcase_name": "different_ranks_sparse",
          "inputs": {
              "x1": tf.ones((3, 4, 1)),
              "x2": tf.sparse.SparseTensor(
                  indices=[[0, 0], [1, 2]], values=[1, 2], dense_shape=[3, 4]
              ),
          },
      },
      {
          "testcase_name": "different_ranks_ragged",
          "inputs": {
              "x1": tf.ones((2, 2, 2, 2)),
              "x2": tf.ragged.constant([[5, 7], [0, 3, 1], [6]]),
          },
      },
      {
          "testcase_name": "keras_input",
          "inputs": {
              "x1": tf_keras.Input(shape=(2, 3, 4), batch_size=4),
              "x2": tf_keras.Input(shape=(2, 3), batch_size=4),
          },
      },
  )
  def test_rank_mismatch(self, inputs):
    layer = concat_features.ConcatFeatures(feature_names=list(inputs.keys()))
    with self.assertRaisesRegex(
        ValueError,
        (
            "All features from the feature_names set must be tensors with the"
            " same shape except for the last dimension"
        ),
    ):
      layer(inputs)

  def test_layer_config(self):
    layer = concat_features.ConcatFeatures(
        feature_names=["feature1", "feature2"], name="encoder", dtype=tf.float64
    )
    self.assertLayerConfigurable(layer=layer, serializable=True)


if __name__ == "__main__":
  tf.test.main()
