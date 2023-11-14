# Copyright 2023 The TensorFlow Authors. All Rights Reserved.
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

"""Defines an encoder for concatenating input features into a single tensor."""

from typing import Mapping, Sequence

import tensorflow as tf, tf_keras

from official.recommendation.uplift import types


@tf_keras.utils.register_keras_serializable(package="Uplift")
class ConcatFeatures(tf_keras.layers.Layer):
  """Concatenates features into a single dense tensor.

  Takes a dictionary of feature tensors as input and concatenates the specified
  features into a single tensor. The tensors are concatenated along their last
  axis. Sparse and ragged tensors are converted to dense tensors before being
  concatenated.
  """

  def __init__(self, feature_names: Sequence[str], **kwargs):
    """Initializes a feature concatenation encoder.

    Args:
      feature_names: names of the input features to concatenate together.
      **kwargs: base layer keyword arguments.
    """
    super().__init__(**kwargs)
    self._feature_names = feature_names

    # Validate feature names.
    if not feature_names:
      raise ValueError(
          "feature_names must be a non-empty list of strings but got"
          f" {feature_names} instead."
      )
    if not all(isinstance(name, str) for name in feature_names):
      raise TypeError(
          "feature_names must be a list of strings, but got types"
          f" {list(map(type, feature_names))}"
      )

  def build(self, input_shapes: Mapping[str, tf.TensorShape]) -> None:
    missing_features = set(self._feature_names) - input_shapes.keys()
    if missing_features:
      raise ValueError(f"Layer inputs is missing features: {missing_features}")

    feature_shapes = {
        feature_name: tensor_shape
        for feature_name, tensor_shape in input_shapes.items()
        if feature_name in self._feature_names
    }

    most_specific_shape = tf.TensorShape(None)
    for feature_name, shape in feature_shapes.items():
      if not isinstance(shape, tf.TensorShape):
        raise TypeError(
            f"Got unsupported tensor shape type for feature {feature_name}. The"
            " feature tensor must be one of `tf.Tensor`, `tf.SparseTensor` or"
            " `tf.RaggedTensor`, with a well defined tensor shape but got shape"
            f" {shape} instead."
        )

      shape = shape[:-1]
      if shape.is_subtype_of(most_specific_shape):
        most_specific_shape = shape

      elif not most_specific_shape.is_subtype_of(shape):
        raise ValueError(
            "All features from the feature_names set must be tensors with the"
            " same shape except for the last dimension, but got features with"
            f" incompatible shapes {feature_shapes}"
        )

    super().build(input_shapes)

  def call(self, inputs: types.DictOfTensors) -> tf.Tensor:
    features = []

    for feature_name, feature in inputs.items():
      if feature_name in self._feature_names:
        if isinstance(feature, tf.Tensor):
          features.append(feature)
        elif isinstance(feature, tf.SparseTensor):
          features.append(tf.sparse.to_dense(feature))
        elif isinstance(feature, tf.RaggedTensor):
          features.append(feature.to_tensor())
        else:
          raise TypeError(
              f"Got unsupported tensor type for feature {feature_name}. The"
              " feature tensor must be one of `tf.Tensor`, `tf.SparseTensor` or"
              f" `tf.RaggedTensor`, but got {feature} instead."
          )

    return tf.concat(features, axis=-1)

  def get_config(self):
    config = super().get_config()
    config.update({"feature_names": self._feature_names})
    return config
