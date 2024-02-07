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

"""MoE model definitions."""


from typing import Any, Optional

import tensorflow as tf, tf_keras

from official.projects.yt8m.modeling import nn_layers


layers = tf_keras.layers


class MoeModel(layers.Layer):
  """A softmax over a mixture of logistic models (with L2 regularization)."""

  def __init__(
      self,
      vocab_size: int = 3862,
      num_mixtures: int = 2,
      use_input_context_gate: bool = False,
      use_output_context_gate: bool = False,
      normalizer_params: Optional[dict[str, Any]] = None,
      vocab_as_last_dim: bool = False,
      l2_regularizer: Optional[tf_keras.regularizers.Regularizer] = None,
      **kwargs,
  ):
    """Creates a Mixture of (Logistic) Experts model.

     The model consists of a per-class softmax distribution over a
     configurable number of logistic classifiers. One of the classifiers
     in the mixture is not trained, and always predicts 0.

    Args:
      vocab_size: The number of classes in the dataset.
      num_mixtures: The number of mixtures (excluding a dummy 'expert' that
        always predicts the non-existence of an entity).
      use_input_context_gate: if True apply context gate layer to the input.
      use_output_context_gate: if True apply context gate layer to the output.
      normalizer_params: parameters of the batch normalization.
      vocab_as_last_dim: if True reshape `activations` and make `vocab_size` as
        the last dimension to avoid small `num_mixtures` as the last dimension.
        XLA pads up the dimensions of tensors: typically the last dimension will
        be padded to 128, and the second to last will be padded to 8.
      l2_regularizer: An optional L2 weight regularizer.
      **kwargs: extra key word args.
    """
    super().__init__(**kwargs)
    self._vocab_size = vocab_size
    self._num_mixtures = num_mixtures
    self._use_input_context_gate = use_input_context_gate
    self._use_output_context_gate = use_output_context_gate
    self._vocab_as_last_dim = vocab_as_last_dim
    self._normalizer_params = normalizer_params
    self._l2_regularizer = l2_regularizer

    if use_input_context_gate:
      self._input_context_gate = nn_layers.ContextGate(
          normalizer_fn=layers.BatchNormalization,
          normalizer_params=normalizer_params,
          name="input_context_gate",
      )
    if use_output_context_gate:
      self._output_context_gate = nn_layers.ContextGate(
          normalizer_fn=layers.BatchNormalization,
          normalizer_params=normalizer_params,
          name="output_context_gate",
      )

    self._gate_dense = layers.Dense(
        vocab_size * (num_mixtures + 1),
        activation=None,
        bias_initializer=None,
        kernel_regularizer=l2_regularizer,
        name="gate",
    )

    self._expert_dense = layers.Dense(
        vocab_size * num_mixtures,
        activation=None,
        kernel_regularizer=l2_regularizer,
        name="expert",
    )

  def call(self, inputs: tf.Tensor) -> dict[str, tf.Tensor]:
    """MoE forward call.

    Args:
      inputs: 'batch_size' x 'num_features' matrix of input features.

    Returns:
      A dictionary with a tensor containing the probability predictions
      of the model in the 'predictions' key. The dimensions of the tensor
      are batch_size x num_classes.
    """

    if self._use_input_context_gate:
      inputs = self._input_context_gate(inputs)

    gate_activations = self._gate_dense(inputs)
    expert_activations = self._expert_dense(inputs)

    if self._vocab_as_last_dim:
      # Batch x (num_mixtures + 1) x #Labels
      gate_activations = tf.reshape(
          gate_activations, [-1, self._num_mixtures + 1, self._vocab_size]
      )
      # Batch x num_mixtures x #Labels
      expert_activations = tf.reshape(
          expert_activations,
          [-1, self._num_mixtures, self._vocab_size],
      )
    else:
      # (Batch * #Labels) x (num_mixtures + 1)
      gate_activations = tf.reshape(
          gate_activations,
          [-1, self._num_mixtures + 1],
      )
      # (Batch * #Labels) x num_mixtures
      expert_activations = tf.reshape(
          expert_activations,
          [-1, self._num_mixtures],
      )

    gating_distribution = tf.nn.softmax(gate_activations, axis=1)
    expert_distribution = tf.nn.sigmoid(expert_activations)
    final_probabilities = tf.reduce_sum(
        gating_distribution[:, : self._num_mixtures] * expert_distribution,
        axis=1,
    )

    if not self._vocab_as_last_dim:
      final_probabilities = tf.reshape(
          final_probabilities,
          [-1, self._vocab_size],
      )

    return {"predictions": final_probabilities}
