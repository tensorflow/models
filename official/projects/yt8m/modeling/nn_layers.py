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

"""Contains model definitions."""

import functools
from typing import Any, Dict, Optional

import tensorflow as tf

from official.modeling import tf_utils
from official.projects.yt8m.configs import yt8m as yt8m_cfg
from official.projects.yt8m.modeling import yt8m_model_utils as utils


layers = tf.keras.layers


class Dbof(tf.keras.Model):
  """A YT8M model class builder.

  Creates a Deep Bag of Frames model.
  The model projects the features for each frame into a higher dimensional
  'clustering' space, pools across frames in that space, and then
  uses a configurable video-level model to classify the now aggregated features.
  The model will randomly sample either frames or sequences of frames during
  training to speed up convergence.
  """

  def __init__(
      self,
      params: yt8m_cfg.DbofModel,
      num_classes: int = 3862,
      input_specs: layers.InputSpec = layers.InputSpec(
          shape=[None, None, 1152]),
      l2_weight_decay: Optional[float] = None,
      **kwargs):
    """YT8M initialization function.

    Args:
      params: model configuration parameters
      num_classes: `int` number of classes in dataset.
      input_specs: `tf.keras.layers.InputSpec` specs of the input tensor.
        [batch_size x num_frames x num_features]
      l2_weight_decay: An optional `float` of kernel regularizer weight decay.
      **kwargs: keyword arguments to be passed.
    """
    self._self_setattr_tracking = False
    self._num_classes = num_classes
    self._input_specs = input_specs
    self._params = params
    self._l2_weight_decay = l2_weight_decay
    self._act_fn = tf_utils.get_activation(params.norm_activation.activation)
    self._norm = functools.partial(
        layers.BatchNormalization,
        momentum=params.norm_activation.norm_momentum,
        epsilon=params.norm_activation.norm_epsilon,
        synchronized=params.norm_activation.use_sync_bn)

    # Divide weight decay by 2.0 to match the implementation of tf.nn.l2_loss.
    # (https://www.tensorflow.org/api_docs/python/tf/keras/regularizers/l2)
    # (https://www.tensorflow.org/api_docs/python/tf/nn/l2_loss)
    l2_regularizer = (
        tf.keras.regularizers.l2(l2_weight_decay / 2.0)
        if l2_weight_decay
        else None
    )

    # [batch_size x num_frames x num_features]
    feature_size = input_specs.shape[-1]
    # shape 'excluding' batch_size
    model_input = tf.keras.Input(shape=self._input_specs.shape[1:])
    # normalize input features
    input_data = tf.nn.l2_normalize(model_input, -1)
    tf.summary.histogram("input_hist", input_data)

    # configure model
    if params.add_batch_norm:
      input_data = self._norm(name="input_bn")(input_data)

    # activation = reshaped input * cluster weights
    if params.cluster_size > 0:
      activation = layers.Dense(
          params.cluster_size,
          kernel_regularizer=l2_regularizer,
          kernel_initializer=tf.random_normal_initializer(
              stddev=1 / tf.sqrt(tf.cast(feature_size, tf.float32))))(
                  input_data)

    if params.add_batch_norm:
      activation = self._norm(name="cluster_bn")(activation)
    else:
      cluster_biases = tf.Variable(
          tf.random_normal_initializer(stddev=1 / tf.math.sqrt(feature_size))(
              shape=[params.cluster_size]),
          name="cluster_biases")
      tf.summary.histogram("cluster_biases", cluster_biases)
      activation += cluster_biases

    activation = self._act_fn(activation)
    tf.summary.histogram("cluster_output", activation)

    if params.use_context_gate_cluster_layer:
      pooling_method = None
      norm_args = dict(name="context_gate_bn")
      activation = utils.context_gate(
          activation,
          normalizer_fn=self._norm,
          normalizer_params=norm_args,
          pooling_method=pooling_method,
          hidden_layer_size=params.context_gate_cluster_bottleneck_size,
          kernel_regularizer=l2_regularizer)

    activation = utils.frame_pooling(activation, params.pooling_method)

    # activation = activation * hidden1_weights
    activation = layers.Dense(
        params.hidden_size,
        kernel_regularizer=l2_regularizer,
        kernel_initializer=tf.random_normal_initializer(
            stddev=1 / tf.sqrt(tf.cast(params.cluster_size, tf.float32))))(
                activation)

    if params.add_batch_norm:
      activation = self._norm(name="hidden1_bn")(activation)

    else:
      hidden1_biases = tf.Variable(
          tf.random_normal_initializer(stddev=0.01)(shape=[params.hidden_size]),
          name="hidden1_biases")

      tf.summary.histogram("hidden1_biases", hidden1_biases)
      activation += hidden1_biases

    activation = self._act_fn(activation)
    tf.summary.histogram("hidden1_output", activation)

    super().__init__(inputs=model_input, outputs=activation, **kwargs)


class LogisticModel(tf.keras.Model):
  """Logistic prediction head model with L2 regularization."""

  def __init__(
      self,
      input_specs: layers.InputSpec = layers.InputSpec(shape=[None, 128]),
      vocab_size: int = 3862,
      l2_penalty: float = 1e-8,
      **kwargs,
  ):
    """Creates a logistic model.

    Args:
      input_specs: 'batch' x 'num_features' matrix of input features.
      vocab_size: The number of classes in the dataset.
      l2_penalty: L2 weight regularization ratio.
      **kwargs: extra key word args.

    Returns:
      A dictionary with a tensor containing the probability predictions of the
      model in the 'predictions' key. The dimensions of the tensor are
      batch_size x num_classes.
    """
    inputs = tf.keras.Input(shape=input_specs.shape[1:])
    output = layers.Dense(
        vocab_size,
        activation=tf.nn.sigmoid,
        kernel_regularizer=tf.keras.regularizers.l2(l2_penalty))(
            inputs)

    super().__init__(inputs=inputs, outputs={"predictions": output}, **kwargs)


class MoeModel(tf.keras.Model):
  """A softmax over a mixture of logistic models (with L2 regularization)."""

  def __init__(
      self,
      input_specs: layers.InputSpec = layers.InputSpec(shape=[None, 128]),
      vocab_size: int = 3862,
      num_mixtures: int = 2,
      use_input_context_gate: bool = False,
      use_output_context_gate: bool = False,
      normalizer_params: Optional[Dict[str, Any]] = None,
      vocab_as_last_dim: bool = False,
      l2_penalty: float = 1e-5,
      **kwargs,
  ):
    """Creates a Mixture of (Logistic) Experts model.

     The model consists of a per-class softmax distribution over a
     configurable number of logistic classifiers. One of the classifiers
     in the mixture is not trained, and always predicts 0.
    Args:
      input_specs: 'batch_size' x 'num_features' matrix of input features.
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
      l2_penalty: How much to penalize the squared magnitudes of parameter
        values.
      **kwargs: extra key word args.

    Returns:
      A dictionary with a tensor containing the probability predictions
      of the model in the 'predictions' key. The dimensions of the tensor
      are batch_size x num_classes.
    """
    inputs = tf.keras.Input(shape=input_specs.shape[1:])
    model_input = inputs

    if use_input_context_gate:
      model_input = utils.context_gate(
          model_input,
          normalizer_fn=layers.BatchNormalization,
          normalizer_params=normalizer_params,
      )

    gate_activations = layers.Dense(
        vocab_size * (num_mixtures + 1),
        activation=None,
        bias_initializer=None,
        kernel_regularizer=tf.keras.regularizers.l2(l2_penalty))(
            model_input)
    expert_activations = layers.Dense(
        vocab_size * num_mixtures,
        activation=None,
        kernel_regularizer=tf.keras.regularizers.l2(l2_penalty))(
            model_input)

    if vocab_as_last_dim:
      # Batch x (num_mixtures + 1) x #Labels
      gate_activations = tf.reshape(
          gate_activations, [-1, num_mixtures + 1, vocab_size])
      # Batch x num_mixtures x #Labels
      expert_activations = tf.reshape(
          expert_activations, [-1, num_mixtures, vocab_size])
    else:
      # (Batch * #Labels) x (num_mixtures + 1)
      gate_activations = tf.reshape(gate_activations, [-1, num_mixtures + 1])
      # (Batch * #Labels) x num_mixtures
      expert_activations = tf.reshape(expert_activations, [-1, num_mixtures])

    gating_distribution = tf.nn.softmax(gate_activations, axis=1)
    expert_distribution = tf.nn.sigmoid(expert_activations)
    final_probabilities = tf.reduce_sum(
        gating_distribution[:, :num_mixtures] * expert_distribution, axis=1)

    if not vocab_as_last_dim:
      final_probabilities = tf.reshape(final_probabilities, [-1, vocab_size])

    if use_output_context_gate:
      final_probabilities = utils.context_gate(
          final_probabilities,
          normalizer_fn=layers.BatchNormalization,
          normalizer_params=normalizer_params,
      )
    super().__init__(
        inputs=inputs,
        outputs={"predictions": final_probabilities},
        **kwargs,
    )
