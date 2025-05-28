# Copyright 2025 The TensorFlow Authors. All Rights Reserved.
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

"""Keras dropout layer that is aware of `RecomputeContext`."""

import numpy as np
import tensorflow as tf, tf_keras

from official.projects.bigbird import recompute_grad as recompute_grad_lib
from official.projects.bigbird import stateless_dropout as stateless_dropout_lib


# Reimplements internal function
# https://github.com/tensorflow/tensorflow/blob/master/tensorflow/python/framework/smart_cond.py.
def smart_cond(pred, true_fn=None, false_fn=None, name=None):
  """Return either `true_fn()` if predicate `pred` is true else `false_fn()`.

  If `pred` is a bool or has a constant value, we return either `true_fn()`
  or `false_fn()`, otherwise we use `tf.cond` to dynamically route to both.

  Arguments:
    pred: A scalar determining whether to return the result of `true_fn` or
      `false_fn`.
    true_fn: The callable to be performed if pred is true.
    false_fn: The callable to be performed if pred is false.
    name: Optional name prefix when using `tf.cond`.

  Returns:
    Tensors returned by the call to either `true_fn` or `false_fn`.

  Raises:
    TypeError: If `true_fn` or `false_fn` is not callable.
  """
  if not callable(true_fn):
    raise TypeError('`true_fn` must be callable.')
  if not callable(false_fn):
    raise TypeError('`false_fn` must be callable.')
  pred_value = tf.get_static_value(pred)
  if isinstance(pred, tf.Variable) or pred_value is None:
    return tf.cond(
        pred, true_fn=true_fn, false_fn=false_fn, name=name)
  if pred_value:
    return true_fn()
  else:
    return false_fn()


# See https://www.tensorflow.org/api_docs/python/tf/keras/layers/Dropout.
class RecomputingDropout(tf_keras.layers.Layer):
  """`tf_keras.layers.Dropout` that supports `recompute_grad`."""

  def __init__(self,
               rate,
               noise_shape=None,
               seed=None,
               force_recomputation=False,
               **kwargs):
    """Initializes `RecomputingDropout`.

    Args:
      rate: Float between 0 and 1. Fraction of the input units to drop.
      noise_shape: 1D integer tensor representing the shape of the binary
        dropout mask that will be multiplied with the input. For instance, if
        inputs have shape `(batch_size, timesteps, features)` and you want the
        dropout mask to be the same for all timesteps, you can use
        `noise_shape=(batch_size, 1, features)`.
      seed: A Python integer to use as random seed.
      force_recomputation: If `True`, then raises an error if called outside a
        recompute context.
      **kwargs: Keyword arguments for `tf_keras.layers.Layer`.
    """

    super(RecomputingDropout, self).__init__(**kwargs)
    self.rate = rate
    self.noise_shape = noise_shape
    self.seed = seed
    self.force_recomputation = force_recomputation
    self.supports_masking = True
    # Create a layer-specific seed to combine with the global recompute seed.
    self._recompute_seed = (
        np.random.randint(-2**31, 2**31, dtype=np.int32)
        if seed is None else seed)

  def _get_noise_shape(self, inputs):
    # Subclasses of `Dropout` may implement `_get_noise_shape(self, inputs)`,
    # which will override `self.noise_shape`, and allows for custom noise
    # shapes with dynamically sized inputs.
    if self.noise_shape is None:
      return None

    concrete_inputs_shape = tf.shape(inputs)
    noise_shape = []
    for i, value in enumerate(self.noise_shape):
      noise_shape.append(concrete_inputs_shape[i] if value is None else value)
    return tf.convert_to_tensor(noise_shape)

  def call(self, inputs, training=None):
    """Builds computation graph.

    Args:
      inputs: Input tensor (of any rank).
      training: Python boolean indicating whether the layer should behave in
        training mode (adding dropout) or in inference mode (doing nothing).

    Returns:
      `inputs` masked according to layer configuration.

    Raises:
      ValueError: If `force_recomputation` is `True` and called outside a
        a recompute context.
    """
    if training is None:
      training = tf_keras.backend.learning_phase()

    def dropped_inputs():
      """Randomly drops elements of `inputs` when `training=True`."""
      recompute_context = recompute_grad_lib.get_recompute_context()
      if recompute_context is None:
        if self.force_recomputation:
          raise ValueError(
              'RecomputeContext is required when force_recomputation=True.')
        return tf.nn.dropout(
            inputs,
            noise_shape=self._get_noise_shape(inputs),
            seed=self.seed,
            rate=self.rate)
      seed = tf.stack([recompute_context.seed, self._recompute_seed])
      return stateless_dropout_lib.stateless_dropout(
          inputs,
          rate=self.rate,
          seed=seed,
          noise_shape=self._get_noise_shape(inputs))

    output = smart_cond(training, dropped_inputs, lambda: tf.identity(inputs))
    return output

  def compute_output_shape(self, input_shape):
    return input_shape

  def get_config(self):
    config = {
        'rate': self.rate,
        'noise_shape': self.noise_shape,
        'seed': self.seed,
        'force_recomputation': self.force_recomputation,
    }
    base_config = super(RecomputingDropout, self).get_config()
    return dict(list(base_config.items()) + list(config.items()))
