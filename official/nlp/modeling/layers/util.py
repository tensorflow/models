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

"""Keras-based transformer block layer."""

import functools

import tensorflow as tf, tf_keras


class TfFunctionIfEagerDecorator(object):
  """Helper decorator function to optionally apply the @tf.function annotation."""

  def __init__(self, **kwargs):
    self.func_kwargs = kwargs

  def __call__(self, func):

    @functools.wraps(func)
    def wrapped_func(*args):
      # TODO(b/150147476, b/150024785): Fix tf.function in TF1 crash.
      if not hasattr(tf.compat.v1, 'executing_eagerly_outside_functions'
                    ) or tf.compat.v1.executing_eagerly_outside_functions():
        return tf.function(func=func, **self.func_kwargs)(*args)
      return func(*args)

    # Cache the created function in self._call_impl.
    if not hasattr(self, '_call_impl'):
      self._call_impl = wrapped_func
    return self._call_impl


def tf_function_if_eager(**kwargs):
  """Applies the @tf.function decorator only if running in eager mode."""
  return TfFunctionIfEagerDecorator(**kwargs)


def filter_kwargs(kwargs):
  """In place removes unused options in kwargs.

  This function removes the construction signatures: e.g.
  number_attention_heads... in TransformerEncoderBlock. This is needed,
  otherwise base_layer.py in Keras will complain.
  Args:
    kwargs: keyword arguments to be filtered.
  """
  # This is the union of signatures of TransformerEncoderBlock and
  # ReZeroTransformer. Every Transformer
  # block that uses compatible signature with TransformerEncoderBlock should
  # call this function before base constructor super().__init__(**kwargs).
  denylist = [
      'num_attention_heads', 'intermediate_size', 'intermediate_activation',
      'inner_dim', 'inner_activation', 'output_range', 'kernel_initializer',
      'bias_initializer', 'kernel_regularizer', 'bias_regularizer',
      'activity_regularizer', 'kernel_constraint', 'bias_constraint',
      'use_bias', 'norm_first', 'norm_epsilon', 'output_dropout',
      'attention_dropout', 'inner_dropout', 'attention_initializer',
      'attention_axes', 'share_rezero'
  ]
  for unused_key in denylist:
    kwargs.pop(unused_key, None)
