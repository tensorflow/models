# Copyright 2020 The TensorFlow Authors. All Rights Reserved.
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
"""Keras-based transformer block layer."""

import functools

import tensorflow as tf


class TfFunctionIfEagerDecorator(object):
  """Helper decorator function to optionally apply the @tf.function annotation."""

  def __init__(self, **kwargs):
    self.func_kwargs = kwargs

  def __call__(self, func):

    @functools.wraps(func)
    def wrapped_func(*args):
      # TODO(b/150147476, b/150024785): Fix tf.function in TF1 crash.
      if not hasattr(tf.compat.v1, "executing_eagerly_outside_functions"
                    ) or tf.compat.v1.executing_eagerly_outside_functions():
        return tf.function(func=func, **self.func_kwargs)(*args)
      return func(*args)

    # Cache the created function in self._call_impl.
    if not hasattr(self, "_call_impl"):
      self._call_impl = wrapped_func
    return self._call_impl


def tf_function_if_eager(**kwargs):
  """Applies the @tf.function decorator only if running in eager mode."""
  return TfFunctionIfEagerDecorator(**kwargs)
