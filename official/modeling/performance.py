# Lint as: python3
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
"""Functions and classes related to training performance."""

import tensorflow as tf


def configure_optimizer(optimizer,
                        use_float16=False,
                        use_graph_rewrite=False,
                        loss_scale='dynamic'):
  """Configures optimizer object with performance options."""
  if use_float16:
    # Wraps optimizer with a LossScaleOptimizer. This is done automatically
    # in compile() with the "mixed_float16" policy, but since we do not call
    # compile(), we must wrap the optimizer manually.
    optimizer = (
        tf.keras.mixed_precision.experimental.LossScaleOptimizer(
            optimizer, loss_scale=loss_scale))
  if use_graph_rewrite:
    # Note: the model dtype must be 'float32', which will ensure
    # tf.ckeras.mixed_precision and
    # tf.train.experimental.enable_mixed_precision_graph_rewrite do not double
    # up.
    optimizer = tf.train.experimental.enable_mixed_precision_graph_rewrite(
        optimizer)
  return optimizer


def set_mixed_precision_policy(dtype, loss_scale=None):
  """Sets mix precision policy."""
  if dtype == tf.float16:
    policy = tf.keras.mixed_precision.experimental.Policy(
        'mixed_float16', loss_scale=loss_scale)
    tf.keras.mixed_precision.experimental.set_policy(policy)
  elif dtype == tf.bfloat16:
    policy = tf.keras.mixed_precision.experimental.Policy('mixed_bfloat16')
    tf.keras.mixed_precision.experimental.set_policy(policy)
  elif dtype == tf.float32:
    tf.keras.mixed_precision.experimental.set_policy('float32')
  else:
    raise ValueError('Unexpected dtype: %s' % dtype)
