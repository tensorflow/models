# Copyright 2021 The TensorFlow Authors. All Rights Reserved.
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

"""Functions and classes related to training performance."""

import tensorflow as tf


def configure_optimizer(optimizer,
                        use_float16=False,
                        use_graph_rewrite=False,
                        loss_scale=None):
  """Configures optimizer object with performance options."""
  if use_float16:
    if loss_scale in (None, 'dynamic'):
      optimizer = tf.keras.mixed_precision.LossScaleOptimizer(optimizer)
    else:
      # loss_scale is a number. We interpret that as a fixed loss scale.
      optimizer = tf.keras.mixed_precision.LossScaleOptimizer(
          optimizer, dynamic=False, initial_scale=loss_scale)
  if use_graph_rewrite:
    # Note: the model dtype must be 'float32', which will ensure
    # tf.keras.mixed_precision and enable_mixed_precision_graph_rewrite do not
    # double up.
    optimizer = (
        tf.compat.v1.mixed_precision.enable_mixed_precision_graph_rewrite(
            optimizer))
  return optimizer


def set_mixed_precision_policy(dtype, loss_scale=None):
  """Sets the global `tf.keras.mixed_precision.Policy`."""
  # TODO(b/191894773): Remove loss_scale argument
  assert loss_scale is None, (
      'The loss_scale argument must be None. The argument exists for '
      'historical reasons and will be removed soon.')
  if dtype == tf.float16:
    tf.keras.mixed_precision.set_global_policy('mixed_float16')
  elif dtype == tf.bfloat16:
    tf.keras.mixed_precision.set_global_policy('mixed_bfloat16')
  elif dtype == tf.float32:
    tf.keras.mixed_precision.set_global_policy('float32')
  else:
    raise ValueError('Unexpected dtype: %s' % dtype)
