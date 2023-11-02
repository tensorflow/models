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

"""Functions and classes related to training performance."""

from absl import logging
import tensorflow as tf, tf_keras


def configure_optimizer(optimizer,
                        use_float16=False,
                        loss_scale=None,
                        use_graph_rewrite=None):
  """Configures optimizer object with performance options."""
  if use_graph_rewrite is not None:
    logging.warning('`use_graph_rewrite` is deprecated inside '
                    '`configure_optimizer`. Please remove the usage.')
  del use_graph_rewrite
  if use_float16:
    if loss_scale in (None, 'dynamic'):
      optimizer = tf_keras.mixed_precision.LossScaleOptimizer(optimizer)
    else:
      # loss_scale is a number. We interpret that as a fixed loss scale.
      optimizer = tf_keras.mixed_precision.LossScaleOptimizer(
          optimizer, dynamic=False, initial_scale=loss_scale)
  return optimizer


def set_mixed_precision_policy(dtype, loss_scale=None):
  """Sets the global `tf_keras.mixed_precision.Policy`."""
  # TODO(b/191894773): Remove loss_scale argument
  assert loss_scale is None, (
      'The loss_scale argument must be None. The argument exists for '
      'historical reasons and will be removed soon.')
  if dtype == tf.float16:
    tf_keras.mixed_precision.set_global_policy('mixed_float16')
  elif dtype == tf.bfloat16:
    tf_keras.mixed_precision.set_global_policy('mixed_bfloat16')
  elif dtype == tf.float32:
    tf_keras.mixed_precision.set_global_policy('float32')
  else:
    raise ValueError('Unexpected dtype: %s' % dtype)
