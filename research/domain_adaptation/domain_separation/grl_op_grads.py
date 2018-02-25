# Copyright 2016 The TensorFlow Authors All Rights Reserved.
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

"""Gradients for operators defined in grl_ops.py."""
import tensorflow as tf


@tf.RegisterGradient("GradientReversal")
def _GradientReversalGrad(_, grad):
  """The gradients for `gradient_reversal`.

  Args:
    _: The `gradient_reversal` `Operation` that we are differentiating,
      which we can use to find the inputs and outputs of the original op.
    grad: Gradient with respect to the output of the `gradient_reversal` op.

  Returns:
    Gradient with respect to the input of `gradient_reversal`, which is simply
    the negative of the input gradient.

  """
  return tf.negative(grad)
