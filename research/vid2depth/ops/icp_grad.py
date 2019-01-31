# Copyright 2017 The TensorFlow Authors All Rights Reserved.
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

"""The gradient of the icp op."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from tensorflow.python.framework import ops


@ops.RegisterGradient('Icp')
def _icp_grad(op, grad_transform, grad_residual):
  """The gradients for `icp`.

  Args:
    op: The `icp` `Operation` that we are differentiating, which we can use
      to find the inputs and outputs of the original op.
    grad_transform: Gradient with respect to `transform` output of the `icp` op.
    grad_residual: Gradient with respect to `residual` output of the
      `icp` op.

  Returns:
    Gradients with respect to the inputs of `icp`.
  """
  unused_transform = op.outputs[0]
  unused_residual = op.outputs[1]
  unused_source = op.inputs[0]
  unused_ego_motion = op.inputs[1]
  unused_target = op.inputs[2]

  grad_p = -grad_residual
  grad_ego_motion = -grad_transform

  return [grad_p, grad_ego_motion, None]
