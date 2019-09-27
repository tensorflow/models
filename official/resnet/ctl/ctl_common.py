# Copyright 2019 The TensorFlow Authors. All Rights Reserved.
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
"""Common util functions and classes used by CTL."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from absl import flags


def define_ctl_flags():
  """Define flags for CTL."""

  flags.DEFINE_boolean(name='use_tf_function', default=True,
                       help='Wrap the train and test step inside a '
                       'tf.function.')
  flags.DEFINE_boolean(name='single_l2_loss_op', default=False,
                       help='Calculate L2_loss on concatenated weights, '
                       'instead of using Keras per-layer L2 loss.')
