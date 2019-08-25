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

"""Utils for Keras models."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import collections

# This is to specify the custom config of model structures. For example,
# ConvDefs(conv_name='conv_pw_12', filters=512) for Mobilenet V1 is to specify
# the filters of the conv layer with name 'conv_pw_12' as 512.s
ConvDefs = collections.namedtuple('ConvDefs', ['conv_name', 'filters'])


def get_conv_def(conv_defs, layer_name):
  """Get the custom config for some layer of the model structure.

  Args:
    conv_defs: A named tuple to specify the custom config of the model
      network. See `ConvDefs` for details.
    layer_name: A string, the name of the layer to be customized.

  Returns:
    The number of filters for the layer, or `None` if there is no custom
    config for the requested layer.
  """
  for conv_def in conv_defs:
    if layer_name == conv_def.conv_name:
      return conv_def.filters
  return None
