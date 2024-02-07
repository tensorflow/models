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

"""Configurations for loading checkpoints."""

import dataclasses
from typing import Dict, Optional

import numpy as np

from official.projects.centernet.utils.checkpoints import config_classes

Conv2DBNCFG = config_classes.Conv2DBNCFG
HeadConvCFG = config_classes.HeadConvCFG
ResidualBlockCFG = config_classes.ResidualBlockCFG
HourglassCFG = config_classes.HourglassCFG


@dataclasses.dataclass
class BackboneConfigData:
  """Backbone Config."""

  weights_dict: Optional[Dict[str, np.ndarray]] = dataclasses.field(
      repr=False, default=None)

  def get_cfg_list(self, name):
    """Get list of block configs for the module."""

    if name == 'hourglass104_512':
      return [
          # Downsampling Layers
          Conv2DBNCFG(
              weights_dict=self.weights_dict['downsample_input']['conv_block']),
          ResidualBlockCFG(
              weights_dict=self.weights_dict['downsample_input'][
                  'residual_block']),
          # Hourglass
          HourglassCFG(
              weights_dict=self.weights_dict['hourglass_network']['0']),
          Conv2DBNCFG(
              weights_dict=self.weights_dict['output_conv']['0']),
          # Intermediate
          Conv2DBNCFG(
              weights_dict=self.weights_dict['intermediate_conv1']['0']),
          Conv2DBNCFG(
              weights_dict=self.weights_dict['intermediate_conv2']['0']),
          ResidualBlockCFG(
              weights_dict=self.weights_dict['intermediate_residual']['0']),
          # Hourglass
          HourglassCFG(
              weights_dict=self.weights_dict['hourglass_network']['1']),
          Conv2DBNCFG(
              weights_dict=self.weights_dict['output_conv']['1']),
      ]

    elif name == 'extremenet':
      return [
          # Downsampling Layers
          Conv2DBNCFG(
              weights_dict=self.weights_dict['downsample_input']['conv_block']),
          ResidualBlockCFG(
              weights_dict=self.weights_dict['downsample_input'][
                  'residual_block']),
          # Hourglass
          HourglassCFG(
              weights_dict=self.weights_dict['hourglass_network']['0']),
          Conv2DBNCFG(
              weights_dict=self.weights_dict['output_conv']['0']),
          # Intermediate
          Conv2DBNCFG(
              weights_dict=self.weights_dict['intermediate_conv1']['0']),
          Conv2DBNCFG(
              weights_dict=self.weights_dict['intermediate_conv2']['0']),
          ResidualBlockCFG(
              weights_dict=self.weights_dict['intermediate_residual']['0']),
          # Hourglass
          HourglassCFG(
              weights_dict=self.weights_dict['hourglass_network']['1']),
          Conv2DBNCFG(
              weights_dict=self.weights_dict['output_conv']['1']),
      ]


@dataclasses.dataclass
class HeadConfigData:
  """Head Config."""

  weights_dict: Optional[Dict[str, np.ndarray]] = dataclasses.field(
      repr=False, default=None)

  def get_cfg_list(self, name):
    if name == 'detection_2d':
      return [
          HeadConvCFG(weights_dict=self.weights_dict['object_center']['0']),
          HeadConvCFG(weights_dict=self.weights_dict['object_center']['1']),
          HeadConvCFG(weights_dict=self.weights_dict['box.Soffset']['0']),
          HeadConvCFG(weights_dict=self.weights_dict['box.Soffset']['1']),
          HeadConvCFG(weights_dict=self.weights_dict['box.Sscale']['0']),
          HeadConvCFG(weights_dict=self.weights_dict['box.Sscale']['1'])
      ]
