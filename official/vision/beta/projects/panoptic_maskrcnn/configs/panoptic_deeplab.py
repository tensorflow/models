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

"""Panoptic Mask R-CNN configuration definition."""

import dataclasses
from typing import List, Optional, Union

from official.modeling import hyperparams
from official.vision.beta.configs import common
from official.vision.beta.configs import backbones
from official.vision.beta.configs import decoders
from official.vision.beta.configs import semantic_segmentation

SEGMENTATION_HEAD = semantic_segmentation.SegmentationHead

_COCO_INPUT_PATH_BASE = 'coco/tfrecords'
_COCO_TRAIN_EXAMPLES = 118287
_COCO_VAL_EXAMPLES = 5000


@dataclasses.dataclass
class InstanceCenterHead(semantic_segmentation.SegmentationHead):
  """Instance Center head config."""
  # None, deeplabv3plus, panoptic_fpn_fusion, 
  # panoptic_deeplab_fusion or pyramid_fusion
  kernel_size: int = 5
  feature_fusion: Optional[str] = None  
  low_level: Union[int, List[int]] = dataclasses.field(
      default_factory=lambda: [3, 2])
  low_level_num_filters: Union[int, List[int]] = dataclasses.field(
      default_factory=lambda: [64, 32])


# pytype: disable=wrong-keyword-args
@dataclasses.dataclass
class PanopticDeeplab(hyperparams.Config):
  """Panoptic Mask R-CNN model config."""
  num_classes: int = 0
  input_size: List[int] = dataclasses.field(default_factory=list)
  min_level: int = 3
  max_level: int = 6
  norm_activation: common.NormActivation = common.NormActivation()
  backbone: backbones.Backbone = backbones.Backbone(
      type='resnet', resnet=backbones.ResNet())
  decoder: decoders.Decoder = decoders.Decoder(type='aspp')
  semantic_head: SEGMENTATION_HEAD = SEGMENTATION_HEAD()
  instance_head: InstanceCenterHead = InstanceCenterHead(
      low_level=[3, 2])
  shared_decoder: bool = False
