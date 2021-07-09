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
from typing import List

import dataclasses
from official.vision.beta.configs import maskrcnn
from official.vision.beta.configs import semantic_segmentation

# pylint: disable=missing-class-docstring

@dataclasses.dataclass
class Parser(maskrcnn.Parser):
  # If resize_eval_groundtruth is set to False, original image sizes are used
  # for eval. In that case, groundtruth_padded_size has to be specified too to
  # allow for batching the variable input sizes of images.
  resize_eval_segmentation_groundtruth: bool = True
  segmentation_groundtruth_padded_size: List[int] = dataclasses.field(
      default_factory=list)
  segmentation_ignore_label: int = 255

@dataclasses.dataclass
class DataConfig(maskrcnn.DataConfig):
  """Input config for training."""
  parser: Parser = Parser()

@dataclasses.dataclass
class PanopticMaskRCNN(maskrcnn.MaskRCNN):
  """Panoptic Mask R-CNN model config."""
  segmentation_model: semantic_segmentation.SemanticSegmentationModel = (
      semantic_segmentation.SemanticSegmentationModel(num_classes=2))
  shared_backbone: bool = True
  shared_decoder: bool = True
