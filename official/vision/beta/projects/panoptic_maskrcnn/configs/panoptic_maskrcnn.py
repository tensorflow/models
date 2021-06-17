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

# Lint as: python3
"""Panoptic Mask R-CNN configuration definition."""

import dataclasses

from official.vision.beta.configs import maskrcnn as maskrcnn_config
from official.vision.beta.configs import semantic_segmentation


@dataclasses.dataclass
class PanopticMaskRCNN(maskrcnn_config.MaskRCNN):
  """Panoptic Mask R-CNN model config."""
  segmentation_model: semantic_segmentation.SemanticSegmentationModel =\
      semantic_segmentation.SemanticSegmentationModel(num_classes=2)
  shared_backbone: bool = True
  shared_decoder: bool = True
