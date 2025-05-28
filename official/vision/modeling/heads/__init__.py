# Copyright 2025 The TensorFlow Authors. All Rights Reserved.
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

"""Heads package definition."""

from official.vision.modeling.heads.dense_prediction_heads import RetinaNetHead
from official.vision.modeling.heads.dense_prediction_heads import RPNHead
from official.vision.modeling.heads.instance_heads import DetectionHead
from official.vision.modeling.heads.instance_heads import MaskHead
from official.vision.modeling.heads.segmentation_heads import MaskScoring
from official.vision.modeling.heads.segmentation_heads import SegmentationHead
