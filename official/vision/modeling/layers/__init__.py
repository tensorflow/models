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

"""Layers package definition."""

from official.vision.modeling.layers.box_sampler import BoxSampler
from official.vision.modeling.layers.detection_generator import DetectionGenerator
from official.vision.modeling.layers.detection_generator import MultilevelDetectionGenerator
from official.vision.modeling.layers.mask_sampler import MaskSampler
from official.vision.modeling.layers.nn_blocks import BottleneckBlock
from official.vision.modeling.layers.nn_blocks import BottleneckResidualInner
from official.vision.modeling.layers.nn_blocks import DepthwiseSeparableConvBlock
from official.vision.modeling.layers.nn_blocks import InvertedBottleneckBlock
from official.vision.modeling.layers.nn_blocks import ResidualBlock
from official.vision.modeling.layers.nn_blocks import ResidualInner
from official.vision.modeling.layers.nn_blocks import ReversibleLayer
from official.vision.modeling.layers.nn_blocks_3d import BottleneckBlock3D
from official.vision.modeling.layers.nn_blocks_3d import SelfGating
from official.vision.modeling.layers.nn_layers import CausalConvMixin
from official.vision.modeling.layers.nn_layers import Conv2D
from official.vision.modeling.layers.nn_layers import Conv3D
from official.vision.modeling.layers.nn_layers import DepthwiseConv2D
from official.vision.modeling.layers.nn_layers import GlobalAveragePool3D
from official.vision.modeling.layers.nn_layers import PositionalEncoding
from official.vision.modeling.layers.nn_layers import Scale
from official.vision.modeling.layers.nn_layers import SpatialAveragePool3D
from official.vision.modeling.layers.nn_layers import SqueezeExcitation
from official.vision.modeling.layers.nn_layers import StochasticDepth
from official.vision.modeling.layers.nn_layers import TemporalSoftmaxPool
from official.vision.modeling.layers.roi_aligner import MultilevelROIAligner
from official.vision.modeling.layers.roi_generator import MultilevelROIGenerator
from official.vision.modeling.layers.roi_sampler import ROISampler
