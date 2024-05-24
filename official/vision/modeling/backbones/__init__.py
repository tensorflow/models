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

"""Backbones package definition."""

from official.vision.modeling.backbones.efficientnet import EfficientNet
from official.vision.modeling.backbones.mobiledet import MobileDet
from official.vision.modeling.backbones.mobilenet import MobileNet
from official.vision.modeling.backbones.resnet import ResNet
from official.vision.modeling.backbones.resnet_3d import ResNet3D
from official.vision.modeling.backbones.resnet_deeplab import DilatedResNet
from official.vision.modeling.backbones.resnet_unet import ResNetUNet
from official.vision.modeling.backbones.revnet import RevNet
from official.vision.modeling.backbones.spinenet import SpineNet
from official.vision.modeling.backbones.spinenet_mobile import SpineNetMobile
from official.vision.modeling.backbones.vit import VisionTransformer
