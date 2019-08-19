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

"""Bring in the shared ResNet modules into this module.

The TensorFlow v1 official models are moved under official/r1/resnet. In order
to be backward compatible with models that directly import v1 modules, we import
the v1 ResNet modules under official.resnet.

New TF models should not depend on modules directly under this path (which will
soon be deprecated and removed).
"""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from official.r1.resnet import cifar10_main
from official.r1.resnet import imagenet_main
from official.r1.resnet import imagenet_preprocessing
from official.r1.resnet import resnet_model
from official.r1.resnet import resnet_run_loop

del absolute_import
del division
del print_function
