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

"""Bring in the shared Keras ResNet modules into this module.

The TensorFlow official Keras models are moved under
    official/vision/image_classification
In order to be backward compatible with models that directly import its modules,
we import the Keras ResNet modules under official.resnet.keras.

New TF models should not depend on modules directly under this path.
"""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from official.vision.image_classification import cifar_preprocessing
from official.vision.image_classification import common as keras_common
from official.vision.image_classification import imagenet_preprocessing
from official.vision.image_classification import resnet_cifar_main as keras_cifar_main
from official.vision.image_classification import resnet_cifar_model
from official.vision.image_classification import resnet_imagenet_main as keras_imagenet_main
from official.vision.image_classification import resnet_model

del absolute_import
del division
del print_function
