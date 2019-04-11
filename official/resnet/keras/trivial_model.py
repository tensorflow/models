# Copyright 2018 The TensorFlow Authors. All Rights Reserved.
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
"""A trivial model for Keras."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from tensorflow.python.keras import backend
from tensorflow.python.keras import layers
from tensorflow.python.keras import models


def trivial_model(num_classes):
  """Trivial model for ImageNet dataset."""

  input_shape = (224, 224, 3)
  img_input = layers.Input(shape=input_shape)

  x = layers.Lambda(lambda x: backend.reshape(x, [-1, 224 * 224 * 3]),
                    name='reshape')(img_input)
  x = layers.Dense(num_classes, activation='softmax', name='fc1000')(x)

  return models.Model(img_input, x, name='trivial')
