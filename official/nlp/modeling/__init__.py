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

"""NLP Modeling Library.

This library provides a set of Keras primitives (`tf.keras.Layer` and
`tf.keras.Model`) that can be assembled into transformer-based models.
They are flexible, validated, interoperable, and both TF1 and TF2 compatible.
"""
from official.nlp.modeling import layers
from official.nlp.modeling import losses
from official.nlp.modeling import models
from official.nlp.modeling import networks
