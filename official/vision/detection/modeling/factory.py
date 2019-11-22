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
"""Factory to build detection model."""


from official.vision.detection.modeling import retinanet_model


def model_generator(params):
  """Model function generator."""
  if params.type == 'retinanet':
    model_fn = retinanet_model.RetinanetModel(params)
  else:
    raise ValueError('Model %s is not supported.'% params.type)

  return model_fn
