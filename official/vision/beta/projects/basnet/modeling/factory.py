# Copyright 2020 The TensorFlow Authors. All Rights Reserved.
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
"""Factory methods to build models."""

# Import libraries

import tensorflow as tf


from official.vision.beta.projects.basnet.configs import basnet as basnet_cfg
from official.vision.beta.projects.basnet.modeling import backbones
from official.vision.beta.projects.basnet.modeling import basnet_model
from official.vision.beta.projects.basnet.modeling.decoders import factory as decoder_factory
from official.vision.beta.projects.basnet.modeling.modules import refunet

def build_basnet_model(
    input_specs: tf.keras.layers.InputSpec,
    model_config: basnet_cfg.BASNetModel,
    l2_regularizer: tf.keras.regularizers.Regularizer = None):
  """Builds BASNet model."""
  backbone = backbones.factory.build_backbone(
      input_specs=input_specs,
      model_config=model_config,
      l2_regularizer=l2_regularizer)

  decoder = decoder_factory.build_decoder(
      input_specs=backbone.output_specs,
      model_config=model_config,
      l2_regularizer=l2_regularizer)

  refinement = refunet.RefUnet()

  norm_activation_config = model_config.norm_activation
  
  model = basnet_model.BASNetModel(backbone, decoder, refinement)
  return model
