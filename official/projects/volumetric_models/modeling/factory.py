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

"""Factory methods to build models."""
from typing import Sequence, Union
import tensorflow as tf, tf_keras

from official.modeling import hyperparams
from official.projects.volumetric_models.modeling.decoders import factory as decoder_factory
from official.projects.volumetric_models.modeling.heads import segmentation_heads_3d
from official.vision.modeling import segmentation_model
from official.vision.modeling.backbones import factory as backbone_factory


def build_segmentation_model_3d(
    input_specs: Union[tf_keras.layers.InputSpec,
                       Sequence[tf_keras.layers.InputSpec]],
    model_config: hyperparams.Config,
    l2_regularizer: tf_keras.regularizers.Regularizer = None
) -> tf_keras.Model:  # pytype: disable=annotation-type-mismatch  # typed-keras
  """Builds Segmentation model."""
  norm_activation_config = model_config.norm_activation
  backbone = backbone_factory.build_backbone(
      input_specs=input_specs,
      backbone_config=model_config.backbone,
      norm_activation_config=norm_activation_config,
      l2_regularizer=l2_regularizer)

  decoder = decoder_factory.build_decoder(
      input_specs=backbone.output_specs,
      model_config=model_config,
      l2_regularizer=l2_regularizer)

  head_config = model_config.head

  head = segmentation_heads_3d.SegmentationHead3D(
      num_classes=model_config.num_classes,
      level=head_config.level,
      num_convs=head_config.num_convs,
      num_filters=head_config.num_filters,
      upsample_factor=head_config.upsample_factor,
      activation=norm_activation_config.activation,
      use_sync_bn=norm_activation_config.use_sync_bn,
      norm_momentum=norm_activation_config.norm_momentum,
      norm_epsilon=norm_activation_config.norm_epsilon,
      use_batch_normalization=head_config.use_batch_normalization,
      kernel_regularizer=l2_regularizer,
      output_logits=head_config.output_logits)

  model = segmentation_model.SegmentationModel(backbone, decoder, head)
  return model
