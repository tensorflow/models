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

"""Builds the overall MOSAIC segmentation models."""
from typing import Any, Dict, Optional, Union

import tensorflow as tf

from official.projects.mosaic.configs import mosaic_config
from official.projects.mosaic.modeling import mosaic_blocks
from official.projects.mosaic.modeling import mosaic_head
from official.vision.modeling import backbones
from official.vision.modeling.heads import segmentation_heads


@tf.keras.utils.register_keras_serializable(package='Vision')
class MosaicSegmentationModel(tf.keras.Model):
  """A model class for segmentation using MOSAIC.

  Input images are passed through a backbone first. A MOSAIC neck encoder
  network is then applied, and finally a MOSAIC segmentation head is applied on
  the outputs of the backbone and neck encoder network. Feature fusion and
  decoding is done in the segmentation head.

  Reference:
   [MOSAIC: Mobile Segmentation via decoding Aggregated Information and encoded
   Context](https://arxiv.org/pdf/2112.11623.pdf)
  """

  def __init__(self,
               backbone: tf.keras.Model,
               head: tf.keras.layers.Layer,
               neck: Optional[tf.keras.layers.Layer] = None,
               mask_scoring_head: Optional[tf.keras.layers.Layer] = None,
               **kwargs):
    """Segmentation initialization function.

    Args:
      backbone: A backbone network.
      head: A segmentation head, e.g. MOSAIC decoder.
      neck: An optional neck encoder network, e.g. MOSAIC encoder. If it is not
        provided, the decoder head will be connected directly with the backbone.
      mask_scoring_head: An optional mask scoring head.
      **kwargs: keyword arguments to be passed.
    """
    super(MosaicSegmentationModel, self).__init__(**kwargs)
    self._config_dict = {
        'backbone': backbone,
        'neck': neck,
        'head': head,
        'mask_scoring_head': mask_scoring_head,
    }
    self.backbone = backbone
    self.neck = neck
    self.head = head
    self.mask_scoring_head = mask_scoring_head

  def call(self,  # pytype: disable=signature-mismatch  # overriding-parameter-count-checks
           inputs: tf.Tensor,
           training: bool = None) -> Dict[str, tf.Tensor]:
    backbone_features = self.backbone(inputs)

    if self.neck is not None:
      neck_features = self.neck(backbone_features, training=training)
    else:
      neck_features = backbone_features

    logits = self.head([neck_features, backbone_features], training=training)
    outputs = {'logits': logits}

    if self.mask_scoring_head:
      mask_scores = self.mask_scoring_head(logits)
      outputs.update({'mask_scores': mask_scores})

    return outputs

  @property
  def checkpoint_items(
      self) -> Dict[str, Union[tf.keras.Model, tf.keras.layers.Layer]]:
    """Returns a dictionary of items to be additionally checkpointed."""
    items = dict(backbone=self.backbone, head=self.head)
    if self.neck is not None:
      items.update(neck=self.neck)
    if self.mask_scoring_head is not None:
      items.update(mask_scoring_head=self.mask_scoring_head)
    return items

  def get_config(self) -> Dict[str, Any]:
    """Returns a config dictionary for initialization from serialization."""
    base_config = super().get_config()
    model_config = base_config
    model_config.update(self._config_dict)
    return model_config

  @classmethod
  def from_config(cls, config, custom_objects=None):
    return cls(**config)


def build_mosaic_segmentation_model(
    input_specs: tf.keras.layers.InputSpec,
    model_config: mosaic_config.MosaicSemanticSegmentationModel,
    l2_regularizer: Optional[tf.keras.regularizers.Regularizer] = None,
    backbone: Optional[tf.keras.Model] = None,
    neck: Optional[tf.keras.layers.Layer] = None
) -> tf.keras.Model:
  """Builds MOSAIC Segmentation model."""
  norm_activation_config = model_config.norm_activation
  if backbone is None:
    backbone = backbones.factory.build_backbone(
        input_specs=input_specs,
        backbone_config=model_config.backbone,
        norm_activation_config=norm_activation_config,
        l2_regularizer=l2_regularizer)

  if neck is None:
    neck_config = model_config.neck
    neck = mosaic_blocks.MosaicEncoderBlock(
        encoder_input_level=neck_config.encoder_input_level,
        branch_filter_depths=neck_config.branch_filter_depths,
        conv_kernel_sizes=neck_config.conv_kernel_sizes,
        pyramid_pool_bin_nums=neck_config.pyramid_pool_bin_nums,
        use_sync_bn=norm_activation_config.use_sync_bn,
        batchnorm_momentum=norm_activation_config.norm_momentum,
        batchnorm_epsilon=norm_activation_config.norm_epsilon,
        activation=neck_config.activation,
        dropout_rate=neck_config.dropout_rate,
        kernel_initializer=neck_config.kernel_initializer,
        kernel_regularizer=l2_regularizer,
        interpolation=neck_config.interpolation,
        use_depthwise_convolution=neck_config.use_depthwise_convolution)

  head_config = model_config.head
  head = mosaic_head.MosaicDecoderHead(
      num_classes=model_config.num_classes,
      decoder_input_levels=head_config.decoder_input_levels,
      decoder_stage_merge_styles=head_config.decoder_stage_merge_styles,
      decoder_filters=head_config.decoder_filters,
      decoder_projected_filters=head_config.decoder_projected_filters,
      encoder_end_level=head_config.encoder_end_level,
      use_additional_classifier_layer=head_config
      .use_additional_classifier_layer,
      classifier_kernel_size=head_config.classifier_kernel_size,
      activation=head_config.activation,
      use_sync_bn=norm_activation_config.use_sync_bn,
      batchnorm_momentum=norm_activation_config.norm_momentum,
      batchnorm_epsilon=norm_activation_config.norm_epsilon,
      kernel_initializer=head_config.kernel_initializer,
      kernel_regularizer=l2_regularizer,
      interpolation=head_config.interpolation)

  mask_scoring_head = None
  if model_config.mask_scoring_head:
    mask_scoring_head = segmentation_heads.MaskScoring(
        num_classes=model_config.num_classes,
        **model_config.mask_scoring_head.as_dict(),
        activation=norm_activation_config.activation,
        use_sync_bn=norm_activation_config.use_sync_bn,
        norm_momentum=norm_activation_config.norm_momentum,
        norm_epsilon=norm_activation_config.norm_epsilon,
        kernel_regularizer=l2_regularizer)

  model = MosaicSegmentationModel(
      backbone=backbone,
      neck=neck,
      head=head,
      mask_scoring_head=mask_scoring_head)
  return model
