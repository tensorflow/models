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

"""Build Panoptic Deeplab model."""
from typing import Any, Mapping, Optional, Union

import tensorflow as tf, tf_keras
from official.projects.panoptic.modeling.layers import panoptic_deeplab_merge


@tf_keras.utils.register_keras_serializable(package='Vision')
class PanopticDeeplabModel(tf_keras.Model):
  """Panoptic Deeplab model."""

  def __init__(
      self,
      backbone: tf_keras.Model,
      semantic_decoder: tf_keras.Model,
      semantic_head: tf_keras.layers.Layer,
      instance_head: tf_keras.layers.Layer,
      instance_decoder: Optional[tf_keras.Model] = None,
      post_processor: Optional[panoptic_deeplab_merge.PostProcessor] = None,
      **kwargs):
    """Panoptic deeplab model initializer.

    Args:
      backbone: a backbone network.
      semantic_decoder: a decoder network. E.g. FPN.
      semantic_head: segmentation head.
      instance_head: instance center head.
      instance_decoder: Optional decoder network for instance predictions.
      post_processor: Optional post processor layer.
      **kwargs: keyword arguments to be passed.
    """
    super(PanopticDeeplabModel, self).__init__(**kwargs)

    self._config_dict = {
        'backbone': backbone,
        'semantic_decoder': semantic_decoder,
        'instance_decoder': instance_decoder,
        'semantic_head': semantic_head,
        'instance_head': instance_head,
        'post_processor': post_processor
    }
    self.backbone = backbone
    self.semantic_decoder = semantic_decoder
    self.instance_decoder = instance_decoder
    self.semantic_head = semantic_head
    self.instance_head = instance_head
    self.post_processor = post_processor

  def call(  # pytype: disable=annotation-type-mismatch,signature-mismatch
      self, inputs: tf.Tensor,
      image_info: tf.Tensor,
      training: bool = None):
    if training is None:
      training = tf_keras.backend.learning_phase()

    backbone_features = self.backbone(inputs, training=training)

    semantic_features = self.semantic_decoder(
        backbone_features, training=training)

    if self.instance_decoder is None:
      instance_features = semantic_features
    else:
      instance_features = self.instance_decoder(
          backbone_features, training=training)

    segmentation_outputs = self.semantic_head(
        (backbone_features, semantic_features),
        training=training)
    instance_outputs = self.instance_head(
        (backbone_features, instance_features),
        training=training)

    outputs = {
        'segmentation_outputs': segmentation_outputs,
        'instance_centers_heatmap':
            instance_outputs['instance_centers_heatmap'],
        'instance_centers_offset':
            instance_outputs['instance_centers_offset'],
    }
    if training:
      return outputs

    if self.post_processor is not None:
      panoptic_masks = self.post_processor(outputs, image_info)
      outputs.update(panoptic_masks)
    return outputs

  @property
  def checkpoint_items(
      self) -> Mapping[str, Union[tf_keras.Model, tf_keras.layers.Layer]]:
    """Returns a dictionary of items to be additionally checkpointed."""
    items = dict(
        backbone=self.backbone,
        semantic_decoder=self.semantic_decoder,
        semantic_head=self.semantic_head,
        instance_head=self.instance_head)
    if self.instance_decoder is not None:
      items.update(instance_decoder=self.instance_decoder)

    return items

  def get_config(self) -> Mapping[str, Any]:
    return self._config_dict

  @classmethod
  def from_config(cls, config, custom_objects=None):
    return cls(**config)
