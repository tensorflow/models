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

"""Panoptic Segmentation model."""

from typing import Mapping, Union, Any, Dict, Optional, List

import tensorflow as tf, tf_keras

layers = tf_keras.layers


@tf_keras.utils.register_keras_serializable(package='Vision')
class MaskConverModel(tf_keras.Model):
  """A MaskConver class model."""

  def __init__(
      self,
      backbone: tf_keras.Model,
      decoder: tf_keras.Model,
      # panoptic_fpn_fusion: tf_keras.layers.Layer,
      class_head: tf_keras.layers.Layer,
      embedding_head: tf_keras.layers.Layer,
      per_pixel_embeddings_head: tf_keras.layers.Layer,
      mlp_embedding_head: tf_keras.layers.Layer,
      proposal_generator: tf_keras.layers.Layer,
      panoptic_generator: Optional[tf_keras.layers.Layer] = None,
      level: int = 3,
      padded_output_size: Optional[List[int]] = None,
      score_threshold: float = 0.1,
      l2_regularizer: Optional[Any] = None,
      embedding_size: int = 256,
      num_classes: int = 201,
      **kwargs):
    """MaskConver initialization function.

    Args:
      backbone: a backbone network.
      decoder: a decoder network. E.g. FPN.
      #  panoptic_fpn_fusion: a panoptic_fpn_fusion layer.
      class_head: class head.
      embedding_head: embedding head.
      per_pixel_embeddings_head: per_pixel_embeddings_head.
      mlp_embedding_head: mlp embedding head.
      proposal_generator: proposal_generator.
      panoptic_generator: panoptic generator.
      level: int.
      padded_output_size: padded output size. GPU or CPU only.
      score_threshold: score threshold, used for filtering.
      l2_regularizer: l2 regularizer.
      embedding_size: `int`, embedding size.
      num_classes: `int`, the total number of classes.
      **kwargs: keyword arguments to be passed.
    """
    super(MaskConverModel, self).__init__(**kwargs)
    self._config_dict = {
        'backbone': backbone,
        'decoder': decoder,
        'class_head': class_head,
        'embedding_head': embedding_head,
        'mlp_embedding_head': mlp_embedding_head,
        'proposal_generator': proposal_generator,
        'level': level,
        'padded_output_size': padded_output_size,
        'score_threshold': score_threshold,
        'per_pixel_embeddings_head': per_pixel_embeddings_head,
    }
    self.backbone = backbone
    self.decoder = decoder
    self.class_head = class_head
    self.embedding_head = embedding_head
    self.embedding_size = embedding_size
    self.mlp = mlp_embedding_head
    self.proposal_generator = proposal_generator
    self.panoptic_generator = panoptic_generator
    self.num_classes = num_classes

    self.level = level
    self.padded_output_size = padded_output_size
    self.score_threshold = score_threshold
    self.per_pixel_embeddings_head = per_pixel_embeddings_head
    self.class_embeddings = tf_keras.layers.Embedding(
        num_classes,
        self.embedding_size,
        embeddings_regularizer=l2_regularizer)

  def call(self, inputs: tf.Tensor,  # pytype: disable=annotation-type-mismatch
           image_info: Optional[tf.Tensor] = None,
           box_indices: Optional[tf.Tensor] = None,
           classes: Optional[tf.Tensor] = None,
           training: bool = None
           ) -> Dict[str, Optional[Any]]:
    backbone_features = self.backbone(inputs)

    if self.decoder:
      decoder_features = self.decoder(backbone_features)
    else:
      decoder_features = backbone_features

    class_heatmaps = self.class_head((backbone_features, decoder_features),
                                     training=training)
    dense_mask_embeddings = self.embedding_head(
        (backbone_features, decoder_features), training=training)
    per_pixel_embeddings = self.per_pixel_embeddings_head(
        (backbone_features, decoder_features), training=training)

    if not training:
      proposals = self.proposal_generator(class_heatmaps)
      classes = proposals['classes']
      confidence = proposals['confidence']
      box_indices = proposals['embedding_indices']
      _ = proposals['num_proposals']

    mask_embeddings = tf.gather_nd(
        dense_mask_embeddings, box_indices, batch_dims=1)
    class_embeddings = self.class_embeddings(tf.maximum(classes, 0))
    mask_embeddings = mask_embeddings * tf.cast(
        class_embeddings, mask_embeddings.dtype)
    mask_embeddings = self.mlp(mask_embeddings)

    mask_proposal_logits = tf.einsum('bqc,bhwc->bhwq',
                                     mask_embeddings,
                                     per_pixel_embeddings)
    mask_proposal_logits = tf.cast(mask_proposal_logits, tf.float32)

    if not training:
      outputs = {'classes': classes,
                 'confidence': confidence,
                 'mask_proposal_logits': mask_proposal_logits,
                 'class_heatmaps': class_heatmaps}
      if self.panoptic_generator is not None:
        panoptic_outputs = self.panoptic_generator(
            outputs, images_info=image_info)
        outputs.update({'panoptic_outputs': panoptic_outputs})
      else:
        outputs['mask_proposal_logits'] = tf.image.resize(
            mask_proposal_logits, self.padded_output_size, 'bilinear')
    else:
      outputs = {'class_heatmaps': class_heatmaps,
                 'mask_proposal_logits': mask_proposal_logits}
    return outputs

  @property
  def checkpoint_items(
      self) -> Mapping[str, Union[tf_keras.Model, tf_keras.layers.Layer]]:
    """Returns a dictionary of items to be additionally checkpointed."""
    items = dict(backbone=self.backbone,
                 class_head=self.class_head,
                 embedding_head=self.embedding_head,
                 per_pixel_embeddings_head=self.per_pixel_embeddings_head)
    if self.decoder is not None:
      items.update(decoder=self.decoder)
    return items

  def get_config(self) -> Mapping[str, Any]:
    return self._config_dict

  @classmethod
  def from_config(cls, config, custom_objects=None):
    return cls(**config)
