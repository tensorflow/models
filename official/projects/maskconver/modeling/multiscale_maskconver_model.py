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

"""Panoptic Segmentation MaskConver model."""

from typing import Mapping, Union, Any, Dict, Optional, List

import tensorflow as tf, tf_keras

layers = tf_keras.layers


@tf_keras.utils.register_keras_serializable(package='Vision')
class MultiScaleMaskConverModel(tf_keras.Model):
  """Multiscale MaskConver class model."""

  def __init__(self,
               backbone: tf_keras.Model,
               decoder: tf_keras.Model,
               mask_decoder: tf_keras.Model,
               class_head: tf_keras.layers.Layer,
               embedding_head: tf_keras.layers.Layer,
               per_pixel_embeddings_head: tf_keras.layers.Layer,
               mlp_embedding_head: tf_keras.layers.Layer,
               panoptic_generator: Optional[tf_keras.layers.Layer] = None,
               min_level: int = 3,
               max_level: int = 5,
               max_proposals: int = 100,
               padded_output_size: Optional[List[int]] = None,
               score_threshold: float = 0.1,
               l2_regularizer: Optional[Any] = None,
               embedding_size: int = 256,
               num_classes: int = 201,
               **kwargs):
    """MaskConver initialization function."""
    super().__init__(**kwargs)
    self._config_dict = {
        'backbone': backbone,
        'decoder': decoder,
        'mask_decoder': mask_decoder,
        'class_head': class_head,
        'embedding_head': embedding_head,
        'mlp_embedding_head': mlp_embedding_head,
        'min_level': min_level,
        'max_level': max_level,
        'max_proposals': max_proposals,
        'padded_output_size': padded_output_size,
        'score_threshold': score_threshold,
        'per_pixel_embeddings_head': per_pixel_embeddings_head,
    }
    self.backbone = backbone
    self.decoder = decoder
    self.mask_decoder = mask_decoder
    self.class_head = class_head
    self.embedding_head = embedding_head
    self.embedding_size = embedding_size
    self.mlp = mlp_embedding_head
    self.panoptic_generator = panoptic_generator
    self.num_classes = num_classes

    self.min_level = min_level
    self.max_level = max_level
    self.max_proposals = max_proposals
    self.padded_output_size = padded_output_size
    self.score_threshold = score_threshold
    self.per_pixel_embeddings_head = per_pixel_embeddings_head
    self.class_embeddings = tf_keras.layers.Embedding(
        num_classes,
        self.embedding_size,
        embeddings_regularizer=l2_regularizer)

  def process_heatmap(self, heatmap: tf.Tensor) -> tf.Tensor:
    scoremap = tf.sigmoid(heatmap)
    scoremap_max_pool = tf.nn.max_pool(
        scoremap, ksize=3, strides=1, padding='SAME')
    valid_mask = tf.abs(scoremap - scoremap_max_pool) < 1e-6
    return scoremap * tf.cast(valid_mask, scoremap.dtype)

  def call(self, inputs: tf.Tensor,  # pytype: disable=annotation-type-mismatch
           image_info: Optional[tf.Tensor] = None,
           box_indices: Optional[tf.Tensor] = None,
           classes: Optional[tf.Tensor] = None,
           training: bool = None
           ) -> Dict[str, Optional[Any]]:
    batch_size = tf.shape(inputs)[0]
    backbone_features = self.backbone(inputs, training=training)

    if self.decoder:
      decoder_features = self.decoder(backbone_features)
      if self.mask_decoder:
        decoder_features2 = self.mask_decoder(backbone_features)
      else:
        decoder_features2 = decoder_features
    else:
      decoder_features = backbone_features

    level_class_heatmaps = self.class_head(decoder_features, training=training)
    level_dense_mask_embeddings = self.embedding_head(
        decoder_features, training=training)
    embedding_size = self.embedding_size

    class_heatmaps = []
    dense_mask_embeddings = []
    class_scoremaps = []
    for level in range(self.min_level, self.max_level + 1):
      if not training:
        class_scoremaps.append(
            tf.reshape(
                self.process_heatmap(level_class_heatmaps[level]),
                [batch_size, -1],
            )
        )
      class_heatmaps.append(
          tf.reshape(
              level_class_heatmaps[level], [batch_size, -1, self.num_classes]
          )
      )
      dense_mask_embeddings.append(
          tf.reshape(
              level_dense_mask_embeddings[level],
              [batch_size, -1, embedding_size],
          )
      )

    class_heatmaps = tf.concat(class_heatmaps, axis=1)
    dense_mask_embeddings = tf.concat(dense_mask_embeddings, axis=1)

    per_pixel_embeddings = self.per_pixel_embeddings_head(
        (backbone_features, decoder_features2), training=training
    )

    if not training:
      class_scoremaps = tf.concat(class_scoremaps, axis=1)
      confidence, top_indices = tf.nn.top_k(
          class_scoremaps, k=self.max_proposals
      )
      box_indices = top_indices // self.num_classes
      classes = top_indices % self.num_classes

    mask_embeddings = tf.gather(
        dense_mask_embeddings, box_indices, batch_dims=1
    )
    class_embeddings = tf.cast(
        self.class_embeddings(tf.maximum(classes, 0)), mask_embeddings.dtype
    )
    mask_embeddings_inputs = mask_embeddings + class_embeddings
    mask_embeddings = self.mlp(mask_embeddings_inputs)

    mask_proposal_logits = tf.einsum(
        'bqc,bhwc->bhwq', mask_embeddings, per_pixel_embeddings
    )
    mask_proposal_logits = tf.cast(mask_proposal_logits, tf.float32)

    if not training:
      outputs = {
          'classes': classes,
          'confidence': confidence,
          'mask_embeddings': mask_embeddings,
          'mask_proposal_logits': mask_proposal_logits,
          'class_heatmaps': class_heatmaps,
      }
      if self.panoptic_generator is not None:
        panoptic_outputs = self.panoptic_generator(
            outputs, images_info=image_info
        )
        outputs.update({'panoptic_outputs': panoptic_outputs})
      else:
        outputs['mask_proposal_logits'] = tf.image.resize(
            mask_proposal_logits, self.padded_output_size, 'bilinear'
        )
    else:
      outputs = {
          'class_heatmaps': class_heatmaps,
          'mask_proposal_logits': mask_proposal_logits,
          'mask_embeddings': mask_embeddings,
      }
    return outputs

  @property
  def checkpoint_items(
      self,
  ) -> Mapping[str, Union[tf_keras.Model, tf_keras.layers.Layer]]:
    """Returns a dictionary of items to be additionally checkpointed."""
    items = dict(
        backbone=self.backbone,
        heads=self.heads,
        class_head=self.class_head,
        embedding_head=self.embedding_head,
        per_pixel_embeddings_head=self.per_pixel_embeddings_head,
    )
    if self.decoder is not None:
      items.update(decoder=self.decoder)
    return items

  def get_config(self) -> Mapping[str, Any]:
    return self._config_dict

  @classmethod
  def from_config(cls, config, custom_objects=None):
    return cls(**config)
