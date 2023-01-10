# Copyright 2022 The TensorFlow Authors. All Rights Reserved.
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

"""Models for MAE."""

import tensorflow as tf

from official.projects.mae.modeling import utils
from official.vision.modeling.backbones import vit


class MaskedAE(tf.keras.Model):
  """MAE model."""

  def __init__(self,
               encoder,
               name=None,
               **kwargs):
    super(MaskedAE, self).__init__(name=name, **kwargs)
    self.encoder = encoder
    self.pixels_per_patch = self.encoder.patch_h * self.encoder.patch_w * 3

  def build(self, input_shape):
    self.decoder = vit.Encoder(
        num_layers=8,
        mlp_dim=2048,
        num_heads=16,
        dropout_rate=0.0,
        attention_dropout_rate=0.0,
        add_pos_embed=False
        )
    self.mask = self.add_weight(
        'mask', (1, 1, 512),
        initializer=tf.keras.initializers.RandomNormal(stddev=0.02))
    self.to_pixels = tf.keras.layers.Dense(self.pixels_per_patch)
    self.linear = tf.keras.layers.Dense(512)
    super().build(input_shape)

  def add_position_embed(self, patch_embeds, num_rows, num_cols):
    # patch_embeds is 1d (N, 1+H*W, D) with cls token.
    shape = tf.shape(patch_embeds)
    position_embedding = utils.position_embedding_sine(
        tf.ones((shape[0], num_rows, num_cols), dtype=patch_embeds.dtype),
        512, normalize=False)
    position_embedding = tf.reshape(
        position_embedding, (shape[0], num_rows * num_cols, -1))
    return patch_embeds + tf.concat(
        [tf.zeros((shape[0], 1, shape[2]), dtype=patch_embeds.dtype),
         position_embedding
        ], axis=1)

  def call(self, inputs, training=None, masking=None):
    patches = inputs['patches']
    masked_indices = tf.cast(inputs['masked_indices'], tf.int32)
    unmasked_indices = tf.cast(inputs['unmasked_indices'], tf.int32)
    batch_size = tf.shape(patches)[0]
    num_h_patches = tf.shape(patches)[1]
    num_w_patches = tf.shape(patches)[2]
    num_patches = num_h_patches * num_w_patches
    num_masks = tf.shape(masked_indices)[1]
    patch_embeds = self.encoder.to_embed(patches)
    patch_embeds = self.encoder.add_position_embed(patch_embeds)
    patch_embeds = tf.reshape(
        patch_embeds,
        (batch_size, num_patches, -1))
    patch_embeds = self.encoder.insert_cls(patch_embeds)

    unmasked_indices = tf.concat(
        [tf.zeros((batch_size, 1), unmasked_indices.dtype),
         unmasked_indices + 1],
        axis=1)
    masked_indices = masked_indices + 1
    unmasked_patch_embeds = tf.gather(
        patch_embeds, unmasked_indices, batch_dims=1)
    encoded = self.encoder({'embeddings': unmasked_patch_embeds})
    encoded = self.linear(encoded)

    zeros = tf.zeros((batch_size, num_patches + 1, 512))

    unmasked_embed = tf.tensor_scatter_nd_add(
        zeros,
        tf.stack([
            tf.tile(
                tf.expand_dims(tf.range(batch_size), axis=1),
                [1, num_patches + 1 - num_masks]), unmasked_indices
        ],
                 axis=-1),
        encoded)
    mask_embeds = tf.tile(self.mask, [batch_size, num_masks, 1])
    full_embed = tf.tensor_scatter_nd_add(
        unmasked_embed,
        tf.stack([
            tf.tile(
                tf.expand_dims(tf.range(batch_size), axis=1),
                [1, num_masks]), masked_indices
        ],
                 axis=-1),
        mask_embeds)
    full_embed = self.add_position_embed(
        full_embed, num_h_patches, num_w_patches)

    decoded = self.decoder(full_embed)
    pred_pixel_values = self.to_pixels(
        tf.gather(decoded, masked_indices, batch_dims=1))
    return pred_pixel_values

  @property
  def checkpoint_items(self):
    """Returns a dictionary of items to be additionally checkpointed."""
    items = dict(encoder=self.encoder)
    return items
