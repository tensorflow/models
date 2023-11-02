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

"""Pixel models."""

import tensorflow as tf, tf_keras

from official.vision.modeling.backbones import vit

layers = tf_keras.layers


class ViTEncoder(vit.Encoder):
  """ViT Encoder.

  The original vit implementation in official/vision/modeling/backbones/vit.py
  does not support attention masks. This version allows passing the attention
  mask in call along with inputs as a (bs, seqlen) tensor.
  """

  def call(self, inputs, training=None):
    x, mask = inputs
    if self._add_pos_embed:
      x = self._pos_embed(x, inputs_positions=self._inputs_positions)
    x = self._dropout(x, training=training)

    for encoder_layer in self._encoder_layers:
      x = encoder_layer((x, mask), training=training)
    x = self._norm(x)
    return x


class VisionTransformer(tf_keras.layers.Layer):
  """ViT backbone."""

  def __init__(
      self,
      patch_h,
      patch_w,
      filters,
      num_layers,
      mlp_dim,
      num_heads,
      dropout_rate,
      attention_dropout_rate,
      init_stochastic_depth_rate,
      **kwargs
  ):
    super().__init__(**kwargs)
    self.patch_h = patch_h
    self.patch_w = patch_w

    self.filters = filters
    self.num_layers = num_layers
    self.mlp_dim = mlp_dim
    self.num_heads = num_heads
    self.dropout_rate = dropout_rate
    self.attention_dropout_rate = attention_dropout_rate
    self.init_stochastic_depth_rate = init_stochastic_depth_rate

  def build(self, input_shape):
    self.patch_to_embed = tf_keras.layers.Conv2D(
        filters=self.filters,
        kernel_size=(self.patch_h, self.patch_w),
        strides=(self.patch_h, self.patch_w),
        padding='valid',
        kernel_initializer='lecun_normal',
    )

    self.encoder = ViTEncoder(
        num_layers=self.num_layers,
        mlp_dim=self.mlp_dim,
        num_heads=self.num_heads,
        dropout_rate=self.dropout_rate,
        attention_dropout_rate=self.attention_dropout_rate,
        init_stochastic_depth_rate=self.init_stochastic_depth_rate,
        add_pos_embed=True,
    )
    self.token_cls = vit.TokenLayer()
    super().build(input_shape)

  def to_embed(self, patches):
    return self.patch_to_embed(patches)

  def insert_cls(self, patch_embeds):
    return self.token_cls(patch_embeds)

  def call(self, inputs):  # pylint:disable=signature-mismatch
    if isinstance(inputs, dict):
      images = inputs.get('pixel_values', None)
      attention_mask = inputs.get('attention_mask', None)
      attention_mask = tf.transpose(
          tf.concat(
              values=[
                  tf.ones((1, tf.shape(attention_mask)[0]), tf.float32),
                  tf.transpose(attention_mask),
              ],
              axis=0,
          )
      )
      attention_mask = tf.einsum('ij,ik->ijk', attention_mask, attention_mask)
      attention_mask = tf.cast(attention_mask, tf.int32)
    else:
      raise ValueError('Unexpected inputs type to %s.' % self.__class__)

    images = tf.transpose(images, perm=[0, 2, 3, 1])
    patch_embeds = self.to_embed(images)
    patch_shape = tf.shape(patch_embeds)
    patch_embeds = tf.reshape(
        patch_embeds, (patch_shape[0], -1, patch_shape[-1])
    )
    patch_embeds = self.insert_cls(patch_embeds)

    return self.encoder((patch_embeds, attention_mask))


class PixelClassifier(tf_keras.layers.Layer):
  """Pixel classifier for finetuning. Uses the cls token."""

  def __init__(self, encoder, num_classes, **kwargs):
    super().__init__(**kwargs)
    self.encoder = encoder
    self.linear = tf_keras.layers.Dense(
        num_classes,
        kernel_initializer=tf_keras.initializers.TruncatedNormal(stddev=0.01),
    )

  def call(self, inputs):
    encoded = self.encoder(inputs)
    return self.linear(encoded[:, 0])


class PixelLinearClassifier(tf_keras.layers.Layer):
  """Pixel classifier for finetuning.

  This is a layer with additional layer norm and linear layer in the
  classification head. Uses the average of all token representations
  """

  def __init__(self, encoder, num_classes, num_filters, **kwargs):
    super().__init__(**kwargs)
    self.encoder = encoder
    self.num_filters = num_filters
    self.linear_clas = tf_keras.layers.Dense(
        num_classes,
        kernel_initializer=tf_keras.initializers.TruncatedNormal(stddev=0.01),
    )

    self.norm = tf_keras.layers.LayerNormalization(
        name='classification_layer_norm',
        axis=-1,
        epsilon=1e-6,
        dtype=tf.float32,
    )

    self.linear_trans = tf_keras.layers.Dense(
        num_filters,
        kernel_initializer=tf_keras.initializers.TruncatedNormal(stddev=0.01),
    )
    self.activation = tf_keras.layers.Activation('gelu')
    self.dropout = tf_keras.layers.Dropout(0.1)

  def call(self, inputs, training=False):
    attention_mask = inputs.get('attention_mask')
    mask_lengths = tf.expand_dims(tf.reduce_sum(attention_mask, axis=1), 1)
    attention_mask = tf.tile(
        tf.expand_dims(attention_mask, 2), [1, 1, self.num_filters]
    )
    encoded = self.encoder(inputs)
    encoded = self.norm(self.activation(self.linear_trans(encoded)))
    encoded = self.dropout(encoded, training=training)

    mean_pooling = (
        tf.reduce_sum(encoded[:, 1:, :] * attention_mask, axis=1) / mask_lengths
    )
    return self.linear_clas(mean_pooling)
