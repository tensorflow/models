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

"""Models for ViT."""

import tensorflow as tf

from official.modeling import tf_utils
from official.projects.mae.modeling import utils
from official.vision.modeling.backbones import vit


def to_patch(images, patch_height, patch_width):
  """Image (NHWC) to patches (N(H' W')(patch_height patch_width c))."""
  batch_size, h, w, c = tf_utils.get_shape_list(images)
  num_h = h // patch_height
  num_w = w // patch_width
  x = tf.reshape(images,
                 (batch_size, num_h, patch_height, num_w, patch_width, c))
  x = tf.einsum('nhpwqc->nhwpqc', x)
  x = tf.reshape(x, (batch_size, num_h, num_w, patch_height * patch_width * c))
  return x


class ViTClassifier(tf.keras.Model):
  """ViT classifier for finetune."""

  def __init__(self, encoder, num_classes, **kwargs):
    super().__init__(**kwargs)
    self.encoder = encoder
    self.linear = tf.keras.layers.Dense(
        num_classes,
        kernel_initializer=tf.keras.initializers.TruncatedNormal(stddev=2e-5))

  def call(self, inputs):  # pytype: disable=signature-mismatch  # overriding-parameter-count-checks
    encoded = self.encoder({'images': inputs})
    return self.linear(encoded[:, 0])


class ViTLinearClassifier(tf.keras.Model):
  """ViT classifier for linear probing."""

  def __init__(self, encoder, num_classes, use_sync_bn=True, **kwargs):
    super().__init__(**kwargs)
    self.encoder = encoder
    self.linear = tf.keras.layers.Dense(
        num_classes,
        kernel_initializer=tf.keras.initializers.TruncatedNormal(stddev=0.01))
    if use_sync_bn:
      self._norm = tf.keras.layers.experimental.SyncBatchNormalization
    else:
      self._norm = tf.keras.layers.BatchNormalization
    self.batch_norm = self._norm(
        axis=-1, epsilon=1e-6, center=False, scale=False, momentum=0.9)

  def call(self, inputs, training=False):  # pytype: disable=signature-mismatch  # overriding-parameter-count-checks
    encoded = self.encoder({'images': inputs})
    features = self.batch_norm(encoded[:, 0], training=training)
    return self.linear(features)


class VisionTransformer(tf.keras.Model):
  """ViT backbone."""

  def __init__(self,
               patch_h,
               patch_w,
               init_stochastic_depth_rate=0.0,
               **kwargs):
    super().__init__(**kwargs)
    self.patch_h = patch_h
    self.patch_w = patch_w
    self.init_stochastic_depth_rate = init_stochastic_depth_rate

  def build(self, input_shape):
    self.patch_to_embed = tf.keras.layers.Dense(1024)
    # ViT-L
    self.encoder = vit.Encoder(
        num_layers=24,
        mlp_dim=4096,
        num_heads=16,
        dropout_rate=0.0,
        attention_dropout_rate=0.0,
        init_stochastic_depth_rate=self.init_stochastic_depth_rate,
        add_pos_embed=False,
    )
    self.token_cls = vit.TokenLayer()
    super().build(input_shape)

  def to_embed(self, patches):
    return self.patch_to_embed(patches)

  def insert_cls(self, patch_embeds):
    return self.token_cls(patch_embeds)

  def add_position_embed(self, patch_embeds):
    return patch_embeds + utils.position_embedding_sine(
        tf.ones_like(patch_embeds[..., 0]), 1024, normalize=False)

  def call(self, inputs):  # pytype: disable=signature-mismatch  # overriding-parameter-count-checks
    if isinstance(inputs, dict):
      images = inputs.get('images', None)
      patch_embeds = inputs.get('embeddings', None)
    else:
      raise ValueError('Unexpected inputs type to %s.' % self.__class__)
    if images is not None:
      patches = to_patch(images, self.patch_h, self.patch_w)
      patch_embeds = self.to_embed(patches)
      patch_shape = tf.shape(patch_embeds)
      patch_embeds = self.add_position_embed(patch_embeds)
      patch_embeds = tf.reshape(patch_embeds,
                                (patch_shape[0], -1, patch_shape[-1]))
      patch_embeds = self.insert_cls(patch_embeds)
    return self.encoder(patch_embeds)
