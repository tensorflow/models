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

"""Implements End-to-End Object Detection with Transformers.

Model paper: https://arxiv.org/abs/2005.12872
This module does not support Keras de/serialization. Please use
tf.train.Checkpoint for object based saving and loading and tf.saved_model.save
for graph serializaiton.
"""
import math
from typing import Any, List

import tensorflow as tf

from official.modeling import tf_utils
from official.projects.detr.modeling import transformer
from official.vision.ops import box_ops


def position_embedding_sine(attention_mask,
                            num_pos_features=256,
                            temperature=10000.,
                            normalize=True,
                            scale=2 * math.pi):
  """Sine-based positional embeddings for 2D images.

  Args:
    attention_mask: a `bool` Tensor specifying the size of the input image to
      the Transformer and which elements are padded, of size [batch_size,
      height, width]
    num_pos_features: a `int` specifying the number of positional features,
      should be equal to the hidden size of the Transformer network
    temperature: a `float` specifying the temperature of the positional
      embedding. Any type that is converted to a `float` can also be accepted.
    normalize: a `bool` determining whether the positional embeddings should be
      normalized between [0, scale] before application of the sine and cos
      functions.
    scale: a `float` if normalize is True specifying the scale embeddings before
      application of the embedding function.

  Returns:
    embeddings: a `float` tensor of the same shape as input_tensor specifying
      the positional embeddings based on sine features.
  """
  if num_pos_features % 2 != 0:
    raise ValueError(
        "Number of embedding features (num_pos_features) must be even when "
        "column and row embeddings are concatenated.")
  num_pos_features = num_pos_features // 2

  # Produce row and column embeddings based on total size of the image
  # <tf.float>[batch_size, height, width]
  attention_mask = tf.cast(attention_mask, tf.float32)
  row_embedding = tf.cumsum(attention_mask, 1)
  col_embedding = tf.cumsum(attention_mask, 2)

  if normalize:
    eps = 1e-6
    row_embedding = row_embedding / (row_embedding[:, -1:, :] + eps) * scale
    col_embedding = col_embedding / (col_embedding[:, :, -1:] + eps) * scale

  dim_t = tf.range(num_pos_features, dtype=row_embedding.dtype)
  dim_t = tf.pow(temperature, 2 * (dim_t // 2) / num_pos_features)

  # Creates positional embeddings for each row and column position
  # <tf.float>[batch_size, height, width, num_pos_features]
  pos_row = tf.expand_dims(row_embedding, -1) / dim_t
  pos_col = tf.expand_dims(col_embedding, -1) / dim_t
  pos_row = tf.stack(
      [tf.sin(pos_row[:, :, :, 0::2]),
       tf.cos(pos_row[:, :, :, 1::2])], axis=4)
  pos_col = tf.stack(
      [tf.sin(pos_col[:, :, :, 0::2]),
       tf.cos(pos_col[:, :, :, 1::2])], axis=4)

  # final_shape = pos_row.shape.as_list()[:3] + [-1]
  final_shape = tf_utils.get_shape_list(pos_row)[:3] + [-1]
  pos_row = tf.reshape(pos_row, final_shape)
  pos_col = tf.reshape(pos_col, final_shape)
  output = tf.concat([pos_row, pos_col], -1)

  embeddings = tf.cast(output, tf.float32)
  return embeddings


def postprocess(outputs: dict[str, tf.Tensor]) -> dict[str, tf.Tensor]:
  """Performs post-processing on model output.

  Args:
    outputs: The raw model output.

  Returns:
    Postprocessed model output.
  """
  predictions = {
      "detection_boxes":  # Box coordinates are relative values here.
          box_ops.cycxhw_to_yxyx(outputs["box_outputs"]),
      "detection_scores":
          tf.math.reduce_max(
              tf.nn.softmax(outputs["cls_outputs"])[:, :, 1:], axis=-1),
      "detection_classes":
          tf.math.argmax(outputs["cls_outputs"][:, :, 1:], axis=-1) + 1,
      # Fix this. It's not being used at the moment.
      "num_detections":
          tf.reduce_sum(
              tf.cast(
                  tf.math.greater(
                      tf.math.reduce_max(outputs["cls_outputs"], axis=-1), 0),
                  tf.int32),
              axis=-1)
  }
  return predictions


class DETR(tf.keras.Model):
  """DETR model with Keras.

  DETR consists of backbone, query embedding, DETRTransformer,
  class and box heads.
  """

  def __init__(self,
               backbone,
               backbone_endpoint_name,
               num_queries,
               hidden_size,
               num_classes,
               num_encoder_layers=6,
               num_decoder_layers=6,
               dropout_rate=0.1,
               **kwargs):
    super().__init__(**kwargs)
    self._num_queries = num_queries
    self._hidden_size = hidden_size
    self._num_classes = num_classes
    self._num_encoder_layers = num_encoder_layers
    self._num_decoder_layers = num_decoder_layers
    self._dropout_rate = dropout_rate
    if hidden_size % 2 != 0:
      raise ValueError("hidden_size must be a multiple of 2.")
    self._backbone = backbone
    self._backbone_endpoint_name = backbone_endpoint_name

  def build(self, input_shape=None):
    self._input_proj = tf.keras.layers.Conv2D(
        self._hidden_size, 1, name="detr/conv2d")
    self._build_detection_decoder()
    super().build(input_shape)

  def _build_detection_decoder(self):
    """Builds detection decoder."""
    self._transformer = DETRTransformer(
        num_encoder_layers=self._num_encoder_layers,
        num_decoder_layers=self._num_decoder_layers,
        dropout_rate=self._dropout_rate)
    self._query_embeddings = self.add_weight(
        "detr/query_embeddings",
        shape=[self._num_queries, self._hidden_size],
        initializer=tf.keras.initializers.RandomNormal(mean=0., stddev=1.),
        dtype=tf.float32)
    sqrt_k = math.sqrt(1.0 / self._hidden_size)
    self._class_embed = tf.keras.layers.Dense(
        self._num_classes,
        kernel_initializer=tf.keras.initializers.RandomUniform(-sqrt_k, sqrt_k),
        name="detr/cls_dense")
    self._bbox_embed = [
        tf.keras.layers.Dense(
            self._hidden_size, activation="relu",
            kernel_initializer=tf.keras.initializers.RandomUniform(
                -sqrt_k, sqrt_k),
            name="detr/box_dense_0"),
        tf.keras.layers.Dense(
            self._hidden_size, activation="relu",
            kernel_initializer=tf.keras.initializers.RandomUniform(
                -sqrt_k, sqrt_k),
            name="detr/box_dense_1"),
        tf.keras.layers.Dense(
            4, kernel_initializer=tf.keras.initializers.RandomUniform(
                -sqrt_k, sqrt_k),
            name="detr/box_dense_2")]
    self._sigmoid = tf.keras.layers.Activation("sigmoid")

  @property
  def backbone(self) -> tf.keras.Model:
    return self._backbone

  def get_config(self):
    return {
        "backbone": self._backbone,
        "backbone_endpoint_name": self._backbone_endpoint_name,
        "num_queries": self._num_queries,
        "hidden_size": self._hidden_size,
        "num_classes": self._num_classes,
        "num_encoder_layers": self._num_encoder_layers,
        "num_decoder_layers": self._num_decoder_layers,
        "dropout_rate": self._dropout_rate,
    }

  @classmethod
  def from_config(cls, config):
    return cls(**config)

  def _generate_image_mask(self, inputs: tf.Tensor,
                           target_shape: tf.Tensor) -> tf.Tensor:
    """Generates image mask from input image."""
    mask = tf.expand_dims(
        tf.cast(tf.not_equal(tf.reduce_sum(inputs, axis=-1), 0), inputs.dtype),
        axis=-1)
    mask = tf.image.resize(
        mask, target_shape, method=tf.image.ResizeMethod.NEAREST_NEIGHBOR)
    return mask

  def call(self, inputs: tf.Tensor, training: bool = None) -> List[Any]:
    batch_size = tf.shape(inputs)[0]
    features = self._backbone(inputs)[self._backbone_endpoint_name]
    shape = tf.shape(features)
    mask = self._generate_image_mask(inputs, shape[1: 3])

    pos_embed = position_embedding_sine(
        mask[:, :, :, 0], num_pos_features=self._hidden_size)
    pos_embed = tf.reshape(pos_embed, [batch_size, -1, self._hidden_size])

    features = tf.reshape(
        self._input_proj(features), [batch_size, -1, self._hidden_size])
    mask = tf.reshape(mask, [batch_size, -1])

    decoded_list = self._transformer({
        "inputs":
            features,
        "targets":
            tf.tile(
                tf.expand_dims(self._query_embeddings, axis=0),
                (batch_size, 1, 1)),
        "pos_embed": pos_embed,
        "mask": mask,
    })
    out_list = []
    for decoded in decoded_list:
      decoded = tf.stack(decoded)
      output_class = self._class_embed(decoded)
      box_out = decoded
      for layer in self._bbox_embed:
        box_out = layer(box_out)
      output_coord = self._sigmoid(box_out)
      out = {"cls_outputs": output_class, "box_outputs": output_coord}
      if not training:
        out.update(postprocess(out))
      out_list.append(out)

    return out_list


class DETRTransformer(tf.keras.layers.Layer):
  """Encoder and Decoder of DETR."""

  def __init__(self, num_encoder_layers=6, num_decoder_layers=6,
               dropout_rate=0.1, **kwargs):
    super().__init__(**kwargs)
    self._dropout_rate = dropout_rate
    self._num_encoder_layers = num_encoder_layers
    self._num_decoder_layers = num_decoder_layers

  def build(self, input_shape=None):
    if self._num_encoder_layers > 0:
      self._encoder = transformer.TransformerEncoder(
          attention_dropout_rate=self._dropout_rate,
          dropout_rate=self._dropout_rate,
          intermediate_dropout=self._dropout_rate,
          norm_first=False,
          num_layers=self._num_encoder_layers)
    else:
      self._encoder = None

    self._decoder = transformer.TransformerDecoder(
        attention_dropout_rate=self._dropout_rate,
        dropout_rate=self._dropout_rate,
        intermediate_dropout=self._dropout_rate,
        norm_first=False,
        num_layers=self._num_decoder_layers)
    super().build(input_shape)

  def get_config(self):
    return {
        "num_encoder_layers": self._num_encoder_layers,
        "num_decoder_layers": self._num_decoder_layers,
        "dropout_rate": self._dropout_rate,
    }

  def call(self, inputs):
    sources = inputs["inputs"]
    targets = inputs["targets"]
    pos_embed = inputs["pos_embed"]
    mask = inputs["mask"]
    input_shape = tf_utils.get_shape_list(sources)
    source_attention_mask = tf.tile(
        tf.expand_dims(mask, axis=1), [1, input_shape[1], 1])
    if self._encoder is not None:
      memory = self._encoder(
          sources, attention_mask=source_attention_mask, pos_embed=pos_embed)
    else:
      memory = sources

    target_shape = tf_utils.get_shape_list(targets)
    cross_attention_mask = tf.tile(
        tf.expand_dims(mask, axis=1), [1, target_shape[1], 1])
    target_shape = tf.shape(targets)
    decoded = self._decoder(
        tf.zeros_like(targets),
        memory,
        # TODO(b/199545430): self_attention_mask could be set to None when this
        # bug is resolved. Passing ones for now.
        self_attention_mask=tf.ones(
            (target_shape[0], target_shape[1], target_shape[1])),
        cross_attention_mask=cross_attention_mask,
        return_all_decoder_outputs=True,
        input_pos_embed=targets,
        memory_pos_embed=pos_embed)
    return decoded
