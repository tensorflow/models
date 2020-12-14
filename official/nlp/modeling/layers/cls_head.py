# Lint as: python3
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
"""A Classification head layer which is common used with sequence encoders."""

import tensorflow as tf

from official.modeling import tf_utils


class ClassificationHead(tf.keras.layers.Layer):
  """Pooling head for sentence-level classification tasks."""

  def __init__(self,
               inner_dim,
               num_classes,
               cls_token_idx=0,
               activation="tanh",
               dropout_rate=0.0,
               initializer="glorot_uniform",
               **kwargs):
    """Initializes the `ClassificationHead`.

    Args:
      inner_dim: The dimensionality of inner projection layer.
      num_classes: Number of output classes.
      cls_token_idx: The index inside the sequence to pool.
      activation: Dense layer activation.
      dropout_rate: Dropout probability.
      initializer: Initializer for dense layer kernels.
      **kwargs: Keyword arguments.
    """
    super().__init__(**kwargs)
    self.dropout_rate = dropout_rate
    self.inner_dim = inner_dim
    self.num_classes = num_classes
    self.activation = tf_utils.get_activation(activation)
    self.initializer = tf.keras.initializers.get(initializer)
    self.cls_token_idx = cls_token_idx

    self.dense = tf.keras.layers.Dense(
        units=inner_dim,
        activation=self.activation,
        kernel_initializer=self.initializer,
        name="pooler_dense")
    self.dropout = tf.keras.layers.Dropout(rate=self.dropout_rate)
    self.out_proj = tf.keras.layers.Dense(
        units=num_classes, kernel_initializer=self.initializer, name="logits")

  def call(self, features):
    x = features[:, self.cls_token_idx, :]  # take <CLS> token.
    x = self.dense(x)
    x = self.dropout(x)
    x = self.out_proj(x)
    return x

  def get_config(self):
    config = {
        "cls_token_idx": self.cls_token_idx,
        "dropout_rate": self.dropout_rate,
        "num_classes": self.num_classes,
        "inner_dim": self.inner_dim,
        "activation": tf.keras.activations.serialize(self.activation),
        "initializer": tf.keras.initializers.serialize(self.initializer),
    }
    config.update(super(ClassificationHead, self).get_config())
    return config

  @classmethod
  def from_config(cls, config, custom_objects=None):
    return cls(**config)

  @property
  def checkpoint_items(self):
    return {self.dense.name: self.dense}


class MultiClsHeads(tf.keras.layers.Layer):
  """Pooling heads sharing the same pooling stem."""

  def __init__(self,
               inner_dim,
               cls_list,
               cls_token_idx=0,
               activation="tanh",
               dropout_rate=0.0,
               initializer="glorot_uniform",
               **kwargs):
    """Initializes the `MultiClsHeads`.

    Args:
      inner_dim: The dimensionality of inner projection layer.
      cls_list: a list of pairs of (classification problem name and the numbers
        of classes.
      cls_token_idx: The index inside the sequence to pool.
      activation: Dense layer activation.
      dropout_rate: Dropout probability.
      initializer: Initializer for dense layer kernels.
      **kwargs: Keyword arguments.
    """
    super().__init__(**kwargs)
    self.dropout_rate = dropout_rate
    self.inner_dim = inner_dim
    self.cls_list = cls_list
    self.activation = tf_utils.get_activation(activation)
    self.initializer = tf.keras.initializers.get(initializer)
    self.cls_token_idx = cls_token_idx

    self.dense = tf.keras.layers.Dense(
        units=inner_dim,
        activation=self.activation,
        kernel_initializer=self.initializer,
        name="pooler_dense")
    self.dropout = tf.keras.layers.Dropout(rate=self.dropout_rate)
    self.out_projs = []
    for name, num_classes in cls_list:
      self.out_projs.append(
          tf.keras.layers.Dense(
              units=num_classes, kernel_initializer=self.initializer,
              name=name))

  def call(self, features):
    x = features[:, self.cls_token_idx, :]  # take <CLS> token.
    x = self.dense(x)
    x = self.dropout(x)
    outputs = {}
    for proj_layer in self.out_projs:
      outputs[proj_layer.name] = proj_layer(x)
    return outputs

  def get_config(self):
    config = {
        "dropout_rate": self.dropout_rate,
        "cls_token_idx": self.cls_token_idx,
        "cls_list": self.cls_list,
        "inner_dim": self.inner_dim,
        "activation": tf.keras.activations.serialize(self.activation),
        "initializer": tf.keras.initializers.serialize(self.initializer),
    }
    config.update(super().get_config())
    return config

  @classmethod
  def from_config(cls, config, custom_objects=None):
    return cls(**config)

  @property
  def checkpoint_items(self):
    # TODO(hongkuny): add output projects to the checkpoint items.
    return {self.dense.name: self.dense}
