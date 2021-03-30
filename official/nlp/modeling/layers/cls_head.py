# Copyright 2021 The TensorFlow Authors. All Rights Reserved.
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

"""A Classification head layer which is common used with sequence encoders."""

import tensorflow as tf

from official.modeling import tf_utils

from official.nlp.modeling.layers import gaussian_process
from official.nlp.modeling.layers import spectral_normalization


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
      inner_dim: The dimensionality of inner projection layer. If 0 or `None`
        then only the output projection layer is created.
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

    if self.inner_dim:
      self.dense = tf.keras.layers.Dense(
          units=self.inner_dim,
          activation=self.activation,
          kernel_initializer=self.initializer,
          name="pooler_dense")
      self.dropout = tf.keras.layers.Dropout(rate=self.dropout_rate)

    self.out_proj = tf.keras.layers.Dense(
        units=num_classes, kernel_initializer=self.initializer, name="logits")

  def call(self, features):
    if not self.inner_dim:
      x = features
    else:
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
      inner_dim: The dimensionality of inner projection layer. If 0 or `None`
        then only the output projection layer is created.
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

    if self.inner_dim:
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
    if not self.inner_dim:
      x = features
    else:
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
    items = {self.dense.name: self.dense}
    items.update({v.name: v for v in self.out_projs})
    return items


class GaussianProcessClassificationHead(ClassificationHead):
  """Gaussian process-based pooling head for sentence classification.

  This class implements a classifier head for BERT encoder that is based on the
  spectral-normalized neural Gaussian process (SNGP) [1]. SNGP is a simple
  method to improve a neural network's uncertainty quantification ability
  without sacrificing accuracy or lantency. It applies spectral normalization to
  the hidden pooler layer, and then replaces the dense output layer with a
  Gaussian process.


  [1]: Jeremiah Liu et al. Simple and Principled Uncertainty Estimation with
       Deterministic Deep Learning via Distance Awareness.
       In _Neural Information Processing Systems_, 2020.
       https://arxiv.org/abs/2006.10108
  """

  def __init__(self,
               inner_dim,
               num_classes,
               cls_token_idx=0,
               activation="tanh",
               dropout_rate=0.0,
               initializer="glorot_uniform",
               use_spec_norm=True,
               use_gp_layer=True,
               **kwargs):
    """Initializes the `GaussianProcessClassificationHead`.

    Args:
      inner_dim: The dimensionality of inner projection layer. If 0 or `None`
        then only the output projection layer is created.
      num_classes: Number of output classes.
      cls_token_idx: The index inside the sequence to pool.
      activation: Dense layer activation.
      dropout_rate: Dropout probability.
      initializer: Initializer for dense layer kernels.
      use_spec_norm: Whether to apply spectral normalization to pooler layer.
      use_gp_layer: Whether to use Gaussian process as the output layer.
      **kwargs: Additional keyword arguments.
    """
    # Collects spectral normalization and Gaussian process args from kwargs.
    self.use_spec_norm = use_spec_norm
    self.use_gp_layer = use_gp_layer
    self.spec_norm_kwargs = extract_spec_norm_kwargs(kwargs)
    self.gp_layer_kwargs = extract_gp_layer_kwargs(kwargs)

    super().__init__(
        inner_dim=inner_dim,
        num_classes=num_classes,
        cls_token_idx=cls_token_idx,
        activation=activation,
        dropout_rate=dropout_rate,
        initializer=initializer,
        **kwargs)

    # Applies spectral normalization to the dense pooler layer.
    if self.use_spec_norm and hasattr(self, "dense"):
      self.dense = spectral_normalization.SpectralNormalization(
          self.dense, inhere_layer_name=True, **self.spec_norm_kwargs)

    # Replace Dense output layer with the Gaussian process layer.
    if use_gp_layer:
      self.out_proj = gaussian_process.RandomFeatureGaussianProcess(
          self.num_classes,
          kernel_initializer=self.initializer,
          name="logits",
          **self.gp_layer_kwargs)

  def get_config(self):
    config = dict(
        use_spec_norm=self.use_spec_norm, use_gp_layer=self.use_gp_layer)

    config.update(self.spec_norm_kwargs)
    config.update(self.gp_layer_kwargs)

    config.update(super(GaussianProcessClassificationHead, self).get_config())
    return config


def extract_gp_layer_kwargs(kwargs):
  """Extracts Gaussian process layer configs from a given kwarg."""

  return dict(
      num_inducing=kwargs.pop("num_inducing", 1024),
      normalize_input=kwargs.pop("normalize_input", True),
      gp_cov_momentum=kwargs.pop("gp_cov_momentum", 0.999),
      gp_cov_ridge_penalty=kwargs.pop("gp_cov_ridge_penalty", 1e-6),
      scale_random_features=kwargs.pop("scale_random_features", False),
      l2_regularization=kwargs.pop("l2_regularization", 0.),
      gp_cov_likelihood=kwargs.pop("gp_cov_likelihood", "gaussian"),
      return_gp_cov=kwargs.pop("return_gp_cov", True),
      return_random_features=kwargs.pop("return_random_features", False),
      use_custom_random_features=kwargs.pop("use_custom_random_features", True),
      custom_random_features_initializer=kwargs.pop(
          "custom_random_features_initializer", "random_normal"),
      custom_random_features_activation=kwargs.pop(
          "custom_random_features_activation", None))


def extract_spec_norm_kwargs(kwargs):
  """Extracts spectral normalization configs from a given kwarg."""

  return dict(
      iteration=kwargs.pop("iteration", 1),
      norm_multiplier=kwargs.pop("norm_multiplier", .99))
