# Copyright 2025 The TensorFlow Authors. All Rights Reserved.
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

import tensorflow as tf, tf_keras

from official.modeling import tf_utils

from official.nlp.modeling.layers import gaussian_process
from official.nlp.modeling.layers import spectral_normalization


class ClassificationHead(tf_keras.layers.Layer):
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
    self.initializer = tf_keras.initializers.get(initializer)
    self.cls_token_idx = cls_token_idx

    if self.inner_dim:
      self.dense = tf_keras.layers.Dense(
          units=self.inner_dim,
          activation=self.activation,
          kernel_initializer=tf_utils.clone_initializer(self.initializer),
          name="pooler_dense")
    self.dropout = tf_keras.layers.Dropout(rate=self.dropout_rate)

    self.out_proj = tf_keras.layers.Dense(
        units=num_classes,
        kernel_initializer=tf_utils.clone_initializer(self.initializer),
        name="logits")

  def call(self, features: tf.Tensor, only_project: bool = False):
    """Implements call().

    Args:
      features: a rank-3 Tensor when self.inner_dim is specified, otherwise
        it is a rank-2 Tensor.
      only_project: a boolean. If True, we return the intermediate Tensor
        before projecting to class logits.

    Returns:
      a Tensor, if only_project is True, shape= [batch size, hidden size].
      If only_project is False, shape= [batch size, num classes].
    """
    if not self.inner_dim:
      x = features
    else:
      x = features[:, self.cls_token_idx, :]  # take <CLS> token.
      x = self.dense(x)

    if only_project:
      return x
    x = self.dropout(x)
    x = self.out_proj(x)
    return x

  def get_config(self):
    config = {
        "cls_token_idx": self.cls_token_idx,
        "dropout_rate": self.dropout_rate,
        "num_classes": self.num_classes,
        "inner_dim": self.inner_dim,
        "activation": tf_keras.activations.serialize(self.activation),
        "initializer": tf_keras.initializers.serialize(self.initializer),
    }
    config.update(super(ClassificationHead, self).get_config())
    return config

  @classmethod
  def from_config(cls, config, custom_objects=None):
    return cls(**config)

  @property
  def checkpoint_items(self):
    return {self.dense.name: self.dense}


class MultiClsHeads(tf_keras.layers.Layer):
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
    self.initializer = tf_keras.initializers.get(initializer)
    self.cls_token_idx = cls_token_idx

    if self.inner_dim:
      self.dense = tf_keras.layers.Dense(
          units=inner_dim,
          activation=self.activation,
          kernel_initializer=tf_utils.clone_initializer(self.initializer),
          name="pooler_dense")
    self.dropout = tf_keras.layers.Dropout(rate=self.dropout_rate)
    self.out_projs = []
    for name, num_classes in cls_list:
      self.out_projs.append(
          tf_keras.layers.Dense(
              units=num_classes,
              kernel_initializer=tf_utils.clone_initializer(self.initializer),
              name=name))

  def call(self, features: tf.Tensor, only_project: bool = False):
    """Implements call().

    Args:
      features: a rank-3 Tensor when self.inner_dim is specified, otherwise
        it is a rank-2 Tensor.
      only_project: a boolean. If True, we return the intermediate Tensor
        before projecting to class logits.

    Returns:
      If only_project is True, a Tensor with shape= [batch size, hidden size].
      If only_project is False, a dictionary of Tensors.
    """
    if not self.inner_dim:
      x = features
    else:
      x = features[:, self.cls_token_idx, :]  # take <CLS> token.
      x = self.dense(x)

    if only_project:
      return x
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
        "activation": tf_keras.activations.serialize(self.activation),
        "initializer": tf_keras.initializers.serialize(self.initializer),
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
               temperature=None,
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
      temperature: The temperature parameter to be used for mean-field
        approximation during inference. If None then no mean-field adjustment is
        applied.
      **kwargs: Additional keyword arguments.
    """
    # Collects spectral normalization and Gaussian process args from kwargs.
    self.use_spec_norm = use_spec_norm
    self.use_gp_layer = use_gp_layer
    self.spec_norm_kwargs = extract_spec_norm_kwargs(kwargs)
    self.gp_layer_kwargs = extract_gp_layer_kwargs(kwargs)
    self.temperature = temperature

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
          kernel_initializer=tf_utils.clone_initializer(self.initializer),
          name="logits",
          **self.gp_layer_kwargs)

  def call(self, features, training=False, return_covmat=False):
    """Returns model output.

    Dring training, the model returns raw logits. During evaluation, the model
    returns uncertainty adjusted logits, and (optionally) the covariance matrix.

    Arguments:
      features: A tensor of input features, shape (batch_size, feature_dim).
      training: Whether the model is in training mode.
      return_covmat: Whether the model should also return covariance matrix if
        `use_gp_layer=True`. During training, it is recommended to set
        `return_covmat=False` to be compatible with the standard Keras pipelines
        (e.g., `model.fit()`).

    Returns:
      logits: Uncertainty-adjusted predictive logits, shape
        (batch_size, num_classes).
      covmat: (Optional) Covariance matrix, shape (batch_size, batch_size).
        Returned only when return_covmat=True.
    """
    logits = super().call(features)

    # Extracts logits and covariance matrix from model output.
    if self.use_gp_layer:
      logits, covmat = logits
    else:
      covmat = None

    # Computes the uncertainty-adjusted logits during evaluation.
    if not training:
      logits = gaussian_process.mean_field_logits(
          logits, covmat, mean_field_factor=self.temperature)

    if return_covmat and covmat is not None:
      return logits, covmat
    return logits

  def reset_covariance_matrix(self):
    """Resets covariance matrix of the Gaussian process layer."""
    if hasattr(self.out_proj, "reset_covariance_matrix"):
      self.out_proj.reset_covariance_matrix()

  def get_config(self):
    config = dict(
        use_spec_norm=self.use_spec_norm, use_gp_layer=self.use_gp_layer)

    config.update(self.spec_norm_kwargs)
    config.update(self.gp_layer_kwargs)
    config["temperature"] = self.temperature

    config.update(super(GaussianProcessClassificationHead, self).get_config())
    return config


def extract_gp_layer_kwargs(kwargs):
  """Extracts Gaussian process layer configs from a given kwarg."""

  return dict(
      num_inducing=kwargs.pop("num_inducing", 1024),
      normalize_input=kwargs.pop("normalize_input", True),
      gp_cov_momentum=kwargs.pop("gp_cov_momentum", 0.999),
      gp_cov_ridge_penalty=kwargs.pop("gp_cov_ridge_penalty", 1.),
      scale_random_features=kwargs.pop("scale_random_features", False),
      l2_regularization=kwargs.pop("l2_regularization", 1e-6),
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


class PerQueryDenseHead(tf_keras.layers.Layer):
  """Pooling head used for EncT5 style models.

    This module projects each query to use a different projection.

    For a input shape= [bs, num_queries, hidden_size], it projects each query to
    (features). Ending up with shape= [bs, num_queries, features].

    For example, for classification with a few classes, one may use num_queries
    as 1 and features as number of classes. For multilabel classification, one
    may use num_queries as number of classes and features as 2. So each query
    represents a binary classification of one label.
  """

  def __init__(self,
               num_queries: int,
               features: int,
               use_bias: bool = False,
               kernel_initializer: str = "glorot_uniform",
               **kwargs):
    """Initializes the `PerQueryDenseHead`.

    Args:
      num_queries: number of queries (the learnable embeddings in the input
        sequences) from the decoder.
      features: int with numbers of output features. Each query with be
        projected to this number with a different projection.
      use_bias: whether to add a bias to the output.
      kernel_initializer: Initializer for dense layer kernels.
      **kwargs: Keyword arguments.
    """
    super().__init__(**kwargs)
    self.num_queries = num_queries
    self.features = features

    self.use_bias = use_bias
    self.kernel_initializer = tf_keras.initializers.get(kernel_initializer)

  def build(self, input_shape):
    input_shape = tf.TensorShape(input_shape)
    # Hidden size.
    last_dim = tf.compat.dimension_value(input_shape[-1])

    self.hidden_size = last_dim
    self.kernel = self.add_weight(
        "kernel",
        shape=[self.num_queries, last_dim, self.features],
        initializer=self.kernel_initializer,
        dtype=self.dtype,
        trainable=True)
    if self.use_bias:
      self.bias = self.add_weight(
          "bias",
          shape=[
              self.num_queries,
              self.features,
          ],
          dtype=self.dtype,
          trainable=True)
    else:
      self.bias = None

  def call(self, inputs: tf.Tensor) -> tf.Tensor:
    """Implements call().

    Args:
      inputs: a rank-3 Tensor of shape= [bs, num_queries, hidden_size].

    Returns:
      A Tensor, shape= [batch size, num_queries, features].
    """

    outputs = tf.einsum("bqh,qhf->bqf", inputs, self.kernel)
    if self.use_bias:
      outputs += self.bias
    return outputs

  def get_config(self):
    config = {
        "num_queries":
            self.num_queries,
        "features":
            self.features,
        "kernel_initializer":
            tf_keras.activations.serialize(self.kernel_initializer),
    }
    config.update(super(PerQueryDenseHead, self).get_config())
    return config

  @classmethod
  def from_config(cls, config, custom_objects=None):
    return cls(**config)
