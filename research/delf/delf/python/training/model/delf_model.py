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
"""DELF model implementation based on the following paper.

  Large-Scale Image Retrieval with Attentive Deep Local Features
  https://arxiv.org/abs/1612.06321
"""

import tensorflow as tf

from delf.python.training.model import resnet50 as resnet

layers = tf.keras.layers
reg = tf.keras.regularizers

_DECAY = 0.0001


class AttentionModel(tf.keras.Model):
  """Instantiates attention model.

  Uses two [kernel_size x kernel_size] convolutions and softplus as activation
  to compute an attention map with the same resolution as the featuremap.
  Features l2-normalized and aggregated using attention probabilites as weights.
  """

  def __init__(self, kernel_size=1, decay=_DECAY, name='attention'):
    """Initialization of attention model.

    Args:
      kernel_size: int, kernel size of convolutions.
      decay: float, decay for l2 regularization of kernel weights.
      name: str, name to identify model.
    """
    super(AttentionModel, self).__init__(name=name)

    # First convolutional layer (called with relu activation).
    self.conv1 = layers.Conv2D(
        512,
        kernel_size,
        kernel_regularizer=reg.l2(decay),
        padding='same',
        name='attn_conv1')
    self.bn_conv1 = layers.BatchNormalization(axis=3, name='bn_conv1')

    # Second convolutional layer, with softplus activation.
    self.conv2 = layers.Conv2D(
        1,
        kernel_size,
        kernel_regularizer=reg.l2(decay),
        padding='same',
        name='attn_conv2')
    self.activation_layer = layers.Activation('softplus')

  def call(self, inputs, training=True):
    x = self.conv1(inputs)
    x = self.bn_conv1(x, training=training)
    x = tf.nn.relu(x)

    score = self.conv2(x)
    prob = self.activation_layer(score)

    # L2-normalize the featuremap before pooling.
    inputs = tf.nn.l2_normalize(inputs, axis=-1)
    feat = tf.reduce_mean(tf.multiply(inputs, prob), [1, 2], keepdims=False)

    return feat, prob, score


class AutoencoderModel(tf.keras.Model):
  """Instantiates the Keras Autoencoder model."""

  def __init__(self, reduced_dimension, expand_dimension, kernel_size=1,
               name='autoencoder'):
    """Initialization of Autoencoder model.

    Args:
      reduced_dimension: int, the output dimension of the autoencoder layer.
      expand_dimension: int, the input dimension of the autoencoder layer.
      kernel_size: int or tuple, height and width of the 2D convolution window.
      name: str, name to identify model.
    """
    super(AutoencoderModel, self).__init__(name=name)
    self.conv1 = layers.Conv2D(
        reduced_dimension,
        kernel_size,
        padding='same',
        name='autoenc_conv1')
    self.conv2 = layers.Conv2D(
        expand_dimension,
        kernel_size,
        activation=tf.keras.activations.relu,
        padding='same',
        name='autoenc_conv2')

  def call(self, inputs):
    dim_reduced_features = self.conv1(inputs)
    dim_expanded_features = self.conv2(dim_reduced_features)
    return dim_expanded_features, dim_reduced_features


class Delf(tf.keras.Model):
  """Instantiates Keras DELF model using ResNet50 as backbone.

  This class implements the [DELF](https://arxiv.org/abs/1612.06321) model for
  extracting local features from images. The backbone is a ResNet50 network
  that extracts featuremaps from both conv_4 and conv_5 layers. Activations
  from conv_4 are used to compute an attention map of the same resolution.
  """

  def __init__(self,
               block3_strides=True,
               name='DELF',
               pooling='avg',
               gem_power=3.0,
               embedding_layer=False,
               embedding_layer_dim=2048,
               use_dim_reduction=False,
               reduced_dimension=128,
               dim_expand_channels=1024):
    """Initialization of DELF model.

    Args:
      block3_strides: bool, whether to add strides to the output of block3.
      name: str, name to identify model.
      pooling: str, pooling mode for global feature extraction; possible values
        are 'None', 'avg', 'max', 'gem.'
      gem_power: float, GeM power for GeM pooling. Only used if pooling ==
        'gem'.
      embedding_layer: bool, whether to create an embedding layer (FC whitening
        layer).
      embedding_layer_dim: int, size of the embedding layer.
      use_dim_reduction: Whether to integrate dimensionality reduction layers.
        If True, extra layers are added to reduce the dimensionality of the
        extracted features.
      reduced_dimension: int, only used if use_dim_reduction is True. The output
        dimension of the autoencoder layer.
      dim_expand_channels: int, only used if use_dim_reduction is True. The
        number of channels of the backbone block used. Default value 1024 is the
        number of channels of backbone block 'block3'.
    """
    super(Delf, self).__init__(name=name)

    # Backbone using Keras ResNet50.
    self.backbone = resnet.ResNet50(
        'channels_last',
        name='backbone',
        include_top=False,
        pooling=pooling,
        block3_strides=block3_strides,
        average_pooling=False,
        gem_power=gem_power,
        embedding_layer=embedding_layer,
        embedding_layer_dim=embedding_layer_dim)

    # Attention model.
    self.attention = AttentionModel(name='attention')

    # Autoencoder model.
    self._use_dim_reduction = use_dim_reduction
    if self._use_dim_reduction:
      self.autoencoder = AutoencoderModel(reduced_dimension,
                                          dim_expand_channels,
                                          name='autoencoder')

  def init_classifiers(self, num_classes, desc_classification=None):
    """Define classifiers for training backbone and attention models."""
    self.num_classes = num_classes
    if desc_classification is None:
      self.desc_classification = layers.Dense(
          num_classes, activation=None, kernel_regularizer=None, name='desc_fc')
    else:
      self.desc_classification = desc_classification
    self.attn_classification = layers.Dense(
        num_classes, activation=None, kernel_regularizer=None, name='att_fc')

  def global_and_local_forward_pass(self, images, training=True):
    """Run a forward to calculate global descriptor and attention prelogits.

    Args:
      images: Tensor containing the dataset on which to run the forward pass.
      training: Indicator of wether the forward pass is running in training mode
        or not.

    Returns:
      Global descriptor prelogits, attention prelogits, attention scores,
        backbone weights.
    """
    backbone_blocks = {}
    desc_prelogits = self.backbone.build_call(
        images, intermediates_dict=backbone_blocks, training=training)
    # Prevent gradients from propagating into the backbone. See DELG paper:
    # https://arxiv.org/abs/2001.05027.
    block3 = backbone_blocks['block3']  # pytype: disable=key-error
    block3 = tf.stop_gradient(block3)
    if self._use_dim_reduction:
      (dim_expanded_features, dim_reduced_features) = self.autoencoder(block3)
      attn_prelogits, attn_scores, _ = self.attention(dim_expanded_features,
                                                      training=training)
    else:
      attn_prelogits, attn_scores, _ = self.attention(block3, training=training)
      dim_expanded_features = None
      dim_reduced_features = None
    return (desc_prelogits, attn_prelogits, attn_scores, backbone_blocks,
            dim_expanded_features, dim_reduced_features)

  def build_call(self, input_image, training=True):
    (global_feature, _, attn_scores, backbone_blocks, _,
     dim_reduced_features) = self.global_and_local_forward_pass(input_image,
                                                                training)
    if self._use_dim_reduction:
      features = dim_reduced_features
    else:
      features = backbone_blocks['block3']  # pytype: disable=key-error
    return global_feature, attn_scores, features

  def call(self, input_image, training=True):
    _, probs, features = self.build_call(input_image, training=training)
    return probs, features
