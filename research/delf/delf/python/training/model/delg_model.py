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
"""DELG model implementation based on the following paper.

  Unifying Deep Local and Global Features for Image Search
  https://arxiv.org/abs/2001.05027
"""

import functools
import math

from absl import logging
import tensorflow as tf

from delf.python.training.model import delf_model

layers = tf.keras.layers


class Delg(delf_model.Delf):
  """Instantiates Keras DELG model using ResNet50 as backbone.

  This class implements the [DELG](https://arxiv.org/abs/2001.05027) model for
  extracting local and global features from images. The same attention layer
  is trained as in the DELF model. In addition, the extraction of global
  features is trained using GeMPooling, a FC whitening layer also called
  "embedding layer" and ArcFace loss.
  """

  def __init__(self,
               block3_strides=True,
               name='DELG',
               gem_power=3.0,
               embedding_layer_dim=2048,
               scale_factor_init=45.25,  # sqrt(2048)
               arcface_margin=0.1):
    """Initialization of DELG model.

    Args:
      block3_strides: bool, whether to add strides to the output of block3.
      name: str, name to identify model.
      gem_power: float, GeM power parameter.
      embedding_layer_dim : int, dimension of the embedding layer.
      scale_factor_init: float.
      arcface_margin: float, ArcFace margin.
    """
    logging.info('Creating Delg model, gem_power %d, embedding_layer_dim %d',
                 gem_power, embedding_layer_dim)
    super(Delg, self).__init__(block3_strides=block3_strides,
                               name=name,
                               pooling='gem',
                               gem_power=gem_power,
                               embedding_layer=True,
                               embedding_layer_dim=embedding_layer_dim)
    self._embedding_layer_dim = embedding_layer_dim
    self._scale_factor_init = scale_factor_init
    self._arcface_margin = arcface_margin

  def init_classifiers(self, num_classes):
    """Define classifiers for training backbone and attention models."""
    logging.info('Initializing Delg backbone and attention models classifiers')
    backbone_classifier_func = self._create_backbone_classifier(num_classes)
    super(Delg, self).init_classifiers(
        num_classes,
        desc_classification=backbone_classifier_func)

  def _create_backbone_classifier(self, num_classes):
    """Define the classifier for training the backbone model."""
    logging.info('Creating cosine classifier')
    self.cosine_weights = tf.Variable(
        initial_value=tf.initializers.GlorotUniform()(
            shape=[self._embedding_layer_dim, num_classes]),
        name='cosine_weights',
        trainable=True)
    self.scale_factor = tf.Variable(self._scale_factor_init,
                                    name='scale_factor',
                                    trainable=False)
    classifier_func = functools.partial(cosine_classifier_logits,
                                        num_classes=num_classes,
                                        cosine_weights=self.cosine_weights,
                                        scale_factor=self.scale_factor,
                                        arcface_margin=self._arcface_margin)
    classifier_func.trainable_weights = [self.cosine_weights]
    return classifier_func


def cosine_classifier_logits(prelogits,
                             labels,
                             num_classes,
                             cosine_weights,
                             scale_factor,
                             arcface_margin,
                             training=True):
  """Compute cosine classifier logits using ArFace margin.

  Args:
    prelogits: float tensor of shape [batch_size, embedding_layer_dim].
    labels: int tensor of shape [batch_size].
    num_classes: int, number of classes.
    cosine_weights: float tensor of shape [embedding_layer_dim, num_classes].
    scale_factor: float.
    arcface_margin: float. Only used if greater than zero, and training is True.
    training: bool, True if training, False if eval.

  Returns:
    logits: Float tensor [batch_size, num_classes].
  """
  # L2-normalize prelogits, then obtain cosine similarity.
  normalized_prelogits = tf.math.l2_normalize(prelogits, axis=1)
  normalized_weights = tf.math.l2_normalize(cosine_weights, axis=0)
  cosine_sim = tf.matmul(normalized_prelogits, normalized_weights)

  # Optionally use ArcFace margin.
  if training and arcface_margin > 0.0:
    # Reshape labels tensor from [batch_size] to [batch_size, num_classes].
    one_hot_labels = tf.one_hot(labels, num_classes)
    cosine_sim = apply_arcface_margin(cosine_sim,
                                      one_hot_labels,
                                      arcface_margin)

  # Apply the scale factor to logits and return.
  logits = scale_factor * cosine_sim
  return logits


def apply_arcface_margin(cosine_sim, one_hot_labels, arcface_margin):
  """Applies ArcFace margin to cosine similarity inputs.

  For a reference, see https://arxiv.org/pdf/1801.07698.pdf. ArFace margin is
  applied to angles from correct classes (as per the ArcFace paper), and only
  if they are <= (pi - margin). Otherwise, applying the margin may actually
  improve their cosine similarity.

  Args:
    cosine_sim: float tensor with shape [batch_size, num_classes].
    one_hot_labels: int tensor with shape [batch_size, num_classes].
    arcface_margin: float.

  Returns:
    cosine_sim_with_margin: Float tensor with shape [batch_size, num_classes].
  """
  theta = tf.acos(cosine_sim, name='acos')
  selected_labels = tf.where(tf.greater(theta, math.pi - arcface_margin),
                             tf.zeros_like(one_hot_labels),
                             one_hot_labels,
                             name='selected_labels')
  final_theta = tf.where(tf.cast(selected_labels, dtype=tf.bool),
                         theta + arcface_margin,
                         theta,
                         name='final_theta')
  return tf.cos(final_theta, name='cosine_sim_with_margin')
