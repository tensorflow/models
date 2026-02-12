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

"""Resnet based DETR implementation in Keras.

See Deep Residual Learning for Image Recognition by He et al.
https://arxiv.org/abs/1512.03385
"""

import tensorflow.compat.v1 as tf

from object_detection.meta_architectures import detr_meta_arch
from object_detection.models.keras_models import resnet_v1
from object_detection.utils import model_util


_RESNET_MODEL_CONV5_LAST_LAYERS = {
    'resnet_v1_50': 'conv5_block3_out'
}


class DETRResnetKerasFeatureExtractor(
    detr_meta_arch.DETRKerasFeatureExtractor):
  """DETR with Resnet feature extractor implementation."""

  def __init__(self,
               is_training,
               resnet_v1_base_model,
               resnet_v1_base_model_name,
               features_stride=32,
               batch_norm_trainable=False,
               weight_decay=0.0):
    """Constructor.

    Args:
      is_training: See base class.
      resnet_v1_base_model: base resnet v1 network to use. Only
        the resnet_v1.resnet_v1_{50} is supported currently.
      resnet_v1_base_model_name: model name under which to construct resnet v1.
      features_stride: See base class.
      batch_norm_trainable: See base class.
      weight_decay: See base class.

    Raises:
      ValueError: If `features_stride` is not 32.
    """
    if features_stride != 32:
      raise ValueError('`features_stride` must be 32.')
    super(DETRResnetKerasFeatureExtractor, self).__init__(
        is_training, features_stride, batch_norm_trainable,
        weight_decay)
    self.classification_backbone = None
    self._variable_dict = {}
    self._resnet_v1_base_model = resnet_v1_base_model
    self._resnet_v1_base_model_name = resnet_v1_base_model_name

  def preprocess(self, resized_inputs):
    """DETR Resnet V1 preprocessing.


    VGG style channel mean subtraction as described here:
    https://gist.github.com/ksimonyan/211839e770f7b538e2d8#file-readme-md
    Note that if the number of channels is not equal to 3, the mean subtraction
    will be skipped and the original resized_inputs will be returned.

    Args:
      resized_inputs: A [batch, height_in, width_in, channels] float32 tensor
        representing a batch of images with values between 0 and 255.0.

    Returns:
      preprocessed_inputs: A [batch, height_out, width_out, channels] float32
        tensor representing a batch of images.

    """
    if resized_inputs.shape.as_list()[3] == 3:
      channel_means = [123.68, 116.779, 103.939]
      return resized_inputs - [[channel_means]]
    else:
      return resized_inputs

  def get_proposal_feature_extractor_model(self, name=None):
    """Returns a model that extracts first stage RPN features.

    Extracts features using the first half of the Resnet v1 network.

    Args:
      name: A scope name to construct all variables within.

    Returns:
      A Keras model that takes preprocessed_inputs:
        A [batch, height, width, channels] float32 tensor
        representing a batch of images.

      And returns rpn_feature_map:
        A tensor with shape [batch, height, width, depth]
    """
    if not self.classification_backbone:
      self.classification_backbone = self._resnet_v1_base_model(
          batchnorm_training=self._train_batch_norm,
          conv_hyperparams=None,
          weight_decay=self._weight_decay,
          classes=None,
          weights=None,
          include_top=False
          )
    with tf.name_scope(name):
      with tf.name_scope('ResnetV1'):
        conv5_last_layer = _RESNET_MODEL_CONV5_LAST_LAYERS[
            self._resnet_v1_base_model_name]
        proposal_features = self.classification_backbone.get_layer(
            name=conv5_last_layer).output
        keras_model = tf.keras.Model(
            inputs=self.classification_backbone.inputs,
            outputs=proposal_features)
        for variable in keras_model.variables:
          self._variable_dict[variable.name[:-2]] = variable
        return keras_model

class DETRResnet50KerasFeatureExtractor(
    DETRResnetKerasFeatureExtractor):
  """DETR with Resnet50 feature extractor implementation."""

  def __init__(self,
               is_training,
               features_stride=32,
               batch_norm_trainable=False,
               weight_decay=0.0):
    """Constructor.

    Args:
      is_training: See base class.
      features_stride: See base class.
      batch_norm_trainable: See base class.
      weight_decay: See base class.
    """
    super(DETRResnet50KerasFeatureExtractor, self).__init__(
        is_training=is_training,
        resnet_v1_base_model=resnet_v1.resnet_v1_50,
        resnet_v1_base_model_name='resnet_v1_50',
        features_stride=features_stride,
        batch_norm_trainable=batch_norm_trainable,
        weight_decay=weight_decay)


class DETRResnet101KerasFeatureExtractor(
    DETRResnetKerasFeatureExtractor):
  """DETR with Resnet101 feature extractor implementation."
  
  Currently unsupported.
  """

  def __init__(self,
               is_training,
               features_stride=32,
               batch_norm_trainable=False,
               weight_decay=0.0):
    """Constructor.

    Args:
      is_training: See base class.
      features_stride: See base class.
      batch_norm_trainable: See base class.
      weight_decay: See base class.
    """
    super(DETRResnet101KerasFeatureExtractor, self).__init__(
        is_training=is_training,
        resnet_v1_base_model=resnet_v1.resnet_v1_101,
        resnet_v1_base_model_name='resnet_v1_101',
        features_stride=features_stride,
        batch_norm_trainable=batch_norm_trainable,
        weight_decay=weight_decay)


class DETRResnet152KerasFeatureExtractor(
    DETRResnetKerasFeatureExtractor):
  """DETR with Resnet152 feature extractor implementation.
  
  Currently unsupported.
  """

  def __init__(self,
               is_training,
               features_stride=32,
               batch_norm_trainable=False,
               weight_decay=0.0):
    """Constructor.

    Args:
      is_training: See base class.
      features_stride: See base class.
      batch_norm_trainable: See base class.
      weight_decay: See base class.
    """
    super(DETRResnet152KerasFeatureExtractor, self).__init__(
        is_training=is_training,
        resnet_v1_base_model=resnet_v1.resnet_v1_152,
        resnet_v1_base_model_name='resnet_v1_152',
        first_stage_features_stride=features_stride,
        batch_norm_trainable=batch_norm_trainable,
        weight_decay=weight_decay)
