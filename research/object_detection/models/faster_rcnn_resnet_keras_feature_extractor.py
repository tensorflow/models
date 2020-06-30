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

"""Resnet based Faster R-CNN implementation in Keras.

See Deep Residual Learning for Image Recognition by He et al.
https://arxiv.org/abs/1512.03385
"""

import tensorflow.compat.v1 as tf

from object_detection.meta_architectures import faster_rcnn_meta_arch
from object_detection.models.keras_models import resnet_v1
from object_detection.utils import model_util


_RESNET_MODEL_CONV4_LAST_LAYERS = {
    'resnet_v1_50': 'conv4_block6_out',
    'resnet_v1_101': 'conv4_block23_out',
    'resnet_v1_152': 'conv4_block36_out',
}


class FasterRCNNResnetKerasFeatureExtractor(
    faster_rcnn_meta_arch.FasterRCNNKerasFeatureExtractor):
  """Faster R-CNN with Resnet feature extractor implementation."""

  def __init__(self,
               is_training,
               resnet_v1_base_model,
               resnet_v1_base_model_name,
               first_stage_features_stride=16,
               batch_norm_trainable=False,
               weight_decay=0.0):
    """Constructor.

    Args:
      is_training: See base class.
      resnet_v1_base_model: base resnet v1 network to use. One of
        the resnet_v1.resnet_v1_{50,101,152} models.
      resnet_v1_base_model_name: model name under which to construct resnet v1.
      first_stage_features_stride: See base class.
      batch_norm_trainable: See base class.
      weight_decay: See base class.

    Raises:
      ValueError: If `first_stage_features_stride` is not 8 or 16.
    """
    if first_stage_features_stride != 16:
      raise ValueError('`first_stage_features_stride` must be 16.')
    super(FasterRCNNResnetKerasFeatureExtractor, self).__init__(
        is_training, first_stage_features_stride, batch_norm_trainable,
        weight_decay)
    self.classification_backbone = None
    self._variable_dict = {}
    self._resnet_v1_base_model = resnet_v1_base_model
    self._resnet_v1_base_model_name = resnet_v1_base_model_name

  def preprocess(self, resized_inputs):
    """Faster R-CNN Resnet V1 preprocessing.

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

        conv4_last_layer = _RESNET_MODEL_CONV4_LAST_LAYERS[
            self._resnet_v1_base_model_name]
        proposal_features = self.classification_backbone.get_layer(
            name=conv4_last_layer).output
        keras_model = tf.keras.Model(
            inputs=self.classification_backbone.inputs,
            outputs=proposal_features)
        for variable in keras_model.variables:
          self._variable_dict[variable.name[:-2]] = variable
        return keras_model

  def get_box_classifier_feature_extractor_model(self, name=None):
    """Returns a model that extracts second stage box classifier features.

    This function reconstructs the "second half" of the ResNet v1
    network after the part defined in `get_proposal_feature_extractor_model`.

    Args:
      name: A scope name to construct all variables within.

    Returns:
      A Keras model that takes proposal_feature_maps:
        A 4-D float tensor with shape
        [batch_size * self.max_num_proposals, crop_height, crop_width, depth]
        representing the feature map cropped to each proposal.
      And returns proposal_classifier_features:
        A 4-D float tensor with shape
        [batch_size * self.max_num_proposals, height, width, depth]
        representing box classifier features for each proposal.
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
        conv4_last_layer = _RESNET_MODEL_CONV4_LAST_LAYERS[
            self._resnet_v1_base_model_name]
        proposal_feature_maps = self.classification_backbone.get_layer(
            name=conv4_last_layer).output
        proposal_classifier_features = self.classification_backbone.get_layer(
            name='conv5_block3_out').output

        keras_model = model_util.extract_submodel(
            model=self.classification_backbone,
            inputs=proposal_feature_maps,
            outputs=proposal_classifier_features)
        for variable in keras_model.variables:
          self._variable_dict[variable.name[:-2]] = variable
        return keras_model


class FasterRCNNResnet50KerasFeatureExtractor(
    FasterRCNNResnetKerasFeatureExtractor):
  """Faster R-CNN with Resnet50 feature extractor implementation."""

  def __init__(self,
               is_training,
               first_stage_features_stride=16,
               batch_norm_trainable=False,
               weight_decay=0.0):
    """Constructor.

    Args:
      is_training: See base class.
      first_stage_features_stride: See base class.
      batch_norm_trainable: See base class.
      weight_decay: See base class.
    """
    super(FasterRCNNResnet50KerasFeatureExtractor, self).__init__(
        is_training=is_training,
        resnet_v1_base_model=resnet_v1.resnet_v1_50,
        resnet_v1_base_model_name='resnet_v1_50',
        first_stage_features_stride=first_stage_features_stride,
        batch_norm_trainable=batch_norm_trainable,
        weight_decay=weight_decay)


class FasterRCNNResnet101KerasFeatureExtractor(
    FasterRCNNResnetKerasFeatureExtractor):
  """Faster R-CNN with Resnet101 feature extractor implementation."""

  def __init__(self,
               is_training,
               first_stage_features_stride=16,
               batch_norm_trainable=False,
               weight_decay=0.0):
    """Constructor.

    Args:
      is_training: See base class.
      first_stage_features_stride: See base class.
      batch_norm_trainable: See base class.
      weight_decay: See base class.
    """
    super(FasterRCNNResnet101KerasFeatureExtractor, self).__init__(
        is_training=is_training,
        resnet_v1_base_model=resnet_v1.resnet_v1_101,
        resnet_v1_base_model_name='resnet_v1_101',
        first_stage_features_stride=first_stage_features_stride,
        batch_norm_trainable=batch_norm_trainable,
        weight_decay=weight_decay)


class FasterRCNNResnet152KerasFeatureExtractor(
    FasterRCNNResnetKerasFeatureExtractor):
  """Faster R-CNN with Resnet152 feature extractor implementation."""

  def __init__(self,
               is_training,
               first_stage_features_stride=16,
               batch_norm_trainable=False,
               weight_decay=0.0):
    """Constructor.

    Args:
      is_training: See base class.
      first_stage_features_stride: See base class.
      batch_norm_trainable: See base class.
      weight_decay: See base class.
    """
    super(FasterRCNNResnet152KerasFeatureExtractor, self).__init__(
        is_training=is_training,
        resnet_v1_base_model=resnet_v1.resnet_v1_152,
        resnet_v1_base_model_name='resnet_v1_152',
        first_stage_features_stride=first_stage_features_stride,
        batch_norm_trainable=batch_norm_trainable,
        weight_decay=weight_decay)
