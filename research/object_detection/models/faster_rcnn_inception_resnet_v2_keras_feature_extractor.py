# Copyright 2019 The TensorFlow Authors. All Rights Reserved.
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

"""Inception Resnet v2 Faster R-CNN implementation in Keras.

See "Inception-v4, Inception-ResNet and the Impact of Residual Connections on
Learning" by Szegedy et al. (https://arxiv.org/abs/1602.07261)
as well as
"Speed/accuracy trade-offs for modern convolutional object detectors" by
Huang et al. (https://arxiv.org/abs/1611.10012)
"""

# Skip pylint for this file because it times out
# pylint: skip-file

import tensorflow.compat.v1 as tf

from object_detection.meta_architectures import faster_rcnn_meta_arch
from object_detection.models.keras_models import inception_resnet_v2
from object_detection.utils import model_util
from object_detection.utils import variables_helper


class FasterRCNNInceptionResnetV2KerasFeatureExtractor(
    faster_rcnn_meta_arch.FasterRCNNKerasFeatureExtractor):
  """Faster R-CNN with Inception Resnet v2 feature extractor implementation."""

  def __init__(self,
               is_training,
               first_stage_features_stride,
               batch_norm_trainable=False,
               weight_decay=0.0):
    """Constructor.

    Args:
      is_training: See base class.
      first_stage_features_stride: See base class.
      batch_norm_trainable: See base class.
      weight_decay: See base class.

    Raises:
      ValueError: If `first_stage_features_stride` is not 8 or 16.
    """
    if first_stage_features_stride != 8 and first_stage_features_stride != 16:
      raise ValueError('`first_stage_features_stride` must be 8 or 16.')
    super(FasterRCNNInceptionResnetV2KerasFeatureExtractor, self).__init__(
        is_training, first_stage_features_stride, batch_norm_trainable,
        weight_decay)
    self._variable_dict = {}
    self.classification_backbone = None

  def preprocess(self, resized_inputs):
    """Faster R-CNN with Inception Resnet v2 preprocessing.

    Maps pixel values to the range [-1, 1].

    Args:
      resized_inputs: A [batch, height_in, width_in, channels] float32 tensor
        representing a batch of images with values between 0 and 255.0.

    Returns:
      preprocessed_inputs: A [batch, height_out, width_out, channels] float32
        tensor representing a batch of images.

    """
    return (2.0 / 255.0) * resized_inputs - 1.0

  def get_proposal_feature_extractor_model(self, name=None):
    """Returns a model that extracts first stage RPN features.

    Extracts features using the first half of the Inception Resnet v2 network.
    We construct the network in `align_feature_maps=True` mode, which means
    that all VALID paddings in the network are changed to SAME padding so that
    the feature maps are aligned.

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
      self.classification_backbone = inception_resnet_v2.inception_resnet_v2(
              self._train_batch_norm,
              output_stride=self._first_stage_features_stride,
              align_feature_maps=True,
              weight_decay=self._weight_decay,
              weights=None,
              include_top=False)
    with tf.name_scope(name):
      with tf.name_scope('InceptionResnetV2'):
        proposal_features = self.classification_backbone.get_layer(
            name='block17_20_ac').output
        keras_model = tf.keras.Model(
            inputs=self.classification_backbone.inputs,
            outputs=proposal_features)
        for variable in keras_model.variables:
          self._variable_dict[variable.name[:-2]] = variable
        return keras_model

  def get_box_classifier_feature_extractor_model(self, name=None):
    """Returns a model that extracts second stage box classifier features.

    This function reconstructs the "second half" of the Inception ResNet v2
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
      self.classification_backbone = inception_resnet_v2.inception_resnet_v2(
              self._train_batch_norm,
              output_stride=self._first_stage_features_stride,
              align_feature_maps=True,
              weight_decay=self._weight_decay,
              weights=None,
              include_top=False)
    with tf.name_scope(name):
      with tf.name_scope('InceptionResnetV2'):
        proposal_feature_maps = self.classification_backbone.get_layer(
            name='block17_20_ac').output
        proposal_classifier_features = self.classification_backbone.get_layer(
            name='conv_7b_ac').output

        keras_model = model_util.extract_submodel(
            model=self.classification_backbone,
            inputs=proposal_feature_maps,
            outputs=proposal_classifier_features)
        for variable in keras_model.variables:
          self._variable_dict[variable.name[:-2]] = variable
        return keras_model

