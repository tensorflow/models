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

"""Faster RCNN Keras-based Resnet V1 FPN Feature Extractor."""

import tensorflow.compat.v1 as tf

from object_detection.meta_architectures import faster_rcnn_meta_arch
from object_detection.models import feature_map_generators
from object_detection.models.keras_models import resnet_v1


_RESNET_MODEL_OUTPUT_LAYERS = {
    'resnet_v1_50': ['conv2_block3_out', 'conv3_block4_out',
                     'conv4_block6_out', 'conv5_block3_out'],
    'resnet_v1_101': ['conv2_block3_out', 'conv3_block4_out',
                      'conv4_block23_out', 'conv5_block3_out'],
    'resnet_v1_152': ['conv2_block3_out', 'conv3_block8_out',
                      'conv4_block36_out', 'conv5_block3_out'],
}


class FasterRCNNResnetV1FpnKerasFeatureExtractor(
    faster_rcnn_meta_arch.FasterRCNNKerasFeatureExtractor):
  """Faster RCNN Feature Extractor using Keras-based Resnet V1 FPN features."""

  def __init__(self,
               is_training,
               resnet_v1_base_model,
               resnet_v1_base_model_name,
               first_stage_features_stride,
               conv_hyperparams,
               batch_norm_trainable=False,
               weight_decay=0.0,
               fpn_min_level=2,
               fpn_max_level=6,
               additional_layer_depth=256,
               override_base_feature_extractor_hyperparams=False):
    """Constructor.

    Args:
      is_training: See base class.
      resnet_v1_base_model: base resnet v1 network to use. One of
        the resnet_v1.resnet_v1_{50,101,152} models.
      resnet_v1_base_model_name: model name under which to construct resnet v1.
      first_stage_features_stride: See base class.
      conv_hyperparams: a `hyperparams_builder.KerasLayerHyperparams` object
        containing convolution hyperparameters for the layers added on top of
        the base feature extractor.
      batch_norm_trainable: See base class.
      weight_decay: See base class.
      fpn_min_level: the highest resolution feature map to use in FPN. The valid
        values are {2, 3, 4, 5} which map to Resnet v1 layers.
      fpn_max_level: the smallest resolution feature map to construct or use in
        FPN. FPN constructions uses features maps starting from fpn_min_level
        upto the fpn_max_level. In the case that there are not enough feature
        maps in the backbone network, additional feature maps are created by
        applying stride 2 convolutions until we get the desired number of fpn
        levels.
      additional_layer_depth: additional feature map layer channel depth.
      override_base_feature_extractor_hyperparams: Whether to override
        hyperparameters of the base feature extractor with the one from
        `conv_hyperparams`.

    Raises:
      ValueError: If `first_stage_features_stride` is not 8 or 16.
    """
    if first_stage_features_stride != 8 and first_stage_features_stride != 16:
      raise ValueError('`first_stage_features_stride` must be 8 or 16.')

    super(FasterRCNNResnetV1FpnKerasFeatureExtractor, self).__init__(
        is_training=is_training,
        first_stage_features_stride=first_stage_features_stride,
        batch_norm_trainable=batch_norm_trainable,
        weight_decay=weight_decay)

    self._resnet_v1_base_model = resnet_v1_base_model
    self._resnet_v1_base_model_name = resnet_v1_base_model_name
    self._conv_hyperparams = conv_hyperparams
    self._fpn_min_level = fpn_min_level
    self._fpn_max_level = fpn_max_level
    self._additional_layer_depth = additional_layer_depth
    self._freeze_batchnorm = (not batch_norm_trainable)
    self._override_base_feature_extractor_hyperparams = \
                    override_base_feature_extractor_hyperparams
    self._resnet_block_names = ['block1', 'block2', 'block3', 'block4']
    self.classification_backbone = None
    self._fpn_features_generator = None
    self._coarse_feature_layers = []

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

    Extracts features using the Resnet v1 FPN network.

    Args:
      name: A scope name to construct all variables within.

    Returns:
      A Keras model that takes preprocessed_inputs:
        A [batch, height, width, channels] float32 tensor
        representing a batch of images.

      And returns rpn_feature_map:
        A list of tensors with shape [batch, height, width, depth]
    """
    with tf.name_scope(name):
      with tf.name_scope('ResnetV1FPN'):
        full_resnet_v1_model = self._resnet_v1_base_model(
            batchnorm_training=self._train_batch_norm,
            conv_hyperparams=(self._conv_hyperparams if
                              self._override_base_feature_extractor_hyperparams
                              else None),
            classes=None,
            weights=None,
            include_top=False)
        output_layers = _RESNET_MODEL_OUTPUT_LAYERS[
            self._resnet_v1_base_model_name]
        outputs = [full_resnet_v1_model.get_layer(output_layer_name).output
                   for output_layer_name in output_layers]
        self.classification_backbone = tf.keras.Model(
            inputs=full_resnet_v1_model.inputs,
            outputs=outputs)
        backbone_outputs = self.classification_backbone(
            full_resnet_v1_model.inputs)

        # construct FPN feature generator
        self._base_fpn_max_level = min(self._fpn_max_level, 5)
        self._num_levels = self._base_fpn_max_level + 1 - self._fpn_min_level
        self._fpn_features_generator = (
            feature_map_generators.KerasFpnTopDownFeatureMaps(
                num_levels=self._num_levels,
                depth=self._additional_layer_depth,
                is_training=self._is_training,
                conv_hyperparams=self._conv_hyperparams,
                freeze_batchnorm=self._freeze_batchnorm,
                name='FeatureMaps'))

        feature_block_list = []
        for level in range(self._fpn_min_level, self._base_fpn_max_level + 1):
          feature_block_list.append('block{}'.format(level - 1))
        feature_block_map = dict(
            list(zip(self._resnet_block_names, backbone_outputs)))
        fpn_input_image_features = [
            (feature_block, feature_block_map[feature_block])
            for feature_block in feature_block_list]
        fpn_features = self._fpn_features_generator(fpn_input_image_features)

        # Construct coarse feature layers
        for i in range(self._base_fpn_max_level, self._fpn_max_level):
          layers = []
          layer_name = 'bottom_up_block{}'.format(i)
          layers.append(
              tf.keras.layers.Conv2D(
                  self._additional_layer_depth,
                  [3, 3],
                  padding='SAME',
                  strides=2,
                  name=layer_name + '_conv',
                  **self._conv_hyperparams.params()))
          layers.append(
              self._conv_hyperparams.build_batch_norm(
                  training=(self._is_training and not self._freeze_batchnorm),
                  name=layer_name + '_batchnorm'))
          layers.append(
              self._conv_hyperparams.build_activation_layer(
                  name=layer_name))
          self._coarse_feature_layers.append(layers)

        feature_maps = []
        for level in range(self._fpn_min_level, self._base_fpn_max_level + 1):
          feature_maps.append(fpn_features['top_down_block{}'.format(level-1)])
        last_feature_map = fpn_features['top_down_block{}'.format(
            self._base_fpn_max_level - 1)]

        for coarse_feature_layers in self._coarse_feature_layers:
          for layer in coarse_feature_layers:
            last_feature_map = layer(last_feature_map)
          feature_maps.append(last_feature_map)

        feature_extractor_model = tf.keras.models.Model(
            inputs=full_resnet_v1_model.inputs, outputs=feature_maps)
        return feature_extractor_model

  def get_box_classifier_feature_extractor_model(self, name=None):
    """Returns a model that extracts second stage box classifier features.

    Construct two fully connected layer to extract the box classifier features.

    Args:
      name: A scope name to construct all variables within.

    Returns:
      A Keras model that takes proposal_feature_maps:
        A 4-D float tensor with shape
        [batch_size * self.max_num_proposals, crop_height, crop_width, depth]
        representing the feature map cropped to each proposal.

      And returns proposal_classifier_features:
        A 4-D float tensor with shape
        [batch_size * self.max_num_proposals, 1024]
        representing box classifier features for each proposal.
    """
    with tf.name_scope(name):
      with tf.name_scope('ResnetV1FPN'):
        # TODO(yiming): Add a batchnorm layer between two fc layers.
        feature_extractor_model = tf.keras.models.Sequential([
            tf.keras.layers.Flatten(),
            tf.keras.layers.Dense(units=1024, activation='relu'),
            tf.keras.layers.Dense(units=1024, activation='relu')
        ])
        return feature_extractor_model


class FasterRCNNResnet50FpnKerasFeatureExtractor(
    FasterRCNNResnetV1FpnKerasFeatureExtractor):
  """Faster RCNN with Resnet50 FPN feature extractor."""

  def __init__(self,
               is_training,
               first_stage_features_stride=16,
               conv_hyperparams=None,
               batch_norm_trainable=False,
               weight_decay=0.0,
               fpn_min_level=2,
               fpn_max_level=6,
               additional_layer_depth=256,
               override_base_feature_extractor_hyperparams=False):
    """Constructor.

    Args:
      is_training: See base class.
      first_stage_features_stride: See base class.
      conv_hyperparams: See base class.
      batch_norm_trainable: See base class.
      weight_decay: See base class.
      fpn_min_level: See base class.
      fpn_max_level: See base class.
      additional_layer_depth: See base class.
      override_base_feature_extractor_hyperparams: See base class.
    """
    super(FasterRCNNResnet50FpnKerasFeatureExtractor, self).__init__(
        is_training=is_training,
        first_stage_features_stride=first_stage_features_stride,
        conv_hyperparams=conv_hyperparams,
        resnet_v1_base_model=resnet_v1.resnet_v1_50,
        resnet_v1_base_model_name='resnet_v1_50',
        batch_norm_trainable=batch_norm_trainable,
        weight_decay=weight_decay,
        fpn_min_level=fpn_min_level,
        fpn_max_level=fpn_max_level,
        additional_layer_depth=additional_layer_depth,
        override_base_feature_extractor_hyperparams=
        override_base_feature_extractor_hyperparams
    )


class FasterRCNNResnet101FpnKerasFeatureExtractor(
    FasterRCNNResnetV1FpnKerasFeatureExtractor):
  """Faster RCNN with Resnet101 FPN feature extractor."""

  def __init__(self,
               is_training,
               first_stage_features_stride=16,
               conv_hyperparams=None,
               batch_norm_trainable=False,
               weight_decay=0.0,
               fpn_min_level=2,
               fpn_max_level=6,
               additional_layer_depth=256,
               override_base_feature_extractor_hyperparams=False):
    """Constructor.

    Args:
      is_training: See base class.
      first_stage_features_stride: See base class.
      conv_hyperparams: See base class.
      batch_norm_trainable: See base class.
      weight_decay: See base class.
      fpn_min_level: See base class.
      fpn_max_level: See base class.
      additional_layer_depth: See base class.
      override_base_feature_extractor_hyperparams: See base class.
    """
    super(FasterRCNNResnet101FpnKerasFeatureExtractor, self).__init__(
        is_training=is_training,
        first_stage_features_stride=first_stage_features_stride,
        conv_hyperparams=conv_hyperparams,
        resnet_v1_base_model=resnet_v1.resnet_v1_101,
        resnet_v1_base_model_name='resnet_v1_101',
        batch_norm_trainable=batch_norm_trainable,
        weight_decay=weight_decay,
        fpn_min_level=fpn_min_level,
        fpn_max_level=fpn_max_level,
        additional_layer_depth=additional_layer_depth,
        override_base_feature_extractor_hyperparams=
        override_base_feature_extractor_hyperparams)


class FasterRCNNResnet152FpnKerasFeatureExtractor(
    FasterRCNNResnetV1FpnKerasFeatureExtractor):
  """Faster RCNN with Resnet152 FPN feature extractor."""

  def __init__(self,
               is_training,
               first_stage_features_stride=16,
               conv_hyperparams=None,
               batch_norm_trainable=False,
               weight_decay=0.0,
               fpn_min_level=2,
               fpn_max_level=6,
               additional_layer_depth=256,
               override_base_feature_extractor_hyperparams=False):
    """Constructor.

    Args:
      is_training: See base class.
      first_stage_features_stride: See base class.
      conv_hyperparams: See base class.
      batch_norm_trainable: See base class.
      weight_decay: See base class.
      fpn_min_level: See base class.
      fpn_max_level: See base class.
      additional_layer_depth: See base class.
      override_base_feature_extractor_hyperparams: See base class.
    """
    super(FasterRCNNResnet152FpnKerasFeatureExtractor, self).__init__(
        is_training=is_training,
        first_stage_features_stride=first_stage_features_stride,
        conv_hyperparams=conv_hyperparams,
        resnet_v1_base_model=resnet_v1.resnet_v1_152,
        resnet_v1_base_model_name='resnet_v1_152',
        batch_norm_trainable=batch_norm_trainable,
        weight_decay=weight_decay,
        fpn_min_level=fpn_min_level,
        fpn_max_level=fpn_max_level,
        additional_layer_depth=additional_layer_depth,
        override_base_feature_extractor_hyperparams=
        override_base_feature_extractor_hyperparams)
