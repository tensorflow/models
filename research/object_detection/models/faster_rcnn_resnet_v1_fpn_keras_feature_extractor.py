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
from object_detection.models.keras_models import model_utils
from object_detection.utils import ops
from object_detection.utils import shape_utils

_RESNET_MODEL_OUTPUT_LAYERS = {
    'resnet_v1_50': ['conv2_block3_out', 'conv3_block4_out',
                     'conv4_block6_out', 'conv5_block3_out'],
    'resnet_v1_101': ['conv2_block3_out', 'conv3_block4_out',
                      'conv4_block23_out', 'conv5_block3_out'],
    'resnet_v1_152': ['conv2_block3_out', 'conv3_block8_out',
                      'conv4_block36_out', 'conv5_block3_out'],
}


class FasterRCNNResnetV1FPNKerasFeatureExtractor(
    faster_rcnn_meta_arch.FasterRCNNFeatureExtractor):
  """Faster RCNN Feature Extractor using Keras-based Resnet V1 FPN features."""

  def __init__(self,
               is_training,
               resnet_v1_base_model,
               resnet_v1_base_model_name,
               first_stage_features_stride,
               conv_hyperparams,
               min_depth,
               depth_multiplier,
               batch_norm_trainable=False,
               weight_decay=0.0,
               fpn_min_level=3,
               fpn_max_level=7,
               additional_layer_depth=256,
               override_base_feature_extractor_hyperparams=False):
    # FIXME: fix doc string for fpn min level and fpn max level
    """Constructor.

    Args:
      is_training: See base class.
      first_stage_features_stride: See base class.

      conv_hyperparameters: a `hyperparams_builder.KerasLayerHyperparams` object
        containing convolution hyperparameters for the layers added on top of
        the base feature extractor.
      min_depth: Minimum number of filters in the convolutional layers.
      depth_multiplier: The depth multiplier to modify the number of filters
        in the convolutional layers.
      resnet_v1_base_model: base resnet v1 network to use. One of
        the resnet_v1.resnet_v1_{50,101,152} models.
      resnet_v1_base_model_name: model name under which to construct resnet v1.

      batch_norm_trainable: See base class.
      weight_decay: See base class.

      fpn_min_level: the highest resolution feature map to use in FPN. The valid
        values are {2, 3, 4, 5} which map to MobileNet v1 layers
        {Conv2d_3_pointwise, Conv2d_5_pointwise, Conv2d_11_pointwise,
        Conv2d_13_pointwise}, respectively.
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
    super(FasterRCNNResnetV1FPNKerasFeatureExtractor, self).__init__(
          is_training=is_training,
          first_stage_features_stride=first_stage_features_stride,
          batch_norm_trainable=batch_norm_trainable,
          weight_decay=weight_decay)
    self._conv_hyperparams = conv_hyperparams
    self._min_depth = min_depth
    self._depth_multiplier = depth_multiplier
    self._additional_layer_depth = additional_layer_depth
    self._freeze_batchnorm = (not batch_norm_trainable)
    self._override_base_feature_extractor_hyperparams = 
                    override_base_feature_extractor_hyperparams
    self._fpn_min_level = fpn_min_level
    self._fpn_max_level = fpn_max_level
    self._resnet_v1_base_model = resnet_v1_base_model
    self._resnet_v1_base_model_name = resnet_v1_base_model_name
    self._resnet_block_names = ['block1', 'block2', 'block3', 'block4']
    self.classification_backbone = None
    self._fpn_features_generator = None

  def build(self,):
    # TODO: Refine doc string
    """Build Resnet V1 FPN architecture."""
    # full_resnet_v1_model = self._resnet_v1_base_model(
    #     batchnorm_training=self._train_batch_norm,
    #     conv_hyperparams=(self._conv_hyperparams
    #                       if self._override_base_feature_extractor_hyperparams
    #                       else None),
    #     min_depth=self._min_depth,
    #     depth_multiplier=self._depth_multiplier,
    #     classes=None,
    #     weights=None,
    #     include_top=False)
    # output_layers = _RESNET_MODEL_OUTPUT_LAYERS[self._resnet_v1_base_model_name]
    # outputs = [full_resnet_v1_model.get_layer(output_layer_name).output
    #            for output_layer_name in output_layers]
    # self.classification_backbone = tf.keras.Model(
    #     inputs=full_resnet_v1_model.inputs,
    #     outputs=outputs)
    # self._depth_fn = lambda d: max(
    #     int(d * self._depth_multiplier), self._min_depth)
    # self._base_fpn_max_level = min(self._fpn_max_level, 5)
    # self._num_levels = self._base_fpn_max_level + 1 - self._fpn_min_level
    # self._fpn_features_generator = (
    #     feature_map_generators.KerasFpnTopDownFeatureMaps(
    #         num_levels=self._num_levels,
    #         depth=self._depth_fn(self._additional_layer_depth),
    #         is_training=self._is_training,
    #         conv_hyperparams=self._conv_hyperparams,
    #         freeze_batchnorm=self._freeze_batchnorm,
    #         name='FeatureMaps'))
    # Construct coarse feature layers
    depth = self._depth_fn(self._additional_layer_depth)
    for i in range(self._base_fpn_max_level, self._fpn_max_level):
      layers = []
      layer_name = 'bottom_up_block{}'.format(i)
      layers.append(
          tf.keras.layers.Conv2D(
              depth,
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
    self.built = True

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
  
  # def _extract_proposal_features(self, preprocessed_inputs, scope=None):
  #   # TODO: doc string
  #   """"""
  #   preprocessed_inputs = shape_utils.check_min_image_dim(
  #       129, preprocessed_inputs)

  #   with tf.name_scope(scope):
  #     with tf.name_scope('ResnetV1FPN'):
  #       image_features = self.classification_backbone(preprocessed_inputs)

  #       feature_block_list = []
  #       for level in range(self._fpn_min_level, self._base_fpn_max_level + 1):
  #         feature_block_list.append('block{}'.format(level - 1))
  #       feature_block_map = dict(
  #           list(zip(self._resnet_block_names, image_features)))
  #       fpn_input_image_features = [
  #           (feature_block, feature_block_map[feature_block])
  #           for feature_block in feature_block_list]
  #       fpn_features = self._fpn_features_generator(fpn_input_image_features)

  #       return fpn_features
        
  def get_proposal_feature_extractor_model(self, name=None):
    """Returns a model that extracts first stage RPN features.

    Extracts features using the first half of the Resnet v1 network.

    Args:
      name: A scope name to construct all variables within.

    Returns:
      A Keras model that takes preprocessed_inputs:
        A [batch, height, width, channels] float32 tensor
        representing a batch of images.
    """
    with tf.name_scope(name):
      with tf.name_scope('ResnetV1FPN'):
        full_resnet_v1_model = self._resnet_v1_base_model(
            batchnorm_training=self._train_batch_norm,
            conv_hyperparams=(self._conv_hyperparams
                              if self._override_base_feature_extractor_hyperparams
                              else None),
            min_depth=self._min_depth,
            depth_multiplier=self._depth_multiplier,
            classes=None,
            weights=None,
            include_top=False)
        output_layers = _RESNET_MODEL_OUTPUT_LAYERS[self._resnet_v1_base_model_name]
        outputs = [full_resnet_v1_model.get_layer(output_layer_name).output
                  for output_layer_name in output_layers]
        self.classification_backbone = tf.keras.Model(
            inputs=full_resnet_v1_model.inputs,
            outputs=outputs)
        backbone_outputs = self.classification_backbone(full_resnet_v1_model.inputs)

        # construct FPN feature generator
        self._depth_fn = lambda d: max(
            int(d * self._depth_multiplier), self._min_depth)
        self._base_fpn_max_level = min(self._fpn_max_level, 5)
        self._num_levels = self._base_fpn_max_level + 1 - self._fpn_min_level
        self._fpn_features_generator = (
            feature_map_generators.KerasFpnTopDownFeatureMaps(
                num_levels=self._num_levels,
                depth=self._depth_fn(self._additional_layer_depth),
                is_training=self._is_training,
                conv_hyperparams=self._conv_hyperparams,
                freeze_batchnorm=self._freeze_batchnorm,
                name='FeatureMaps'))
        
        feature_block_list = []
        for level in range(self._fpn_min_level, self._base_fpn_max_level + 1):
          feature_block_list.append('block{}'.format(level - 1))
        feature_block_map = dict(
            list(zip(self._resnet_block_names, image_features)))
        fpn_input_image_features = [
            (feature_block, feature_block_map[feature_block])
            for feature_block in feature_block_list]
        fpn_features = self._fpn_features_generator(fpn_input_image_features)

        feature_extractor_model = tf.keras.models.Model(
            inputs=self.full_resnet_v1_model.inputs, outputs=fpn_features)
        return feature_extractor_model

  # def _extract_box_classifier_features(self, proposal_feature_maps, scope=None):
  #   with tf.name_scope(scope):
  #     with tf.name_scope('ResnetV1FPN'):
  #       feature_maps = []
  #       for level in range(self._fpn_min_level, self._base_fpn_max_level + 1):
  #         feature_maps.append(proposal_feature_maps['top_down_block{}'.format(level-1)])
  #       self.last_feature_map = proposal_feature_maps['top_down_block{}'.format(
  #           self._base_fpn_max_level - 1)]

  #       for coarse_feature_layers in self._coarse_feature_layers:
  #         for layer in coarse_feature_layers:
  #           last_feature_map = layer(last_feature_map)
  #         feature_maps.append(self.last_feature_map)

  #       return feature_maps

  def get_box_classifier_feature_extractor_model(self, name=None):



class FasterRCNNResnet50FPNKerasFeatureExtractor(
    FasterRCNNResnetV1FPNKerasFeatureExtractor):
  """Faster RCNN with Resnet50 FPN feature extractor implementation."""
  
  def __init__(self,
               is_training,
               first_stage_features_stride=16,
               conv_hyperparams=None,
               min_depth=16,
               depth_multiplier=1,
               batch_norm_trainable=False,
               weight_decay=0.0,
               fpn_min_level=3,
               fpn_max_level=7,
               additional_layer_depth=256,
               override_base_feature_extractor_hyperparams=False):
    """Constructor.

    Args:
      is_training: See base class.
      first_stage_features_stride: See base class.
      conv_hyperparams: See base class.
      min_depth: See base class.
      depth_multiplier: See base class.
      batch_norm_trainable: See base class.
      weight_decay: See base class.
      fpn_min_level: See base class.
      fpn_max_level: See base class.
      additional_layer_depth: See base class.
      override_base_feature_extractor_hyperparams: See base class.
    """
    super(FasterRCNNResnet50KerasFeatureExtractor, self).__init__(
        is_training=is_training,
        first_stage_features_stride=first_stage_features_stride,
        conv_hyperparams=conv_hyperparameters,
        min_depth=min_depth,
        depth_multiplier=depth_multiplier,
        resnet_v1_base_model=resnet_v1.resnet_v1_50,
        resnet_v1_base_model_name='resnet_v1_50',
        batch_norm_trainable=batch_norm_trainable,
        weight_decay=weight_decay,
        fpn_min_level=fpn_min_level,
        fpn_max_level=fpn_max_level,
        additional_layer_depth=additional_layer_depth,
        override_base_feature_extractor_hyperparams=override_base_feature_extractor_hyperparams)

class FasterRCNNResnet101FPNKerasFeatureExtractor(
    FasterRCNNResnetV1FPNKerasFeatureExtractor):
  """Faster RCNN with Resnet101 FPN feature extractor implementation."""
  def __init__(self,
               is_training,
               first_stage_features_stride=16,
               conv_hyperparams=None,
               min_depth=16,
               depth_multiplier=1,
               batch_norm_trainable=False,
               weight_decay=0.0,
               fpn_min_level=3,
               fpn_max_level=7,
               additional_layer_depth=256,
               override_base_feature_extractor_hyperparams=False):
    """Constructor.

    Args:
      is_training: See base class.
      first_stage_features_stride: See base class.
      conv_hyperparams: See base class.
      min_depth: See base class.
      depth_multiplier: See base class.
      batch_norm_trainable: See base class.
      weight_decay: See base class.
      fpn_min_level: See base class.
      fpn_max_level: See base class.
      additional_layer_depth: See base class.
      override_base_feature_extractor_hyperparams: See base class.
    """
    super(FasterRCNNResnet50KerasFeatureExtractor, self).__init__(
        is_training=is_training,
        first_stage_features_stride=first_stage_features_stride,
        conv_hyperparams=conv_hyperparameters,
        min_depth=min_depth,
        depth_multiplier=depth_multiplier,
        resnet_v1_base_model=resnet_v1.resnet_v1_101,
        resnet_v1_base_model_name='resnet_v1_101',
        batch_norm_trainable=batch_norm_trainable,
        weight_decay=weight_decay,
        fpn_min_level=fpn_min_level,
        fpn_max_level=fpn_max_level,
        additional_layer_depth=additional_layer_depth,
        override_base_feature_extractor_hyperparams=override_base_feature_extractor_hyperparams)


class FasterRCNNResnet152FPNKerasFeatureExtractor(
    FasterRCNNResnetV1FPNKerasFeatureExtractor):
  """Faster RCNN with Resnet152 FPN feature extractor implementation."""

  def __init__(self,
               is_training,
               first_stage_features_stride=16,
               conv_hyperparams=None,
               min_depth=16,
               depth_multiplier=1,
               batch_norm_trainable=False,
               weight_decay=0.0,
               fpn_min_level=3,
               fpn_max_level=7,
               additional_layer_depth=256,
               override_base_feature_extractor_hyperparams=False):
    """Constructor.

    Args:
      is_training: See base class.
      first_stage_features_stride: See base class.
      conv_hyperparams: See base class.
      min_depth: See base class.
      depth_multiplier: See base class.
      batch_norm_trainable: See base class.
      weight_decay: See base class.
      fpn_min_level: See base class.
      fpn_max_level: See base class.
      additional_layer_depth: See base class.
      override_base_feature_extractor_hyperparams: See base class.
    """
    super(FasterRCNNResnet50KerasFeatureExtractor, self).__init__(
        is_training=is_training,
        first_stage_features_stride=first_stage_features_stride,
        conv_hyperparams=conv_hyperparameters,
        min_depth=min_depth,
        depth_multiplier=depth_multiplier,
        resnet_v1_base_model=resnet_v1.resnet_v1_152,
        resnet_v1_base_model_name='resnet_v1_152',
        batch_norm_trainable=batch_norm_trainable,
        weight_decay=weight_decay,
        fpn_min_level=fpn_min_level,
        fpn_max_level=fpn_max_level,
        additional_layer_depth=additional_layer_depth,
        override_base_feature_extractor_hyperparams=override_base_feature_extractor_hyperparams)