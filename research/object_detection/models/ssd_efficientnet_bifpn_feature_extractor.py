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
"""SSD Keras-based EfficientNet + BiFPN (EfficientDet) Feature Extractor."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from absl import logging
from six.moves import range
from six.moves import zip
import tensorflow.compat.v2 as tf

from tensorflow.python.keras import backend as keras_backend
from object_detection.meta_architectures import ssd_meta_arch
from object_detection.models import bidirectional_feature_pyramid_generators as bifpn_generators
from object_detection.utils import ops
from object_detection.utils import shape_utils
from object_detection.utils import tf_version
# pylint: disable=g-import-not-at-top
if tf_version.is_tf2():
  from official.vision.image_classification.efficientnet import efficientnet_model

_EFFICIENTNET_LEVEL_ENDPOINTS = {
    1: 'stack_0/block_0/project_bn',
    2: 'stack_1/block_1/add',
    3: 'stack_2/block_1/add',
    4: 'stack_4/block_2/add',
    5: 'stack_6/block_0/project_bn',
}


class SSDEfficientNetBiFPNKerasFeatureExtractor(
    ssd_meta_arch.SSDKerasFeatureExtractor):
  """SSD Keras-based EfficientNetBiFPN (EfficientDet) Feature Extractor."""

  def __init__(self,
               is_training,
               depth_multiplier,
               min_depth,
               pad_to_multiple,
               conv_hyperparams,
               freeze_batchnorm,
               inplace_batchnorm_update,
               bifpn_min_level,
               bifpn_max_level,
               bifpn_num_iterations,
               bifpn_num_filters,
               bifpn_combine_method,
               efficientnet_version,
               use_explicit_padding=None,
               use_depthwise=None,
               override_base_feature_extractor_hyperparams=None,
               name=None):
    """SSD Keras-based EfficientNetBiFPN (EfficientDet) feature extractor.

    Args:
      is_training: whether the network is in training mode.
      depth_multiplier: unsupported by EfficientNetBiFPN. float, depth
        multiplier for the feature extractor.
      min_depth: minimum feature extractor depth.
      pad_to_multiple: the nearest multiple to zero pad the input height and
        width dimensions to.
      conv_hyperparams: a `hyperparams_builder.KerasLayerHyperparams` object
        containing convolution hyperparameters for the layers added on top of
        the base feature extractor.
      freeze_batchnorm: whether to freeze batch norm parameters during training
        or not. When training with a small batch size (e.g. 1), it is desirable
        to freeze batch norm update and use pretrained batch norm params.
      inplace_batchnorm_update: whether to update batch norm moving average
        values inplace. When this is false train op must add a control
        dependency on tf.graphkeys.UPDATE_OPS collection in order to update
        batch norm statistics.
      bifpn_min_level: the highest resolution feature map to use in BiFPN. The
        valid values are {2, 3, 4, 5} which map to Resnet blocks {1, 2, 3, 4}
        respectively.
      bifpn_max_level: the smallest resolution feature map to use in the BiFPN.
        BiFPN constructions uses features maps starting from bifpn_min_level
        upto the bifpn_max_level. In the case that there are not enough feature
        maps in the backbone network, additional feature maps are created by
        applying stride 2 convolutions until we get the desired number of BiFPN
        levels.
      bifpn_num_iterations: number of BiFPN iterations. Overrided if
        efficientdet_version is provided.
      bifpn_num_filters: number of filters (channels) in all BiFPN layers.
        Overrided if efficientdet_version is provided.
      bifpn_combine_method: the method used to combine BiFPN nodes.
      efficientnet_version: the EfficientNet version to use for this feature
        extractor's backbone.
      use_explicit_padding: unsupported by EfficientNetBiFPN. Whether to use
        explicit padding when extracting features.
      use_depthwise: unsupported by EfficientNetBiFPN, since BiFPN uses regular
        convolutions when inputs to a node have a differing number of channels,
        and use separable convolutions after combine operations.
      override_base_feature_extractor_hyperparams: Whether to override the
        efficientnet backbone's default weight decay with the weight decay
        defined by `conv_hyperparams`. Note, only overriding of weight decay is
        currently supported.
      name: a string name scope to assign to the model. If 'None', Keras will
        auto-generate one from the class name.
    """
    super(SSDEfficientNetBiFPNKerasFeatureExtractor, self).__init__(
        is_training=is_training,
        depth_multiplier=depth_multiplier,
        min_depth=min_depth,
        pad_to_multiple=pad_to_multiple,
        conv_hyperparams=conv_hyperparams,
        freeze_batchnorm=freeze_batchnorm,
        inplace_batchnorm_update=inplace_batchnorm_update,
        use_explicit_padding=None,
        use_depthwise=None,
        override_base_feature_extractor_hyperparams=
        override_base_feature_extractor_hyperparams,
        name=name)
    if depth_multiplier != 1.0:
      raise ValueError('EfficientNetBiFPN does not support a non-default '
                       'depth_multiplier.')
    if use_explicit_padding:
      raise ValueError('EfficientNetBiFPN does not support explicit padding.')
    if use_depthwise:
      raise ValueError('EfficientNetBiFPN does not support use_depthwise.')

    self._bifpn_min_level = bifpn_min_level
    self._bifpn_max_level = bifpn_max_level
    self._bifpn_num_iterations = bifpn_num_iterations
    self._bifpn_num_filters = max(bifpn_num_filters, min_depth)
    self._bifpn_node_params = {'combine_method': bifpn_combine_method}
    self._efficientnet_version = efficientnet_version

    logging.info('EfficientDet EfficientNet backbone version: %s',
                 self._efficientnet_version)
    logging.info('EfficientDet BiFPN num filters: %d', self._bifpn_num_filters)
    logging.info('EfficientDet BiFPN num iterations: %d',
                 self._bifpn_num_iterations)

    self._backbone_max_level = min(
        max(_EFFICIENTNET_LEVEL_ENDPOINTS.keys()), bifpn_max_level)
    self._output_layer_names = [
        _EFFICIENTNET_LEVEL_ENDPOINTS[i]
        for i in range(bifpn_min_level, self._backbone_max_level + 1)]
    self._output_layer_alias = [
        'level_{}'.format(i)
        for i in range(bifpn_min_level, self._backbone_max_level + 1)]

    # Initialize the EfficientNet backbone.
    # Note, this is currently done in the init method rather than in the build
    # method, since doing so introduces an error which is not well understood.
    efficientnet_overrides = {'rescale_input': False}
    if override_base_feature_extractor_hyperparams:
      efficientnet_overrides[
          'weight_decay'] = conv_hyperparams.get_regularizer_weight()
    if (conv_hyperparams.use_sync_batch_norm() and
        keras_backend.is_tpu_strategy(tf.distribute.get_strategy())):
      efficientnet_overrides['batch_norm'] = 'tpu'
    efficientnet_base = efficientnet_model.EfficientNet.from_name(
        model_name=self._efficientnet_version, overrides=efficientnet_overrides)
    outputs = [efficientnet_base.get_layer(output_layer_name).output
               for output_layer_name in self._output_layer_names]
    self._efficientnet = tf.keras.Model(
        inputs=efficientnet_base.inputs, outputs=outputs)
    self.classification_backbone = efficientnet_base
    self._bifpn_stage = None

  def build(self, input_shape):
    self._bifpn_stage = bifpn_generators.KerasBiFpnFeatureMaps(
        bifpn_num_iterations=self._bifpn_num_iterations,
        bifpn_num_filters=self._bifpn_num_filters,
        fpn_min_level=self._bifpn_min_level,
        fpn_max_level=self._bifpn_max_level,
        input_max_level=self._backbone_max_level,
        is_training=self._is_training,
        conv_hyperparams=self._conv_hyperparams,
        freeze_batchnorm=self._freeze_batchnorm,
        bifpn_node_params=self._bifpn_node_params,
        name='bifpn')
    self.built = True

  def preprocess(self, inputs):
    """SSD preprocessing.

    Channel-wise mean subtraction and scaling.

    Args:
      inputs: a [batch, height, width, channels] float tensor representing a
        batch of images.

    Returns:
      preprocessed_inputs: a [batch, height, width, channels] float tensor
        representing a batch of images.
    """
    if inputs.shape.as_list()[3] == 3:
      # Input images are expected to be in the range [0, 255].
      channel_offset = [0.485, 0.456, 0.406]
      channel_scale = [0.229, 0.224, 0.225]
      return ((inputs / 255.0) - [[channel_offset]]) / [[channel_scale]]
    else:
      return inputs

  def _extract_features(self, preprocessed_inputs):
    """Extract features from preprocessed inputs.

    Args:
      preprocessed_inputs: a [batch, height, width, channels] float tensor
        representing a batch of images.

    Returns:
      feature_maps: a list of tensors where the ith tensor has shape
        [batch, height_i, width_i, depth_i]
    """
    preprocessed_inputs = shape_utils.check_min_image_dim(
        129, preprocessed_inputs)

    base_feature_maps = self._efficientnet(
        ops.pad_to_multiple(preprocessed_inputs, self._pad_to_multiple))

    output_feature_map_dict = self._bifpn_stage(
        list(zip(self._output_layer_alias, base_feature_maps)))

    return list(output_feature_map_dict.values())


class SSDEfficientNetB0BiFPNKerasFeatureExtractor(
    SSDEfficientNetBiFPNKerasFeatureExtractor):
  """SSD Keras EfficientNet-b0 BiFPN (EfficientDet-d0) Feature Extractor."""

  def __init__(self,
               is_training,
               depth_multiplier,
               min_depth,
               pad_to_multiple,
               conv_hyperparams,
               freeze_batchnorm,
               inplace_batchnorm_update,
               bifpn_min_level=3,
               bifpn_max_level=7,
               bifpn_num_iterations=3,
               bifpn_num_filters=64,
               bifpn_combine_method='fast_attention',
               use_explicit_padding=None,
               use_depthwise=None,
               override_base_feature_extractor_hyperparams=None,
               name='EfficientDet-D0'):
    """SSD Keras EfficientNet-b0 BiFPN (EfficientDet-d0) Feature Extractor.

    Args:
      is_training: whether the network is in training mode.
      depth_multiplier: unsupported by EfficientNetBiFPN. float, depth
        multiplier for the feature extractor.
      min_depth: minimum feature extractor depth.
      pad_to_multiple: the nearest multiple to zero pad the input height and
        width dimensions to.
      conv_hyperparams: a `hyperparams_builder.KerasLayerHyperparams` object
        containing convolution hyperparameters for the layers added on top of
        the base feature extractor.
      freeze_batchnorm: whether to freeze batch norm parameters during training
        or not. When training with a small batch size (e.g. 1), it is desirable
        to freeze batch norm update and use pretrained batch norm params.
      inplace_batchnorm_update: whether to update batch norm moving average
        values inplace. When this is false train op must add a control
        dependency on tf.graphkeys.UPDATE_OPS collection in order to update
        batch norm statistics.
      bifpn_min_level: the highest resolution feature map to use in BiFPN. The
        valid values are {2, 3, 4, 5} which map to Resnet blocks {1, 2, 3, 4}
        respectively.
      bifpn_max_level: the smallest resolution feature map to use in the BiFPN.
        BiFPN constructions uses features maps starting from bifpn_min_level
        upto the bifpn_max_level. In the case that there are not enough feature
        maps in the backbone network, additional feature maps are created by
        applying stride 2 convolutions until we get the desired number of BiFPN
        levels.
      bifpn_num_iterations: number of BiFPN iterations. Overrided if
        efficientdet_version is provided.
      bifpn_num_filters: number of filters (channels) in all BiFPN layers.
        Overrided if efficientdet_version is provided.
      bifpn_combine_method: the method used to combine BiFPN nodes.
      use_explicit_padding: unsupported by EfficientNetBiFPN. Whether to use
        explicit padding when extracting features.
      use_depthwise: unsupported by EfficientNetBiFPN, since BiFPN uses regular
        convolutions when inputs to a node have a differing number of channels,
        and use separable convolutions after combine operations.
      override_base_feature_extractor_hyperparams: unsupported. Whether to
        override hyperparameters of the base feature extractor with the one from
        `conv_hyperparams`.
      name: a string name scope to assign to the model. If 'None', Keras will
        auto-generate one from the class name.
    """
    super(SSDEfficientNetB0BiFPNKerasFeatureExtractor, self).__init__(
        is_training=is_training,
        depth_multiplier=depth_multiplier,
        min_depth=min_depth,
        pad_to_multiple=pad_to_multiple,
        conv_hyperparams=conv_hyperparams,
        freeze_batchnorm=freeze_batchnorm,
        inplace_batchnorm_update=inplace_batchnorm_update,
        bifpn_min_level=bifpn_min_level,
        bifpn_max_level=bifpn_max_level,
        bifpn_num_iterations=bifpn_num_iterations,
        bifpn_num_filters=bifpn_num_filters,
        bifpn_combine_method=bifpn_combine_method,
        efficientnet_version='efficientnet-b0',
        use_explicit_padding=use_explicit_padding,
        use_depthwise=use_depthwise,
        override_base_feature_extractor_hyperparams=
        override_base_feature_extractor_hyperparams,
        name=name)


class SSDEfficientNetB1BiFPNKerasFeatureExtractor(
    SSDEfficientNetBiFPNKerasFeatureExtractor):
  """SSD Keras EfficientNet-b1 BiFPN (EfficientDet-d1) Feature Extractor."""

  def __init__(self,
               is_training,
               depth_multiplier,
               min_depth,
               pad_to_multiple,
               conv_hyperparams,
               freeze_batchnorm,
               inplace_batchnorm_update,
               bifpn_min_level=3,
               bifpn_max_level=7,
               bifpn_num_iterations=4,
               bifpn_num_filters=88,
               bifpn_combine_method='fast_attention',
               use_explicit_padding=None,
               use_depthwise=None,
               override_base_feature_extractor_hyperparams=None,
               name='EfficientDet-D1'):
    """SSD Keras EfficientNet-b1 BiFPN (EfficientDet-d1) Feature Extractor.

    Args:
      is_training: whether the network is in training mode.
      depth_multiplier: unsupported by EfficientNetBiFPN. float, depth
        multiplier for the feature extractor.
      min_depth: minimum feature extractor depth.
      pad_to_multiple: the nearest multiple to zero pad the input height and
        width dimensions to.
      conv_hyperparams: a `hyperparams_builder.KerasLayerHyperparams` object
        containing convolution hyperparameters for the layers added on top of
        the base feature extractor.
      freeze_batchnorm: whether to freeze batch norm parameters during training
        or not. When training with a small batch size (e.g. 1), it is desirable
        to freeze batch norm update and use pretrained batch norm params.
      inplace_batchnorm_update: whether to update batch norm moving average
        values inplace. When this is false train op must add a control
        dependency on tf.graphkeys.UPDATE_OPS collection in order to update
        batch norm statistics.
      bifpn_min_level: the highest resolution feature map to use in BiFPN. The
        valid values are {2, 3, 4, 5} which map to Resnet blocks {1, 2, 3, 4}
        respectively.
      bifpn_max_level: the smallest resolution feature map to use in the BiFPN.
        BiFPN constructions uses features maps starting from bifpn_min_level
        upto the bifpn_max_level. In the case that there are not enough feature
        maps in the backbone network, additional feature maps are created by
        applying stride 2 convolutions until we get the desired number of BiFPN
        levels.
      bifpn_num_iterations: number of BiFPN iterations. Overrided if
        efficientdet_version is provided.
      bifpn_num_filters: number of filters (channels) in all BiFPN layers.
        Overrided if efficientdet_version is provided.
      bifpn_combine_method: the method used to combine BiFPN nodes.
      use_explicit_padding: unsupported by EfficientNetBiFPN. Whether to use
        explicit padding when extracting features.
      use_depthwise: unsupported by EfficientNetBiFPN, since BiFPN uses regular
        convolutions when inputs to a node have a differing number of channels,
        and use separable convolutions after combine operations.
      override_base_feature_extractor_hyperparams: unsupported. Whether to
        override hyperparameters of the base feature extractor with the one from
        `conv_hyperparams`.
      name: a string name scope to assign to the model. If 'None', Keras will
        auto-generate one from the class name.
    """
    super(SSDEfficientNetB1BiFPNKerasFeatureExtractor, self).__init__(
        is_training=is_training,
        depth_multiplier=depth_multiplier,
        min_depth=min_depth,
        pad_to_multiple=pad_to_multiple,
        conv_hyperparams=conv_hyperparams,
        freeze_batchnorm=freeze_batchnorm,
        inplace_batchnorm_update=inplace_batchnorm_update,
        bifpn_min_level=bifpn_min_level,
        bifpn_max_level=bifpn_max_level,
        bifpn_num_iterations=bifpn_num_iterations,
        bifpn_num_filters=bifpn_num_filters,
        bifpn_combine_method=bifpn_combine_method,
        efficientnet_version='efficientnet-b1',
        use_explicit_padding=use_explicit_padding,
        use_depthwise=use_depthwise,
        override_base_feature_extractor_hyperparams=
        override_base_feature_extractor_hyperparams,
        name=name)


class SSDEfficientNetB2BiFPNKerasFeatureExtractor(
    SSDEfficientNetBiFPNKerasFeatureExtractor):
  """SSD Keras EfficientNet-b2 BiFPN (EfficientDet-d2) Feature Extractor."""

  def __init__(self,
               is_training,
               depth_multiplier,
               min_depth,
               pad_to_multiple,
               conv_hyperparams,
               freeze_batchnorm,
               inplace_batchnorm_update,
               bifpn_min_level=3,
               bifpn_max_level=7,
               bifpn_num_iterations=5,
               bifpn_num_filters=112,
               bifpn_combine_method='fast_attention',
               use_explicit_padding=None,
               use_depthwise=None,
               override_base_feature_extractor_hyperparams=None,
               name='EfficientDet-D2'):

    """SSD Keras EfficientNet-b2 BiFPN (EfficientDet-d2) Feature Extractor.

    Args:
      is_training: whether the network is in training mode.
      depth_multiplier: unsupported by EfficientNetBiFPN. float, depth
        multiplier for the feature extractor.
      min_depth: minimum feature extractor depth.
      pad_to_multiple: the nearest multiple to zero pad the input height and
        width dimensions to.
      conv_hyperparams: a `hyperparams_builder.KerasLayerHyperparams` object
        containing convolution hyperparameters for the layers added on top of
        the base feature extractor.
      freeze_batchnorm: whether to freeze batch norm parameters during training
        or not. When training with a small batch size (e.g. 1), it is desirable
        to freeze batch norm update and use pretrained batch norm params.
      inplace_batchnorm_update: whether to update batch norm moving average
        values inplace. When this is false train op must add a control
        dependency on tf.graphkeys.UPDATE_OPS collection in order to update
        batch norm statistics.
      bifpn_min_level: the highest resolution feature map to use in BiFPN. The
        valid values are {2, 3, 4, 5} which map to Resnet blocks {1, 2, 3, 4}
        respectively.
      bifpn_max_level: the smallest resolution feature map to use in the BiFPN.
        BiFPN constructions uses features maps starting from bifpn_min_level
        upto the bifpn_max_level. In the case that there are not enough feature
        maps in the backbone network, additional feature maps are created by
        applying stride 2 convolutions until we get the desired number of BiFPN
        levels.
      bifpn_num_iterations: number of BiFPN iterations. Overrided if
        efficientdet_version is provided.
      bifpn_num_filters: number of filters (channels) in all BiFPN layers.
        Overrided if efficientdet_version is provided.
      bifpn_combine_method: the method used to combine BiFPN nodes.
      use_explicit_padding: unsupported by EfficientNetBiFPN. Whether to use
        explicit padding when extracting features.
      use_depthwise: unsupported by EfficientNetBiFPN, since BiFPN uses regular
        convolutions when inputs to a node have a differing number of channels,
        and use separable convolutions after combine operations.
      override_base_feature_extractor_hyperparams: unsupported. Whether to
        override hyperparameters of the base feature extractor with the one from
        `conv_hyperparams`.
      name: a string name scope to assign to the model. If 'None', Keras will
        auto-generate one from the class name.
    """
    super(SSDEfficientNetB2BiFPNKerasFeatureExtractor, self).__init__(
        is_training=is_training,
        depth_multiplier=depth_multiplier,
        min_depth=min_depth,
        pad_to_multiple=pad_to_multiple,
        conv_hyperparams=conv_hyperparams,
        freeze_batchnorm=freeze_batchnorm,
        inplace_batchnorm_update=inplace_batchnorm_update,
        bifpn_min_level=bifpn_min_level,
        bifpn_max_level=bifpn_max_level,
        bifpn_num_iterations=bifpn_num_iterations,
        bifpn_num_filters=bifpn_num_filters,
        bifpn_combine_method=bifpn_combine_method,
        efficientnet_version='efficientnet-b2',
        use_explicit_padding=use_explicit_padding,
        use_depthwise=use_depthwise,
        override_base_feature_extractor_hyperparams=
        override_base_feature_extractor_hyperparams,
        name=name)


class SSDEfficientNetB3BiFPNKerasFeatureExtractor(
    SSDEfficientNetBiFPNKerasFeatureExtractor):
  """SSD Keras EfficientNet-b3 BiFPN (EfficientDet-d3) Feature Extractor."""

  def __init__(self,
               is_training,
               depth_multiplier,
               min_depth,
               pad_to_multiple,
               conv_hyperparams,
               freeze_batchnorm,
               inplace_batchnorm_update,
               bifpn_min_level=3,
               bifpn_max_level=7,
               bifpn_num_iterations=6,
               bifpn_num_filters=160,
               bifpn_combine_method='fast_attention',
               use_explicit_padding=None,
               use_depthwise=None,
               override_base_feature_extractor_hyperparams=None,
               name='EfficientDet-D3'):

    """SSD Keras EfficientNet-b3 BiFPN (EfficientDet-d3) Feature Extractor.

    Args:
      is_training: whether the network is in training mode.
      depth_multiplier: unsupported by EfficientNetBiFPN. float, depth
        multiplier for the feature extractor.
      min_depth: minimum feature extractor depth.
      pad_to_multiple: the nearest multiple to zero pad the input height and
        width dimensions to.
      conv_hyperparams: a `hyperparams_builder.KerasLayerHyperparams` object
        containing convolution hyperparameters for the layers added on top of
        the base feature extractor.
      freeze_batchnorm: whether to freeze batch norm parameters during training
        or not. When training with a small batch size (e.g. 1), it is desirable
        to freeze batch norm update and use pretrained batch norm params.
      inplace_batchnorm_update: whether to update batch norm moving average
        values inplace. When this is false train op must add a control
        dependency on tf.graphkeys.UPDATE_OPS collection in order to update
        batch norm statistics.
      bifpn_min_level: the highest resolution feature map to use in BiFPN. The
        valid values are {2, 3, 4, 5} which map to Resnet blocks {1, 2, 3, 4}
        respectively.
      bifpn_max_level: the smallest resolution feature map to use in the BiFPN.
        BiFPN constructions uses features maps starting from bifpn_min_level
        upto the bifpn_max_level. In the case that there are not enough feature
        maps in the backbone network, additional feature maps are created by
        applying stride 2 convolutions until we get the desired number of BiFPN
        levels.
      bifpn_num_iterations: number of BiFPN iterations. Overrided if
        efficientdet_version is provided.
      bifpn_num_filters: number of filters (channels) in all BiFPN layers.
        Overrided if efficientdet_version is provided.
      bifpn_combine_method: the method used to combine BiFPN nodes.
      use_explicit_padding: unsupported by EfficientNetBiFPN. Whether to use
        explicit padding when extracting features.
      use_depthwise: unsupported by EfficientNetBiFPN, since BiFPN uses regular
        convolutions when inputs to a node have a differing number of channels,
        and use separable convolutions after combine operations.
      override_base_feature_extractor_hyperparams: unsupported. Whether to
        override hyperparameters of the base feature extractor with the one from
        `conv_hyperparams`.
      name: a string name scope to assign to the model. If 'None', Keras will
        auto-generate one from the class name.
    """
    super(SSDEfficientNetB3BiFPNKerasFeatureExtractor, self).__init__(
        is_training=is_training,
        depth_multiplier=depth_multiplier,
        min_depth=min_depth,
        pad_to_multiple=pad_to_multiple,
        conv_hyperparams=conv_hyperparams,
        freeze_batchnorm=freeze_batchnorm,
        inplace_batchnorm_update=inplace_batchnorm_update,
        bifpn_min_level=bifpn_min_level,
        bifpn_max_level=bifpn_max_level,
        bifpn_num_iterations=bifpn_num_iterations,
        bifpn_num_filters=bifpn_num_filters,
        bifpn_combine_method=bifpn_combine_method,
        efficientnet_version='efficientnet-b3',
        use_explicit_padding=use_explicit_padding,
        use_depthwise=use_depthwise,
        override_base_feature_extractor_hyperparams=
        override_base_feature_extractor_hyperparams,
        name=name)


class SSDEfficientNetB4BiFPNKerasFeatureExtractor(
    SSDEfficientNetBiFPNKerasFeatureExtractor):
  """SSD Keras EfficientNet-b4 BiFPN (EfficientDet-d4) Feature Extractor."""

  def __init__(self,
               is_training,
               depth_multiplier,
               min_depth,
               pad_to_multiple,
               conv_hyperparams,
               freeze_batchnorm,
               inplace_batchnorm_update,
               bifpn_min_level=3,
               bifpn_max_level=7,
               bifpn_num_iterations=7,
               bifpn_num_filters=224,
               bifpn_combine_method='fast_attention',
               use_explicit_padding=None,
               use_depthwise=None,
               override_base_feature_extractor_hyperparams=None,
               name='EfficientDet-D4'):

    """SSD Keras EfficientNet-b4 BiFPN (EfficientDet-d4) Feature Extractor.

    Args:
      is_training: whether the network is in training mode.
      depth_multiplier: unsupported by EfficientNetBiFPN. float, depth
        multiplier for the feature extractor.
      min_depth: minimum feature extractor depth.
      pad_to_multiple: the nearest multiple to zero pad the input height and
        width dimensions to.
      conv_hyperparams: a `hyperparams_builder.KerasLayerHyperparams` object
        containing convolution hyperparameters for the layers added on top of
        the base feature extractor.
      freeze_batchnorm: whether to freeze batch norm parameters during training
        or not. When training with a small batch size (e.g. 1), it is desirable
        to freeze batch norm update and use pretrained batch norm params.
      inplace_batchnorm_update: whether to update batch norm moving average
        values inplace. When this is false train op must add a control
        dependency on tf.graphkeys.UPDATE_OPS collection in order to update
        batch norm statistics.
      bifpn_min_level: the highest resolution feature map to use in BiFPN. The
        valid values are {2, 3, 4, 5} which map to Resnet blocks {1, 2, 3, 4}
        respectively.
      bifpn_max_level: the smallest resolution feature map to use in the BiFPN.
        BiFPN constructions uses features maps starting from bifpn_min_level
        upto the bifpn_max_level. In the case that there are not enough feature
        maps in the backbone network, additional feature maps are created by
        applying stride 2 convolutions until we get the desired number of BiFPN
        levels.
      bifpn_num_iterations: number of BiFPN iterations. Overrided if
        efficientdet_version is provided.
      bifpn_num_filters: number of filters (channels) in all BiFPN layers.
        Overrided if efficientdet_version is provided.
      bifpn_combine_method: the method used to combine BiFPN nodes.
      use_explicit_padding: unsupported by EfficientNetBiFPN. Whether to use
        explicit padding when extracting features.
      use_depthwise: unsupported by EfficientNetBiFPN, since BiFPN uses regular
        convolutions when inputs to a node have a differing number of channels,
        and use separable convolutions after combine operations.
      override_base_feature_extractor_hyperparams: unsupported. Whether to
        override hyperparameters of the base feature extractor with the one from
        `conv_hyperparams`.
      name: a string name scope to assign to the model. If 'None', Keras will
        auto-generate one from the class name.
    """
    super(SSDEfficientNetB4BiFPNKerasFeatureExtractor, self).__init__(
        is_training=is_training,
        depth_multiplier=depth_multiplier,
        min_depth=min_depth,
        pad_to_multiple=pad_to_multiple,
        conv_hyperparams=conv_hyperparams,
        freeze_batchnorm=freeze_batchnorm,
        inplace_batchnorm_update=inplace_batchnorm_update,
        bifpn_min_level=bifpn_min_level,
        bifpn_max_level=bifpn_max_level,
        bifpn_num_iterations=bifpn_num_iterations,
        bifpn_num_filters=bifpn_num_filters,
        bifpn_combine_method=bifpn_combine_method,
        efficientnet_version='efficientnet-b4',
        use_explicit_padding=use_explicit_padding,
        use_depthwise=use_depthwise,
        override_base_feature_extractor_hyperparams=
        override_base_feature_extractor_hyperparams,
        name=name)


class SSDEfficientNetB5BiFPNKerasFeatureExtractor(
    SSDEfficientNetBiFPNKerasFeatureExtractor):
  """SSD Keras EfficientNet-b5 BiFPN (EfficientDet-d5) Feature Extractor."""

  def __init__(self,
               is_training,
               depth_multiplier,
               min_depth,
               pad_to_multiple,
               conv_hyperparams,
               freeze_batchnorm,
               inplace_batchnorm_update,
               bifpn_min_level=3,
               bifpn_max_level=7,
               bifpn_num_iterations=7,
               bifpn_num_filters=288,
               bifpn_combine_method='fast_attention',
               use_explicit_padding=None,
               use_depthwise=None,
               override_base_feature_extractor_hyperparams=None,
               name='EfficientDet-D5'):

    """SSD Keras EfficientNet-b5 BiFPN (EfficientDet-d5) Feature Extractor.

    Args:
      is_training: whether the network is in training mode.
      depth_multiplier: unsupported by EfficientNetBiFPN. float, depth
        multiplier for the feature extractor.
      min_depth: minimum feature extractor depth.
      pad_to_multiple: the nearest multiple to zero pad the input height and
        width dimensions to.
      conv_hyperparams: a `hyperparams_builder.KerasLayerHyperparams` object
        containing convolution hyperparameters for the layers added on top of
        the base feature extractor.
      freeze_batchnorm: whether to freeze batch norm parameters during training
        or not. When training with a small batch size (e.g. 1), it is desirable
        to freeze batch norm update and use pretrained batch norm params.
      inplace_batchnorm_update: whether to update batch norm moving average
        values inplace. When this is false train op must add a control
        dependency on tf.graphkeys.UPDATE_OPS collection in order to update
        batch norm statistics.
      bifpn_min_level: the highest resolution feature map to use in BiFPN. The
        valid values are {2, 3, 4, 5} which map to Resnet blocks {1, 2, 3, 4}
        respectively.
      bifpn_max_level: the smallest resolution feature map to use in the BiFPN.
        BiFPN constructions uses features maps starting from bifpn_min_level
        upto the bifpn_max_level. In the case that there are not enough feature
        maps in the backbone network, additional feature maps are created by
        applying stride 2 convolutions until we get the desired number of BiFPN
        levels.
      bifpn_num_iterations: number of BiFPN iterations. Overrided if
        efficientdet_version is provided.
      bifpn_num_filters: number of filters (channels) in all BiFPN layers.
        Overrided if efficientdet_version is provided.
      bifpn_combine_method: the method used to combine BiFPN nodes.
      use_explicit_padding: unsupported by EfficientNetBiFPN. Whether to use
        explicit padding when extracting features.
      use_depthwise: unsupported by EfficientNetBiFPN, since BiFPN uses regular
        convolutions when inputs to a node have a differing number of channels,
        and use separable convolutions after combine operations.
      override_base_feature_extractor_hyperparams: unsupported. Whether to
        override hyperparameters of the base feature extractor with the one from
        `conv_hyperparams`.
      name: a string name scope to assign to the model. If 'None', Keras will
        auto-generate one from the class name.
    """
    super(SSDEfficientNetB5BiFPNKerasFeatureExtractor, self).__init__(
        is_training=is_training,
        depth_multiplier=depth_multiplier,
        min_depth=min_depth,
        pad_to_multiple=pad_to_multiple,
        conv_hyperparams=conv_hyperparams,
        freeze_batchnorm=freeze_batchnorm,
        inplace_batchnorm_update=inplace_batchnorm_update,
        bifpn_min_level=bifpn_min_level,
        bifpn_max_level=bifpn_max_level,
        bifpn_num_iterations=bifpn_num_iterations,
        bifpn_num_filters=bifpn_num_filters,
        bifpn_combine_method=bifpn_combine_method,
        efficientnet_version='efficientnet-b5',
        use_explicit_padding=use_explicit_padding,
        use_depthwise=use_depthwise,
        override_base_feature_extractor_hyperparams=
        override_base_feature_extractor_hyperparams,
        name=name)


class SSDEfficientNetB6BiFPNKerasFeatureExtractor(
    SSDEfficientNetBiFPNKerasFeatureExtractor):
  """SSD Keras EfficientNet-b6 BiFPN (EfficientDet-d[6,7]) Feature Extractor."""

  def __init__(self,
               is_training,
               depth_multiplier,
               min_depth,
               pad_to_multiple,
               conv_hyperparams,
               freeze_batchnorm,
               inplace_batchnorm_update,
               bifpn_min_level=3,
               bifpn_max_level=7,
               bifpn_num_iterations=8,
               bifpn_num_filters=384,
               bifpn_combine_method='sum',
               use_explicit_padding=None,
               use_depthwise=None,
               override_base_feature_extractor_hyperparams=None,
               name='EfficientDet-D6-D7'):

    """SSD Keras EfficientNet-b6 BiFPN (EfficientDet-d[6,7]) Feature Extractor.

    SSD Keras EfficientNet-b6 BiFPN Feature Extractor, a.k.a. EfficientDet-d6
    and EfficientDet-d7. The EfficientDet-d[6,7] models use the same backbone
    EfficientNet-b6 and the same BiFPN architecture, and therefore have the same
    number of parameters. They only differ in their input resolutions.

    Args:
      is_training: whether the network is in training mode.
      depth_multiplier: unsupported by EfficientNetBiFPN. float, depth
        multiplier for the feature extractor.
      min_depth: minimum feature extractor depth.
      pad_to_multiple: the nearest multiple to zero pad the input height and
        width dimensions to.
      conv_hyperparams: a `hyperparams_builder.KerasLayerHyperparams` object
        containing convolution hyperparameters for the layers added on top of
        the base feature extractor.
      freeze_batchnorm: whether to freeze batch norm parameters during training
        or not. When training with a small batch size (e.g. 1), it is desirable
        to freeze batch norm update and use pretrained batch norm params.
      inplace_batchnorm_update: whether to update batch norm moving average
        values inplace. When this is false train op must add a control
        dependency on tf.graphkeys.UPDATE_OPS collection in order to update
        batch norm statistics.
      bifpn_min_level: the highest resolution feature map to use in BiFPN. The
        valid values are {2, 3, 4, 5} which map to Resnet blocks {1, 2, 3, 4}
        respectively.
      bifpn_max_level: the smallest resolution feature map to use in the BiFPN.
        BiFPN constructions uses features maps starting from bifpn_min_level
        upto the bifpn_max_level. In the case that there are not enough feature
        maps in the backbone network, additional feature maps are created by
        applying stride 2 convolutions until we get the desired number of BiFPN
        levels.
      bifpn_num_iterations: number of BiFPN iterations. Overrided if
        efficientdet_version is provided.
      bifpn_num_filters: number of filters (channels) in all BiFPN layers.
        Overrided if efficientdet_version is provided.
      bifpn_combine_method: the method used to combine BiFPN nodes.
      use_explicit_padding: unsupported by EfficientNetBiFPN. Whether to use
        explicit padding when extracting features.
      use_depthwise: unsupported by EfficientNetBiFPN, since BiFPN uses regular
        convolutions when inputs to a node have a differing number of channels,
        and use separable convolutions after combine operations.
      override_base_feature_extractor_hyperparams: unsupported. Whether to
        override hyperparameters of the base feature extractor with the one from
        `conv_hyperparams`.
      name: a string name scope to assign to the model. If 'None', Keras will
        auto-generate one from the class name.
    """
    super(SSDEfficientNetB6BiFPNKerasFeatureExtractor, self).__init__(
        is_training=is_training,
        depth_multiplier=depth_multiplier,
        min_depth=min_depth,
        pad_to_multiple=pad_to_multiple,
        conv_hyperparams=conv_hyperparams,
        freeze_batchnorm=freeze_batchnorm,
        inplace_batchnorm_update=inplace_batchnorm_update,
        bifpn_min_level=bifpn_min_level,
        bifpn_max_level=bifpn_max_level,
        bifpn_num_iterations=bifpn_num_iterations,
        bifpn_num_filters=bifpn_num_filters,
        bifpn_combine_method=bifpn_combine_method,
        efficientnet_version='efficientnet-b6',
        use_explicit_padding=use_explicit_padding,
        use_depthwise=use_depthwise,
        override_base_feature_extractor_hyperparams=
        override_base_feature_extractor_hyperparams,
        name=name)


class SSDEfficientNetB7BiFPNKerasFeatureExtractor(
    SSDEfficientNetBiFPNKerasFeatureExtractor):
  """SSD Keras EfficientNet-b7 BiFPN Feature Extractor."""

  def __init__(self,
               is_training,
               depth_multiplier,
               min_depth,
               pad_to_multiple,
               conv_hyperparams,
               freeze_batchnorm,
               inplace_batchnorm_update,
               bifpn_min_level=3,
               bifpn_max_level=7,
               bifpn_num_iterations=8,
               bifpn_num_filters=384,
               bifpn_combine_method='sum',
               use_explicit_padding=None,
               use_depthwise=None,
               override_base_feature_extractor_hyperparams=None,
               name='EfficientNet-B7_BiFPN'):

    """SSD Keras EfficientNet-b7 BiFPN Feature Extractor.

    Args:
      is_training: whether the network is in training mode.
      depth_multiplier: unsupported by EfficientNetBiFPN. float, depth
        multiplier for the feature extractor.
      min_depth: minimum feature extractor depth.
      pad_to_multiple: the nearest multiple to zero pad the input height and
        width dimensions to.
      conv_hyperparams: a `hyperparams_builder.KerasLayerHyperparams` object
        containing convolution hyperparameters for the layers added on top of
        the base feature extractor.
      freeze_batchnorm: whether to freeze batch norm parameters during training
        or not. When training with a small batch size (e.g. 1), it is desirable
        to freeze batch norm update and use pretrained batch norm params.
      inplace_batchnorm_update: whether to update batch norm moving average
        values inplace. When this is false train op must add a control
        dependency on tf.graphkeys.UPDATE_OPS collection in order to update
        batch norm statistics.
      bifpn_min_level: the highest resolution feature map to use in BiFPN. The
        valid values are {2, 3, 4, 5} which map to Resnet blocks {1, 2, 3, 4}
        respectively.
      bifpn_max_level: the smallest resolution feature map to use in the BiFPN.
        BiFPN constructions uses features maps starting from bifpn_min_level
        upto the bifpn_max_level. In the case that there are not enough feature
        maps in the backbone network, additional feature maps are created by
        applying stride 2 convolutions until we get the desired number of BiFPN
        levels.
      bifpn_num_iterations: number of BiFPN iterations. Overrided if
        efficientdet_version is provided.
      bifpn_num_filters: number of filters (channels) in all BiFPN layers.
        Overrided if efficientdet_version is provided.
      bifpn_combine_method: the method used to combine BiFPN nodes.
      use_explicit_padding: unsupported by EfficientNetBiFPN. Whether to use
        explicit padding when extracting features.
      use_depthwise: unsupported by EfficientNetBiFPN, since BiFPN uses regular
        convolutions when inputs to a node have a differing number of channels,
        and use separable convolutions after combine operations.
      override_base_feature_extractor_hyperparams: unsupported. Whether to
        override hyperparameters of the base feature extractor with the one from
        `conv_hyperparams`.
      name: a string name scope to assign to the model. If 'None', Keras will
        auto-generate one from the class name.
    """
    super(SSDEfficientNetB7BiFPNKerasFeatureExtractor, self).__init__(
        is_training=is_training,
        depth_multiplier=depth_multiplier,
        min_depth=min_depth,
        pad_to_multiple=pad_to_multiple,
        conv_hyperparams=conv_hyperparams,
        freeze_batchnorm=freeze_batchnorm,
        inplace_batchnorm_update=inplace_batchnorm_update,
        bifpn_min_level=bifpn_min_level,
        bifpn_max_level=bifpn_max_level,
        bifpn_num_iterations=bifpn_num_iterations,
        bifpn_num_filters=bifpn_num_filters,
        bifpn_combine_method=bifpn_combine_method,
        efficientnet_version='efficientnet-b7',
        use_explicit_padding=use_explicit_padding,
        use_depthwise=use_depthwise,
        override_base_feature_extractor_hyperparams=
        override_base_feature_extractor_hyperparams,
        name=name)
