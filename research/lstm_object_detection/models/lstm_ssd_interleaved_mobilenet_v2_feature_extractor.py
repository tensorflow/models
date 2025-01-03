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

"""LSTDInterleavedFeatureExtractor which interleaves multiple MobileNet V2."""

import tensorflow.compat.v1 as tf
import tf_slim as slim

from tensorflow.python.framework import ops as tf_ops
from lstm_object_detection.lstm import lstm_cells
from lstm_object_detection.lstm import rnn_decoder
from lstm_object_detection.meta_architectures import lstm_ssd_meta_arch
from lstm_object_detection.models import mobilenet_defs
from object_detection.models import feature_map_generators
from object_detection.utils import ops
from object_detection.utils import shape_utils
from nets.mobilenet import mobilenet
from nets.mobilenet import mobilenet_v2


class LSTMSSDInterleavedMobilenetV2FeatureExtractor(
    lstm_ssd_meta_arch.LSTMSSDInterleavedFeatureExtractor):
  """LSTM-SSD Interleaved Feature Extractor using MobilenetV2 features."""

  def __init__(self,
               is_training,
               depth_multiplier,
               min_depth,
               pad_to_multiple,
               conv_hyperparams_fn,
               reuse_weights=None,
               use_explicit_padding=False,
               use_depthwise=True,
               override_base_feature_extractor_hyperparams=False):
    """Interleaved Feature Extractor for LSTD Models with MobileNet v2.

    Args:
      is_training: whether the network is in training mode.
      depth_multiplier: float depth multiplier for feature extractor.
      min_depth: minimum feature extractor depth.
      pad_to_multiple: the nearest multiple to zero pad the input height and
        width dimensions to.
      conv_hyperparams_fn: A function to construct tf slim arg_scope for conv2d
        and separable_conv2d ops in the layers that are added on top of the
        base feature extractor.
      reuse_weights: Whether to reuse variables. Default is None.
      use_explicit_padding: Whether to use explicit padding when extracting
        features. Default is False.
      use_depthwise: Whether to use depthwise convolutions. Default is True.
      override_base_feature_extractor_hyperparams: Whether to override
        hyperparameters of the base feature extractor with the one from
        `conv_hyperparams_fn`.
    """
    super(LSTMSSDInterleavedMobilenetV2FeatureExtractor, self).__init__(
        is_training=is_training,
        depth_multiplier=depth_multiplier,
        min_depth=min_depth,
        pad_to_multiple=pad_to_multiple,
        conv_hyperparams_fn=conv_hyperparams_fn,
        reuse_weights=reuse_weights,
        use_explicit_padding=use_explicit_padding,
        use_depthwise=use_depthwise,
        override_base_feature_extractor_hyperparams=
        override_base_feature_extractor_hyperparams)
    # RANDOM_SKIP_SMALL means the training policy is random and the small model
    # does not update state during training.
    if self._is_training:
      self._interleave_method = 'RANDOM_SKIP_SMALL'
    else:
      self._interleave_method = 'SKIP9'

    self._flatten_state = False
    self._scale_state = False
    self._clip_state = True
    self._pre_bottleneck = True
    self._feature_map_layout = {
        'from_layer': ['layer_19', '', '', '', ''],
        'layer_depth': [-1, 256, 256, 256, 256],
        'use_depthwise': self._use_depthwise,
        'use_explicit_padding': self._use_explicit_padding,
    }
    self._low_res = True
    self._base_network_scope = 'MobilenetV2'

  def extract_base_features_large(self, preprocessed_inputs):
    """Extract the large base model features.

    Variables are created under the scope of <scope>/MobilenetV2_1/

    Args:
      preprocessed_inputs: preprocessed input images of shape:
        [batch, width, height, depth].

    Returns:
      net: the last feature map created from the base feature extractor.
      end_points: a dictionary of feature maps created.
    """
    scope_name = self._base_network_scope + '_1'
    with tf.variable_scope(scope_name, reuse=self._reuse_weights) as base_scope:
      net, end_points = mobilenet_v2.mobilenet_base(
          preprocessed_inputs,
          depth_multiplier=self._depth_multipliers[0],
          conv_defs=mobilenet_defs.mobilenet_v2_lite_def(
              is_quantized=self._is_quantized),
          use_explicit_padding=self._use_explicit_padding,
          scope=base_scope)
      return net, end_points

  def extract_base_features_small(self, preprocessed_inputs):
    """Extract the small base model features.

    Variables are created under the scope of <scope>/MobilenetV2_2/

    Args:
      preprocessed_inputs: preprocessed input images of shape:
        [batch, width, height, depth].

    Returns:
      net: the last feature map created from the base feature extractor.
      end_points: a dictionary of feature maps created.
    """
    scope_name = self._base_network_scope + '_2'
    with tf.variable_scope(scope_name, reuse=self._reuse_weights) as base_scope:
      if self._low_res:
        height_small = preprocessed_inputs.get_shape().as_list()[1] // 2
        width_small = preprocessed_inputs.get_shape().as_list()[2] // 2
        inputs_small = tf.image.resize_images(preprocessed_inputs,
                                              [height_small, width_small])
        # Create end point handle for tflite deployment.
        with tf.name_scope(None):
          inputs_small = tf.identity(
              inputs_small, name='normalized_input_image_tensor_small')
      else:
        inputs_small = preprocessed_inputs
      net, end_points = mobilenet_v2.mobilenet_base(
          inputs_small,
          depth_multiplier=self._depth_multipliers[1],
          conv_defs=mobilenet_defs.mobilenet_v2_lite_def(
              is_quantized=self._is_quantized, low_res=self._low_res),
          use_explicit_padding=self._use_explicit_padding,
          scope=base_scope)
      return net, end_points

  def create_lstm_cell(self, batch_size, output_size, state_saver, state_name,
                       dtype=tf.float32):
    """Create the LSTM cell, and initialize state if necessary.

    Args:
      batch_size: input batch size.
      output_size: output size of the lstm cell, [width, height].
      state_saver: a state saver object with methods `state` and `save_state`.
      state_name: string, the name to use with the state_saver.
      dtype: dtype to initialize lstm state.

    Returns:
      lstm_cell: the lstm cell unit.
      init_state: initial state representations.
      step: the step
    """
    lstm_cell = lstm_cells.GroupedConvLSTMCell(
        filter_size=(3, 3),
        output_size=output_size,
        num_units=max(self._min_depth, self._lstm_state_depth),
        is_training=self._is_training,
        activation=tf.nn.relu6,
        flatten_state=self._flatten_state,
        scale_state=self._scale_state,
        clip_state=self._clip_state,
        output_bottleneck=True,
        pre_bottleneck=self._pre_bottleneck,
        is_quantized=self._is_quantized,
        visualize_gates=False)

    if state_saver is None:
      init_state = lstm_cell.init_state('lstm_state', batch_size, dtype)
      step = None
    else:
      step = state_saver.state(state_name + '_step')
      c = state_saver.state(state_name + '_c')
      h = state_saver.state(state_name + '_h')
      c.set_shape([batch_size] + c.get_shape().as_list()[1:])
      h.set_shape([batch_size] + h.get_shape().as_list()[1:])
      init_state = (c, h)
    return lstm_cell, init_state, step

  def extract_features(self, preprocessed_inputs, state_saver=None,
                       state_name='lstm_state', unroll_length=10, scope=None):
    """Extract features from preprocessed inputs.

    The features include the base network features, lstm features and SSD
    features, organized in the following name scope:

    <scope>/MobilenetV2_1/...
    <scope>/MobilenetV2_2/...
    <scope>/LSTM/...
    <scope>/FeatureMap/...

    Args:
      preprocessed_inputs: a [batch, height, width, channels] float tensor
        representing a batch of consecutive frames from video clips.
      state_saver: A state saver object with methods `state` and `save_state`.
      state_name: Python string, the name to use with the state_saver.
      unroll_length: number of steps to unroll the lstm.
      scope: Scope for the base network of the feature extractor.

    Returns:
      feature_maps: a list of tensors where the ith tensor has shape
        [batch, height_i, width_i, depth_i]
    Raises:
      ValueError: if interleave_method not recognized or large and small base
        network output feature maps of different sizes.
    """
    preprocessed_inputs = shape_utils.check_min_image_dim(
        33, preprocessed_inputs)
    preprocessed_inputs = ops.pad_to_multiple(
        preprocessed_inputs, self._pad_to_multiple)
    batch_size = preprocessed_inputs.shape[0].value // unroll_length
    batch_axis = 0
    nets = []

    # Batch processing of mobilenet features.
    with slim.arg_scope(mobilenet_v2.training_scope(
        is_training=self._is_training,
        bn_decay=0.9997)), \
        slim.arg_scope([mobilenet.depth_multiplier],
                       min_depth=self._min_depth, divisible_by=8):
      # Big model.
      net, _ = self.extract_base_features_large(preprocessed_inputs)
      nets.append(net)
      large_base_feature_shape = net.shape

      # Small models
      net, _ = self.extract_base_features_small(preprocessed_inputs)
      nets.append(net)
      small_base_feature_shape = net.shape
      if not (large_base_feature_shape[1] == small_base_feature_shape[1] and
              large_base_feature_shape[2] == small_base_feature_shape[2]):
        raise ValueError('Large and Small base network feature map dimension '
                         'not equal!')

    with slim.arg_scope(self._conv_hyperparams_fn()):
      with tf.variable_scope('LSTM', reuse=self._reuse_weights):
        output_size = (large_base_feature_shape[1], large_base_feature_shape[2])
        lstm_cell, init_state, step = self.create_lstm_cell(
            batch_size, output_size, state_saver, state_name,
            dtype=preprocessed_inputs.dtype)

        nets_seq = [
            tf.split(net, unroll_length, axis=batch_axis) for net in nets
        ]

        net_seq, states_out = rnn_decoder.multi_input_rnn_decoder(
            nets_seq,
            init_state,
            lstm_cell,
            step,
            selection_strategy=self._interleave_method,
            is_training=self._is_training,
            is_quantized=self._is_quantized,
            pre_bottleneck=self._pre_bottleneck,
            flatten_state=self._flatten_state,
            scope=None)
        self._states_out = states_out

      image_features = {}
      if state_saver is not None:
        self._step = state_saver.state(state_name + '_step')
        batcher_ops = [
            state_saver.save_state(state_name + '_c', states_out[-1][0]),
            state_saver.save_state(state_name + '_h', states_out[-1][1]),
            state_saver.save_state(state_name + '_step', self._step + 1)]
        with tf_ops.control_dependencies(batcher_ops):
          image_features['layer_19'] = tf.concat(net_seq, 0)
      else:
        image_features['layer_19'] = tf.concat(net_seq, 0)

      # SSD layers.
      with tf.variable_scope('FeatureMap'):
        feature_maps = feature_map_generators.multi_resolution_feature_maps(
            feature_map_layout=self._feature_map_layout,
            depth_multiplier=self._depth_multiplier,
            min_depth=self._min_depth,
            insert_1x1_conv=True,
            image_features=image_features,
            pool_residual=True)
    return list(feature_maps.values())
