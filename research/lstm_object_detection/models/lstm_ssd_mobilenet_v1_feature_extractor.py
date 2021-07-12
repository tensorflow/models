# Copyright 2018 The TensorFlow Authors. All Rights Reserved.
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

"""LSTMSSDFeatureExtractor for MobilenetV1 features."""

import tensorflow.compat.v1 as tf
import tf_slim as slim
from tensorflow.python.framework import ops as tf_ops
from lstm_object_detection.lstm import lstm_cells
from lstm_object_detection.lstm import rnn_decoder
from lstm_object_detection.meta_architectures import lstm_ssd_meta_arch
from object_detection.models import feature_map_generators
from object_detection.utils import context_manager
from object_detection.utils import ops
from object_detection.utils import shape_utils
from nets import mobilenet_v1


class LSTMSSDMobileNetV1FeatureExtractor(
    lstm_ssd_meta_arch.LSTMSSDFeatureExtractor):
  """LSTM Feature Extractor using MobilenetV1 features."""

  def __init__(self,
               is_training,
               depth_multiplier,
               min_depth,
               pad_to_multiple,
               conv_hyperparams_fn,
               reuse_weights=None,
               use_explicit_padding=False,
               use_depthwise=True,
               override_base_feature_extractor_hyperparams=False,
               lstm_state_depth=256):
    """Initializes instance of MobileNetV1 Feature Extractor for LSTMSSD Models.

    Args:
      is_training: A boolean whether the network is in training mode.
      depth_multiplier: A float depth multiplier for feature extractor.
      min_depth: A number representing minimum feature extractor depth.
      pad_to_multiple: The nearest multiple to zero pad the input height and
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
      lstm_state_depth: An integter of the depth of the lstm state.
    """
    super(LSTMSSDMobileNetV1FeatureExtractor, self).__init__(
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
    self._feature_map_layout = {
        'from_layer': ['Conv2d_13_pointwise_lstm', '', '', '', ''],
        'layer_depth': [-1, 512, 256, 256, 128],
        'use_explicit_padding': self._use_explicit_padding,
        'use_depthwise': self._use_depthwise,
    }
    self._base_network_scope = 'MobilenetV1'
    self._lstm_state_depth = lstm_state_depth

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
    lstm_cell = lstm_cells.BottleneckConvLSTMCell(
        filter_size=(3, 3),
        output_size=output_size,
        num_units=max(self._min_depth, self._lstm_state_depth),
        activation=tf.nn.relu6,
        visualize_gates=False)

    if state_saver is None:
      init_state = lstm_cell.init_state(state_name, batch_size, dtype)
      step = None
    else:
      step = state_saver.state(state_name + '_step')
      c = state_saver.state(state_name + '_c')
      h = state_saver.state(state_name + '_h')
      init_state = (c, h)
    return lstm_cell, init_state, step

  def extract_features(self,
                       preprocessed_inputs,
                       state_saver=None,
                       state_name='lstm_state',
                       unroll_length=5,
                       scope=None):
    """Extracts features from preprocessed inputs.

    The features include the base network features, lstm features and SSD
    features, organized in the following name scope:

    <parent scope>/MobilenetV1/...
    <parent scope>/LSTM/...
    <parent scope>/FeatureMaps/...

    Args:
      preprocessed_inputs: A [batch, height, width, channels] float tensor
        representing a batch of consecutive frames from video clips.
      state_saver: A state saver object with methods `state` and `save_state`.
      state_name: A python string for the name to use with the state_saver.
      unroll_length: The number of steps to unroll the lstm.
      scope: The scope for the base network of the feature extractor.

    Returns:
      A list of tensors where the ith tensor has shape [batch, height_i,
      width_i, depth_i]
    """
    preprocessed_inputs = shape_utils.check_min_image_dim(
        33, preprocessed_inputs)
    with slim.arg_scope(
        mobilenet_v1.mobilenet_v1_arg_scope(is_training=self._is_training)):
      with (slim.arg_scope(self._conv_hyperparams_fn())
            if self._override_base_feature_extractor_hyperparams else
            context_manager.IdentityContextManager()):
        with slim.arg_scope([slim.batch_norm], fused=False):
          # Base network.
          with tf.variable_scope(
              scope, self._base_network_scope,
              reuse=self._reuse_weights) as scope:
            net, image_features = mobilenet_v1.mobilenet_v1_base(
                ops.pad_to_multiple(preprocessed_inputs, self._pad_to_multiple),
                final_endpoint='Conv2d_13_pointwise',
                min_depth=self._min_depth,
                depth_multiplier=self._depth_multiplier,
                scope=scope)

    with slim.arg_scope(self._conv_hyperparams_fn()):
      with slim.arg_scope(
          [slim.batch_norm], fused=False, is_training=self._is_training):
        # ConvLSTM layers.
        batch_size = net.shape[0].value // unroll_length
        with tf.variable_scope('LSTM', reuse=self._reuse_weights) as lstm_scope:
          lstm_cell, init_state, _ = self.create_lstm_cell(
              batch_size,
              (net.shape[1].value, net.shape[2].value),
              state_saver,
              state_name,
              dtype=preprocessed_inputs.dtype)
          net_seq = list(tf.split(net, unroll_length))

          # Identities added for inputing state tensors externally.
          c_ident = tf.identity(init_state[0], name='lstm_state_in_c')
          h_ident = tf.identity(init_state[1], name='lstm_state_in_h')
          init_state = (c_ident, h_ident)

          net_seq, states_out = rnn_decoder.rnn_decoder(
              net_seq, init_state, lstm_cell, scope=lstm_scope)
          batcher_ops = None
          self._states_out = states_out
          if state_saver is not None:
            self._step = state_saver.state('%s_step' % state_name)
            batcher_ops = [
                state_saver.save_state('%s_c' % state_name, states_out[-1][0]),
                state_saver.save_state('%s_h' % state_name, states_out[-1][1]),
                state_saver.save_state('%s_step' % state_name, self._step + 1)
            ]
          with tf_ops.control_dependencies(batcher_ops):
            image_features['Conv2d_13_pointwise_lstm'] = tf.concat(net_seq, 0)

          # Identities added for reading output states, to be reused externally.
          tf.identity(states_out[-1][0], name='lstm_state_out_c')
          tf.identity(states_out[-1][1], name='lstm_state_out_h')

        # SSD layers.
        with tf.variable_scope('FeatureMaps', reuse=self._reuse_weights):
          feature_maps = feature_map_generators.multi_resolution_feature_maps(
              feature_map_layout=self._feature_map_layout,
              depth_multiplier=(self._depth_multiplier),
              min_depth=self._min_depth,
              insert_1x1_conv=True,
              image_features=image_features)

    return list(feature_maps.values())
