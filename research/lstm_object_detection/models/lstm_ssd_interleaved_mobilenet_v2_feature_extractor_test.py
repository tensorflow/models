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

"""Tests for lstm_ssd_interleaved_mobilenet_v2_feature_extractor."""

import itertools
import numpy as np
import tensorflow.compat.v1 as tf
from tensorflow.contrib import slim
from tensorflow.contrib import training as contrib_training

from lstm_object_detection.models import lstm_ssd_interleaved_mobilenet_v2_feature_extractor
from object_detection.models import ssd_feature_extractor_test


class LSTMSSDInterleavedMobilenetV2FeatureExtractorTest(
    ssd_feature_extractor_test.SsdFeatureExtractorTestBase):

  def _create_feature_extractor(self,
                                depth_multiplier,
                                pad_to_multiple,
                                is_quantized=False):
    """Constructs a new feature extractor.

    Args:
      depth_multiplier: float depth multiplier for feature extractor
      pad_to_multiple: the nearest multiple to zero pad the input height and
        width dimensions to.
      is_quantized: whether to quantize the graph.
    Returns:
      an ssd_meta_arch.SSDFeatureExtractor object.
    """
    min_depth = 32
    def conv_hyperparams_fn():
      with slim.arg_scope([slim.conv2d], normalizer_fn=slim.batch_norm), \
        slim.arg_scope([slim.batch_norm], is_training=False) as sc:
        return sc
    feature_extractor = (
        lstm_ssd_interleaved_mobilenet_v2_feature_extractor
        .LSTMSSDInterleavedMobilenetV2FeatureExtractor(False, depth_multiplier,
                                                       min_depth,
                                                       pad_to_multiple,
                                                       conv_hyperparams_fn))
    feature_extractor.lstm_state_depth = int(320 * depth_multiplier)
    feature_extractor.depth_multipliers = [
        depth_multiplier, depth_multiplier / 4.0
    ]
    feature_extractor.is_quantized = is_quantized
    return feature_extractor

  def test_feature_extractor_construct_with_expected_params(self):
    def conv_hyperparams_fn():
      with (slim.arg_scope([slim.conv2d], normalizer_fn=slim.batch_norm) and
            slim.arg_scope([slim.batch_norm], decay=0.97, epsilon=1e-3)) as sc:
        return sc

    params = {
        'is_training': True,
        'depth_multiplier': .55,
        'min_depth': 9,
        'pad_to_multiple': 3,
        'conv_hyperparams_fn': conv_hyperparams_fn,
        'reuse_weights': False,
        'use_explicit_padding': True,
        'use_depthwise': False,
        'override_base_feature_extractor_hyperparams': True}

    feature_extractor = (
        lstm_ssd_interleaved_mobilenet_v2_feature_extractor
        .LSTMSSDInterleavedMobilenetV2FeatureExtractor(**params))

    self.assertEqual(params['is_training'],
                     feature_extractor._is_training)
    self.assertEqual(params['depth_multiplier'],
                     feature_extractor._depth_multiplier)
    self.assertEqual(params['min_depth'],
                     feature_extractor._min_depth)
    self.assertEqual(params['pad_to_multiple'],
                     feature_extractor._pad_to_multiple)
    self.assertEqual(params['conv_hyperparams_fn'],
                     feature_extractor._conv_hyperparams_fn)
    self.assertEqual(params['reuse_weights'],
                     feature_extractor._reuse_weights)
    self.assertEqual(params['use_explicit_padding'],
                     feature_extractor._use_explicit_padding)
    self.assertEqual(params['use_depthwise'],
                     feature_extractor._use_depthwise)
    self.assertEqual(params['override_base_feature_extractor_hyperparams'],
                     (feature_extractor.
                      _override_base_feature_extractor_hyperparams))

  def test_extract_features_returns_correct_shapes_128(self):
    image_height = 128
    image_width = 128
    depth_multiplier = 1.0
    pad_to_multiple = 1
    expected_feature_map_shape = [(2, 4, 4, 640),
                                  (2, 2, 2, 256), (2, 1, 1, 256),
                                  (2, 1, 1, 256), (2, 1, 1, 256)]
    self.check_extract_features_returns_correct_shape(
        2, image_height, image_width, depth_multiplier, pad_to_multiple,
        expected_feature_map_shape)

  def test_extract_features_returns_correct_shapes_unroll10(self):
    image_height = 128
    image_width = 128
    depth_multiplier = 1.0
    pad_to_multiple = 1
    expected_feature_map_shape = [(10, 4, 4, 640),
                                  (10, 2, 2, 256), (10, 1, 1, 256),
                                  (10, 1, 1, 256), (10, 1, 1, 256)]
    self.check_extract_features_returns_correct_shape(
        10, image_height, image_width, depth_multiplier, pad_to_multiple,
        expected_feature_map_shape, unroll_length=10)

  def test_extract_features_returns_correct_shapes_320(self):
    image_height = 320
    image_width = 320
    depth_multiplier = 1.0
    pad_to_multiple = 1
    expected_feature_map_shape = [(2, 10, 10, 640),
                                  (2, 5, 5, 256), (2, 3, 3, 256),
                                  (2, 2, 2, 256), (2, 1, 1, 256)]
    self.check_extract_features_returns_correct_shape(
        2, image_height, image_width, depth_multiplier, pad_to_multiple,
        expected_feature_map_shape)

  def test_extract_features_returns_correct_shapes_enforcing_min_depth(self):
    image_height = 320
    image_width = 320
    depth_multiplier = 0.5**12
    pad_to_multiple = 1
    expected_feature_map_shape = [(2, 10, 10, 64),
                                  (2, 5, 5, 32), (2, 3, 3, 32),
                                  (2, 2, 2, 32), (2, 1, 1, 32)]
    self.check_extract_features_returns_correct_shape(
        2, image_height, image_width, depth_multiplier, pad_to_multiple,
        expected_feature_map_shape)

  def test_extract_features_returns_correct_shapes_with_pad_to_multiple(self):
    image_height = 299
    image_width = 299
    depth_multiplier = 1.0
    pad_to_multiple = 32
    expected_feature_map_shape = [(2, 10, 10, 640),
                                  (2, 5, 5, 256), (2, 3, 3, 256),
                                  (2, 2, 2, 256), (2, 1, 1, 256)]
    self.check_extract_features_returns_correct_shape(
        2, image_height, image_width, depth_multiplier, pad_to_multiple,
        expected_feature_map_shape)

  def test_preprocess_returns_correct_value_range(self):
    image_height = 128
    image_width = 128
    depth_multiplier = 1
    pad_to_multiple = 1
    test_image = np.random.rand(4, image_height, image_width, 3)
    feature_extractor = self._create_feature_extractor(depth_multiplier,
                                                       pad_to_multiple)
    preprocessed_image = feature_extractor.preprocess(test_image)
    self.assertTrue(np.all(np.less_equal(np.abs(preprocessed_image), 1.0)))

  def test_variables_only_created_in_scope(self):
    depth_multiplier = 1
    pad_to_multiple = 1
    scope_names = ['MobilenetV2', 'LSTM', 'FeatureMap']
    self.check_feature_extractor_variables_under_scopes(
        depth_multiplier, pad_to_multiple, scope_names)

  def test_has_fused_batchnorm(self):
    image_height = 40
    image_width = 40
    depth_multiplier = 1
    pad_to_multiple = 32
    image_placeholder = tf.placeholder(tf.float32,
                                       [1, image_height, image_width, 3])
    feature_extractor = self._create_feature_extractor(depth_multiplier,
                                                       pad_to_multiple)
    preprocessed_image = feature_extractor.preprocess(image_placeholder)
    _ = feature_extractor.extract_features(preprocessed_image, unroll_length=1)
    self.assertTrue(any(op.type.startswith('FusedBatchNorm')
                        for op in tf.get_default_graph().get_operations()))

  def test_variables_for_tflite(self):
    image_height = 40
    image_width = 40
    depth_multiplier = 1
    pad_to_multiple = 32
    image_placeholder = tf.placeholder(tf.float32,
                                       [1, image_height, image_width, 3])
    feature_extractor = self._create_feature_extractor(depth_multiplier,
                                                       pad_to_multiple)
    preprocessed_image = feature_extractor.preprocess(image_placeholder)
    tflite_unsupported = ['SquaredDifference']
    _ = feature_extractor.extract_features(preprocessed_image, unroll_length=1)
    self.assertFalse(any(op.type in tflite_unsupported
                         for op in tf.get_default_graph().get_operations()))

  def test_output_nodes_for_tflite(self):
    image_height = 64
    image_width = 64
    depth_multiplier = 1.0
    pad_to_multiple = 1
    image_placeholder = tf.placeholder(tf.float32,
                                       [1, image_height, image_width, 3])
    feature_extractor = self._create_feature_extractor(depth_multiplier,
                                                       pad_to_multiple)
    preprocessed_image = feature_extractor.preprocess(image_placeholder)
    _ = feature_extractor.extract_features(preprocessed_image, unroll_length=1)

    tflite_nodes = [
        'raw_inputs/init_lstm_c',
        'raw_inputs/init_lstm_h',
        'raw_inputs/base_endpoint',
        'raw_outputs/lstm_c',
        'raw_outputs/lstm_h',
        'raw_outputs/base_endpoint_1',
        'raw_outputs/base_endpoint_2'
    ]
    ops_names = [op.name for op in tf.get_default_graph().get_operations()]
    for node in tflite_nodes:
      self.assertTrue(any(node in s for s in ops_names))

  def test_fixed_concat_nodes(self):
    image_height = 64
    image_width = 64
    depth_multiplier = 1.0
    pad_to_multiple = 1
    image_placeholder = tf.placeholder(tf.float32,
                                       [1, image_height, image_width, 3])
    feature_extractor = self._create_feature_extractor(
        depth_multiplier, pad_to_multiple, is_quantized=True)
    preprocessed_image = feature_extractor.preprocess(image_placeholder)
    _ = feature_extractor.extract_features(preprocessed_image, unroll_length=1)

    concat_nodes = [
        'MobilenetV2_1/expanded_conv_16/project/Relu6',
        'MobilenetV2_2/expanded_conv_16/project/Relu6'
    ]
    ops_names = [op.name for op in tf.get_default_graph().get_operations()]
    for node in concat_nodes:
      self.assertTrue(any(node in s for s in ops_names))

  def test_lstm_states(self):
    image_height = 256
    image_width = 256
    depth_multiplier = 1
    pad_to_multiple = 1
    state_channel = 320
    init_state1 = {
        'lstm_state_c': tf.zeros(
            [image_height/32, image_width/32, state_channel]),
        'lstm_state_h': tf.zeros(
            [image_height/32, image_width/32, state_channel]),
        'lstm_state_step': tf.zeros([1])
    }
    init_state2 = {
        'lstm_state_c': tf.random_uniform(
            [image_height/32, image_width/32, state_channel]),
        'lstm_state_h': tf.random_uniform(
            [image_height/32, image_width/32, state_channel]),
        'lstm_state_step': tf.zeros([1])
    }
    seq = {'dummy': tf.random_uniform([2, 1, 1, 1])}
    stateful_reader1 = contrib_training.SequenceQueueingStateSaver(
        batch_size=1,
        num_unroll=1,
        input_length=2,
        input_key='',
        input_sequences=seq,
        input_context={},
        initial_states=init_state1,
        capacity=1)
    stateful_reader2 = contrib_training.SequenceQueueingStateSaver(
        batch_size=1,
        num_unroll=1,
        input_length=2,
        input_key='',
        input_sequences=seq,
        input_context={},
        initial_states=init_state2,
        capacity=1)
    image = tf.random_uniform([1, image_height, image_width, 3])
    feature_extractor = self._create_feature_extractor(depth_multiplier,
                                                       pad_to_multiple)
    with tf.variable_scope('zero_state'):
      feature_maps1 = feature_extractor.extract_features(
          image, stateful_reader1.next_batch, unroll_length=1)
    with tf.variable_scope('random_state'):
      feature_maps2 = feature_extractor.extract_features(
          image, stateful_reader2.next_batch, unroll_length=1)
    with tf.Session() as sess:
      sess.run(tf.global_variables_initializer())
      sess.run(tf.local_variables_initializer())
      sess.run(tf.get_collection(tf.GraphKeys.TABLE_INITIALIZERS))
      sess.run([stateful_reader1.prefetch_op, stateful_reader2.prefetch_op])
      maps1, maps2 = sess.run([feature_maps1, feature_maps2])
      state = sess.run(stateful_reader1.next_batch.state('lstm_state_c'))
    # feature maps should be different because states are different
    self.assertFalse(np.all(np.equal(maps1[0], maps2[0])))
    # state should no longer be zero after update
    self.assertTrue(state.any())

  def check_extract_features_returns_correct_shape(
      self, batch_size, image_height, image_width, depth_multiplier,
      pad_to_multiple, expected_feature_map_shapes, unroll_length=1):
    def graph_fn(image_tensor):
      feature_extractor = self._create_feature_extractor(depth_multiplier,
                                                         pad_to_multiple)
      feature_maps = feature_extractor.extract_features(
          image_tensor, unroll_length=unroll_length)
      return feature_maps

    image_tensor = np.random.rand(batch_size, image_height, image_width,
                                  3).astype(np.float32)
    feature_maps = self.execute(graph_fn, [image_tensor])
    for feature_map, expected_shape in itertools.izip(
        feature_maps, expected_feature_map_shapes):
      self.assertAllEqual(feature_map.shape, expected_shape)

  def check_feature_extractor_variables_under_scopes(
      self, depth_multiplier, pad_to_multiple, scope_names):
    g = tf.Graph()
    with g.as_default():
      feature_extractor = self._create_feature_extractor(
          depth_multiplier, pad_to_multiple)
      preprocessed_inputs = tf.placeholder(tf.float32, (4, 320, 320, 3))
      feature_extractor.extract_features(
          preprocessed_inputs, unroll_length=1)
      variables = g.get_collection(tf.GraphKeys.GLOBAL_VARIABLES)
      for variable in variables:
        self.assertTrue(
            any([
                variable.name.startswith(scope_name)
                for scope_name in scope_names
            ]), 'Variable name: ' + variable.name +
            ' is not under any provided scopes: ' + ','.join(scope_names))


if __name__ == '__main__':
  tf.test.main()
