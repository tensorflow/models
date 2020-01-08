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

"""Tests for models.lstm_ssd_mobilenet_v1_feature_extractor."""

import numpy as np
import tensorflow as tf
from tensorflow.contrib import slim as contrib_slim
from tensorflow.contrib import training as contrib_training

from lstm_object_detection.models import lstm_ssd_mobilenet_v1_feature_extractor as feature_extactor
from object_detection.models import ssd_feature_extractor_test

slim = contrib_slim


class LstmSsdMobilenetV1FeatureExtractorTest(
    ssd_feature_extractor_test.SsdFeatureExtractorTestBase):

  def _create_feature_extractor(self,
                                depth_multiplier=1.0,
                                pad_to_multiple=1,
                                is_training=True,
                                use_explicit_padding=False):
    """Constructs a new feature extractor.

    Args:
      depth_multiplier: A float depth multiplier for feature extractor.
      pad_to_multiple: The nearest multiple to zero pad the input height and
        width dimensions to.
      is_training: A boolean whether the network is in training mode.
      use_explicit_padding: A boolean whether to use explicit padding.

    Returns:
      An lstm_ssd_meta_arch.LSTMSSDMobileNetV1FeatureExtractor object.
    """
    min_depth = 32
    extractor = (
        feature_extactor.LSTMSSDMobileNetV1FeatureExtractor(
            is_training,
            depth_multiplier,
            min_depth,
            pad_to_multiple,
            self.conv_hyperparams_fn,
            use_explicit_padding=use_explicit_padding))
    extractor.lstm_state_depth = int(256 * depth_multiplier)
    return extractor

  def test_extract_features_returns_correct_shapes_256(self):
    image_height = 256
    image_width = 256
    depth_multiplier = 1.0
    pad_to_multiple = 1
    batch_size = 5
    expected_feature_map_shape = [(batch_size, 8, 8, 256), (batch_size, 4, 4,
                                                            512),
                                  (batch_size, 2, 2, 256), (batch_size, 1, 1,
                                                            256)]
    self.check_extract_features_returns_correct_shape(
        batch_size,
        image_height,
        image_width,
        depth_multiplier,
        pad_to_multiple,
        expected_feature_map_shape,
        use_explicit_padding=False)
    self.check_extract_features_returns_correct_shape(
        batch_size,
        image_height,
        image_width,
        depth_multiplier,
        pad_to_multiple,
        expected_feature_map_shape,
        use_explicit_padding=True)

  def test_preprocess_returns_correct_value_range(self):
    test_image = np.random.rand(5, 128, 128, 3)
    feature_extractor = self._create_feature_extractor()
    preprocessed_image = feature_extractor.preprocess(test_image)
    self.assertTrue(np.all(np.less_equal(np.abs(preprocessed_image), 1.0)))

  def test_variables_only_created_in_scope(self):
    scope_name = 'MobilenetV1'
    g = tf.Graph()
    with g.as_default():
      preprocessed_inputs = tf.placeholder(tf.float32, (5, 256, 256, 3))
      feature_extractor = self._create_feature_extractor()
      feature_extractor.extract_features(preprocessed_inputs)
      variables = g.get_collection(tf.GraphKeys.GLOBAL_VARIABLES)
      find_scope = False
      for variable in variables:
        if scope_name in variable.name:
          find_scope = True
          break
      self.assertTrue(find_scope)

  def test_lstm_non_zero_state(self):
    init_state = {
        'lstm_state_c': tf.zeros([8, 8, 256]),
        'lstm_state_h': tf.zeros([8, 8, 256]),
        'lstm_state_step': tf.zeros([1])
    }
    seq = {'test': tf.random_uniform([3, 1, 1, 1])}
    stateful_reader = contrib_training.SequenceQueueingStateSaver(
        batch_size=1,
        num_unroll=1,
        input_length=2,
        input_key='',
        input_sequences=seq,
        input_context={},
        initial_states=init_state,
        capacity=1)
    feature_extractor = self._create_feature_extractor()
    image = tf.random_uniform([5, 256, 256, 3])
    with tf.variable_scope('zero_state'):
      feature_map = feature_extractor.extract_features(
          image, stateful_reader.next_batch)
    with tf.Session() as sess:
      sess.run(tf.global_variables_initializer())
      sess.run([stateful_reader.prefetch_op])
      _ = sess.run([feature_map])
      # Update states with the next batch.
      state = sess.run(stateful_reader.next_batch.state('lstm_state_c'))
    # State should no longer be zero after update.
    self.assertTrue(state.any())


if __name__ == '__main__':
  tf.test.main()
