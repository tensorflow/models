# Copyright 2017 The TensorFlow Authors. All Rights Reserved.
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

"""Tests for object_detection.meta_architectures.ssd_meta_arch."""
import functools
import numpy as np
import tensorflow as tf

from tensorflow.python.training import saver as tf_saver
from object_detection.core import anchor_generator
from object_detection.core import box_list
from object_detection.core import losses
from object_detection.core import post_processing
from object_detection.core import region_similarity_calculator as sim_calc
from object_detection.meta_architectures import ssd_meta_arch
from object_detection.utils import test_utils

slim = tf.contrib.slim


class FakeSSDFeatureExtractor(ssd_meta_arch.SSDFeatureExtractor):

  def __init__(self):
    super(FakeSSDFeatureExtractor, self).__init__(
        depth_multiplier=0, min_depth=0, conv_hyperparams=None)

  def preprocess(self, resized_inputs):
    return tf.identity(resized_inputs)

  def extract_features(self, preprocessed_inputs):
    with tf.variable_scope('mock_model'):
      features = slim.conv2d(inputs=preprocessed_inputs, num_outputs=32,
                             kernel_size=[1, 1], scope='layer1')
      return [features]


class MockAnchorGenerator2x2(anchor_generator.AnchorGenerator):
  """Sets up a simple 2x2 anchor grid on the unit square."""

  def name_scope(self):
    return 'MockAnchorGenerator'

  def num_anchors_per_location(self):
    return [1]

  def _generate(self, feature_map_shape_list):
    return box_list.BoxList(
        tf.constant([[0, 0, .5, .5],
                     [0, .5, .5, 1],
                     [.5, 0, 1, .5],
                     [.5, .5, 1, 1]], tf.float32))


class SsdMetaArchTest(tf.test.TestCase):

  def setUp(self):
    """Set up mock SSD model.

    Here we set up a simple mock SSD model that will always predict 4
    detections that happen to always be exactly the anchors that are set up
    in the above MockAnchorGenerator.  Because we let max_detections=5,
    we will also always end up with an extra padded row in the detection
    results.
    """
    is_training = False
    self._num_classes = 1
    mock_anchor_generator = MockAnchorGenerator2x2()
    mock_box_predictor = test_utils.MockBoxPredictor(
        is_training, self._num_classes)
    mock_box_coder = test_utils.MockBoxCoder()
    fake_feature_extractor = FakeSSDFeatureExtractor()
    mock_matcher = test_utils.MockMatcher()
    region_similarity_calculator = sim_calc.IouSimilarity()

    def image_resizer_fn(image):
      return tf.identity(image)

    classification_loss = losses.WeightedSigmoidClassificationLoss(
        anchorwise_output=True)
    localization_loss = losses.WeightedSmoothL1LocalizationLoss(
        anchorwise_output=True)
    non_max_suppression_fn = functools.partial(
        post_processing.batch_multiclass_non_max_suppression,
        score_thresh=-20.0,
        iou_thresh=1.0,
        max_size_per_class=5,
        max_total_size=5)
    classification_loss_weight = 1.0
    localization_loss_weight = 1.0
    normalize_loss_by_num_matches = False

    # This hard example miner is expected to be a no-op.
    hard_example_miner = losses.HardExampleMiner(
        num_hard_examples=None,
        iou_threshold=1.0)

    self._num_anchors = 4
    self._code_size = 4
    self._model = ssd_meta_arch.SSDMetaArch(
        is_training, mock_anchor_generator, mock_box_predictor, mock_box_coder,
        fake_feature_extractor, mock_matcher, region_similarity_calculator,
        image_resizer_fn, non_max_suppression_fn, tf.identity,
        classification_loss, localization_loss, classification_loss_weight,
        localization_loss_weight, normalize_loss_by_num_matches,
        hard_example_miner)

  def test_predict_results_have_correct_keys_and_shapes(self):
    batch_size = 3
    preprocessed_input = tf.random_uniform((batch_size, 2, 2, 3),
                                           dtype=tf.float32)
    prediction_dict = self._model.predict(preprocessed_input)

    self.assertTrue('box_encodings' in prediction_dict)
    self.assertTrue('class_predictions_with_background' in prediction_dict)
    self.assertTrue('feature_maps' in prediction_dict)

    expected_box_encodings_shape_out = (
        batch_size, self._num_anchors, self._code_size)
    expected_class_predictions_with_background_shape_out = (
        batch_size, self._num_anchors, self._num_classes+1)
    init_op = tf.global_variables_initializer()
    with self.test_session() as sess:
      sess.run(init_op)
      prediction_out = sess.run(prediction_dict)
      self.assertAllEqual(prediction_out['box_encodings'].shape,
                          expected_box_encodings_shape_out)
      self.assertAllEqual(
          prediction_out['class_predictions_with_background'].shape,
          expected_class_predictions_with_background_shape_out)

  def test_postprocess_results_are_correct(self):
    batch_size = 2
    preprocessed_input = tf.random_uniform((batch_size, 2, 2, 3),
                                           dtype=tf.float32)
    prediction_dict = self._model.predict(preprocessed_input)
    detections = self._model.postprocess(prediction_dict)

    expected_boxes = np.array([[[0, 0, .5, .5],
                                [0, .5, .5, 1],
                                [.5, 0, 1, .5],
                                [.5, .5, 1, 1],
                                [0, 0, 0, 0]],
                               [[0, 0, .5, .5],
                                [0, .5, .5, 1],
                                [.5, 0, 1, .5],
                                [.5, .5, 1, 1],
                                [0, 0, 0, 0]]])
    expected_scores = np.array([[0, 0, 0, 0, 0],
                                [0, 0, 0, 0, 0]])
    expected_classes = np.array([[0, 0, 0, 0, 0],
                                 [0, 0, 0, 0, 0]])
    expected_num_detections = np.array([4, 4])

    self.assertTrue('detection_boxes' in detections)
    self.assertTrue('detection_scores' in detections)
    self.assertTrue('detection_classes' in detections)
    self.assertTrue('num_detections' in detections)

    init_op = tf.global_variables_initializer()
    with self.test_session() as sess:
      sess.run(init_op)
      detections_out = sess.run(detections)
      self.assertAllClose(detections_out['detection_boxes'], expected_boxes)
      self.assertAllClose(detections_out['detection_scores'], expected_scores)
      self.assertAllClose(detections_out['detection_classes'], expected_classes)
      self.assertAllClose(detections_out['num_detections'],
                          expected_num_detections)

  def test_loss_results_are_correct(self):
    batch_size = 2
    preprocessed_input = tf.random_uniform((batch_size, 2, 2, 3),
                                           dtype=tf.float32)
    groundtruth_boxes_list = [tf.constant([[0, 0, .5, .5]], dtype=tf.float32),
                              tf.constant([[0, 0, .5, .5]], dtype=tf.float32)]
    groundtruth_classes_list = [tf.constant([[1]], dtype=tf.float32),
                                tf.constant([[1]], dtype=tf.float32)]
    self._model.provide_groundtruth(groundtruth_boxes_list,
                                    groundtruth_classes_list)
    prediction_dict = self._model.predict(preprocessed_input)
    loss_dict = self._model.loss(prediction_dict)

    self.assertTrue('localization_loss' in loss_dict)
    self.assertTrue('classification_loss' in loss_dict)

    expected_localization_loss = 0.0
    expected_classification_loss = (batch_size * self._num_anchors
                                    * (self._num_classes+1) * np.log(2.0))
    init_op = tf.global_variables_initializer()
    with self.test_session() as sess:
      sess.run(init_op)
      losses_out = sess.run(loss_dict)

      self.assertAllClose(losses_out['localization_loss'],
                          expected_localization_loss)
      self.assertAllClose(losses_out['classification_loss'],
                          expected_classification_loss)

  def test_restore_fn_detection(self):
    init_op = tf.global_variables_initializer()
    saver = tf_saver.Saver()
    save_path = self.get_temp_dir()
    with self.test_session() as sess:
      sess.run(init_op)
      saved_model_path = saver.save(sess, save_path)
      restore_fn = self._model.restore_fn(saved_model_path,
                                          from_detection_checkpoint=True)
      restore_fn(sess)
      for var in sess.run(tf.report_uninitialized_variables()):
        self.assertNotIn('FeatureExtractor', var.name)

  def test_restore_fn_classification(self):
    # Define mock tensorflow classification graph and save variables.
    test_graph_classification = tf.Graph()
    with test_graph_classification.as_default():
      image = tf.placeholder(dtype=tf.float32, shape=[1, 20, 20, 3])
      with tf.variable_scope('mock_model'):
        net = slim.conv2d(image, num_outputs=32, kernel_size=1, scope='layer1')
        slim.conv2d(net, num_outputs=3, kernel_size=1, scope='layer2')

      init_op = tf.global_variables_initializer()
      saver = tf.train.Saver()
      save_path = self.get_temp_dir()
      with self.test_session() as sess:
        sess.run(init_op)
        saved_model_path = saver.save(sess, save_path)

    # Create tensorflow detection graph and load variables from
    # classification checkpoint.
    test_graph_detection = tf.Graph()
    with test_graph_detection.as_default():
      inputs_shape = [2, 2, 2, 3]
      inputs = tf.to_float(tf.random_uniform(
          inputs_shape, minval=0, maxval=255, dtype=tf.int32))
      preprocessed_inputs = self._model.preprocess(inputs)
      prediction_dict = self._model.predict(preprocessed_inputs)
      self._model.postprocess(prediction_dict)
      restore_fn = self._model.restore_fn(saved_model_path,
                                          from_detection_checkpoint=False)
      with self.test_session() as sess:
        restore_fn(sess)
        for var in sess.run(tf.report_uninitialized_variables()):
          self.assertNotIn('FeatureExtractor', var.name)


if __name__ == '__main__':
  tf.test.main()
