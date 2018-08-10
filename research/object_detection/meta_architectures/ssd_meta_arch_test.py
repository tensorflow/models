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
from absl.testing import parameterized

import numpy as np
import tensorflow as tf

from object_detection.core import anchor_generator
from object_detection.core import balanced_positive_negative_sampler as sampler
from object_detection.core import box_list
from object_detection.core import losses
from object_detection.core import post_processing
from object_detection.core import region_similarity_calculator as sim_calc
from object_detection.core import target_assigner
from object_detection.meta_architectures import ssd_meta_arch
from object_detection.utils import ops
from object_detection.utils import test_case
from object_detection.utils import test_utils

slim = tf.contrib.slim
keras = tf.keras.layers


class FakeSSDFeatureExtractor(ssd_meta_arch.SSDFeatureExtractor):

  def __init__(self):
    super(FakeSSDFeatureExtractor, self).__init__(
        is_training=True,
        depth_multiplier=0,
        min_depth=0,
        pad_to_multiple=1,
        conv_hyperparams_fn=None)

  def preprocess(self, resized_inputs):
    return tf.identity(resized_inputs)

  def extract_features(self, preprocessed_inputs):
    with tf.variable_scope('mock_model'):
      features = slim.conv2d(inputs=preprocessed_inputs, num_outputs=32,
                             kernel_size=1, scope='layer1')
      return [features]


class FakeSSDKerasFeatureExtractor(ssd_meta_arch.SSDKerasFeatureExtractor):

  def __init__(self):
    with tf.name_scope('mock_model'):
      super(FakeSSDKerasFeatureExtractor, self).__init__(
          is_training=True,
          depth_multiplier=0,
          min_depth=0,
          pad_to_multiple=1,
          conv_hyperparams_config=None,
          freeze_batchnorm=False,
          inplace_batchnorm_update=False,
      )

      self._conv = keras.Conv2D(filters=32, kernel_size=1, name='layer1')

  def preprocess(self, resized_inputs):
    return tf.identity(resized_inputs)

  def _extract_features(self, preprocessed_inputs, **kwargs):
    with tf.name_scope('mock_model'):
      return [self._conv(preprocessed_inputs)]


class MockAnchorGenerator2x2(anchor_generator.AnchorGenerator):
  """Sets up a simple 2x2 anchor grid on the unit square."""

  def name_scope(self):
    return 'MockAnchorGenerator'

  def num_anchors_per_location(self):
    return [1]

  def _generate(self, feature_map_shape_list, im_height, im_width):
    return [box_list.BoxList(
        tf.constant([[0, 0, .5, .5],
                     [0, .5, .5, 1],
                     [.5, 0, 1, .5],
                     [1., 1., 1.5, 1.5]  # Anchor that is outside clip_window.
                    ], tf.float32))]

  def num_anchors(self):
    return 4


def _get_value_for_matching_key(dictionary, suffix):
  for key in dictionary.keys():
    if key.endswith(suffix):
      return dictionary[key]
  raise ValueError('key not found {}'.format(suffix))


@parameterized.parameters(
    {'use_keras': False},
    {'use_keras': True},
)
class SsdMetaArchTest(test_case.TestCase, parameterized.TestCase):

  def _create_model(self,
                    apply_hard_mining=True,
                    normalize_loc_loss_by_codesize=False,
                    add_background_class=True,
                    random_example_sampling=False,
                    weight_regression_loss_by_score=False,
                    use_expected_classification_loss_under_sampling=False,
                    minimum_negative_sampling=1,
                    desired_negative_sampling_ratio=3,
                    use_keras=False):
    is_training = False
    num_classes = 1
    mock_anchor_generator = MockAnchorGenerator2x2()
    if use_keras:
      mock_box_predictor = test_utils.MockKerasBoxPredictor(
          is_training, num_classes)
    else:
      mock_box_predictor = test_utils.MockBoxPredictor(
          is_training, num_classes)
    mock_box_coder = test_utils.MockBoxCoder()
    if use_keras:
      fake_feature_extractor = FakeSSDKerasFeatureExtractor()
    else:
      fake_feature_extractor = FakeSSDFeatureExtractor()
    mock_matcher = test_utils.MockMatcher()
    region_similarity_calculator = sim_calc.IouSimilarity()
    encode_background_as_zeros = False
    def image_resizer_fn(image):
      return [tf.identity(image), tf.shape(image)]

    classification_loss = losses.WeightedSigmoidClassificationLoss()
    localization_loss = losses.WeightedSmoothL1LocalizationLoss()
    non_max_suppression_fn = functools.partial(
        post_processing.batch_multiclass_non_max_suppression,
        score_thresh=-20.0,
        iou_thresh=1.0,
        max_size_per_class=5,
        max_total_size=5)
    classification_loss_weight = 1.0
    localization_loss_weight = 1.0
    negative_class_weight = 1.0
    normalize_loss_by_num_matches = False

    hard_example_miner = None
    if apply_hard_mining:
      # This hard example miner is expected to be a no-op.
      hard_example_miner = losses.HardExampleMiner(
          num_hard_examples=None,
          iou_threshold=1.0)

    random_example_sampler = None
    if random_example_sampling:
      random_example_sampler = sampler.BalancedPositiveNegativeSampler(
          positive_fraction=0.5)

    target_assigner_instance = target_assigner.TargetAssigner(
        region_similarity_calculator,
        mock_matcher,
        mock_box_coder,
        negative_class_weight=negative_class_weight,
        weight_regression_loss_by_score=weight_regression_loss_by_score)

    expected_classification_loss_under_sampling = None
    if use_expected_classification_loss_under_sampling:
      expected_classification_loss_under_sampling = functools.partial(
          ops.expected_classification_loss_under_sampling,
          minimum_negative_sampling=minimum_negative_sampling,
          desired_negative_sampling_ratio=desired_negative_sampling_ratio)

    code_size = 4
    model = ssd_meta_arch.SSDMetaArch(
        is_training,
        mock_anchor_generator,
        mock_box_predictor,
        mock_box_coder,
        fake_feature_extractor,
        mock_matcher,
        region_similarity_calculator,
        encode_background_as_zeros,
        negative_class_weight,
        image_resizer_fn,
        non_max_suppression_fn,
        tf.identity,
        classification_loss,
        localization_loss,
        classification_loss_weight,
        localization_loss_weight,
        normalize_loss_by_num_matches,
        hard_example_miner,
        target_assigner_instance=target_assigner_instance,
        add_summaries=False,
        normalize_loc_loss_by_codesize=normalize_loc_loss_by_codesize,
        freeze_batchnorm=False,
        inplace_batchnorm_update=False,
        add_background_class=add_background_class,
        random_example_sampler=random_example_sampler,
        expected_classification_loss_under_sampling=
        expected_classification_loss_under_sampling)
    return model, num_classes, mock_anchor_generator.num_anchors(), code_size

  def test_preprocess_preserves_shapes_with_dynamic_input_image(
      self, use_keras):
    image_shapes = [(3, None, None, 3),
                    (None, 10, 10, 3),
                    (None, None, None, 3)]
    model, _, _, _ = self._create_model(use_keras=use_keras)
    for image_shape in image_shapes:
      image_placeholder = tf.placeholder(tf.float32, shape=image_shape)
      preprocessed_inputs, _ = model.preprocess(image_placeholder)
      self.assertAllEqual(preprocessed_inputs.shape.as_list(), image_shape)

  def test_preprocess_preserves_shape_with_static_input_image(self, use_keras):
    def graph_fn(input_image):
      model, _, _, _ = self._create_model(use_keras=use_keras)
      return model.preprocess(input_image)
    input_image = np.random.rand(2, 3, 3, 3).astype(np.float32)
    preprocessed_inputs, _ = self.execute(graph_fn, [input_image])
    self.assertAllEqual(preprocessed_inputs.shape, [2, 3, 3, 3])

  def test_predict_result_shapes_on_image_with_dynamic_shape(self, use_keras):
    batch_size = 3
    image_size = 2
    input_shapes = [(None, image_size, image_size, 3),
                    (batch_size, None, None, 3),
                    (None, None, None, 3)]

    for input_shape in input_shapes:
      tf_graph = tf.Graph()
      with tf_graph.as_default():
        model, num_classes, num_anchors, code_size = self._create_model(
            use_keras=use_keras)
        preprocessed_input_placeholder = tf.placeholder(tf.float32,
                                                        shape=input_shape)
        prediction_dict = model.predict(
            preprocessed_input_placeholder, true_image_shapes=None)

        self.assertIn('box_encodings', prediction_dict)
        self.assertIn('class_predictions_with_background', prediction_dict)
        self.assertIn('feature_maps', prediction_dict)
        self.assertIn('anchors', prediction_dict)

        init_op = tf.global_variables_initializer()
      with self.test_session(graph=tf_graph) as sess:
        sess.run(init_op)
        prediction_out = sess.run(prediction_dict,
                                  feed_dict={
                                      preprocessed_input_placeholder:
                                      np.random.uniform(
                                          size=(batch_size, 2, 2, 3))})
      expected_box_encodings_shape_out = (batch_size, num_anchors, code_size)
      expected_class_predictions_with_background_shape_out = (batch_size,
                                                              num_anchors,
                                                              num_classes + 1)

      self.assertAllEqual(prediction_out['box_encodings'].shape,
                          expected_box_encodings_shape_out)
      self.assertAllEqual(
          prediction_out['class_predictions_with_background'].shape,
          expected_class_predictions_with_background_shape_out)

  def test_predict_result_shapes_on_image_with_static_shape(self, use_keras):

    with tf.Graph().as_default():
      _, num_classes, num_anchors, code_size = self._create_model(
          use_keras=use_keras)

    def graph_fn(input_image):
      model, _, _, _ = self._create_model()
      predictions = model.predict(input_image, true_image_shapes=None)
      return (predictions['box_encodings'],
              predictions['class_predictions_with_background'],
              predictions['feature_maps'],
              predictions['anchors'])
    batch_size = 3
    image_size = 2
    channels = 3
    input_image = np.random.rand(batch_size, image_size, image_size,
                                 channels).astype(np.float32)
    expected_box_encodings_shape = (batch_size, num_anchors, code_size)
    expected_class_predictions_shape = (batch_size, num_anchors, num_classes+1)
    (box_encodings, class_predictions, _, _) = self.execute(graph_fn,
                                                            [input_image])
    self.assertAllEqual(box_encodings.shape, expected_box_encodings_shape)
    self.assertAllEqual(class_predictions.shape,
                        expected_class_predictions_shape)

  def test_postprocess_results_are_correct(self, use_keras):
    batch_size = 2
    image_size = 2
    input_shapes = [(batch_size, image_size, image_size, 3),
                    (None, image_size, image_size, 3),
                    (batch_size, None, None, 3),
                    (None, None, None, 3)]

    expected_boxes = [
        [
            [0, 0, .5, .5],
            [0, .5, .5, 1],
            [.5, 0, 1, .5],
            [0, 0, 0, 0],  # pruned prediction
            [0, 0, 0, 0]
        ],  # padding
        [
            [0, 0, .5, .5],
            [0, .5, .5, 1],
            [.5, 0, 1, .5],
            [0, 0, 0, 0],  # pruned prediction
            [0, 0, 0, 0]
        ]
    ]  # padding
    expected_scores = [[0, 0, 0, 0, 0], [0, 0, 0, 0, 0]]
    expected_classes = [[0, 0, 0, 0, 0], [0, 0, 0, 0, 0]]
    expected_num_detections = np.array([3, 3])

    for input_shape in input_shapes:
      tf_graph = tf.Graph()
      with tf_graph.as_default():
        model, _, _, _ = self._create_model(use_keras=use_keras)
        input_placeholder = tf.placeholder(tf.float32, shape=input_shape)
        preprocessed_inputs, true_image_shapes = model.preprocess(
            input_placeholder)
        prediction_dict = model.predict(preprocessed_inputs,
                                        true_image_shapes)
        detections = model.postprocess(prediction_dict, true_image_shapes)
        self.assertIn('detection_boxes', detections)
        self.assertIn('detection_scores', detections)
        self.assertIn('detection_classes', detections)
        self.assertIn('num_detections', detections)
        init_op = tf.global_variables_initializer()
      with self.test_session(graph=tf_graph) as sess:
        sess.run(init_op)
        detections_out = sess.run(detections,
                                  feed_dict={
                                      input_placeholder:
                                      np.random.uniform(
                                          size=(batch_size, 2, 2, 3))})
      for image_idx in range(batch_size):
        self.assertTrue(
            test_utils.first_rows_close_as_set(
                detections_out['detection_boxes'][image_idx].tolist(),
                expected_boxes[image_idx]))
      self.assertAllClose(detections_out['detection_scores'], expected_scores)
      self.assertAllClose(detections_out['detection_classes'], expected_classes)
      self.assertAllClose(detections_out['num_detections'],
                          expected_num_detections)

  def test_loss_results_are_correct(self, use_keras):

    with tf.Graph().as_default():
      _, num_classes, num_anchors, _ = self._create_model(use_keras=use_keras)
    def graph_fn(preprocessed_tensor, groundtruth_boxes1, groundtruth_boxes2,
                 groundtruth_classes1, groundtruth_classes2):
      groundtruth_boxes_list = [groundtruth_boxes1, groundtruth_boxes2]
      groundtruth_classes_list = [groundtruth_classes1, groundtruth_classes2]
      model, _, _, _ = self._create_model(apply_hard_mining=False)
      model.provide_groundtruth(groundtruth_boxes_list,
                                groundtruth_classes_list)
      prediction_dict = model.predict(preprocessed_tensor,
                                      true_image_shapes=None)
      loss_dict = model.loss(prediction_dict, true_image_shapes=None)
      return (
          _get_value_for_matching_key(loss_dict, 'Loss/localization_loss'),
          _get_value_for_matching_key(loss_dict, 'Loss/classification_loss'))

    batch_size = 2
    preprocessed_input = np.random.rand(batch_size, 2, 2, 3).astype(np.float32)
    groundtruth_boxes1 = np.array([[0, 0, .5, .5]], dtype=np.float32)
    groundtruth_boxes2 = np.array([[0, 0, .5, .5]], dtype=np.float32)
    groundtruth_classes1 = np.array([[1]], dtype=np.float32)
    groundtruth_classes2 = np.array([[1]], dtype=np.float32)
    expected_localization_loss = 0.0
    expected_classification_loss = (batch_size * num_anchors
                                    * (num_classes+1) * np.log(2.0))
    (localization_loss,
     classification_loss) = self.execute(graph_fn, [preprocessed_input,
                                                    groundtruth_boxes1,
                                                    groundtruth_boxes2,
                                                    groundtruth_classes1,
                                                    groundtruth_classes2])
    self.assertAllClose(localization_loss, expected_localization_loss)
    self.assertAllClose(classification_loss, expected_classification_loss)

  def test_loss_results_are_correct_with_normalize_by_codesize_true(
      self, use_keras):

    with tf.Graph().as_default():
      _, _, _, _ = self._create_model(use_keras=use_keras)
    def graph_fn(preprocessed_tensor, groundtruth_boxes1, groundtruth_boxes2,
                 groundtruth_classes1, groundtruth_classes2):
      groundtruth_boxes_list = [groundtruth_boxes1, groundtruth_boxes2]
      groundtruth_classes_list = [groundtruth_classes1, groundtruth_classes2]
      model, _, _, _ = self._create_model(apply_hard_mining=False,
                                          normalize_loc_loss_by_codesize=True,
                                          use_keras=use_keras)
      model.provide_groundtruth(groundtruth_boxes_list,
                                groundtruth_classes_list)
      prediction_dict = model.predict(preprocessed_tensor,
                                      true_image_shapes=None)
      loss_dict = model.loss(prediction_dict, true_image_shapes=None)
      return (_get_value_for_matching_key(loss_dict, 'Loss/localization_loss'),)

    batch_size = 2
    preprocessed_input = np.random.rand(batch_size, 2, 2, 3).astype(np.float32)
    groundtruth_boxes1 = np.array([[0, 0, 1, 1]], dtype=np.float32)
    groundtruth_boxes2 = np.array([[0, 0, 1, 1]], dtype=np.float32)
    groundtruth_classes1 = np.array([[1]], dtype=np.float32)
    groundtruth_classes2 = np.array([[1]], dtype=np.float32)
    expected_localization_loss = 0.5 / 4
    localization_loss = self.execute(graph_fn, [preprocessed_input,
                                                groundtruth_boxes1,
                                                groundtruth_boxes2,
                                                groundtruth_classes1,
                                                groundtruth_classes2])
    self.assertAllClose(localization_loss, expected_localization_loss)

  def test_loss_results_are_correct_with_hard_example_mining(self, use_keras):

    with tf.Graph().as_default():
      _, num_classes, num_anchors, _ = self._create_model(use_keras=use_keras)
    def graph_fn(preprocessed_tensor, groundtruth_boxes1, groundtruth_boxes2,
                 groundtruth_classes1, groundtruth_classes2):
      groundtruth_boxes_list = [groundtruth_boxes1, groundtruth_boxes2]
      groundtruth_classes_list = [groundtruth_classes1, groundtruth_classes2]
      model, _, _, _ = self._create_model()
      model.provide_groundtruth(groundtruth_boxes_list,
                                groundtruth_classes_list)
      prediction_dict = model.predict(preprocessed_tensor,
                                      true_image_shapes=None)
      loss_dict = model.loss(prediction_dict, true_image_shapes=None)
      return (
          _get_value_for_matching_key(loss_dict, 'Loss/localization_loss'),
          _get_value_for_matching_key(loss_dict, 'Loss/classification_loss'))

    batch_size = 2
    preprocessed_input = np.random.rand(batch_size, 2, 2, 3).astype(np.float32)
    groundtruth_boxes1 = np.array([[0, 0, .5, .5]], dtype=np.float32)
    groundtruth_boxes2 = np.array([[0, 0, .5, .5]], dtype=np.float32)
    groundtruth_classes1 = np.array([[1]], dtype=np.float32)
    groundtruth_classes2 = np.array([[1]], dtype=np.float32)
    expected_localization_loss = 0.0
    expected_classification_loss = (batch_size * num_anchors
                                    * (num_classes+1) * np.log(2.0))
    (localization_loss, classification_loss) = self.execute_cpu(
        graph_fn, [
            preprocessed_input, groundtruth_boxes1, groundtruth_boxes2,
            groundtruth_classes1, groundtruth_classes2
        ])
    self.assertAllClose(localization_loss, expected_localization_loss)
    self.assertAllClose(classification_loss, expected_classification_loss)

  def test_loss_results_are_correct_without_add_background_class(
      self, use_keras):

    with tf.Graph().as_default():
      _, num_classes, num_anchors, _ = self._create_model(
          add_background_class=False, use_keras=use_keras)

    def graph_fn(preprocessed_tensor, groundtruth_boxes1, groundtruth_boxes2,
                 groundtruth_classes1, groundtruth_classes2):
      groundtruth_boxes_list = [groundtruth_boxes1, groundtruth_boxes2]
      groundtruth_classes_list = [groundtruth_classes1, groundtruth_classes2]
      model, _, _, _ = self._create_model(
          apply_hard_mining=False, add_background_class=False,
          use_keras=use_keras)
      model.provide_groundtruth(groundtruth_boxes_list,
                                groundtruth_classes_list)
      prediction_dict = model.predict(
          preprocessed_tensor, true_image_shapes=None)
      loss_dict = model.loss(prediction_dict, true_image_shapes=None)
      return (loss_dict['Loss/localization_loss'],
              loss_dict['Loss/classification_loss'])

    batch_size = 2
    preprocessed_input = np.random.rand(batch_size, 2, 2, 3).astype(np.float32)
    groundtruth_boxes1 = np.array([[0, 0, .5, .5]], dtype=np.float32)
    groundtruth_boxes2 = np.array([[0, 0, .5, .5]], dtype=np.float32)
    groundtruth_classes1 = np.array([[0, 1]], dtype=np.float32)
    groundtruth_classes2 = np.array([[0, 1]], dtype=np.float32)
    expected_localization_loss = 0.0
    expected_classification_loss = (
        batch_size * num_anchors * (num_classes + 1) * np.log(2.0))
    (localization_loss, classification_loss) = self.execute(
        graph_fn, [
            preprocessed_input, groundtruth_boxes1, groundtruth_boxes2,
            groundtruth_classes1, groundtruth_classes2
        ])

    self.assertAllClose(localization_loss, expected_localization_loss)
    self.assertAllClose(classification_loss, expected_classification_loss)

  def test_loss_with_expected_classification_loss(self, use_keras):

    with tf.Graph().as_default():
      _, num_classes, num_anchors, _ = self._create_model(use_keras=use_keras)

    def graph_fn(preprocessed_tensor, groundtruth_boxes1, groundtruth_boxes2,
                 groundtruth_classes1, groundtruth_classes2):
      groundtruth_boxes_list = [groundtruth_boxes1, groundtruth_boxes2]
      groundtruth_classes_list = [groundtruth_classes1, groundtruth_classes2]
      model, _, _, _ = self._create_model(
          apply_hard_mining=False,
          add_background_class=True,
          use_expected_classification_loss_under_sampling=True,
          minimum_negative_sampling=1,
          desired_negative_sampling_ratio=desired_negative_sampling_ratio)
      model.provide_groundtruth(groundtruth_boxes_list,
                                groundtruth_classes_list)
      prediction_dict = model.predict(
          preprocessed_tensor, true_image_shapes=None)
      loss_dict = model.loss(prediction_dict, true_image_shapes=None)
      return (loss_dict['Loss/localization_loss'],
              loss_dict['Loss/classification_loss'])

    batch_size = 2
    desired_negative_sampling_ratio = 4
    preprocessed_input = np.random.rand(batch_size, 2, 2, 3).astype(np.float32)
    groundtruth_boxes1 = np.array([[0, 0, .5, .5]], dtype=np.float32)
    groundtruth_boxes2 = np.array([[0, 0, .5, .5]], dtype=np.float32)
    groundtruth_classes1 = np.array([[1]], dtype=np.float32)
    groundtruth_classes2 = np.array([[1]], dtype=np.float32)
    expected_localization_loss = 0.0

    expected_classification_loss = (
        batch_size * (desired_negative_sampling_ratio * num_anchors +
                      num_classes * num_anchors) * np.log(2.0))
    (localization_loss, classification_loss) = self.execute(
        graph_fn, [
            preprocessed_input, groundtruth_boxes1, groundtruth_boxes2,
            groundtruth_classes1, groundtruth_classes2
        ])

    self.assertAllClose(localization_loss, expected_localization_loss)
    self.assertAllClose(classification_loss, expected_classification_loss)

  def test_loss_results_are_correct_with_weight_regression_loss_by_score(
      self, use_keras):

    with tf.Graph().as_default():
      _, num_classes, num_anchors, _ = self._create_model(
          use_keras=use_keras,
          add_background_class=False,
          weight_regression_loss_by_score=True)

    def graph_fn(preprocessed_tensor, groundtruth_boxes1, groundtruth_boxes2,
                 groundtruth_classes1, groundtruth_classes2):
      groundtruth_boxes_list = [groundtruth_boxes1, groundtruth_boxes2]
      groundtruth_classes_list = [groundtruth_classes1, groundtruth_classes2]
      model, _, _, _ = self._create_model(
          use_keras=use_keras,
          apply_hard_mining=False,
          add_background_class=False,
          weight_regression_loss_by_score=True)
      model.provide_groundtruth(groundtruth_boxes_list,
                                groundtruth_classes_list)
      prediction_dict = model.predict(
          preprocessed_tensor, true_image_shapes=None)
      loss_dict = model.loss(prediction_dict, true_image_shapes=None)
      return (loss_dict['Loss/localization_loss'],
              loss_dict['Loss/classification_loss'])

    batch_size = 2
    preprocessed_input = np.random.rand(batch_size, 2, 2, 3).astype(np.float32)
    groundtruth_boxes1 = np.array([[0, 0, 1, 1]], dtype=np.float32)
    groundtruth_boxes2 = np.array([[0, 0, 1, 1]], dtype=np.float32)
    groundtruth_classes1 = np.array([[0, 1]], dtype=np.float32)
    groundtruth_classes2 = np.array([[1, 0]], dtype=np.float32)
    expected_localization_loss = 0.25
    expected_classification_loss = (
        batch_size * num_anchors * (num_classes + 1) * np.log(2.0))
    (localization_loss, classification_loss) = self.execute(
        graph_fn, [
            preprocessed_input, groundtruth_boxes1, groundtruth_boxes2,
            groundtruth_classes1, groundtruth_classes2
        ])
    self.assertAllClose(localization_loss, expected_localization_loss)
    self.assertAllClose(classification_loss, expected_classification_loss)

  def test_restore_map_for_detection_ckpt(self, use_keras):
    model, _, _, _ = self._create_model(use_keras=use_keras)
    model.predict(tf.constant(np.array([[[[0, 0], [1, 1]], [[1, 0], [0, 1]]]],
                                       dtype=np.float32)),
                  true_image_shapes=None)
    init_op = tf.global_variables_initializer()
    saver = tf.train.Saver()
    save_path = self.get_temp_dir()
    with self.test_session() as sess:
      sess.run(init_op)
      saved_model_path = saver.save(sess, save_path)
      var_map = model.restore_map(
          fine_tune_checkpoint_type='detection',
          load_all_detection_checkpoint_vars=False)
      self.assertIsInstance(var_map, dict)
      saver = tf.train.Saver(var_map)
      saver.restore(sess, saved_model_path)
      for var in sess.run(tf.report_uninitialized_variables()):
        self.assertNotIn('FeatureExtractor', var)

  def test_restore_map_for_classification_ckpt(self, use_keras):
    # Define mock tensorflow classification graph and save variables.
    test_graph_classification = tf.Graph()
    with test_graph_classification.as_default():
      image = tf.placeholder(dtype=tf.float32, shape=[1, 20, 20, 3])
      if use_keras:
        with tf.name_scope('mock_model'):
          layer_one = keras.Conv2D(32, kernel_size=1, name='layer1')
          net = layer_one(image)
          layer_two = keras.Conv2D(3, kernel_size=1, name='layer2')
          layer_two(net)
      else:
        with tf.variable_scope('mock_model'):
          net = slim.conv2d(image, num_outputs=32, kernel_size=1,
                            scope='layer1')
          slim.conv2d(net, num_outputs=3, kernel_size=1, scope='layer2')

      init_op = tf.global_variables_initializer()
      saver = tf.train.Saver()
      save_path = self.get_temp_dir()
      with self.test_session(graph=test_graph_classification) as sess:
        sess.run(init_op)
        saved_model_path = saver.save(sess, save_path)

    # Create tensorflow detection graph and load variables from
    # classification checkpoint.
    test_graph_detection = tf.Graph()
    with test_graph_detection.as_default():
      model, _, _, _ = self._create_model(use_keras=use_keras)
      inputs_shape = [2, 2, 2, 3]
      inputs = tf.to_float(tf.random_uniform(
          inputs_shape, minval=0, maxval=255, dtype=tf.int32))
      preprocessed_inputs, true_image_shapes = model.preprocess(inputs)
      prediction_dict = model.predict(preprocessed_inputs, true_image_shapes)
      model.postprocess(prediction_dict, true_image_shapes)
      another_variable = tf.Variable([17.0], name='another_variable')  # pylint: disable=unused-variable
      var_map = model.restore_map(fine_tune_checkpoint_type='classification')
      self.assertNotIn('another_variable', var_map)
      self.assertIsInstance(var_map, dict)
      saver = tf.train.Saver(var_map)
      with self.test_session(graph=test_graph_detection) as sess:
        saver.restore(sess, saved_model_path)
        for var in sess.run(tf.report_uninitialized_variables()):
          self.assertNotIn('FeatureExtractor', var)

  def test_load_all_det_checkpoint_vars(self, use_keras):
    test_graph_detection = tf.Graph()
    with test_graph_detection.as_default():
      model, _, _, _ = self._create_model(use_keras=use_keras)
      inputs_shape = [2, 2, 2, 3]
      inputs = tf.to_float(
          tf.random_uniform(inputs_shape, minval=0, maxval=255, dtype=tf.int32))
      preprocessed_inputs, true_image_shapes = model.preprocess(inputs)
      prediction_dict = model.predict(preprocessed_inputs, true_image_shapes)
      model.postprocess(prediction_dict, true_image_shapes)
      another_variable = tf.Variable([17.0], name='another_variable')  # pylint: disable=unused-variable
      var_map = model.restore_map(
          fine_tune_checkpoint_type='detection',
          load_all_detection_checkpoint_vars=True)
      self.assertIsInstance(var_map, dict)
      self.assertIn('another_variable', var_map)

  def test_loss_results_are_correct_with_random_example_sampling(
      self,
      use_keras):

    with tf.Graph().as_default():
      _, num_classes, num_anchors, _ = self._create_model(
          random_example_sampling=True,
          use_keras=use_keras)
    print num_classes, num_anchors

    def graph_fn(preprocessed_tensor, groundtruth_boxes1, groundtruth_boxes2,
                 groundtruth_classes1, groundtruth_classes2):
      groundtruth_boxes_list = [groundtruth_boxes1, groundtruth_boxes2]
      groundtruth_classes_list = [groundtruth_classes1, groundtruth_classes2]
      model, _, _, _ = self._create_model(random_example_sampling=True,
                                          use_keras=use_keras)
      model.provide_groundtruth(groundtruth_boxes_list,
                                groundtruth_classes_list)
      prediction_dict = model.predict(
          preprocessed_tensor, true_image_shapes=None)
      loss_dict = model.loss(prediction_dict, true_image_shapes=None)
      return (_get_value_for_matching_key(loss_dict, 'Loss/localization_loss'),
              _get_value_for_matching_key(loss_dict,
                                          'Loss/classification_loss'))

    batch_size = 2
    preprocessed_input = np.random.rand(batch_size, 2, 2, 3).astype(np.float32)
    groundtruth_boxes1 = np.array([[0, 0, .5, .5]], dtype=np.float32)
    groundtruth_boxes2 = np.array([[0, 0, .5, .5]], dtype=np.float32)
    groundtruth_classes1 = np.array([[1]], dtype=np.float32)
    groundtruth_classes2 = np.array([[1]], dtype=np.float32)
    expected_localization_loss = 0.0
    # Among 4 anchors (1 positive, 3 negative) in this test, only 2 anchors are
    # selected (1 positive, 1 negative) since random sampler will adjust number
    # of negative examples to make sure positive example fraction in the batch
    # is 0.5.
    expected_classification_loss = (
        batch_size * 2 * (num_classes + 1) * np.log(2.0))
    (localization_loss, classification_loss) = self.execute_cpu(
        graph_fn, [
            preprocessed_input, groundtruth_boxes1, groundtruth_boxes2,
            groundtruth_classes1, groundtruth_classes2
        ])
    self.assertAllClose(localization_loss, expected_localization_loss)
    self.assertAllClose(classification_loss, expected_classification_loss)

if __name__ == '__main__':
  tf.test.main()
