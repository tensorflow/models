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

"""Tests for object_detection.meta_architectures.faster_rcnn_meta_arch."""
import numpy as np
import tensorflow as tf
from google.protobuf import text_format
from object_detection.anchor_generators import grid_anchor_generator
from object_detection.builders import box_predictor_builder
from object_detection.builders import hyperparams_builder
from object_detection.builders import post_processing_builder
from object_detection.core import losses
from object_detection.meta_architectures import faster_rcnn_meta_arch
from object_detection.protos import box_predictor_pb2
from object_detection.protos import hyperparams_pb2
from object_detection.protos import post_processing_pb2

slim = tf.contrib.slim
BOX_CODE_SIZE = 4


class FakeFasterRCNNFeatureExtractor(
    faster_rcnn_meta_arch.FasterRCNNFeatureExtractor):
  """Fake feature extracture to use in tests."""

  def __init__(self):
    super(FakeFasterRCNNFeatureExtractor, self).__init__(
        is_training=False,
        first_stage_features_stride=32,
        reuse_weights=None,
        weight_decay=0.0)

  def preprocess(self, resized_inputs):
    return tf.identity(resized_inputs)

  def _extract_proposal_features(self, preprocessed_inputs, scope):
    with tf.variable_scope('mock_model'):
      return 0 * slim.conv2d(preprocessed_inputs,
                             num_outputs=3, kernel_size=1, scope='layer1')

  def _extract_box_classifier_features(self, proposal_feature_maps, scope):
    with tf.variable_scope('mock_model'):
      return 0 * slim.conv2d(proposal_feature_maps,
                             num_outputs=3, kernel_size=1, scope='layer2')


class FasterRCNNMetaArchTestBase(tf.test.TestCase):
  """Base class to test Faster R-CNN and R-FCN meta architectures."""

  def _build_arg_scope_with_hyperparams(self,
                                        hyperparams_text_proto,
                                        is_training):
    hyperparams = hyperparams_pb2.Hyperparams()
    text_format.Merge(hyperparams_text_proto, hyperparams)
    return hyperparams_builder.build(hyperparams, is_training=is_training)

  def _get_second_stage_box_predictor_text_proto(self):
    box_predictor_text_proto = """
      mask_rcnn_box_predictor {
        fc_hyperparams {
          op: FC
          activation: NONE
          regularizer {
            l2_regularizer {
              weight: 0.0005
            }
          }
          initializer {
            variance_scaling_initializer {
              factor: 1.0
              uniform: true
              mode: FAN_AVG
            }
          }
        }
      }
    """
    return box_predictor_text_proto

  def _get_second_stage_box_predictor(self, num_classes, is_training):
    box_predictor_proto = box_predictor_pb2.BoxPredictor()
    text_format.Merge(self._get_second_stage_box_predictor_text_proto(),
                      box_predictor_proto)
    return box_predictor_builder.build(
        hyperparams_builder.build,
        box_predictor_proto,
        num_classes=num_classes,
        is_training=is_training)

  def _get_model(self, box_predictor, **common_kwargs):
    return faster_rcnn_meta_arch.FasterRCNNMetaArch(
        initial_crop_size=3,
        maxpool_kernel_size=1,
        maxpool_stride=1,
        second_stage_mask_rcnn_box_predictor=box_predictor,
        **common_kwargs)

  def _build_model(self,
                   is_training,
                   first_stage_only,
                   second_stage_batch_size,
                   first_stage_max_proposals=8,
                   num_classes=2,
                   hard_mining=False):

    def image_resizer_fn(image):
      return tf.identity(image)

    # anchors in this test are designed so that a subset of anchors are inside
    # the image and a subset of anchors are outside.
    first_stage_anchor_scales = (0.001, 0.005, 0.1)
    first_stage_anchor_aspect_ratios = (0.5, 1.0, 2.0)
    first_stage_anchor_strides = (1, 1)
    first_stage_anchor_generator = grid_anchor_generator.GridAnchorGenerator(
        first_stage_anchor_scales,
        first_stage_anchor_aspect_ratios,
        anchor_stride=first_stage_anchor_strides)

    fake_feature_extractor = FakeFasterRCNNFeatureExtractor()

    first_stage_box_predictor_hyperparams_text_proto = """
      op: CONV
      activation: RELU
      regularizer {
        l2_regularizer {
          weight: 0.00004
        }
      }
      initializer {
        truncated_normal_initializer {
          stddev: 0.03
        }
      }
    """
    first_stage_box_predictor_arg_scope = (
        self._build_arg_scope_with_hyperparams(
            first_stage_box_predictor_hyperparams_text_proto, is_training))

    first_stage_box_predictor_kernel_size = 3
    first_stage_atrous_rate = 1
    first_stage_box_predictor_depth = 512
    first_stage_minibatch_size = 3
    first_stage_positive_balance_fraction = .5

    first_stage_nms_score_threshold = -1.0
    first_stage_nms_iou_threshold = 1.0
    first_stage_max_proposals = first_stage_max_proposals

    first_stage_localization_loss_weight = 1.0
    first_stage_objectness_loss_weight = 1.0

    post_processing_text_proto = """
      batch_non_max_suppression {
        score_threshold: -20.0
        iou_threshold: 1.0
        max_detections_per_class: 5
        max_total_detections: 5
      }
    """
    post_processing_config = post_processing_pb2.PostProcessing()
    text_format.Merge(post_processing_text_proto, post_processing_config)
    second_stage_non_max_suppression_fn, _ = post_processing_builder.build(
        post_processing_config)
    second_stage_balance_fraction = 1.0

    second_stage_score_conversion_fn = tf.identity
    second_stage_localization_loss_weight = 1.0
    second_stage_classification_loss_weight = 1.0

    hard_example_miner = None
    if hard_mining:
      hard_example_miner = losses.HardExampleMiner(
          num_hard_examples=1,
          iou_threshold=0.99,
          loss_type='both',
          cls_loss_weight=second_stage_classification_loss_weight,
          loc_loss_weight=second_stage_localization_loss_weight,
          max_negatives_per_positive=None)

    common_kwargs = {
        'is_training': is_training,
        'num_classes': num_classes,
        'image_resizer_fn': image_resizer_fn,
        'feature_extractor': fake_feature_extractor,
        'first_stage_only': first_stage_only,
        'first_stage_anchor_generator': first_stage_anchor_generator,
        'first_stage_atrous_rate': first_stage_atrous_rate,
        'first_stage_box_predictor_arg_scope':
        first_stage_box_predictor_arg_scope,
        'first_stage_box_predictor_kernel_size':
        first_stage_box_predictor_kernel_size,
        'first_stage_box_predictor_depth': first_stage_box_predictor_depth,
        'first_stage_minibatch_size': first_stage_minibatch_size,
        'first_stage_positive_balance_fraction':
        first_stage_positive_balance_fraction,
        'first_stage_nms_score_threshold': first_stage_nms_score_threshold,
        'first_stage_nms_iou_threshold': first_stage_nms_iou_threshold,
        'first_stage_max_proposals': first_stage_max_proposals,
        'first_stage_localization_loss_weight':
        first_stage_localization_loss_weight,
        'first_stage_objectness_loss_weight':
        first_stage_objectness_loss_weight,
        'second_stage_batch_size': second_stage_batch_size,
        'second_stage_balance_fraction': second_stage_balance_fraction,
        'second_stage_non_max_suppression_fn':
        second_stage_non_max_suppression_fn,
        'second_stage_score_conversion_fn': second_stage_score_conversion_fn,
        'second_stage_localization_loss_weight':
        second_stage_localization_loss_weight,
        'second_stage_classification_loss_weight':
        second_stage_classification_loss_weight,
        'hard_example_miner': hard_example_miner}

    return self._get_model(self._get_second_stage_box_predictor(
        num_classes=num_classes, is_training=is_training), **common_kwargs)

  def test_predict_gives_correct_shapes_in_inference_mode_first_stage_only(
      self):
    test_graph = tf.Graph()
    with test_graph.as_default():
      model = self._build_model(
          is_training=False, first_stage_only=True, second_stage_batch_size=2)
      batch_size = 2
      height = 10
      width = 12
      input_image_shape = (batch_size, height, width, 3)

      preprocessed_inputs = tf.placeholder(dtype=tf.float32,
                                           shape=(batch_size, None, None, 3))
      prediction_dict = model.predict(preprocessed_inputs)

      # In inference mode, anchors are clipped to the image window, but not
      # pruned.  Since MockFasterRCNN.extract_proposal_features returns a
      # tensor with the same shape as its input, the expected number of anchors
      # is height * width * the number of anchors per location (i.e. 3x3).
      expected_num_anchors = height * width * 3 * 3
      expected_output_keys = set([
          'rpn_box_predictor_features', 'rpn_features_to_crop', 'image_shape',
          'rpn_box_encodings', 'rpn_objectness_predictions_with_background',
          'anchors'])
      expected_output_shapes = {
          'rpn_box_predictor_features': (batch_size, height, width, 512),
          'rpn_features_to_crop': (batch_size, height, width, 3),
          'rpn_box_encodings': (batch_size, expected_num_anchors, 4),
          'rpn_objectness_predictions_with_background':
          (batch_size, expected_num_anchors, 2),
          'anchors': (expected_num_anchors, 4)
      }

      init_op = tf.global_variables_initializer()
      with self.test_session() as sess:
        sess.run(init_op)
        prediction_out = sess.run(prediction_dict,
                                  feed_dict={
                                      preprocessed_inputs:
                                      np.zeros(input_image_shape)
                                  })

        self.assertEqual(set(prediction_out.keys()), expected_output_keys)

        self.assertAllEqual(prediction_out['image_shape'], input_image_shape)
        for output_key, expected_shape in expected_output_shapes.iteritems():
          self.assertAllEqual(prediction_out[output_key].shape, expected_shape)

        # Check that anchors are clipped to window.
        anchors = prediction_out['anchors']
        self.assertTrue(np.all(np.greater_equal(anchors, 0)))
        self.assertTrue(np.all(np.less_equal(anchors[:, 0], height)))
        self.assertTrue(np.all(np.less_equal(anchors[:, 1], width)))
        self.assertTrue(np.all(np.less_equal(anchors[:, 2], height)))
        self.assertTrue(np.all(np.less_equal(anchors[:, 3], width)))

  def test_predict_gives_valid_anchors_in_training_mode_first_stage_only(self):
    test_graph = tf.Graph()
    with test_graph.as_default():
      model = self._build_model(
          is_training=True, first_stage_only=True, second_stage_batch_size=2)
      batch_size = 2
      height = 10
      width = 12
      input_image_shape = (batch_size, height, width, 3)
      preprocessed_inputs = tf.placeholder(dtype=tf.float32,
                                           shape=(batch_size, None, None, 3))
      prediction_dict = model.predict(preprocessed_inputs)

      expected_output_keys = set([
          'rpn_box_predictor_features', 'rpn_features_to_crop', 'image_shape',
          'rpn_box_encodings', 'rpn_objectness_predictions_with_background',
          'anchors'])
      # At training time, anchors that exceed image bounds are pruned.  Thus
      # the `expected_num_anchors` in the above inference mode test is now
      # a strict upper bound on the number of anchors.
      num_anchors_strict_upper_bound = height * width * 3 * 3

      init_op = tf.global_variables_initializer()
      with self.test_session() as sess:
        sess.run(init_op)
        prediction_out = sess.run(prediction_dict,
                                  feed_dict={
                                      preprocessed_inputs:
                                      np.zeros(input_image_shape)
                                  })

        self.assertEqual(set(prediction_out.keys()), expected_output_keys)
        self.assertAllEqual(prediction_out['image_shape'], input_image_shape)

        # Check that anchors have less than the upper bound and
        # are clipped to window.
        anchors = prediction_out['anchors']
        self.assertTrue(len(anchors.shape) == 2 and anchors.shape[1] == 4)
        num_anchors_out = anchors.shape[0]
        self.assertTrue(num_anchors_out < num_anchors_strict_upper_bound)

        self.assertTrue(np.all(np.greater_equal(anchors, 0)))
        self.assertTrue(np.all(np.less_equal(anchors[:, 0], height)))
        self.assertTrue(np.all(np.less_equal(anchors[:, 1], width)))
        self.assertTrue(np.all(np.less_equal(anchors[:, 2], height)))
        self.assertTrue(np.all(np.less_equal(anchors[:, 3], width)))

        self.assertAllEqual(prediction_out['rpn_box_encodings'].shape,
                            (batch_size, num_anchors_out, 4))
        self.assertAllEqual(
            prediction_out['rpn_objectness_predictions_with_background'].shape,
            (batch_size, num_anchors_out, 2))

  def test_predict_gives_correct_shapes_in_inference_mode_both_stages(self):
    test_graph = tf.Graph()
    with test_graph.as_default():
      model = self._build_model(
          is_training=False, first_stage_only=False, second_stage_batch_size=2)
      batch_size = 2
      image_size = 10
      image_shape = (batch_size, image_size, image_size, 3)
      preprocessed_inputs = tf.zeros(image_shape, dtype=tf.float32)
      result_tensor_dict = model.predict(preprocessed_inputs)
      expected_num_anchors = image_size * image_size * 3 * 3

      expected_shapes = {
          'rpn_box_predictor_features':
          (2, image_size, image_size, 512),
          'rpn_features_to_crop': (2, image_size, image_size, 3),
          'image_shape': (4,),
          'rpn_box_encodings': (2, expected_num_anchors, 4),
          'rpn_objectness_predictions_with_background':
          (2, expected_num_anchors, 2),
          'anchors': (expected_num_anchors, 4),
          'refined_box_encodings': (2 * 8, 2, 4),
          'class_predictions_with_background': (2 * 8, 2 + 1),
          'num_proposals': (2,),
          'proposal_boxes': (2, 8, 4),
      }
      init_op = tf.global_variables_initializer()
      with self.test_session() as sess:
        sess.run(init_op)
        tensor_dict_out = sess.run(result_tensor_dict)
        self.assertEqual(set(tensor_dict_out.keys()),
                         set(expected_shapes.keys()))
        for key in expected_shapes:
          self.assertAllEqual(tensor_dict_out[key].shape, expected_shapes[key])

  def test_predict_gives_correct_shapes_in_train_mode_both_stages(self):
    test_graph = tf.Graph()
    with test_graph.as_default():
      model = self._build_model(
          is_training=True, first_stage_only=False, second_stage_batch_size=7)
      batch_size = 2
      image_size = 10
      image_shape = (batch_size, image_size, image_size, 3)
      preprocessed_inputs = tf.zeros(image_shape, dtype=tf.float32)
      groundtruth_boxes_list = [
          tf.constant([[0, 0, .5, .5], [.5, .5, 1, 1]], dtype=tf.float32),
          tf.constant([[0, .5, .5, 1], [.5, 0, 1, .5]], dtype=tf.float32)]
      groundtruth_classes_list = [
          tf.constant([[1, 0], [0, 1]], dtype=tf.float32),
          tf.constant([[1, 0], [1, 0]], dtype=tf.float32)]

      model.provide_groundtruth(groundtruth_boxes_list,
                                groundtruth_classes_list)

      result_tensor_dict = model.predict(preprocessed_inputs)
      expected_shapes = {
          'rpn_box_predictor_features':
          (2, image_size, image_size, 512),
          'rpn_features_to_crop': (2, image_size, image_size, 3),
          'image_shape': (4,),
          'refined_box_encodings': (2 * 7, 2, 4),
          'class_predictions_with_background': (2 * 7, 2 + 1),
          'num_proposals': (2,),
          'proposal_boxes': (2, 7, 4),
      }
      init_op = tf.global_variables_initializer()
      with self.test_session() as sess:
        sess.run(init_op)
        tensor_dict_out = sess.run(result_tensor_dict)
        self.assertEqual(set(tensor_dict_out.keys()),
                         set(expected_shapes.keys()).union(set([
                             'rpn_box_encodings',
                             'rpn_objectness_predictions_with_background',
                             'anchors'])))
        for key in expected_shapes:
          self.assertAllEqual(tensor_dict_out[key].shape, expected_shapes[key])

        anchors_shape_out = tensor_dict_out['anchors'].shape
        self.assertEqual(2, len(anchors_shape_out))
        self.assertEqual(4, anchors_shape_out[1])
        num_anchors_out = anchors_shape_out[0]
        self.assertAllEqual(tensor_dict_out['rpn_box_encodings'].shape,
                            (2, num_anchors_out, 4))
        self.assertAllEqual(
            tensor_dict_out['rpn_objectness_predictions_with_background'].shape,
            (2, num_anchors_out, 2))

  def test_postprocess_first_stage_only_inference_mode(self):
    model = self._build_model(
        is_training=False, first_stage_only=True, second_stage_batch_size=6)
    batch_size = 2
    anchors = tf.constant(
        [[0, 0, 16, 16],
         [0, 16, 16, 32],
         [16, 0, 32, 16],
         [16, 16, 32, 32]], dtype=tf.float32)
    rpn_box_encodings = tf.zeros(
        [batch_size, anchors.get_shape().as_list()[0],
         BOX_CODE_SIZE], dtype=tf.float32)
    # use different numbers for the objectness category to break ties in
    # order of boxes returned by NMS
    rpn_objectness_predictions_with_background = tf.constant([
        [[-10, 13],
         [10, -10],
         [10, -11],
         [-10, 12]],
        [[10, -10],
         [-10, 13],
         [-10, 12],
         [10, -11]]], dtype=tf.float32)
    rpn_features_to_crop = tf.ones((batch_size, 8, 8, 10), dtype=tf.float32)
    image_shape = tf.constant([batch_size, 32, 32, 3], dtype=tf.int32)
    proposals = model.postprocess({
        'rpn_box_encodings': rpn_box_encodings,
        'rpn_objectness_predictions_with_background':
        rpn_objectness_predictions_with_background,
        'rpn_features_to_crop': rpn_features_to_crop,
        'anchors': anchors,
        'image_shape': image_shape})
    expected_proposal_boxes = [
        [[0, 0, .5, .5], [.5, .5, 1, 1], [0, .5, .5, 1], [.5, 0, 1.0, .5]]
        + 4 * [4 * [0]],
        [[0, .5, .5, 1], [.5, 0, 1.0, .5], [0, 0, .5, .5], [.5, .5, 1, 1]]
        + 4 * [4 * [0]]]
    expected_proposal_scores = [[1, 1, 0, 0, 0, 0, 0, 0],
                                [1, 1, 0, 0, 0, 0, 0, 0]]
    expected_num_proposals = [4, 4]

    expected_output_keys = set(['detection_boxes', 'detection_scores',
                                'num_detections'])
    self.assertEqual(set(proposals.keys()), expected_output_keys)
    with self.test_session() as sess:
      proposals_out = sess.run(proposals)
      self.assertAllClose(proposals_out['detection_boxes'],
                          expected_proposal_boxes)
      self.assertAllClose(proposals_out['detection_scores'],
                          expected_proposal_scores)
      self.assertAllEqual(proposals_out['num_detections'],
                          expected_num_proposals)

  def test_postprocess_first_stage_only_train_mode(self):
    model = self._build_model(
        is_training=True, first_stage_only=True, second_stage_batch_size=2)
    batch_size = 2
    anchors = tf.constant(
        [[0, 0, 16, 16],
         [0, 16, 16, 32],
         [16, 0, 32, 16],
         [16, 16, 32, 32]], dtype=tf.float32)
    rpn_box_encodings = tf.zeros(
        [batch_size, anchors.get_shape().as_list()[0],
         BOX_CODE_SIZE], dtype=tf.float32)
    # use different numbers for the objectness category to break ties in
    # order of boxes returned by NMS
    rpn_objectness_predictions_with_background = tf.constant([
        [[-10, 13],
         [-10, 12],
         [-10, 11],
         [-10, 10]],
        [[-10, 13],
         [-10, 12],
         [-10, 11],
         [-10, 10]]], dtype=tf.float32)
    rpn_features_to_crop = tf.ones((batch_size, 8, 8, 10), dtype=tf.float32)
    image_shape = tf.constant([batch_size, 32, 32, 3], dtype=tf.int32)
    groundtruth_boxes_list = [
        tf.constant([[0, 0, .5, .5], [.5, .5, 1, 1]], dtype=tf.float32),
        tf.constant([[0, .5, .5, 1], [.5, 0, 1, .5]], dtype=tf.float32)]
    groundtruth_classes_list = [tf.constant([[1, 0], [0, 1]], dtype=tf.float32),
                                tf.constant([[1, 0], [1, 0]], dtype=tf.float32)]

    model.provide_groundtruth(groundtruth_boxes_list,
                              groundtruth_classes_list)
    proposals = model.postprocess({
        'rpn_box_encodings': rpn_box_encodings,
        'rpn_objectness_predictions_with_background':
        rpn_objectness_predictions_with_background,
        'rpn_features_to_crop': rpn_features_to_crop,
        'anchors': anchors,
        'image_shape': image_shape})
    expected_proposal_boxes = [
        [[0, 0, .5, .5], [.5, .5, 1, 1]], [[0, .5, .5, 1], [.5, 0, 1, .5]]]
    expected_proposal_scores = [[1, 1],
                                [1, 1]]
    expected_num_proposals = [2, 2]

    expected_output_keys = set(['detection_boxes', 'detection_scores',
                                'num_detections'])
    self.assertEqual(set(proposals.keys()), expected_output_keys)

    with self.test_session() as sess:
      proposals_out = sess.run(proposals)
      self.assertAllClose(proposals_out['detection_boxes'],
                          expected_proposal_boxes)
      self.assertAllClose(proposals_out['detection_scores'],
                          expected_proposal_scores)
      self.assertAllEqual(proposals_out['num_detections'],
                          expected_num_proposals)

  def test_postprocess_second_stage_only_inference_mode(self):
    model = self._build_model(
        is_training=False, first_stage_only=False, second_stage_batch_size=6)

    batch_size = 2
    total_num_padded_proposals = batch_size * model.max_num_proposals
    proposal_boxes = tf.constant(
        [[[1, 1, 2, 3],
          [0, 0, 1, 1],
          [.5, .5, .6, .6],
          4*[0], 4*[0], 4*[0], 4*[0], 4*[0]],
         [[2, 3, 6, 8],
          [1, 2, 5, 3],
          4*[0], 4*[0], 4*[0], 4*[0], 4*[0], 4*[0]]], dtype=tf.float32)
    num_proposals = tf.constant([3, 2], dtype=tf.int32)
    refined_box_encodings = tf.zeros(
        [total_num_padded_proposals, model.num_classes, 4], dtype=tf.float32)
    class_predictions_with_background = tf.ones(
        [total_num_padded_proposals, model.num_classes+1], dtype=tf.float32)
    image_shape = tf.constant([batch_size, 36, 48, 3], dtype=tf.int32)

    detections = model.postprocess({
        'refined_box_encodings': refined_box_encodings,
        'class_predictions_with_background': class_predictions_with_background,
        'num_proposals': num_proposals,
        'proposal_boxes': proposal_boxes,
        'image_shape': image_shape
    })
    with self.test_session() as sess:
      detections_out = sess.run(detections)
      self.assertAllEqual(detections_out['detection_boxes'].shape, [2, 5, 4])
      self.assertAllClose(detections_out['detection_scores'],
                          [[1, 1, 1, 1, 1], [1, 1, 1, 1, 0]])
      self.assertAllClose(detections_out['detection_classes'],
                          [[0, 0, 0, 1, 1], [0, 0, 1, 1, 0]])
      self.assertAllClose(detections_out['num_detections'], [5, 4])

  def test_loss_first_stage_only_mode(self):
    model = self._build_model(
        is_training=True, first_stage_only=True, second_stage_batch_size=6)
    batch_size = 2
    anchors = tf.constant(
        [[0, 0, 16, 16],
         [0, 16, 16, 32],
         [16, 0, 32, 16],
         [16, 16, 32, 32]], dtype=tf.float32)

    rpn_box_encodings = tf.zeros(
        [batch_size,
         anchors.get_shape().as_list()[0],
         BOX_CODE_SIZE], dtype=tf.float32)
    # use different numbers for the objectness category to break ties in
    # order of boxes returned by NMS
    rpn_objectness_predictions_with_background = tf.constant([
        [[-10, 13],
         [10, -10],
         [10, -11],
         [-10, 12]],
        [[10, -10],
         [-10, 13],
         [-10, 12],
         [10, -11]]], dtype=tf.float32)
    image_shape = tf.constant([batch_size, 32, 32, 3], dtype=tf.int32)

    groundtruth_boxes_list = [
        tf.constant([[0, 0, .5, .5], [.5, .5, 1, 1]], dtype=tf.float32),
        tf.constant([[0, .5, .5, 1], [.5, 0, 1, .5]], dtype=tf.float32)]
    groundtruth_classes_list = [tf.constant([[1, 0], [0, 1]], dtype=tf.float32),
                                tf.constant([[1, 0], [1, 0]], dtype=tf.float32)]

    prediction_dict = {
        'rpn_box_encodings': rpn_box_encodings,
        'rpn_objectness_predictions_with_background':
        rpn_objectness_predictions_with_background,
        'image_shape': image_shape,
        'anchors': anchors
    }
    model.provide_groundtruth(groundtruth_boxes_list,
                              groundtruth_classes_list)
    loss_dict = model.loss(prediction_dict)
    with self.test_session() as sess:
      loss_dict_out = sess.run(loss_dict)
      self.assertAllClose(loss_dict_out['first_stage_localization_loss'], 0)
      self.assertAllClose(loss_dict_out['first_stage_objectness_loss'], 0)
      self.assertTrue('second_stage_localization_loss' not in loss_dict_out)
      self.assertTrue('second_stage_classification_loss' not in loss_dict_out)

  def test_loss_full(self):
    model = self._build_model(
        is_training=True, first_stage_only=False, second_stage_batch_size=6)
    batch_size = 2
    anchors = tf.constant(
        [[0, 0, 16, 16],
         [0, 16, 16, 32],
         [16, 0, 32, 16],
         [16, 16, 32, 32]], dtype=tf.float32)
    rpn_box_encodings = tf.zeros(
        [batch_size,
         anchors.get_shape().as_list()[0],
         BOX_CODE_SIZE], dtype=tf.float32)
    # use different numbers for the objectness category to break ties in
    # order of boxes returned by NMS
    rpn_objectness_predictions_with_background = tf.constant([
        [[-10, 13],
         [10, -10],
         [10, -11],
         [-10, 12]],
        [[10, -10],
         [-10, 13],
         [-10, 12],
         [10, -11]]], dtype=tf.float32)
    image_shape = tf.constant([batch_size, 32, 32, 3], dtype=tf.int32)

    num_proposals = tf.constant([6, 6], dtype=tf.int32)
    proposal_boxes = tf.constant(
        2 * [[[0, 0, 16, 16],
              [0, 16, 16, 32],
              [16, 0, 32, 16],
              [16, 16, 32, 32],
              [0, 0, 16, 16],
              [0, 16, 16, 32]]], dtype=tf.float32)
    refined_box_encodings = tf.zeros(
        (batch_size * model.max_num_proposals,
         model.num_classes,
         BOX_CODE_SIZE), dtype=tf.float32)
    class_predictions_with_background = tf.constant(
        [[-10, 10, -10],  # first image
         [10, -10, -10],
         [10, -10, -10],
         [-10, -10, 10],
         [-10, 10, -10],
         [10, -10, -10],
         [10, -10, -10],  # second image
         [-10, 10, -10],
         [-10, 10, -10],
         [10, -10, -10],
         [10, -10, -10],
         [-10, 10, -10]], dtype=tf.float32)

    groundtruth_boxes_list = [
        tf.constant([[0, 0, .5, .5], [.5, .5, 1, 1]], dtype=tf.float32),
        tf.constant([[0, .5, .5, 1], [.5, 0, 1, .5]], dtype=tf.float32)]
    groundtruth_classes_list = [tf.constant([[1, 0], [0, 1]], dtype=tf.float32),
                                tf.constant([[1, 0], [1, 0]], dtype=tf.float32)]

    prediction_dict = {
        'rpn_box_encodings': rpn_box_encodings,
        'rpn_objectness_predictions_with_background':
        rpn_objectness_predictions_with_background,
        'image_shape': image_shape,
        'anchors': anchors,
        'refined_box_encodings': refined_box_encodings,
        'class_predictions_with_background': class_predictions_with_background,
        'proposal_boxes': proposal_boxes,
        'num_proposals': num_proposals
    }
    model.provide_groundtruth(groundtruth_boxes_list,
                              groundtruth_classes_list)
    loss_dict = model.loss(prediction_dict)

    with self.test_session() as sess:
      loss_dict_out = sess.run(loss_dict)
      self.assertAllClose(loss_dict_out['first_stage_localization_loss'], 0)
      self.assertAllClose(loss_dict_out['first_stage_objectness_loss'], 0)
      self.assertAllClose(loss_dict_out['second_stage_localization_loss'], 0)
      self.assertAllClose(loss_dict_out['second_stage_classification_loss'], 0)

  def test_loss_full_zero_padded_proposals(self):
    model = self._build_model(
        is_training=True, first_stage_only=False, second_stage_batch_size=6)
    batch_size = 1
    anchors = tf.constant(
        [[0, 0, 16, 16],
         [0, 16, 16, 32],
         [16, 0, 32, 16],
         [16, 16, 32, 32]], dtype=tf.float32)
    rpn_box_encodings = tf.zeros(
        [batch_size,
         anchors.get_shape().as_list()[0],
         BOX_CODE_SIZE], dtype=tf.float32)
    # use different numbers for the objectness category to break ties in
    # order of boxes returned by NMS
    rpn_objectness_predictions_with_background = tf.constant([
        [[-10, 13],
         [10, -10],
         [10, -11],
         [10, -12]],], dtype=tf.float32)
    image_shape = tf.constant([batch_size, 32, 32, 3], dtype=tf.int32)

    # box_classifier_batch_size is 6, but here we assume that the number of
    # actual proposals (not counting zero paddings) is fewer (3).
    num_proposals = tf.constant([3], dtype=tf.int32)
    proposal_boxes = tf.constant(
        [[[0, 0, 16, 16],
          [0, 16, 16, 32],
          [16, 0, 32, 16],
          [0, 0, 0, 0],  # begin paddings
          [0, 0, 0, 0],
          [0, 0, 0, 0]]], dtype=tf.float32)

    refined_box_encodings = tf.zeros(
        (batch_size * model.max_num_proposals,
         model.num_classes,
         BOX_CODE_SIZE), dtype=tf.float32)
    class_predictions_with_background = tf.constant(
        [[-10, 10, -10],
         [10, -10, -10],
         [10, -10, -10],
         [0, 0, 0],  # begin paddings
         [0, 0, 0],
         [0, 0, 0]], dtype=tf.float32)

    groundtruth_boxes_list = [
        tf.constant([[0, 0, .5, .5]], dtype=tf.float32)]
    groundtruth_classes_list = [tf.constant([[1, 0]], dtype=tf.float32)]

    prediction_dict = {
        'rpn_box_encodings': rpn_box_encodings,
        'rpn_objectness_predictions_with_background':
        rpn_objectness_predictions_with_background,
        'image_shape': image_shape,
        'anchors': anchors,
        'refined_box_encodings': refined_box_encodings,
        'class_predictions_with_background': class_predictions_with_background,
        'proposal_boxes': proposal_boxes,
        'num_proposals': num_proposals
    }
    model.provide_groundtruth(groundtruth_boxes_list,
                              groundtruth_classes_list)
    loss_dict = model.loss(prediction_dict)

    with self.test_session() as sess:
      loss_dict_out = sess.run(loss_dict)
      self.assertAllClose(loss_dict_out['first_stage_localization_loss'], 0)
      self.assertAllClose(loss_dict_out['first_stage_objectness_loss'], 0)
      self.assertAllClose(loss_dict_out['second_stage_localization_loss'], 0)
      self.assertAllClose(loss_dict_out['second_stage_classification_loss'], 0)

  def test_loss_full_zero_padded_proposals_nonzero_loss_with_two_images(self):
    model = self._build_model(
        is_training=True, first_stage_only=False, second_stage_batch_size=6)
    batch_size = 2
    anchors = tf.constant(
        [[0, 0, 16, 16],
         [0, 16, 16, 32],
         [16, 0, 32, 16],
         [16, 16, 32, 32]], dtype=tf.float32)
    rpn_box_encodings = tf.zeros(
        [batch_size,
         anchors.get_shape().as_list()[0],
         BOX_CODE_SIZE], dtype=tf.float32)
    # use different numbers for the objectness category to break ties in
    # order of boxes returned by NMS
    rpn_objectness_predictions_with_background = tf.constant(
        [[[-10, 13],
          [10, -10],
          [10, -11],
          [10, -12]],
         [[-10, 13],
          [10, -10],
          [10, -11],
          [10, -12]]], dtype=tf.float32)
    image_shape = tf.constant([batch_size, 32, 32, 3], dtype=tf.int32)

    # box_classifier_batch_size is 6, but here we assume that the number of
    # actual proposals (not counting zero paddings) is fewer (3).
    num_proposals = tf.constant([3, 2], dtype=tf.int32)
    proposal_boxes = tf.constant(
        [[[0, 0, 16, 16],
          [0, 16, 16, 32],
          [16, 0, 32, 16],
          [0, 0, 0, 0],  # begin paddings
          [0, 0, 0, 0],
          [0, 0, 0, 0]],
         [[0, 0, 16, 16],
          [0, 16, 16, 32],
          [0, 0, 0, 0],
          [0, 0, 0, 0],  # begin paddings
          [0, 0, 0, 0],
          [0, 0, 0, 0]]], dtype=tf.float32)

    refined_box_encodings = tf.zeros(
        (batch_size * model.max_num_proposals,
         model.num_classes,
         BOX_CODE_SIZE), dtype=tf.float32)
    class_predictions_with_background = tf.constant(
        [[-10, 10, -10],  # first image
         [10, -10, -10],
         [10, -10, -10],
         [0, 0, 0],  # begin paddings
         [0, 0, 0],
         [0, 0, 0],
         [-10, -10, 10],  # second image
         [10, -10, -10],
         [0, 0, 0],  # begin paddings
         [0, 0, 0],
         [0, 0, 0],
         [0, 0, 0],], dtype=tf.float32)

    # The first groundtruth box is 4/5 of the anchor size in both directions
    # experiencing a loss of:
    # 2 * SmoothL1(5 * log(4/5)) / num_proposals
    #   = 2 * (abs(5 * log(1/2)) - .5) / 3
    # The second groundtruth box is identical to the prediction and thus
    # experiences zero loss.
    # Total average loss is (abs(5 * log(1/2)) - .5) / 3.
    groundtruth_boxes_list = [
        tf.constant([[0.05, 0.05, 0.45, 0.45]], dtype=tf.float32),
        tf.constant([[0.0, 0.0, 0.5, 0.5]], dtype=tf.float32)]
    groundtruth_classes_list = [tf.constant([[1, 0]], dtype=tf.float32),
                                tf.constant([[0, 1]], dtype=tf.float32)]
    exp_loc_loss = (-5 * np.log(.8) - 0.5) / 3.0

    prediction_dict = {
        'rpn_box_encodings': rpn_box_encodings,
        'rpn_objectness_predictions_with_background':
        rpn_objectness_predictions_with_background,
        'image_shape': image_shape,
        'anchors': anchors,
        'refined_box_encodings': refined_box_encodings,
        'class_predictions_with_background': class_predictions_with_background,
        'proposal_boxes': proposal_boxes,
        'num_proposals': num_proposals
    }
    model.provide_groundtruth(groundtruth_boxes_list,
                              groundtruth_classes_list)
    loss_dict = model.loss(prediction_dict)

    with self.test_session() as sess:
      loss_dict_out = sess.run(loss_dict)
      self.assertAllClose(loss_dict_out['first_stage_localization_loss'],
                          exp_loc_loss)
      self.assertAllClose(loss_dict_out['first_stage_objectness_loss'], 0)
      self.assertAllClose(loss_dict_out['second_stage_localization_loss'],
                          exp_loc_loss)
      self.assertAllClose(loss_dict_out['second_stage_classification_loss'], 0)

  def test_loss_with_hard_mining(self):
    model = self._build_model(is_training=True,
                              first_stage_only=False,
                              second_stage_batch_size=None,
                              first_stage_max_proposals=6,
                              hard_mining=True)
    batch_size = 1
    anchors = tf.constant(
        [[0, 0, 16, 16],
         [0, 16, 16, 32],
         [16, 0, 32, 16],
         [16, 16, 32, 32]], dtype=tf.float32)
    rpn_box_encodings = tf.zeros(
        [batch_size,
         anchors.get_shape().as_list()[0],
         BOX_CODE_SIZE], dtype=tf.float32)
    # use different numbers for the objectness category to break ties in
    # order of boxes returned by NMS
    rpn_objectness_predictions_with_background = tf.constant(
        [[[-10, 13],
          [-10, 12],
          [10, -11],
          [10, -12]]], dtype=tf.float32)
    image_shape = tf.constant([batch_size, 32, 32, 3], dtype=tf.int32)

    # box_classifier_batch_size is 6, but here we assume that the number of
    # actual proposals (not counting zero paddings) is fewer (3).
    num_proposals = tf.constant([3], dtype=tf.int32)
    proposal_boxes = tf.constant(
        [[[0, 0, 16, 16],
          [0, 16, 16, 32],
          [16, 0, 32, 16],
          [0, 0, 0, 0],  # begin paddings
          [0, 0, 0, 0],
          [0, 0, 0, 0]]], dtype=tf.float32)

    refined_box_encodings = tf.zeros(
        (batch_size * model.max_num_proposals,
         model.num_classes,
         BOX_CODE_SIZE), dtype=tf.float32)
    class_predictions_with_background = tf.constant(
        [[-10, 10, -10],  # first image
         [-10, -10, 10],
         [10, -10, -10],
         [0, 0, 0],  # begin paddings
         [0, 0, 0],
         [0, 0, 0]], dtype=tf.float32)

    # The first groundtruth box is 4/5 of the anchor size in both directions
    # experiencing a loss of:
    # 2 * SmoothL1(5 * log(4/5)) / num_proposals
    #   = 2 * (abs(5 * log(1/2)) - .5) / 3
    # The second groundtruth box is 46/50 of the anchor size in both directions
    # experiencing a loss of:
    # 2 * SmoothL1(5 * log(42/50)) / num_proposals
    #   = 2 * (.5(5 * log(.92))^2 - .5) / 3.
    # Since the first groundtruth box experiences greater loss, and we have
    # set num_hard_examples=1 in the HardMiner, the final localization loss
    # corresponds to that of the first groundtruth box.
    groundtruth_boxes_list = [
        tf.constant([[0.05, 0.05, 0.45, 0.45],
                     [0.02, 0.52, 0.48, 0.98],], dtype=tf.float32)]
    groundtruth_classes_list = [tf.constant([[1, 0], [0, 1]], dtype=tf.float32)]
    exp_loc_loss = 2 * (-5 * np.log(.8) - 0.5) / 3.0

    prediction_dict = {
        'rpn_box_encodings': rpn_box_encodings,
        'rpn_objectness_predictions_with_background':
        rpn_objectness_predictions_with_background,
        'image_shape': image_shape,
        'anchors': anchors,
        'refined_box_encodings': refined_box_encodings,
        'class_predictions_with_background': class_predictions_with_background,
        'proposal_boxes': proposal_boxes,
        'num_proposals': num_proposals
    }
    model.provide_groundtruth(groundtruth_boxes_list,
                              groundtruth_classes_list)
    loss_dict = model.loss(prediction_dict)

    with self.test_session() as sess:
      loss_dict_out = sess.run(loss_dict)
      self.assertAllClose(loss_dict_out['second_stage_localization_loss'],
                          exp_loc_loss)
      self.assertAllClose(loss_dict_out['second_stage_classification_loss'], 0)

  def test_restore_fn_classification(self):
    # Define mock tensorflow classification graph and save variables.
    test_graph_classification = tf.Graph()
    with test_graph_classification.as_default():
      image = tf.placeholder(dtype=tf.float32, shape=[1, 20, 20, 3])
      with tf.variable_scope('mock_model'):
        net = slim.conv2d(image, num_outputs=3, kernel_size=1, scope='layer1')
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
      model = self._build_model(
          is_training=False, first_stage_only=False, second_stage_batch_size=6)

      inputs_shape = (2, 20, 20, 3)
      inputs = tf.to_float(tf.random_uniform(
          inputs_shape, minval=0, maxval=255, dtype=tf.int32))
      preprocessed_inputs = model.preprocess(inputs)
      prediction_dict = model.predict(preprocessed_inputs)
      model.postprocess(prediction_dict)
      restore_fn = model.restore_fn(saved_model_path,
                                    from_detection_checkpoint=False)
      with self.test_session() as sess:
        restore_fn(sess)

  def test_restore_fn_detection(self):
    # Define first detection graph and save variables.
    test_graph_detection1 = tf.Graph()
    with test_graph_detection1.as_default():
      model = self._build_model(
          is_training=False, first_stage_only=False, second_stage_batch_size=6)
      inputs_shape = (2, 20, 20, 3)
      inputs = tf.to_float(tf.random_uniform(
          inputs_shape, minval=0, maxval=255, dtype=tf.int32))
      preprocessed_inputs = model.preprocess(inputs)
      prediction_dict = model.predict(preprocessed_inputs)
      model.postprocess(prediction_dict)
      init_op = tf.global_variables_initializer()
      saver = tf.train.Saver()
      save_path = self.get_temp_dir()
      with self.test_session() as sess:
        sess.run(init_op)
        saved_model_path = saver.save(sess, save_path)

    # Define second detection graph and restore variables.
    test_graph_detection2 = tf.Graph()
    with test_graph_detection2.as_default():
      model2 = self._build_model(is_training=False, first_stage_only=False,
                                 second_stage_batch_size=6, num_classes=42)

      inputs_shape2 = (2, 20, 20, 3)
      inputs2 = tf.to_float(tf.random_uniform(
          inputs_shape2, minval=0, maxval=255, dtype=tf.int32))
      preprocessed_inputs2 = model2.preprocess(inputs2)
      prediction_dict2 = model2.predict(preprocessed_inputs2)
      model2.postprocess(prediction_dict2)
      restore_fn = model2.restore_fn(saved_model_path,
                                     from_detection_checkpoint=True)
      with self.test_session() as sess:
        restore_fn(sess)
        for var in sess.run(tf.report_uninitialized_variables()):
          self.assertNotIn(model2.first_stage_feature_extractor_scope, var.name)
          self.assertNotIn(model2.second_stage_feature_extractor_scope,
                           var.name)

if __name__ == '__main__':
  tf.test.main()
