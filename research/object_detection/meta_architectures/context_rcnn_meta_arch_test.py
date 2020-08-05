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
"""Tests for object_detection.meta_architectures.context_meta_arch."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import functools
import unittest
from unittest import mock  # pylint: disable=g-importing-member
from absl.testing import parameterized
import tensorflow.compat.v1 as tf
import tf_slim as slim

from google.protobuf import text_format

from object_detection.anchor_generators import grid_anchor_generator
from object_detection.builders import box_predictor_builder
from object_detection.builders import hyperparams_builder
from object_detection.builders import post_processing_builder
from object_detection.core import balanced_positive_negative_sampler as sampler
from object_detection.core import losses
from object_detection.core import post_processing
from object_detection.core import standard_fields as fields
from object_detection.core import target_assigner
from object_detection.meta_architectures import context_rcnn_meta_arch
from object_detection.meta_architectures import faster_rcnn_meta_arch
from object_detection.protos import box_predictor_pb2
from object_detection.protos import hyperparams_pb2
from object_detection.protos import post_processing_pb2
from object_detection.utils import spatial_transform_ops as spatial_ops
from object_detection.utils import test_case
from object_detection.utils import test_utils
from object_detection.utils import tf_version


class FakeFasterRCNNFeatureExtractor(
    faster_rcnn_meta_arch.FasterRCNNFeatureExtractor):
  """Fake feature extractor to use in tests."""

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
      proposal_features = 0 * slim.conv2d(
          preprocessed_inputs, num_outputs=3, kernel_size=1, scope='layer1')
      return proposal_features, {}

  def _extract_box_classifier_features(self, proposal_feature_maps, scope):
    with tf.variable_scope('mock_model'):
      return 0 * slim.conv2d(
          proposal_feature_maps, num_outputs=3, kernel_size=1, scope='layer2')


class FakeFasterRCNNKerasFeatureExtractor(
    faster_rcnn_meta_arch.FasterRCNNKerasFeatureExtractor):
  """Fake feature extractor to use in tests."""

  def __init__(self):
    super(FakeFasterRCNNKerasFeatureExtractor, self).__init__(
        is_training=False, first_stage_features_stride=32, weight_decay=0.0)

  def preprocess(self, resized_inputs):
    return tf.identity(resized_inputs)

  def get_proposal_feature_extractor_model(self, name):

    class ProposalFeatureExtractor(tf.keras.Model):
      """Dummy proposal feature extraction."""

      def __init__(self, name):
        super(ProposalFeatureExtractor, self).__init__(name=name)
        self.conv = None

      def build(self, input_shape):
        self.conv = tf.keras.layers.Conv2D(
            3, kernel_size=1, padding='SAME', name='layer1')

      def call(self, inputs):
        return self.conv(inputs)

    return ProposalFeatureExtractor(name=name)

  def get_box_classifier_feature_extractor_model(self, name):
    return tf.keras.Sequential([
        tf.keras.layers.Conv2D(
            3, kernel_size=1, padding='SAME', name=name + '_layer2')
    ])


class ContextRCNNMetaArchTest(test_case.TestCase, parameterized.TestCase):

  def _get_model(self, box_predictor, **common_kwargs):
    return context_rcnn_meta_arch.ContextRCNNMetaArch(
        initial_crop_size=3,
        maxpool_kernel_size=1,
        maxpool_stride=1,
        second_stage_mask_rcnn_box_predictor=box_predictor,
        attention_bottleneck_dimension=10,
        attention_temperature=0.2,
        **common_kwargs)

  def _build_arg_scope_with_hyperparams(self, hyperparams_text_proto,
                                        is_training):
    hyperparams = hyperparams_pb2.Hyperparams()
    text_format.Merge(hyperparams_text_proto, hyperparams)
    return hyperparams_builder.build(hyperparams, is_training=is_training)

  def _build_keras_layer_hyperparams(self, hyperparams_text_proto):
    hyperparams = hyperparams_pb2.Hyperparams()
    text_format.Merge(hyperparams_text_proto, hyperparams)
    return hyperparams_builder.KerasLayerHyperparams(hyperparams)

  def _get_second_stage_box_predictor_text_proto(self,
                                                 share_box_across_classes=False
                                                ):
    share_box_field = 'true' if share_box_across_classes else 'false'
    box_predictor_text_proto = """
      mask_rcnn_box_predictor {{
        fc_hyperparams {{
          op: FC
          activation: NONE
          regularizer {{
            l2_regularizer {{
              weight: 0.0005
            }}
          }}
          initializer {{
            variance_scaling_initializer {{
              factor: 1.0
              uniform: true
              mode: FAN_AVG
            }}
          }}
        }}
        share_box_across_classes: {share_box_across_classes}
      }}
    """.format(share_box_across_classes=share_box_field)
    return box_predictor_text_proto

  def _get_box_classifier_features_shape(self,
                                         image_size,
                                         batch_size,
                                         max_num_proposals,
                                         initial_crop_size,
                                         maxpool_stride,
                                         num_features):
    return (batch_size * max_num_proposals,
            initial_crop_size/maxpool_stride,
            initial_crop_size/maxpool_stride,
            num_features)

  def _get_second_stage_box_predictor(self,
                                      num_classes,
                                      is_training,
                                      predict_masks,
                                      masks_are_class_agnostic,
                                      share_box_across_classes=False,
                                      use_keras=False):
    box_predictor_proto = box_predictor_pb2.BoxPredictor()
    text_format.Merge(
        self._get_second_stage_box_predictor_text_proto(
            share_box_across_classes), box_predictor_proto)
    if predict_masks:
      text_format.Merge(
          self._add_mask_to_second_stage_box_predictor_text_proto(
              masks_are_class_agnostic), box_predictor_proto)

    if use_keras:
      return box_predictor_builder.build_keras(
          hyperparams_builder.KerasLayerHyperparams,
          inplace_batchnorm_update=False,
          freeze_batchnorm=False,
          box_predictor_config=box_predictor_proto,
          num_classes=num_classes,
          num_predictions_per_location_list=None,
          is_training=is_training)
    else:
      return box_predictor_builder.build(
          hyperparams_builder.build,
          box_predictor_proto,
          num_classes=num_classes,
          is_training=is_training)

  def _build_model(self,
                   is_training,
                   number_of_stages,
                   second_stage_batch_size,
                   first_stage_max_proposals=8,
                   num_classes=2,
                   hard_mining=False,
                   softmax_second_stage_classification_loss=True,
                   predict_masks=False,
                   pad_to_max_dimension=None,
                   masks_are_class_agnostic=False,
                   use_matmul_crop_and_resize=False,
                   clip_anchors_to_image=False,
                   use_matmul_gather_in_matcher=False,
                   use_static_shapes=False,
                   calibration_mapping_value=None,
                   share_box_across_classes=False,
                   return_raw_detections_during_predict=False):
    use_keras = tf_version.is_tf2()
    def image_resizer_fn(image, masks=None):
      """Fake image resizer function."""
      resized_inputs = []
      resized_image = tf.identity(image)
      if pad_to_max_dimension is not None:
        resized_image = tf.image.pad_to_bounding_box(image, 0, 0,
                                                     pad_to_max_dimension,
                                                     pad_to_max_dimension)
      resized_inputs.append(resized_image)
      if masks is not None:
        resized_masks = tf.identity(masks)
        if pad_to_max_dimension is not None:
          resized_masks = tf.image.pad_to_bounding_box(
              tf.transpose(masks, [1, 2, 0]), 0, 0, pad_to_max_dimension,
              pad_to_max_dimension)
          resized_masks = tf.transpose(resized_masks, [2, 0, 1])
        resized_inputs.append(resized_masks)
      resized_inputs.append(tf.shape(image))
      return resized_inputs

    # anchors in this test are designed so that a subset of anchors are inside
    # the image and a subset of anchors are outside.
    first_stage_anchor_scales = (0.001, 0.005, 0.1)
    first_stage_anchor_aspect_ratios = (0.5, 1.0, 2.0)
    first_stage_anchor_strides = (1, 1)
    first_stage_anchor_generator = grid_anchor_generator.GridAnchorGenerator(
        first_stage_anchor_scales,
        first_stage_anchor_aspect_ratios,
        anchor_stride=first_stage_anchor_strides)
    first_stage_target_assigner = target_assigner.create_target_assigner(
        'FasterRCNN',
        'proposal',
        use_matmul_gather=use_matmul_gather_in_matcher)

    if use_keras:
      fake_feature_extractor = FakeFasterRCNNKerasFeatureExtractor()
    else:
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
    if use_keras:
      first_stage_box_predictor_arg_scope_fn = (
          self._build_keras_layer_hyperparams(
              first_stage_box_predictor_hyperparams_text_proto))
    else:
      first_stage_box_predictor_arg_scope_fn = (
          self._build_arg_scope_with_hyperparams(
              first_stage_box_predictor_hyperparams_text_proto, is_training))

    first_stage_box_predictor_kernel_size = 3
    first_stage_atrous_rate = 1
    first_stage_box_predictor_depth = 512
    first_stage_minibatch_size = 3
    first_stage_sampler = sampler.BalancedPositiveNegativeSampler(
        positive_fraction=0.5, is_static=use_static_shapes)

    first_stage_nms_score_threshold = -1.0
    first_stage_nms_iou_threshold = 1.0
    first_stage_max_proposals = first_stage_max_proposals
    first_stage_non_max_suppression_fn = functools.partial(
        post_processing.batch_multiclass_non_max_suppression,
        score_thresh=first_stage_nms_score_threshold,
        iou_thresh=first_stage_nms_iou_threshold,
        max_size_per_class=first_stage_max_proposals,
        max_total_size=first_stage_max_proposals,
        use_static_shapes=use_static_shapes)

    first_stage_localization_loss_weight = 1.0
    first_stage_objectness_loss_weight = 1.0

    post_processing_config = post_processing_pb2.PostProcessing()
    post_processing_text_proto = """
      score_converter: IDENTITY
      batch_non_max_suppression {
        score_threshold: -20.0
        iou_threshold: 1.0
        max_detections_per_class: 5
        max_total_detections: 5
        use_static_shapes: """ + '{}'.format(use_static_shapes) + """
      }
    """
    if calibration_mapping_value:
      calibration_text_proto = """
      calibration_config {
        function_approximation {
          x_y_pairs {
            x_y_pair {
              x: 0.0
              y: %f
            }
            x_y_pair {
              x: 1.0
              y: %f
              }}}}""" % (calibration_mapping_value, calibration_mapping_value)
      post_processing_text_proto = (
          post_processing_text_proto + ' ' + calibration_text_proto)
    text_format.Merge(post_processing_text_proto, post_processing_config)
    second_stage_non_max_suppression_fn, second_stage_score_conversion_fn = (
        post_processing_builder.build(post_processing_config))

    second_stage_target_assigner = target_assigner.create_target_assigner(
        'FasterRCNN',
        'detection',
        use_matmul_gather=use_matmul_gather_in_matcher)
    second_stage_sampler = sampler.BalancedPositiveNegativeSampler(
        positive_fraction=1.0, is_static=use_static_shapes)

    second_stage_localization_loss_weight = 1.0
    second_stage_classification_loss_weight = 1.0
    if softmax_second_stage_classification_loss:
      second_stage_classification_loss = (
          losses.WeightedSoftmaxClassificationLoss())
    else:
      second_stage_classification_loss = (
          losses.WeightedSigmoidClassificationLoss())

    hard_example_miner = None
    if hard_mining:
      hard_example_miner = losses.HardExampleMiner(
          num_hard_examples=1,
          iou_threshold=0.99,
          loss_type='both',
          cls_loss_weight=second_stage_classification_loss_weight,
          loc_loss_weight=second_stage_localization_loss_weight,
          max_negatives_per_positive=None)

    crop_and_resize_fn = (
        spatial_ops.multilevel_matmul_crop_and_resize
        if use_matmul_crop_and_resize
        else spatial_ops.multilevel_native_crop_and_resize)
    common_kwargs = {
        'is_training':
            is_training,
        'num_classes':
            num_classes,
        'image_resizer_fn':
            image_resizer_fn,
        'feature_extractor':
            fake_feature_extractor,
        'number_of_stages':
            number_of_stages,
        'first_stage_anchor_generator':
            first_stage_anchor_generator,
        'first_stage_target_assigner':
            first_stage_target_assigner,
        'first_stage_atrous_rate':
            first_stage_atrous_rate,
        'first_stage_box_predictor_arg_scope_fn':
            first_stage_box_predictor_arg_scope_fn,
        'first_stage_box_predictor_kernel_size':
            first_stage_box_predictor_kernel_size,
        'first_stage_box_predictor_depth':
            first_stage_box_predictor_depth,
        'first_stage_minibatch_size':
            first_stage_minibatch_size,
        'first_stage_sampler':
            first_stage_sampler,
        'first_stage_non_max_suppression_fn':
            first_stage_non_max_suppression_fn,
        'first_stage_max_proposals':
            first_stage_max_proposals,
        'first_stage_localization_loss_weight':
            first_stage_localization_loss_weight,
        'first_stage_objectness_loss_weight':
            first_stage_objectness_loss_weight,
        'second_stage_target_assigner':
            second_stage_target_assigner,
        'second_stage_batch_size':
            second_stage_batch_size,
        'second_stage_sampler':
            second_stage_sampler,
        'second_stage_non_max_suppression_fn':
            second_stage_non_max_suppression_fn,
        'second_stage_score_conversion_fn':
            second_stage_score_conversion_fn,
        'second_stage_localization_loss_weight':
            second_stage_localization_loss_weight,
        'second_stage_classification_loss_weight':
            second_stage_classification_loss_weight,
        'second_stage_classification_loss':
            second_stage_classification_loss,
        'hard_example_miner':
            hard_example_miner,
        'crop_and_resize_fn':
            crop_and_resize_fn,
        'clip_anchors_to_image':
            clip_anchors_to_image,
        'use_static_shapes':
            use_static_shapes,
        'resize_masks':
            True,
        'return_raw_detections_during_predict':
            return_raw_detections_during_predict
    }

    return self._get_model(
        self._get_second_stage_box_predictor(
            num_classes=num_classes,
            is_training=is_training,
            use_keras=use_keras,
            predict_masks=predict_masks,
            masks_are_class_agnostic=masks_are_class_agnostic,
            share_box_across_classes=share_box_across_classes), **common_kwargs)

  @unittest.skipIf(tf_version.is_tf2(), 'Skipping TF1.X only test.')
  @mock.patch.object(context_rcnn_meta_arch, 'context_rcnn_lib')
  def test_prediction_mock_tf1(self, mock_context_rcnn_lib_v1):
    """Mocks the context_rcnn_lib_v1 module to test the prediction.

    Using mock object so that we can ensure compute_box_context_attention is
    called in side the prediction function.

    Args:
      mock_context_rcnn_lib_v1: mock module for the context_rcnn_lib_v1.
    """
    model = self._build_model(
        is_training=False,
        number_of_stages=2,
        second_stage_batch_size=6,
        num_classes=42)
    mock_tensor = tf.ones([2, 8, 3, 3, 3], tf.float32)

    mock_context_rcnn_lib_v1.compute_box_context_attention.return_value = mock_tensor
    inputs_shape = (2, 20, 20, 3)
    inputs = tf.cast(
        tf.random_uniform(inputs_shape, minval=0, maxval=255, dtype=tf.int32),
        dtype=tf.float32)
    preprocessed_inputs, true_image_shapes = model.preprocess(inputs)
    context_features = tf.random_uniform((2, 20, 10),
                                         minval=0,
                                         maxval=255,
                                         dtype=tf.float32)
    valid_context_size = tf.random_uniform((2,),
                                           minval=0,
                                           maxval=10,
                                           dtype=tf.int32)
    features = {
        fields.InputDataFields.context_features: context_features,
        fields.InputDataFields.valid_context_size: valid_context_size
    }

    side_inputs = model.get_side_inputs(features)

    _ = model.predict(preprocessed_inputs, true_image_shapes, **side_inputs)
    mock_context_rcnn_lib_v1.compute_box_context_attention.assert_called_once()

  @parameterized.named_parameters(
      {'testcase_name': 'static_shapes', 'static_shapes': True},
      {'testcase_name': 'nostatic_shapes', 'static_shapes': False},
      )
  def test_prediction_end_to_end(self, static_shapes):
    """Runs prediction end to end and test the shape of the results."""
    with test_utils.GraphContextOrNone() as g:
      model = self._build_model(
          is_training=False,
          number_of_stages=2,
          second_stage_batch_size=6,
          use_matmul_crop_and_resize=static_shapes,
          clip_anchors_to_image=static_shapes,
          use_matmul_gather_in_matcher=static_shapes,
          use_static_shapes=static_shapes,
          num_classes=42)

    def graph_fn():
      inputs_shape = (2, 20, 20, 3)
      inputs = tf.cast(
          tf.random_uniform(inputs_shape, minval=0, maxval=255, dtype=tf.int32),
          dtype=tf.float32)
      preprocessed_inputs, true_image_shapes = model.preprocess(inputs)
      context_features = tf.random_uniform((2, 20, 10),
                                           minval=0,
                                           maxval=255,
                                           dtype=tf.float32)
      valid_context_size = tf.random_uniform((2,),
                                             minval=0,
                                             maxval=10,
                                             dtype=tf.int32)
      features = {
          fields.InputDataFields.context_features: context_features,
          fields.InputDataFields.valid_context_size: valid_context_size
      }

      side_inputs = model.get_side_inputs(features)
      prediction_dict = model.predict(preprocessed_inputs, true_image_shapes,
                                      **side_inputs)
      return (prediction_dict['rpn_box_predictor_features'],
              prediction_dict['rpn_box_encodings'],
              prediction_dict['refined_box_encodings'],
              prediction_dict['proposal_boxes_normalized'],
              prediction_dict['proposal_boxes'])
    execute_fn = self.execute if static_shapes else self.execute_cpu
    (rpn_box_predictor_features, rpn_box_encodings, refined_box_encodings,
     proposal_boxes_normalized, proposal_boxes) = execute_fn(graph_fn, [],
                                                             graph=g)
    self.assertAllEqual(rpn_box_predictor_features.shape, [2, 20, 20, 512])
    self.assertAllEqual(rpn_box_encodings.shape, [2, 3600, 4])
    self.assertAllEqual(refined_box_encodings.shape, [16, 42, 4])
    self.assertAllEqual(proposal_boxes_normalized.shape, [2, 8, 4])
    self.assertAllEqual(proposal_boxes.shape, [2, 8, 4])


if __name__ == '__main__':
  tf.test.main()
