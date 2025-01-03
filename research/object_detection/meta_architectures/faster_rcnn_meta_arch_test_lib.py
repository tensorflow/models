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
import functools
from absl.testing import parameterized

import numpy as np
import six
import tensorflow.compat.v1 as tf

from google.protobuf import text_format
from object_detection.anchor_generators import grid_anchor_generator
from object_detection.anchor_generators import multiscale_grid_anchor_generator
from object_detection.builders import box_predictor_builder
from object_detection.builders import hyperparams_builder
from object_detection.builders import post_processing_builder
from object_detection.core import balanced_positive_negative_sampler as sampler
from object_detection.core import losses
from object_detection.core import post_processing
from object_detection.core import target_assigner
from object_detection.meta_architectures import faster_rcnn_meta_arch
from object_detection.protos import box_predictor_pb2
from object_detection.protos import hyperparams_pb2
from object_detection.protos import post_processing_pb2
from object_detection.utils import spatial_transform_ops as spatial_ops
from object_detection.utils import test_case
from object_detection.utils import test_utils
from object_detection.utils import tf_version

# pylint: disable=g-import-not-at-top
try:
  import tf_slim as slim
except ImportError:
  # TF 2.0 doesn't ship with contrib.
  pass
# pylint: enable=g-import-not-at-top

BOX_CODE_SIZE = 4


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


class FakeFasterRCNNMultiLevelFeatureExtractor(
    faster_rcnn_meta_arch.FasterRCNNFeatureExtractor):
  """Fake feature extractor to use in tests."""

  def __init__(self):
    super(FakeFasterRCNNMultiLevelFeatureExtractor, self).__init__(
        is_training=False,
        first_stage_features_stride=32,
        reuse_weights=None,
        weight_decay=0.0)

  def preprocess(self, resized_inputs):
    return tf.identity(resized_inputs)

  def _extract_proposal_features(self, preprocessed_inputs, scope):
    with tf.variable_scope('mock_model'):
      proposal_features_1 = 0 * slim.conv2d(
          preprocessed_inputs, num_outputs=3, kernel_size=3, scope='layer1',
          padding='VALID')
      proposal_features_2 = 0 * slim.conv2d(
          proposal_features_1, num_outputs=3, kernel_size=3, scope='layer2',
          padding='VALID')
      return [proposal_features_1, proposal_features_2], {}

  def _extract_box_classifier_features(self, proposal_feature_maps, scope):
    with tf.variable_scope('mock_model'):
      return 0 * slim.conv2d(
          proposal_feature_maps, num_outputs=3, kernel_size=1, scope='layer3')


class FakeFasterRCNNKerasFeatureExtractor(
    faster_rcnn_meta_arch.FasterRCNNKerasFeatureExtractor):
  """Fake feature extractor to use in tests."""

  def __init__(self):
    super(FakeFasterRCNNKerasFeatureExtractor, self).__init__(
        is_training=False,
        first_stage_features_stride=32,
        weight_decay=0.0)

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
    return tf.keras.Sequential([tf.keras.layers.Conv2D(
        3, kernel_size=1, padding='SAME', name=name + '_layer2')])


class FakeFasterRCNNKerasMultilevelFeatureExtractor(
    faster_rcnn_meta_arch.FasterRCNNKerasFeatureExtractor):
  """Fake feature extractor to use in tests."""

  def __init__(self):
    super(FakeFasterRCNNKerasMultilevelFeatureExtractor, self).__init__(
        is_training=False,
        first_stage_features_stride=32,
        weight_decay=0.0)

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
            3, kernel_size=3, name='layer1')
        self.conv_1 = tf.keras.layers.Conv2D(
            3, kernel_size=3, name='layer1')

      def call(self, inputs):
        output_1 = self.conv(inputs)
        output_2 = self.conv_1(output_1)
        return [output_1, output_2]

    return ProposalFeatureExtractor(name=name)


class FasterRCNNMetaArchTestBase(test_case.TestCase, parameterized.TestCase):
  """Base class to test Faster R-CNN and R-FCN meta architectures."""

  def _build_arg_scope_with_hyperparams(self,
                                        hyperparams_text_proto,
                                        is_training):
    hyperparams = hyperparams_pb2.Hyperparams()
    text_format.Merge(hyperparams_text_proto, hyperparams)
    return hyperparams_builder.build(hyperparams, is_training=is_training)

  def _build_keras_layer_hyperparams(self, hyperparams_text_proto):
    hyperparams = hyperparams_pb2.Hyperparams()
    text_format.Merge(hyperparams_text_proto, hyperparams)
    return hyperparams_builder.KerasLayerHyperparams(hyperparams)

  def _get_second_stage_box_predictor_text_proto(
      self, share_box_across_classes=False):
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

  def _add_mask_to_second_stage_box_predictor_text_proto(
      self, masks_are_class_agnostic=False):
    agnostic = 'true' if masks_are_class_agnostic else 'false'
    box_predictor_text_proto = """
      mask_rcnn_box_predictor {
        predict_instance_masks: true
        masks_are_class_agnostic: """ + agnostic + """
        mask_height: 14
        mask_width: 14
        conv_hyperparams {
          op: CONV
          regularizer {
            l2_regularizer {
              weight: 0.0
            }
          }
          initializer {
            truncated_normal_initializer {
              stddev: 0.01
            }
          }
        }
      }
    """
    return box_predictor_text_proto

  def _get_second_stage_box_predictor(self, num_classes, is_training,
                                      predict_masks, masks_are_class_agnostic,
                                      share_box_across_classes=False,
                                      use_keras=False):
    box_predictor_proto = box_predictor_pb2.BoxPredictor()
    text_format.Merge(self._get_second_stage_box_predictor_text_proto(
        share_box_across_classes), box_predictor_proto)
    if predict_masks:
      text_format.Merge(
          self._add_mask_to_second_stage_box_predictor_text_proto(
              masks_are_class_agnostic),
          box_predictor_proto)

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

  def _get_model(self, box_predictor, keras_model=False, **common_kwargs):
    return faster_rcnn_meta_arch.FasterRCNNMetaArch(
        initial_crop_size=3,
        maxpool_kernel_size=1,
        maxpool_stride=1,
        second_stage_mask_rcnn_box_predictor=box_predictor,
        **common_kwargs)

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
                   return_raw_detections_during_predict=False,
                   output_final_box_features=False,
                   multi_level=False):
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
          resized_masks = tf.image.pad_to_bounding_box(tf.transpose(masks,
                                                                    [1, 2, 0]),
                                                       0, 0,
                                                       pad_to_max_dimension,
                                                       pad_to_max_dimension)
          resized_masks = tf.transpose(resized_masks, [2, 0, 1])
        resized_inputs.append(resized_masks)
      resized_inputs.append(tf.shape(image))
      return resized_inputs

    # anchors in this test are designed so that a subset of anchors are inside
    # the image and a subset of anchors are outside.
    first_stage_anchor_generator = None
    if multi_level:
      min_level = 0
      max_level = 1
      anchor_scale = 0.1
      aspect_ratios = [1.0, 2.0, 0.5]
      scales_per_octave = 2
      normalize_coordinates = False
      (first_stage_anchor_generator
      ) = multiscale_grid_anchor_generator.MultiscaleGridAnchorGenerator(
          min_level, max_level, anchor_scale, aspect_ratios, scales_per_octave,
          normalize_coordinates)
    else:
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
      if multi_level:
        fake_feature_extractor = FakeFasterRCNNKerasMultilevelFeatureExtractor()
      else:
        fake_feature_extractor = FakeFasterRCNNKerasFeatureExtractor()
    else:
      if multi_level:
        fake_feature_extractor = FakeFasterRCNNMultiLevelFeatureExtractor()
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
        use_static_shapes: """ +'{}'.format(use_static_shapes) + """
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
      post_processing_text_proto = (post_processing_text_proto
                                    + ' ' + calibration_text_proto)
    text_format.Merge(post_processing_text_proto, post_processing_config)
    second_stage_non_max_suppression_fn, second_stage_score_conversion_fn = (
        post_processing_builder.build(post_processing_config))

    second_stage_target_assigner = target_assigner.create_target_assigner(
        'FasterRCNN', 'detection',
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
            return_raw_detections_during_predict,
        'output_final_box_features':
            output_final_box_features
    }

    return self._get_model(
        self._get_second_stage_box_predictor(
            num_classes=num_classes,
            is_training=is_training,
            use_keras=use_keras,
            predict_masks=predict_masks,
            masks_are_class_agnostic=masks_are_class_agnostic,
            share_box_across_classes=share_box_across_classes), **common_kwargs)

  @parameterized.parameters(
      {'use_static_shapes': False},
      {'use_static_shapes': True},
  )
  def test_predict_gives_correct_shapes_in_inference_mode_first_stage_only(
      self, use_static_shapes=False):
    batch_size = 2
    height = 10
    width = 12
    input_image_shape = (batch_size, height, width, 3)

    with test_utils.GraphContextOrNone() as g:
      model = self._build_model(
          is_training=False,
          number_of_stages=1,
          second_stage_batch_size=2,
          clip_anchors_to_image=use_static_shapes,
          use_static_shapes=use_static_shapes)
    def graph_fn(images):
      """Function to construct tf graph for the test."""

      preprocessed_inputs, true_image_shapes = model.preprocess(images)
      prediction_dict = model.predict(preprocessed_inputs, true_image_shapes)
      return (prediction_dict['rpn_box_predictor_features'][0],
              prediction_dict['rpn_features_to_crop'][0],
              prediction_dict['image_shape'],
              prediction_dict['rpn_box_encodings'],
              prediction_dict['rpn_objectness_predictions_with_background'],
              prediction_dict['anchors'])

    images = np.zeros(input_image_shape, dtype=np.float32)

    # In inference mode, anchors are clipped to the image window, but not
    # pruned.  Since MockFasterRCNN.extract_proposal_features returns a
    # tensor with the same shape as its input, the expected number of anchors
    # is height * width * the number of anchors per location (i.e. 3x3).
    expected_num_anchors = height * width * 3 * 3
    expected_output_shapes = {
        'rpn_box_predictor_features': (batch_size, height, width, 512),
        'rpn_features_to_crop': (batch_size, height, width, 3),
        'rpn_box_encodings': (batch_size, expected_num_anchors, 4),
        'rpn_objectness_predictions_with_background':
        (batch_size, expected_num_anchors, 2),
        'anchors': (expected_num_anchors, 4)
    }

    if use_static_shapes:
      results = self.execute(graph_fn, [images], graph=g)
    else:
      results = self.execute_cpu(graph_fn, [images], graph=g)

    self.assertAllEqual(results[0].shape,
                        expected_output_shapes['rpn_box_predictor_features'])
    self.assertAllEqual(results[1].shape,
                        expected_output_shapes['rpn_features_to_crop'])
    self.assertAllEqual(results[2],
                        input_image_shape)
    self.assertAllEqual(results[3].shape,
                        expected_output_shapes['rpn_box_encodings'])
    self.assertAllEqual(
        results[4].shape,
        expected_output_shapes['rpn_objectness_predictions_with_background'])
    self.assertAllEqual(results[5].shape,
                        expected_output_shapes['anchors'])

    # Check that anchors are clipped to window.
    anchors = results[5]
    self.assertTrue(np.all(np.greater_equal(anchors, 0)))
    self.assertTrue(np.all(np.less_equal(anchors[:, 0], height)))
    self.assertTrue(np.all(np.less_equal(anchors[:, 1], width)))
    self.assertTrue(np.all(np.less_equal(anchors[:, 2], height)))
    self.assertTrue(np.all(np.less_equal(anchors[:, 3], width)))

  @parameterized.parameters(
      {'use_static_shapes': False},
      {'use_static_shapes': True},
  )
  def test_predict_shape_in_inference_mode_first_stage_only_multi_level(
      self, use_static_shapes):
    batch_size = 2
    height = 50
    width = 52
    input_image_shape = (batch_size, height, width, 3)

    with test_utils.GraphContextOrNone() as g:
      model = self._build_model(
          is_training=False,
          number_of_stages=1,
          second_stage_batch_size=2,
          clip_anchors_to_image=use_static_shapes,
          use_static_shapes=use_static_shapes,
          multi_level=True)
    def graph_fn(images):
      """Function to construct tf graph for the test."""

      preprocessed_inputs, true_image_shapes = model.preprocess(images)
      prediction_dict = model.predict(preprocessed_inputs, true_image_shapes)
      return (prediction_dict['rpn_box_predictor_features'][0],
              prediction_dict['rpn_box_predictor_features'][1],
              prediction_dict['rpn_features_to_crop'][0],
              prediction_dict['rpn_features_to_crop'][1],
              prediction_dict['image_shape'],
              prediction_dict['rpn_box_encodings'],
              prediction_dict['rpn_objectness_predictions_with_background'],
              prediction_dict['anchors'])

    images = np.zeros(input_image_shape, dtype=np.float32)

    # In inference mode, anchors are clipped to the image window, but not
    # pruned.  Since MockFasterRCNN.extract_proposal_features returns a
    # tensor with the same shape as its input, the expected number of anchors
    # is height * width * the number of anchors per location (i.e. 3x3).
    expected_num_anchors = ((height-2) * (width-2) + (height-4) * (width-4)) * 6
    expected_output_shapes = {
        'rpn_box_predictor_features_0': (batch_size, height-2, width-2, 512),
        'rpn_box_predictor_features_1': (batch_size, height-4, width-4, 512),
        'rpn_features_to_crop_0': (batch_size, height-2, width-2, 3),
        'rpn_features_to_crop_1': (batch_size, height-4, width-4, 3),
        'rpn_box_encodings': (batch_size, expected_num_anchors, 4),
        'rpn_objectness_predictions_with_background':
        (batch_size, expected_num_anchors, 2),
    }

    if use_static_shapes:
      expected_output_shapes['anchors'] = (expected_num_anchors, 4)
    else:
      expected_output_shapes['anchors'] = (18300, 4)

    if use_static_shapes:
      results = self.execute(graph_fn, [images], graph=g)
    else:
      results = self.execute_cpu(graph_fn, [images], graph=g)

    self.assertAllEqual(results[0].shape,
                        expected_output_shapes['rpn_box_predictor_features_0'])
    self.assertAllEqual(results[1].shape,
                        expected_output_shapes['rpn_box_predictor_features_1'])
    self.assertAllEqual(results[2].shape,
                        expected_output_shapes['rpn_features_to_crop_0'])
    self.assertAllEqual(results[3].shape,
                        expected_output_shapes['rpn_features_to_crop_1'])
    self.assertAllEqual(results[4],
                        input_image_shape)
    self.assertAllEqual(results[5].shape,
                        expected_output_shapes['rpn_box_encodings'])
    self.assertAllEqual(
        results[6].shape,
        expected_output_shapes['rpn_objectness_predictions_with_background'])
    self.assertAllEqual(results[7].shape,
                        expected_output_shapes['anchors'])

    # Check that anchors are clipped to window.
    anchors = results[5]
    self.assertTrue(np.all(np.greater_equal(anchors, 0)))
    self.assertTrue(np.all(np.less_equal(anchors[:, 0], height)))
    self.assertTrue(np.all(np.less_equal(anchors[:, 1], width)))
    self.assertTrue(np.all(np.less_equal(anchors[:, 2], height)))
    self.assertTrue(np.all(np.less_equal(anchors[:, 3], width)))

  def test_regularization_losses(self):
    with test_utils.GraphContextOrNone() as g:
      model = self._build_model(
          is_training=True, number_of_stages=1, second_stage_batch_size=2)
    def graph_fn():
      batch_size = 2
      height = 10
      width = 12
      input_image_shape = (batch_size, height, width, 3)
      image, true_image_shapes = model.preprocess(tf.zeros(input_image_shape))
      model.predict(image, true_image_shapes)

      reg_losses = tf.math.add_n(model.regularization_losses())
      return reg_losses
    reg_losses = self.execute(graph_fn, [], graph=g)
    self.assertGreaterEqual(reg_losses, 0)

  def test_predict_gives_valid_anchors_in_training_mode_first_stage_only(self):
    expected_output_keys = set([
        'rpn_box_predictor_features', 'rpn_features_to_crop', 'image_shape',
        'rpn_box_encodings', 'rpn_objectness_predictions_with_background',
        'anchors', 'feature_maps'])

    with test_utils.GraphContextOrNone() as g:
      model = self._build_model(
          is_training=True, number_of_stages=1, second_stage_batch_size=2,)

    batch_size = 2
    height = 10
    width = 12
    input_image_shape = (batch_size, height, width, 3)
    def graph_fn():
      image, true_image_shapes = model.preprocess(tf.zeros(input_image_shape))
      prediction_dict = model.predict(image, true_image_shapes)
      self.assertEqual(set(prediction_dict.keys()), expected_output_keys)
      return (prediction_dict['image_shape'], prediction_dict['anchors'],
              prediction_dict['rpn_box_encodings'],
              prediction_dict['rpn_objectness_predictions_with_background'])

    (image_shape, anchors, rpn_box_encodings,
     rpn_objectness_predictions_with_background) = self.execute(graph_fn, [],
                                                                graph=g)
    # At training time, anchors that exceed image bounds are pruned.  Thus
    # the `expected_num_anchors` in the above inference mode test is now
    # a strict upper bound on the number of anchors.
    num_anchors_strict_upper_bound = height * width * 3 * 3
    self.assertAllEqual(image_shape, input_image_shape)
    self.assertTrue(len(anchors.shape) == 2 and anchors.shape[1] == 4)
    num_anchors_out = anchors.shape[0]
    self.assertLess(num_anchors_out, num_anchors_strict_upper_bound)

    self.assertTrue(np.all(np.greater_equal(anchors, 0)))
    self.assertTrue(np.all(np.less_equal(anchors[:, 0], height)))
    self.assertTrue(np.all(np.less_equal(anchors[:, 1], width)))
    self.assertTrue(np.all(np.less_equal(anchors[:, 2], height)))
    self.assertTrue(np.all(np.less_equal(anchors[:, 3], width)))

    self.assertAllEqual(rpn_box_encodings.shape,
                        (batch_size, num_anchors_out, 4))
    self.assertAllEqual(
        rpn_objectness_predictions_with_background.shape,
        (batch_size, num_anchors_out, 2))

  @parameterized.parameters(
      {'use_static_shapes': False},
      {'use_static_shapes': True},
  )
  def test_predict_correct_shapes_in_inference_mode_two_stages(
      self, use_static_shapes):

    def compare_results(results, expected_output_shapes):
      """Checks if the shape of the predictions are as expected."""
      self.assertAllEqual(results[0][0].shape,
                          expected_output_shapes['rpn_box_predictor_features'])
      self.assertAllEqual(results[1][0].shape,
                          expected_output_shapes['rpn_features_to_crop'])
      self.assertAllEqual(results[2].shape,
                          expected_output_shapes['image_shape'])
      self.assertAllEqual(results[3].shape,
                          expected_output_shapes['rpn_box_encodings'])
      self.assertAllEqual(
          results[4].shape,
          expected_output_shapes['rpn_objectness_predictions_with_background'])
      self.assertAllEqual(results[5].shape,
                          expected_output_shapes['anchors'])
      self.assertAllEqual(results[6].shape,
                          expected_output_shapes['refined_box_encodings'])
      self.assertAllEqual(
          results[7].shape,
          expected_output_shapes['class_predictions_with_background'])
      self.assertAllEqual(results[8].shape,
                          expected_output_shapes['num_proposals'])
      self.assertAllEqual(results[9].shape,
                          expected_output_shapes['proposal_boxes'])
      self.assertAllEqual(results[10].shape,
                          expected_output_shapes['proposal_boxes_normalized'])
      self.assertAllEqual(results[11].shape,
                          expected_output_shapes['box_classifier_features'])
      self.assertAllEqual(results[12].shape,
                          expected_output_shapes['final_anchors'])
    batch_size = 2
    image_size = 10
    max_num_proposals = 8
    initial_crop_size = 3
    maxpool_stride = 1

    with test_utils.GraphContextOrNone() as g:
      model = self._build_model(
          is_training=False,
          number_of_stages=2,
          second_stage_batch_size=2,
          predict_masks=False,
          use_matmul_crop_and_resize=use_static_shapes,
          clip_anchors_to_image=use_static_shapes,
          use_static_shapes=use_static_shapes)
    def graph_fn():
      """A function with TF compute."""
      if use_static_shapes:
        images = tf.random_uniform((batch_size, image_size, image_size, 3))
      else:
        images = tf.random_uniform((tf.random_uniform([],
                                                      minval=batch_size,
                                                      maxval=batch_size + 1,
                                                      dtype=tf.int32),
                                    tf.random_uniform([],
                                                      minval=image_size,
                                                      maxval=image_size + 1,
                                                      dtype=tf.int32),
                                    tf.random_uniform([],
                                                      minval=image_size,
                                                      maxval=image_size + 1,
                                                      dtype=tf.int32), 3))
      preprocessed_inputs, true_image_shapes = model.preprocess(images)
      prediction_dict = model.predict(preprocessed_inputs, true_image_shapes)
      return (prediction_dict['rpn_box_predictor_features'],
              prediction_dict['rpn_features_to_crop'],
              prediction_dict['image_shape'],
              prediction_dict['rpn_box_encodings'],
              prediction_dict['rpn_objectness_predictions_with_background'],
              prediction_dict['anchors'],
              prediction_dict['refined_box_encodings'],
              prediction_dict['class_predictions_with_background'],
              prediction_dict['num_proposals'],
              prediction_dict['proposal_boxes'],
              prediction_dict['proposal_boxes_normalized'],
              prediction_dict['box_classifier_features'],
              prediction_dict['final_anchors'])
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
        'refined_box_encodings': (2 * max_num_proposals, 2, 4),
        'class_predictions_with_background': (2 * max_num_proposals, 2 + 1),
        'num_proposals': (2,),
        'proposal_boxes': (2, max_num_proposals, 4),
        'proposal_boxes_normalized': (2, max_num_proposals, 4),
        'box_classifier_features':
        self._get_box_classifier_features_shape(image_size,
                                                batch_size,
                                                max_num_proposals,
                                                initial_crop_size,
                                                maxpool_stride,
                                                3),
        'feature_maps': [(2, image_size, image_size, 512)],
        'final_anchors': (2, max_num_proposals, 4)
    }

    if use_static_shapes:
      results = self.execute(graph_fn, [], graph=g)
    else:
      results = self.execute_cpu(graph_fn, [], graph=g)
    compare_results(results, expected_shapes)

  @parameterized.parameters(
      {'use_static_shapes': False},
      {'use_static_shapes': True},
  )
  def test_predict_gives_correct_shapes_in_train_mode_both_stages(
      self,
      use_static_shapes=False):
    batch_size = 2
    image_size = 10
    max_num_proposals = 7
    initial_crop_size = 3
    maxpool_stride = 1

    with test_utils.GraphContextOrNone() as g:
      model = self._build_model(
          is_training=True,
          number_of_stages=2,
          second_stage_batch_size=7,
          predict_masks=False,
          use_matmul_crop_and_resize=use_static_shapes,
          clip_anchors_to_image=use_static_shapes,
          use_static_shapes=use_static_shapes)

    def graph_fn(images, gt_boxes, gt_classes, gt_weights):
      """Function to construct tf graph for the test."""
      preprocessed_inputs, true_image_shapes = model.preprocess(images)
      model.provide_groundtruth(
          groundtruth_boxes_list=tf.unstack(gt_boxes),
          groundtruth_classes_list=tf.unstack(gt_classes),
          groundtruth_weights_list=tf.unstack(gt_weights))
      result_tensor_dict = model.predict(preprocessed_inputs, true_image_shapes)
      return (result_tensor_dict['refined_box_encodings'],
              result_tensor_dict['class_predictions_with_background'],
              result_tensor_dict['proposal_boxes'],
              result_tensor_dict['proposal_boxes_normalized'],
              result_tensor_dict['anchors'],
              result_tensor_dict['rpn_box_encodings'],
              result_tensor_dict['rpn_objectness_predictions_with_background'],
              result_tensor_dict['rpn_features_to_crop'][0],
              result_tensor_dict['rpn_box_predictor_features'][0],
              result_tensor_dict['final_anchors'],
             )

    image_shape = (batch_size, image_size, image_size, 3)
    images = np.zeros(image_shape, dtype=np.float32)
    gt_boxes = np.stack([
        np.array([[0, 0, .5, .5], [.5, .5, 1, 1]], dtype=np.float32),
        np.array([[0, .5, .5, 1], [.5, 0, 1, .5]], dtype=np.float32)
    ])
    gt_classes = np.stack([
        np.array([[1, 0], [0, 1]], dtype=np.float32),
        np.array([[1, 0], [1, 0]], dtype=np.float32)
    ])
    gt_weights = np.stack([
        np.array([1, 1], dtype=np.float32),
        np.array([1, 1], dtype=np.float32)
    ])
    if use_static_shapes:
      results = self.execute(graph_fn,
                             [images, gt_boxes, gt_classes, gt_weights],
                             graph=g)
    else:
      results = self.execute_cpu(graph_fn,
                                 [images, gt_boxes, gt_classes, gt_weights],
                                 graph=g)

    expected_shapes = {
        'rpn_box_predictor_features': (2, image_size, image_size, 512),
        'rpn_features_to_crop': (2, image_size, image_size, 3),
        'refined_box_encodings': (2 * max_num_proposals, 2, 4),
        'class_predictions_with_background': (2 * max_num_proposals, 2 + 1),
        'proposal_boxes': (2, max_num_proposals, 4),
        'rpn_box_encodings': (2, image_size * image_size * 9, 4),
        'proposal_boxes_normalized': (2, max_num_proposals, 4),
        'box_classifier_features':
            self._get_box_classifier_features_shape(
                image_size, batch_size, max_num_proposals, initial_crop_size,
                maxpool_stride, 3),
        'rpn_objectness_predictions_with_background':
        (2, image_size * image_size * 9, 2),
        'final_anchors': (2, max_num_proposals, 4)
    }
    # TODO(rathodv): Possibly change utils/test_case.py to accept dictionaries
    # and return dicionaries so don't have to rely on the order of tensors.
    self.assertAllEqual(results[0].shape,
                        expected_shapes['refined_box_encodings'])
    self.assertAllEqual(results[1].shape,
                        expected_shapes['class_predictions_with_background'])
    self.assertAllEqual(results[2].shape, expected_shapes['proposal_boxes'])
    self.assertAllEqual(results[3].shape,
                        expected_shapes['proposal_boxes_normalized'])
    anchors_shape = results[4].shape
    self.assertAllEqual(results[5].shape,
                        [batch_size, anchors_shape[0], 4])
    self.assertAllEqual(results[6].shape,
                        [batch_size, anchors_shape[0], 2])
    self.assertAllEqual(results[7].shape,
                        expected_shapes['rpn_features_to_crop'])
    self.assertAllEqual(results[8].shape,
                        expected_shapes['rpn_box_predictor_features'])
    self.assertAllEqual(results[9].shape,
                        expected_shapes['final_anchors'])

  @parameterized.parameters(
      {'use_static_shapes': False, 'pad_to_max_dimension': None},
      {'use_static_shapes': True, 'pad_to_max_dimension': None},
      {'use_static_shapes': False, 'pad_to_max_dimension': 56,},
      {'use_static_shapes': True, 'pad_to_max_dimension': 56},
  )
  def test_postprocess_first_stage_only_inference_mode(
      self, use_static_shapes=False,
      pad_to_max_dimension=None):
    batch_size = 2
    first_stage_max_proposals = 4 if use_static_shapes else 8

    with test_utils.GraphContextOrNone() as g:
      model = self._build_model(
          is_training=False,
          number_of_stages=1, second_stage_batch_size=6,
          use_matmul_crop_and_resize=use_static_shapes,
          clip_anchors_to_image=use_static_shapes,
          use_static_shapes=use_static_shapes,
          use_matmul_gather_in_matcher=use_static_shapes,
          first_stage_max_proposals=first_stage_max_proposals,
          pad_to_max_dimension=pad_to_max_dimension)

    def graph_fn(images,
                 rpn_box_encodings,
                 rpn_objectness_predictions_with_background,
                 rpn_features_to_crop,
                 anchors):
      """Function to construct tf graph for the test."""
      preprocessed_images, true_image_shapes = model.preprocess(images)
      proposals = model.postprocess({
          'rpn_box_encodings': rpn_box_encodings,
          'rpn_objectness_predictions_with_background':
          rpn_objectness_predictions_with_background,
          'rpn_features_to_crop': rpn_features_to_crop,
          'image_shape': tf.shape(preprocessed_images),
          'anchors': anchors}, true_image_shapes)
      return (proposals['num_detections'], proposals['detection_boxes'],
              proposals['detection_scores'], proposals['raw_detection_boxes'],
              proposals['raw_detection_scores'])

    anchors = np.array(
        [[0, 0, 16, 16],
         [0, 16, 16, 32],
         [16, 0, 32, 16],
         [16, 16, 32, 32]], dtype=np.float32)
    rpn_box_encodings = np.zeros(
        (batch_size, anchors.shape[0], BOX_CODE_SIZE), dtype=np.float32)
    # use different numbers for the objectness category to break ties in
    # order of boxes returned by NMS
    rpn_objectness_predictions_with_background = np.array([
        [[-10, 13],
         [10, -10],
         [10, -11],
         [-10, 12]],
        [[10, -10],
         [-10, 13],
         [-10, 12],
         [10, -11]]], dtype=np.float32)
    rpn_features_to_crop = np.ones((batch_size, 8, 8, 10), dtype=np.float32)
    image_shape = (batch_size, 32, 32, 3)
    images = np.zeros(image_shape, dtype=np.float32)

    if use_static_shapes:
      results = self.execute(graph_fn,
                             [images, rpn_box_encodings,
                              rpn_objectness_predictions_with_background,
                              rpn_features_to_crop, anchors], graph=g)
    else:
      results = self.execute_cpu(graph_fn,
                                 [images, rpn_box_encodings,
                                  rpn_objectness_predictions_with_background,
                                  rpn_features_to_crop, anchors], graph=g)

    expected_proposal_boxes = [
        [[0, 0, .5, .5], [.5, .5, 1, 1], [0, .5, .5, 1], [.5, 0, 1.0, .5]]
        + 4 * [4 * [0]],
        [[0, .5, .5, 1], [.5, 0, 1.0, .5], [0, 0, .5, .5], [.5, .5, 1, 1]]
        + 4 * [4 * [0]]]
    expected_proposal_scores = [[1, 1, 0, 0, 0, 0, 0, 0],
                                [1, 1, 0, 0, 0, 0, 0, 0]]
    expected_num_proposals = [4, 4]
    expected_raw_proposal_boxes = [[[0., 0., 0.5, 0.5], [0., 0.5, 0.5, 1.],
                                    [0.5, 0., 1., 0.5], [0.5, 0.5, 1., 1.]],
                                   [[0., 0., 0.5, 0.5], [0., 0.5, 0.5, 1.],
                                    [0.5, 0., 1., 0.5], [0.5, 0.5, 1., 1.]]]
    expected_raw_scores = [[[0., 1.], [1., 0.], [1., 0.], [0., 1.]],
                           [[1., 0.], [0., 1.], [0., 1.], [1., 0.]]]

    if pad_to_max_dimension is not None:
      expected_raw_proposal_boxes = (np.array(expected_raw_proposal_boxes) *
                                     32 / pad_to_max_dimension)
      expected_proposal_boxes = (np.array(expected_proposal_boxes) *
                                 32 / pad_to_max_dimension)

    self.assertAllClose(results[0], expected_num_proposals)
    for indx, num_proposals in enumerate(expected_num_proposals):
      self.assertAllClose(results[1][indx][0:num_proposals],
                          expected_proposal_boxes[indx][0:num_proposals])
      self.assertAllClose(results[2][indx][0:num_proposals],
                          expected_proposal_scores[indx][0:num_proposals])
    self.assertAllClose(results[3], expected_raw_proposal_boxes)
    self.assertAllClose(results[4], expected_raw_scores)

  def _test_postprocess_first_stage_only_train_mode(self,
                                                    pad_to_max_dimension=None):

    with test_utils.GraphContextOrNone() as g:
      model = self._build_model(
          is_training=True,
          number_of_stages=1, second_stage_batch_size=2,
          pad_to_max_dimension=pad_to_max_dimension)
    batch_size = 2

    def graph_fn():
      """A function with TF compute."""
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
      groundtruth_classes_list = [tf.constant([[1, 0], [0, 1]],
                                              dtype=tf.float32),
                                  tf.constant([[1, 0], [1, 0]],
                                              dtype=tf.float32)]
      groundtruth_weights_list = [
          tf.constant([1, 1], dtype=tf.float32),
          tf.constant([1, 1], dtype=tf.float32)
      ]
      _, true_image_shapes = model.preprocess(tf.zeros(image_shape))
      model.provide_groundtruth(
          groundtruth_boxes_list,
          groundtruth_classes_list,
          groundtruth_weights_list=groundtruth_weights_list)
      proposals = model.postprocess({
          'rpn_box_encodings': rpn_box_encodings,
          'rpn_objectness_predictions_with_background':
          rpn_objectness_predictions_with_background,
          'rpn_features_to_crop': rpn_features_to_crop,
          'anchors': anchors,
          'image_shape': image_shape}, true_image_shapes)
      return (proposals['detection_boxes'], proposals['detection_scores'],
              proposals['num_detections'],
              proposals['detection_multiclass_scores'],
              proposals['raw_detection_boxes'],
              proposals['raw_detection_scores'])

    expected_proposal_boxes = [
        [[0, 0, .5, .5], [.5, .5, 1, 1]], [[0, .5, .5, 1], [.5, 0, 1, .5]]]
    expected_proposal_scores = [[1, 1],
                                [1, 1]]
    expected_proposal_multiclass_scores = [[[0., 1.], [0., 1.]],
                                           [[0., 1.], [0., 1.]]]
    expected_raw_proposal_boxes = [[[0., 0., 0.5, 0.5], [0., 0.5, 0.5, 1.],
                                    [0.5, 0., 1., 0.5], [0.5, 0.5, 1., 1.]],
                                   [[0., 0., 0.5, 0.5], [0., 0.5, 0.5, 1.],
                                    [0.5, 0., 1., 0.5], [0.5, 0.5, 1., 1.]]]
    expected_raw_scores = [[[0., 1.], [0., 1.], [0., 1.], [0., 1.]],
                           [[0., 1.], [0., 1.], [0., 1.], [0., 1.]]]

    (proposal_boxes, proposal_scores, batch_num_detections,
     batch_multiclass_scores, raw_detection_boxes,
     raw_detection_scores) = self.execute_cpu(graph_fn, [], graph=g)
    for image_idx in range(batch_size):
      num_detections = int(batch_num_detections[image_idx])
      boxes = proposal_boxes[image_idx][:num_detections, :].tolist()
      scores = proposal_scores[image_idx][:num_detections].tolist()
      multiclass_scores = batch_multiclass_scores[
          image_idx][:num_detections, :].tolist()
      expected_boxes = expected_proposal_boxes[image_idx]
      expected_scores = expected_proposal_scores[image_idx]
      expected_multiclass_scores = expected_proposal_multiclass_scores[
          image_idx]
      self.assertTrue(
          test_utils.first_rows_close_as_set(boxes, expected_boxes))
      self.assertTrue(
          test_utils.first_rows_close_as_set(scores, expected_scores))
      self.assertTrue(
          test_utils.first_rows_close_as_set(multiclass_scores,
                                             expected_multiclass_scores))

    self.assertAllClose(raw_detection_boxes, expected_raw_proposal_boxes)
    self.assertAllClose(raw_detection_scores, expected_raw_scores)

  @parameterized.parameters(
      {'pad_to_max_dimension': 56},
      {'pad_to_max_dimension': None}
  )
  def test_postprocess_first_stage_only_train_mode_padded_image(
      self, pad_to_max_dimension):
    self._test_postprocess_first_stage_only_train_mode(pad_to_max_dimension)

  @parameterized.parameters(
      {'use_static_shapes': False, 'pad_to_max_dimension': None},
      {'use_static_shapes': True, 'pad_to_max_dimension': None},
      {'use_static_shapes': False, 'pad_to_max_dimension': 56},
      {'use_static_shapes': True, 'pad_to_max_dimension': 56},
  )
  def test_postprocess_second_stage_only_inference_mode(
      self, use_static_shapes=False,
      pad_to_max_dimension=None):
    batch_size = 2
    num_classes = 2
    image_shape = np.array((2, 36, 48, 3), dtype=np.int32)
    first_stage_max_proposals = 8
    total_num_padded_proposals = batch_size * first_stage_max_proposals

    with test_utils.GraphContextOrNone() as g:
      model = self._build_model(
          is_training=False,
          number_of_stages=2,
          second_stage_batch_size=6,
          use_matmul_crop_and_resize=use_static_shapes,
          clip_anchors_to_image=use_static_shapes,
          use_static_shapes=use_static_shapes,
          use_matmul_gather_in_matcher=use_static_shapes,
          pad_to_max_dimension=pad_to_max_dimension)
    def graph_fn(images,
                 refined_box_encodings,
                 class_predictions_with_background,
                 num_proposals,
                 proposal_boxes):
      """Function to construct tf graph for the test."""
      _, true_image_shapes = model.preprocess(images)
      detections = model.postprocess({
          'refined_box_encodings': refined_box_encodings,
          'class_predictions_with_background':
          class_predictions_with_background,
          'num_proposals': num_proposals,
          'proposal_boxes': proposal_boxes,
      }, true_image_shapes)
      return (detections['num_detections'], detections['detection_boxes'],
              detections['detection_scores'], detections['detection_classes'],
              detections['raw_detection_boxes'],
              detections['raw_detection_scores'],
              detections['detection_multiclass_scores'],
              detections['detection_anchor_indices'])

    proposal_boxes = np.array(
        [[[1, 1, 2, 3],
          [0, 0, 1, 1],
          [.5, .5, .6, .6],
          4*[0], 4*[0], 4*[0], 4*[0], 4*[0]],
         [[2, 3, 6, 8],
          [1, 2, 5, 3],
          4*[0], 4*[0], 4*[0], 4*[0], 4*[0], 4*[0]]], dtype=np.float32)

    num_proposals = np.array([3, 2], dtype=np.int32)
    refined_box_encodings = np.zeros(
        [total_num_padded_proposals, num_classes, 4], dtype=np.float32)
    class_predictions_with_background = np.ones(
        [total_num_padded_proposals, num_classes+1], dtype=np.float32)
    images = np.zeros(image_shape, dtype=np.float32)

    if use_static_shapes:
      results = self.execute(graph_fn,
                             [images, refined_box_encodings,
                              class_predictions_with_background,
                              num_proposals, proposal_boxes], graph=g)
    else:
      results = self.execute_cpu(graph_fn,
                                 [images, refined_box_encodings,
                                  class_predictions_with_background,
                                  num_proposals, proposal_boxes], graph=g)
    # Note that max_total_detections=5 in the NMS config.
    expected_num_detections = [5, 4]
    expected_detection_classes = [[0, 0, 0, 1, 1], [0, 0, 1, 1, 0]]
    expected_detection_scores = [[1, 1, 1, 1, 1], [1, 1, 1, 1, 0]]
    expected_multiclass_scores = [[[1, 1, 1],
                                   [1, 1, 1],
                                   [1, 1, 1],
                                   [1, 1, 1],
                                   [1, 1, 1]],
                                  [[1, 1, 1],
                                   [1, 1, 1],
                                   [1, 1, 1],
                                   [1, 1, 1],
                                   [0, 0, 0]]]
    # Note that a single anchor can be used for multiple detections (predictions
    # are made independently per class).
    expected_anchor_indices = [[0, 1, 2, 0, 1],
                               [0, 1, 0, 1]]

    h = float(image_shape[1])
    w = float(image_shape[2])
    expected_raw_detection_boxes = np.array(
        [[[1 / h, 1 / w, 2 / h, 3 / w], [0, 0, 1 / h, 1 / w],
          [.5 / h, .5 / w, .6 / h, .6 / w], 4 * [0], 4 * [0], 4 * [0], 4 * [0],
          4 * [0]],
         [[2 / h, 3 / w, 6 / h, 8 / w], [1 / h, 2 / w, 5 / h, 3 / w], 4 * [0],
          4 * [0], 4 * [0], 4 * [0], 4 * [0], 4 * [0]]],
        dtype=np.float32)

    self.assertAllClose(results[0], expected_num_detections)

    for indx, num_proposals in enumerate(expected_num_detections):
      self.assertAllClose(results[2][indx][0:num_proposals],
                          expected_detection_scores[indx][0:num_proposals])
      self.assertAllClose(results[3][indx][0:num_proposals],
                          expected_detection_classes[indx][0:num_proposals])
      self.assertAllClose(results[6][indx][0:num_proposals],
                          expected_multiclass_scores[indx][0:num_proposals])
      self.assertAllClose(results[7][indx][0:num_proposals],
                          expected_anchor_indices[indx][0:num_proposals])

    self.assertAllClose(results[4], expected_raw_detection_boxes)
    self.assertAllClose(results[5],
                        class_predictions_with_background.reshape([-1, 8, 3]))
    if not use_static_shapes:
      self.assertAllEqual(results[1].shape, [2, 5, 4])

  def test_preprocess_preserves_dynamic_input_shapes(self):
    width = tf.random.uniform([], minval=5, maxval=10, dtype=tf.int32)
    batch = tf.random.uniform([], minval=2, maxval=3, dtype=tf.int32)
    shape = tf.stack([batch, 5, width, 3])
    image = tf.random.uniform(shape)
    model = self._build_model(
        is_training=False, number_of_stages=2, second_stage_batch_size=6)
    preprocessed_inputs, _ = model.preprocess(image)
    self.assertTrue(
        preprocessed_inputs.shape.is_compatible_with([None, 5, None, 3]))

  def test_preprocess_preserves_static_input_shapes(self):
    shape = tf.stack([2, 5, 5, 3])
    image = tf.random.uniform(shape)
    model = self._build_model(
        is_training=False, number_of_stages=2, second_stage_batch_size=6)
    preprocessed_inputs, _ = model.preprocess(image)
    self.assertTrue(
        preprocessed_inputs.shape.is_compatible_with([2, 5, 5, 3]))

  # TODO(rathodv): Split test into two - with and without masks.
  def test_loss_first_stage_only_mode(self):
    with test_utils.GraphContextOrNone() as g:
      model = self._build_model(
          is_training=True,
          number_of_stages=1, second_stage_batch_size=6)
    batch_size = 2
    def graph_fn():
      """A function with TF compute."""
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
      groundtruth_classes_list = [tf.constant([[1, 0], [0, 1]],
                                              dtype=tf.float32),
                                  tf.constant([[1, 0], [1, 0]],
                                              dtype=tf.float32)]

      prediction_dict = {
          'rpn_box_encodings': rpn_box_encodings,
          'rpn_objectness_predictions_with_background':
          rpn_objectness_predictions_with_background,
          'image_shape': image_shape,
          'anchors': anchors
      }
      _, true_image_shapes = model.preprocess(tf.zeros(image_shape))
      model.provide_groundtruth(groundtruth_boxes_list,
                                groundtruth_classes_list)
      loss_dict = model.loss(prediction_dict, true_image_shapes)
      self.assertNotIn('Loss/BoxClassifierLoss/localization_loss',
                       loss_dict)
      self.assertNotIn('Loss/BoxClassifierLoss/classification_loss',
                       loss_dict)
      return (loss_dict['Loss/RPNLoss/localization_loss'],
              loss_dict['Loss/RPNLoss/objectness_loss'])
    loc_loss, obj_loss = self.execute_cpu(graph_fn, [], graph=g)
    self.assertAllClose(loc_loss, 0)
    self.assertAllClose(obj_loss, 0)

  # TODO(rathodv): Split test into two - with and without masks.
  def test_loss_full(self):
    with test_utils.GraphContextOrNone() as g:
      model = self._build_model(
          is_training=True,
          number_of_stages=2, second_stage_batch_size=6)
    batch_size = 3
    def graph_fn():
      """A function with TF compute."""
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
          [[[-10, 13], [10, -10], [10, -11], [-10, 12]],
           [[10, -10], [-10, 13], [-10, 12], [10, -11]],
           [[10, -10], [-10, 13], [-10, 12], [10, -11]]],
          dtype=tf.float32)
      image_shape = tf.constant([batch_size, 32, 32, 3], dtype=tf.int32)

      num_proposals = tf.constant([6, 6, 6], dtype=tf.int32)
      proposal_boxes = tf.constant(
          3 * [[[0, 0, 16, 16], [0, 16, 16, 32], [16, 0, 32, 16],
                [16, 16, 32, 32], [0, 0, 16, 16], [0, 16, 16, 32]]],
          dtype=tf.float32)
      refined_box_encodings = tf.zeros(
          (batch_size * model.max_num_proposals,
           model.num_classes,
           BOX_CODE_SIZE), dtype=tf.float32)
      class_predictions_with_background = tf.constant(
          [
              [-10, 10, -10],  # first image
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
              [-10, 10, -10],
              [10, -10, -10],  # third image
              [-10, 10, -10],
              [-10, 10, -10],
              [10, -10, -10],
              [10, -10, -10],
              [-10, 10, -10]
          ],
          dtype=tf.float32)

      mask_predictions_logits = 20 * tf.ones((batch_size *
                                              model.max_num_proposals,
                                              model.num_classes,
                                              14, 14),
                                             dtype=tf.float32)

      groundtruth_boxes_list = [
          tf.constant([[0, 0, .5, .5], [.5, .5, 1, 1]], dtype=tf.float32),
          tf.constant([[0, .5, .5, 1], [.5, 0, 1, .5]], dtype=tf.float32),
          tf.constant([[0, .5, .5, 1], [.5, 0, 1, 1]], dtype=tf.float32)
      ]
      groundtruth_classes_list = [
          tf.constant([[1, 0], [0, 1]], dtype=tf.float32),
          tf.constant([[1, 0], [1, 0]], dtype=tf.float32),
          tf.constant([[1, 0], [0, 1]], dtype=tf.float32)
      ]

      # Set all elements of groundtruth mask to 1.0. In this case all proposal
      # crops of the groundtruth masks should return a mask that covers the
      # entire proposal. Thus, if mask_predictions_logits element values are all
      # greater than 20, the loss should be zero.
      groundtruth_masks_list = [
          tf.convert_to_tensor(np.ones((2, 32, 32)), dtype=tf.float32),
          tf.convert_to_tensor(np.ones((2, 32, 32)), dtype=tf.float32),
          tf.convert_to_tensor(np.ones((2, 32, 32)), dtype=tf.float32)
      ]
      groundtruth_weights_list = [
          tf.constant([1, 1], dtype=tf.float32),
          tf.constant([1, 1], dtype=tf.float32),
          tf.constant([1, 0], dtype=tf.float32)
      ]
      prediction_dict = {
          'rpn_box_encodings': rpn_box_encodings,
          'rpn_objectness_predictions_with_background':
          rpn_objectness_predictions_with_background,
          'image_shape': image_shape,
          'anchors': anchors,
          'refined_box_encodings': refined_box_encodings,
          'class_predictions_with_background':
              class_predictions_with_background,
          'proposal_boxes': proposal_boxes,
          'num_proposals': num_proposals,
          'mask_predictions': mask_predictions_logits
      }
      _, true_image_shapes = model.preprocess(tf.zeros(image_shape))
      model.provide_groundtruth(
          groundtruth_boxes_list,
          groundtruth_classes_list,
          groundtruth_masks_list,
          groundtruth_weights_list=groundtruth_weights_list)
      loss_dict = model.loss(prediction_dict, true_image_shapes)
      return (loss_dict['Loss/RPNLoss/localization_loss'],
              loss_dict['Loss/RPNLoss/objectness_loss'],
              loss_dict['Loss/BoxClassifierLoss/localization_loss'],
              loss_dict['Loss/BoxClassifierLoss/classification_loss'],
              loss_dict['Loss/BoxClassifierLoss/mask_loss'])
    (rpn_loc_loss, rpn_obj_loss, box_loc_loss, box_cls_loss,
     box_mask_loss) = self.execute_cpu(graph_fn, [], graph=g)
    self.assertAllClose(rpn_loc_loss, 0)
    self.assertAllClose(rpn_obj_loss, 0)
    self.assertAllClose(box_loc_loss, 0)
    self.assertAllClose(box_cls_loss, 0)
    self.assertAllClose(box_mask_loss, 0)

  def test_loss_full_zero_padded_proposals(self):
    with test_utils.GraphContextOrNone() as g:
      model = self._build_model(
          is_training=True, number_of_stages=2, second_stage_batch_size=6)
    batch_size = 1
    def graph_fn():
      """A function with TF compute."""
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

      mask_predictions_logits = 20 * tf.ones((batch_size *
                                              model.max_num_proposals,
                                              model.num_classes,
                                              14, 14),
                                             dtype=tf.float32)

      groundtruth_boxes_list = [
          tf.constant([[0, 0, .5, .5]], dtype=tf.float32)]
      groundtruth_classes_list = [tf.constant([[1, 0]], dtype=tf.float32)]

      # Set all elements of groundtruth mask to 1.0. In this case all proposal
      # crops of the groundtruth masks should return a mask that covers the
      # entire proposal. Thus, if mask_predictions_logits element values are all
      # greater than 20, the loss should be zero.
      groundtruth_masks_list = [tf.convert_to_tensor(np.ones((1, 32, 32)),
                                                     dtype=tf.float32)]

      prediction_dict = {
          'rpn_box_encodings': rpn_box_encodings,
          'rpn_objectness_predictions_with_background':
          rpn_objectness_predictions_with_background,
          'image_shape': image_shape,
          'anchors': anchors,
          'refined_box_encodings': refined_box_encodings,
          'class_predictions_with_background':
              class_predictions_with_background,
          'proposal_boxes': proposal_boxes,
          'num_proposals': num_proposals,
          'mask_predictions': mask_predictions_logits
      }
      _, true_image_shapes = model.preprocess(tf.zeros(image_shape))
      model.provide_groundtruth(groundtruth_boxes_list,
                                groundtruth_classes_list,
                                groundtruth_masks_list)
      loss_dict = model.loss(prediction_dict, true_image_shapes)
      return (loss_dict['Loss/RPNLoss/localization_loss'],
              loss_dict['Loss/RPNLoss/objectness_loss'],
              loss_dict['Loss/BoxClassifierLoss/localization_loss'],
              loss_dict['Loss/BoxClassifierLoss/classification_loss'],
              loss_dict['Loss/BoxClassifierLoss/mask_loss'])
    (rpn_loc_loss, rpn_obj_loss, box_loc_loss, box_cls_loss,
     box_mask_loss) = self.execute_cpu(graph_fn, [], graph=g)
    self.assertAllClose(rpn_loc_loss, 0)
    self.assertAllClose(rpn_obj_loss, 0)
    self.assertAllClose(box_loc_loss, 0)
    self.assertAllClose(box_cls_loss, 0)
    self.assertAllClose(box_mask_loss, 0)

  def test_loss_full_multiple_label_groundtruth(self):
    with test_utils.GraphContextOrNone() as g:
      model = self._build_model(
          is_training=True,
          number_of_stages=2, second_stage_batch_size=6,
          softmax_second_stage_classification_loss=False)
    batch_size = 1
    def graph_fn():
      """A function with TF compute."""
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

      # second_stage_localization_loss should only be computed for predictions
      # that match groundtruth. For multiple label groundtruth boxes, the loss
      # should only be computed once for the label with the smaller index.
      refined_box_encodings = tf.constant(
          [[[0, 0, 0, 0], [1, 1, -1, -1]],
           [[1, 1, -1, -1], [1, 1, 1, 1]],
           [[1, 1, -1, -1], [1, 1, 1, 1]],
           [[1, 1, -1, -1], [1, 1, 1, 1]],
           [[1, 1, -1, -1], [1, 1, 1, 1]],
           [[1, 1, -1, -1], [1, 1, 1, 1]]], dtype=tf.float32)
      class_predictions_with_background = tf.constant(
          [[-100, 100, 100],
           [100, -100, -100],
           [100, -100, -100],
           [0, 0, 0],  # begin paddings
           [0, 0, 0],
           [0, 0, 0]], dtype=tf.float32)

      mask_predictions_logits = 20 * tf.ones((batch_size *
                                              model.max_num_proposals,
                                              model.num_classes,
                                              14, 14),
                                             dtype=tf.float32)

      groundtruth_boxes_list = [
          tf.constant([[0, 0, .5, .5]], dtype=tf.float32)]
      # Box contains two ground truth labels.
      groundtruth_classes_list = [tf.constant([[1, 1]], dtype=tf.float32)]

      # Set all elements of groundtruth mask to 1.0. In this case all proposal
      # crops of the groundtruth masks should return a mask that covers the
      # entire proposal. Thus, if mask_predictions_logits element values are all
      # greater than 20, the loss should be zero.
      groundtruth_masks_list = [tf.convert_to_tensor(np.ones((1, 32, 32)),
                                                     dtype=tf.float32)]

      prediction_dict = {
          'rpn_box_encodings': rpn_box_encodings,
          'rpn_objectness_predictions_with_background':
          rpn_objectness_predictions_with_background,
          'image_shape': image_shape,
          'anchors': anchors,
          'refined_box_encodings': refined_box_encodings,
          'class_predictions_with_background':
              class_predictions_with_background,
          'proposal_boxes': proposal_boxes,
          'num_proposals': num_proposals,
          'mask_predictions': mask_predictions_logits
      }
      _, true_image_shapes = model.preprocess(tf.zeros(image_shape))
      model.provide_groundtruth(groundtruth_boxes_list,
                                groundtruth_classes_list,
                                groundtruth_masks_list)
      loss_dict = model.loss(prediction_dict, true_image_shapes)
      return (loss_dict['Loss/RPNLoss/localization_loss'],
              loss_dict['Loss/RPNLoss/objectness_loss'],
              loss_dict['Loss/BoxClassifierLoss/localization_loss'],
              loss_dict['Loss/BoxClassifierLoss/classification_loss'],
              loss_dict['Loss/BoxClassifierLoss/mask_loss'])
    (rpn_loc_loss, rpn_obj_loss, box_loc_loss, box_cls_loss,
     box_mask_loss) = self.execute_cpu(graph_fn, [], graph=g)
    self.assertAllClose(rpn_loc_loss, 0)
    self.assertAllClose(rpn_obj_loss, 0)
    self.assertAllClose(box_loc_loss, 0)
    self.assertAllClose(box_cls_loss, 0)
    self.assertAllClose(box_mask_loss, 0)

  @parameterized.parameters(
      {'use_static_shapes': False, 'shared_boxes': False},
      {'use_static_shapes': False, 'shared_boxes': True},
      {'use_static_shapes': True, 'shared_boxes': False},
      {'use_static_shapes': True, 'shared_boxes': True},
  )
  def test_loss_full_zero_padded_proposals_nonzero_loss_with_two_images(
      self, use_static_shapes=False, shared_boxes=False):
    batch_size = 2
    first_stage_max_proposals = 8
    second_stage_batch_size = 6
    num_classes = 2
    with test_utils.GraphContextOrNone() as g:
      model = self._build_model(
          is_training=True,
          number_of_stages=2,
          second_stage_batch_size=second_stage_batch_size,
          first_stage_max_proposals=first_stage_max_proposals,
          num_classes=num_classes,
          use_matmul_crop_and_resize=use_static_shapes,
          clip_anchors_to_image=use_static_shapes,
          use_static_shapes=use_static_shapes)

    def graph_fn(anchors, rpn_box_encodings,
                 rpn_objectness_predictions_with_background, images,
                 num_proposals, proposal_boxes, refined_box_encodings,
                 class_predictions_with_background, groundtruth_boxes,
                 groundtruth_classes):
      """Function to construct tf graph for the test."""
      prediction_dict = {
          'rpn_box_encodings': rpn_box_encodings,
          'rpn_objectness_predictions_with_background':
          rpn_objectness_predictions_with_background,
          'image_shape': tf.shape(images),
          'anchors': anchors,
          'refined_box_encodings': refined_box_encodings,
          'class_predictions_with_background':
          class_predictions_with_background,
          'proposal_boxes': proposal_boxes,
          'num_proposals': num_proposals
      }
      _, true_image_shapes = model.preprocess(images)
      model.provide_groundtruth(tf.unstack(groundtruth_boxes),
                                tf.unstack(groundtruth_classes))
      loss_dict = model.loss(prediction_dict, true_image_shapes)
      return (loss_dict['Loss/RPNLoss/localization_loss'],
              loss_dict['Loss/RPNLoss/objectness_loss'],
              loss_dict['Loss/BoxClassifierLoss/localization_loss'],
              loss_dict['Loss/BoxClassifierLoss/classification_loss'])

    anchors = np.array(
        [[0, 0, 16, 16],
         [0, 16, 16, 32],
         [16, 0, 32, 16],
         [16, 16, 32, 32]], dtype=np.float32)
    rpn_box_encodings = np.zeros(
        [batch_size, anchors.shape[1], BOX_CODE_SIZE], dtype=np.float32)
    # use different numbers for the objectness category to break ties in
    # order of boxes returned by NMS
    rpn_objectness_predictions_with_background = np.array(
        [[[-10, 13],
          [10, -10],
          [10, -11],
          [10, -12]],
         [[-10, 13],
          [10, -10],
          [10, -11],
          [10, -12]]], dtype=np.float32)
    images = np.zeros([batch_size, 32, 32, 3], dtype=np.float32)

    # box_classifier_batch_size is 6, but here we assume that the number of
    # actual proposals (not counting zero paddings) is fewer.
    num_proposals = np.array([3, 2], dtype=np.int32)
    proposal_boxes = np.array(
        [[[0, 0, 16, 16],
          [0, 16, 16, 32],
          [16, 0, 32, 16],
          [0, 0, 0, 0],  # begin paddings
          [0, 0, 0, 0],
          [0, 0, 0, 0]],
         [[0, 0, 16, 16],
          [0, 16, 16, 32],
          [0, 0, 0, 0],  # begin paddings
          [0, 0, 0, 0],
          [0, 0, 0, 0],
          [0, 0, 0, 0]]], dtype=np.float32)

    refined_box_encodings = np.zeros(
        (batch_size * second_stage_batch_size, 1
         if shared_boxes else num_classes, BOX_CODE_SIZE),
        dtype=np.float32)
    class_predictions_with_background = np.array(
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
         [0, 0, 0],], dtype=np.float32)

    # The first groundtruth box is 4/5 of the anchor size in both directions
    # experiencing a loss of:
    # 2 * SmoothL1(5 * log(4/5)) / num_proposals
    #   = 2 * (abs(5 * log(1/2)) - .5) / 3
    # The second groundtruth box is identical to the prediction and thus
    # experiences zero loss.
    # Total average loss is (abs(5 * log(1/2)) - .5) / 3.
    groundtruth_boxes = np.stack([
        np.array([[0.05, 0.05, 0.45, 0.45]], dtype=np.float32),
        np.array([[0.0, 0.0, 0.5, 0.5]], dtype=np.float32)])
    groundtruth_classes = np.stack([np.array([[1, 0]], dtype=np.float32),
                                    np.array([[0, 1]], dtype=np.float32)])

    execute_fn = self.execute_cpu
    if use_static_shapes:
      execute_fn = self.execute

    results = execute_fn(graph_fn, [
        anchors, rpn_box_encodings, rpn_objectness_predictions_with_background,
        images, num_proposals, proposal_boxes, refined_box_encodings,
        class_predictions_with_background, groundtruth_boxes,
        groundtruth_classes
    ], graph=g)

    exp_loc_loss = (-5 * np.log(.8) - 0.5) / 3.0

    self.assertAllClose(results[0], exp_loc_loss, rtol=1e-4, atol=1e-4)
    self.assertAllClose(results[1], 0.0)
    self.assertAllClose(results[2], exp_loc_loss, rtol=1e-4, atol=1e-4)
    self.assertAllClose(results[3], 0.0)

  def test_loss_with_hard_mining(self):
    with test_utils.GraphContextOrNone() as g:
      model = self._build_model(is_training=True,
                                number_of_stages=2,
                                second_stage_batch_size=None,
                                first_stage_max_proposals=6,
                                hard_mining=True)
    batch_size = 1
    def graph_fn():
      """A function with TF compute."""
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
      # The second groundtruth box is 46/50 of the anchor size in both
      # directions experiencing a loss of:
      # 2 * SmoothL1(5 * log(42/50)) / num_proposals
      #   = 2 * (.5(5 * log(.92))^2 - .5) / 3.
      # Since the first groundtruth box experiences greater loss, and we have
      # set num_hard_examples=1 in the HardMiner, the final localization loss
      # corresponds to that of the first groundtruth box.
      groundtruth_boxes_list = [
          tf.constant([[0.05, 0.05, 0.45, 0.45],
                       [0.02, 0.52, 0.48, 0.98],], dtype=tf.float32)]
      groundtruth_classes_list = [tf.constant([[1, 0], [0, 1]],
                                              dtype=tf.float32)]

      prediction_dict = {
          'rpn_box_encodings': rpn_box_encodings,
          'rpn_objectness_predictions_with_background':
          rpn_objectness_predictions_with_background,
          'image_shape': image_shape,
          'anchors': anchors,
          'refined_box_encodings': refined_box_encodings,
          'class_predictions_with_background':
              class_predictions_with_background,
          'proposal_boxes': proposal_boxes,
          'num_proposals': num_proposals
      }
      _, true_image_shapes = model.preprocess(tf.zeros(image_shape))
      model.provide_groundtruth(groundtruth_boxes_list,
                                groundtruth_classes_list)
      loss_dict = model.loss(prediction_dict, true_image_shapes)
      return (loss_dict['Loss/BoxClassifierLoss/localization_loss'],
              loss_dict['Loss/BoxClassifierLoss/classification_loss'])
    loc_loss, cls_loss = self.execute_cpu(graph_fn, [], graph=g)
    exp_loc_loss = 2 * (-5 * np.log(.8) - 0.5) / 3.0
    self.assertAllClose(loc_loss, exp_loc_loss)
    self.assertAllClose(cls_loss, 0)

  def test_loss_with_hard_mining_and_losses_mask(self):
    with test_utils.GraphContextOrNone() as g:
      model = self._build_model(is_training=True,
                                number_of_stages=2,
                                second_stage_batch_size=None,
                                first_stage_max_proposals=6,
                                hard_mining=True)
    batch_size = 2
    number_of_proposals = 3
    def graph_fn():
      """A function with TF compute."""
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
            [10, -12]],
           [[-10, 13],
            [-10, 12],
            [10, -11],
            [10, -12]]], dtype=tf.float32)
      image_shape = tf.constant([batch_size, 32, 32, 3], dtype=tf.int32)

      # box_classifier_batch_size is 6, but here we assume that the number of
      # actual proposals (not counting zero paddings) is fewer (3).
      num_proposals = tf.constant([number_of_proposals, number_of_proposals],
                                  dtype=tf.int32)
      proposal_boxes = tf.constant(
          [[[0, 0, 16, 16],  # first image
            [0, 16, 16, 32],
            [16, 0, 32, 16],
            [0, 0, 0, 0],  # begin paddings
            [0, 0, 0, 0],
            [0, 0, 0, 0]],
           [[0, 0, 16, 16],  # second image
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
           [0, 0, 0],
           [-10, 10, -10],  # second image
           [-10, -10, 10],
           [10, -10, -10],
           [0, 0, 0],  # begin paddings
           [0, 0, 0],
           [0, 0, 0]], dtype=tf.float32)

      # The first groundtruth box is 4/5 of the anchor size in both directions
      # experiencing a loss of:
      # 2 * SmoothL1(5 * log(4/5)) / (num_proposals * batch_size)
      #   = 2 * (abs(5 * log(1/2)) - .5) / 3
      # The second groundtruth box is 46/50 of the anchor size in both
      # directions experiencing a loss of:
      # 2 * SmoothL1(5 * log(42/50)) / (num_proposals * batch_size)
      #   = 2 * (.5(5 * log(.92))^2 - .5) / 3.
      # Since the first groundtruth box experiences greater loss, and we have
      # set num_hard_examples=1 in the HardMiner, the final localization loss
      # corresponds to that of the first groundtruth box.
      groundtruth_boxes_list = [
          tf.constant([[0.05, 0.05, 0.45, 0.45],
                       [0.02, 0.52, 0.48, 0.98]], dtype=tf.float32),
          tf.constant([[0.05, 0.05, 0.45, 0.45],
                       [0.02, 0.52, 0.48, 0.98]], dtype=tf.float32)]
      groundtruth_classes_list = [
          tf.constant([[1, 0], [0, 1]], dtype=tf.float32),
          tf.constant([[1, 0], [0, 1]], dtype=tf.float32)]
      is_annotated_list = [tf.constant(True, dtype=tf.bool),
                           tf.constant(False, dtype=tf.bool)]

      prediction_dict = {
          'rpn_box_encodings': rpn_box_encodings,
          'rpn_objectness_predictions_with_background':
          rpn_objectness_predictions_with_background,
          'image_shape': image_shape,
          'anchors': anchors,
          'refined_box_encodings': refined_box_encodings,
          'class_predictions_with_background':
              class_predictions_with_background,
          'proposal_boxes': proposal_boxes,
          'num_proposals': num_proposals
      }
      _, true_image_shapes = model.preprocess(tf.zeros(image_shape))
      model.provide_groundtruth(groundtruth_boxes_list,
                                groundtruth_classes_list,
                                is_annotated_list=is_annotated_list)
      loss_dict = model.loss(prediction_dict, true_image_shapes)
      return (loss_dict['Loss/BoxClassifierLoss/localization_loss'],
              loss_dict['Loss/BoxClassifierLoss/classification_loss'])
    exp_loc_loss = (2 * (-5 * np.log(.8) - 0.5) /
                    (number_of_proposals * batch_size))
    loc_loss, cls_loss = self.execute_cpu(graph_fn, [], graph=g)
    self.assertAllClose(loc_loss, exp_loc_loss)
    self.assertAllClose(cls_loss, 0)

  def test_restore_map_for_classification_ckpt(self):
    if tf_version.is_tf2(): self.skipTest('Skipping TF1 only test.')
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
      with self.test_session(graph=test_graph_classification) as sess:
        sess.run(init_op)
        saved_model_path = saver.save(sess, save_path)

    # Create tensorflow detection graph and load variables from
    # classification checkpoint.
    test_graph_detection = tf.Graph()
    with test_graph_detection.as_default():
      model = self._build_model(
          is_training=False,
          number_of_stages=2, second_stage_batch_size=6)

      inputs_shape = (2, 20, 20, 3)
      inputs = tf.cast(tf.random_uniform(
          inputs_shape, minval=0, maxval=255, dtype=tf.int32), dtype=tf.float32)
      preprocessed_inputs, true_image_shapes = model.preprocess(inputs)
      prediction_dict = model.predict(preprocessed_inputs, true_image_shapes)
      model.postprocess(prediction_dict, true_image_shapes)
      var_map = model.restore_map(fine_tune_checkpoint_type='classification')
      self.assertIsInstance(var_map, dict)
      saver = tf.train.Saver(var_map)
      with self.test_session(graph=test_graph_classification) as sess:
        saver.restore(sess, saved_model_path)
        for var in sess.run(tf.report_uninitialized_variables()):
          self.assertNotIn(model.first_stage_feature_extractor_scope, var)
          self.assertNotIn(model.second_stage_feature_extractor_scope, var)

  def test_restore_map_for_detection_ckpt(self):
    if tf_version.is_tf2(): self.skipTest('Skipping TF1 only test.')
    # Define mock tensorflow classification graph and save variables.
    # Define first detection graph and save variables.
    test_graph_detection1 = tf.Graph()
    with test_graph_detection1.as_default():
      model = self._build_model(
          is_training=False,
          number_of_stages=2, second_stage_batch_size=6)
      inputs_shape = (2, 20, 20, 3)
      inputs = tf.cast(tf.random_uniform(
          inputs_shape, minval=0, maxval=255, dtype=tf.int32), dtype=tf.float32)
      preprocessed_inputs, true_image_shapes = model.preprocess(inputs)
      prediction_dict = model.predict(preprocessed_inputs, true_image_shapes)
      model.postprocess(prediction_dict, true_image_shapes)
      another_variable = tf.Variable([17.0], name='another_variable')  # pylint: disable=unused-variable
      init_op = tf.global_variables_initializer()
      saver = tf.train.Saver()
      save_path = self.get_temp_dir()
      with self.test_session(graph=test_graph_detection1) as sess:
        sess.run(init_op)
        saved_model_path = saver.save(sess, save_path)

    # Define second detection graph and restore variables.
    test_graph_detection2 = tf.Graph()
    with test_graph_detection2.as_default():
      model2 = self._build_model(is_training=False,
                                 number_of_stages=2,
                                 second_stage_batch_size=6, num_classes=42)

      inputs_shape2 = (2, 20, 20, 3)
      inputs2 = tf.cast(tf.random_uniform(
          inputs_shape2, minval=0, maxval=255, dtype=tf.int32),
                        dtype=tf.float32)
      preprocessed_inputs2, true_image_shapes = model2.preprocess(inputs2)
      prediction_dict2 = model2.predict(preprocessed_inputs2, true_image_shapes)
      model2.postprocess(prediction_dict2, true_image_shapes)
      another_variable = tf.Variable([17.0], name='another_variable')  # pylint: disable=unused-variable
      var_map = model2.restore_map(fine_tune_checkpoint_type='detection')
      self.assertIsInstance(var_map, dict)
      saver = tf.train.Saver(var_map)
      with self.test_session(graph=test_graph_detection2) as sess:
        saver.restore(sess, saved_model_path)
        uninitialized_vars_list = sess.run(tf.report_uninitialized_variables())
        self.assertIn(six.b('another_variable'), uninitialized_vars_list)
        for var in uninitialized_vars_list:
          self.assertNotIn(
              six.b(model2.first_stage_feature_extractor_scope), var)
          self.assertNotIn(
              six.b(model2.second_stage_feature_extractor_scope), var)

  def test_load_all_det_checkpoint_vars(self):
    if tf_version.is_tf2(): self.skipTest('Skipping TF1 only test.')
    test_graph_detection = tf.Graph()
    with test_graph_detection.as_default():
      model = self._build_model(
          is_training=False,
          number_of_stages=2,
          second_stage_batch_size=6,
          num_classes=42)

      inputs_shape = (2, 20, 20, 3)
      inputs = tf.cast(
          tf.random_uniform(inputs_shape, minval=0, maxval=255, dtype=tf.int32),
          dtype=tf.float32)
      preprocessed_inputs, true_image_shapes = model.preprocess(inputs)
      prediction_dict = model.predict(preprocessed_inputs, true_image_shapes)
      model.postprocess(prediction_dict, true_image_shapes)
      another_variable = tf.Variable([17.0], name='another_variable')  # pylint: disable=unused-variable
      var_map = model.restore_map(
          fine_tune_checkpoint_type='detection',
          load_all_detection_checkpoint_vars=True)
      self.assertIsInstance(var_map, dict)
      self.assertIn('another_variable', var_map)


if __name__ == '__main__':
  tf.test.main()
