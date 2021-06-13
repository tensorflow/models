# Lint as: python2, python3
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
"""Tests for model_builder under TensorFlow 2.X."""

import os
import unittest

from absl.testing import parameterized
import tensorflow.compat.v1 as tf

from google.protobuf import text_format
from object_detection.builders import model_builder
from object_detection.builders import model_builder_test
from object_detection.core import losses
from object_detection.meta_architectures import deepmac_meta_arch
from object_detection.models import center_net_hourglass_feature_extractor
from object_detection.models.keras_models import hourglass_network
from object_detection.protos import center_net_pb2
from object_detection.protos import model_pb2
from object_detection.utils import tf_version


@unittest.skipIf(tf_version.is_tf1(), 'Skipping TF2.X only test.')
class ModelBuilderTF2Test(
    model_builder_test.ModelBuilderTest, parameterized.TestCase):

  def default_ssd_feature_extractor(self):
    return 'ssd_resnet50_v1_fpn_keras'

  def default_faster_rcnn_feature_extractor(self):
    return 'faster_rcnn_resnet101_keras'

  def ssd_feature_extractors(self):
    return model_builder.SSD_KERAS_FEATURE_EXTRACTOR_CLASS_MAP

  def get_override_base_feature_extractor_hyperparams(self, extractor_type):
    return extractor_type in {}

  def faster_rcnn_feature_extractors(self):
    return model_builder.FASTER_RCNN_KERAS_FEATURE_EXTRACTOR_CLASS_MAP

  def get_fake_label_map_file_path(self):
    keypoint_spec_text = """
    item {
      name: "/m/01g317"
      id: 1
      display_name: "person"
      keypoints {
        id: 0
        label: 'nose'
      }
      keypoints {
        id: 1
        label: 'left_shoulder'
      }
      keypoints {
        id: 2
        label: 'right_shoulder'
      }
      keypoints {
        id: 3
        label: 'hip'
      }
    }
    """
    keypoint_label_map_path = os.path.join(
        self.get_temp_dir(), 'keypoint_label_map')
    with tf.gfile.Open(keypoint_label_map_path, 'wb') as f:
      f.write(keypoint_spec_text)
    return keypoint_label_map_path

  def get_fake_keypoint_proto(self, customize_head_params=False):
    task_proto_txt = """
      task_name: "human_pose"
      task_loss_weight: 0.9
      keypoint_regression_loss_weight: 1.0
      keypoint_heatmap_loss_weight: 0.1
      keypoint_offset_loss_weight: 0.5
      heatmap_bias_init: 2.14
      keypoint_class_name: "/m/01g317"
      loss {
        classification_loss {
          penalty_reduced_logistic_focal_loss {
            alpha: 3.0
            beta: 4.0
          }
        }
        localization_loss {
          l1_localization_loss {
          }
        }
      }
      keypoint_label_to_std {
        key: "nose"
        value: 0.3
      }
      keypoint_label_to_std {
        key: "hip"
        value: 0.0
      }
      keypoint_candidate_score_threshold: 0.3
      num_candidates_per_keypoint: 12
      peak_max_pool_kernel_size: 5
      unmatched_keypoint_score: 0.05
      box_scale: 1.7
      candidate_search_scale: 0.2
      candidate_ranking_mode: "score_distance_ratio"
      offset_peak_radius: 3
      per_keypoint_offset: true
      predict_depth: true
      per_keypoint_depth: true
      keypoint_depth_loss_weight: 0.3
    """
    if customize_head_params:
      task_proto_txt += """
      heatmap_head_params {
        num_filters: 64
        num_filters: 32
        kernel_sizes: 5
        kernel_sizes: 3
      }
      offset_head_params {
        num_filters: 128
        num_filters: 64
        kernel_sizes: 5
        kernel_sizes: 3
      }
      """
    config = text_format.Merge(task_proto_txt,
                               center_net_pb2.CenterNet.KeypointEstimation())
    return config

  def get_fake_object_center_proto(self, customize_head_params=False):
    proto_txt = """
      object_center_loss_weight: 0.5
      heatmap_bias_init: 3.14
      min_box_overlap_iou: 0.2
      max_box_predictions: 15
      classification_loss {
        penalty_reduced_logistic_focal_loss {
          alpha: 3.0
          beta: 4.0
        }
      }
    """
    if customize_head_params:
      proto_txt += """
      center_head_params {
        num_filters: 64
        num_filters: 32
        kernel_sizes: 5
        kernel_sizes: 3
      }
      """
    return text_format.Merge(proto_txt,
                             center_net_pb2.CenterNet.ObjectCenterParams())

  def get_fake_object_center_from_keypoints_proto(self):
    proto_txt = """
      object_center_loss_weight: 0.5
      heatmap_bias_init: 3.14
      min_box_overlap_iou: 0.2
      max_box_predictions: 15
      classification_loss {
        penalty_reduced_logistic_focal_loss {
          alpha: 3.0
          beta: 4.0
        }
      }
      keypoint_weights_for_center: 1.0
      keypoint_weights_for_center: 0.0
      keypoint_weights_for_center: 1.0
      keypoint_weights_for_center: 0.0
    """
    return text_format.Merge(proto_txt,
                             center_net_pb2.CenterNet.ObjectCenterParams())

  def get_fake_object_detection_proto(self, customize_head_params=False):
    proto_txt = """
      task_loss_weight: 0.5
      offset_loss_weight: 0.1
      scale_loss_weight: 0.2
      localization_loss {
        l1_localization_loss {
        }
      }
    """
    if customize_head_params:
      proto_txt += """
      scale_head_params {
        num_filters: 128
        num_filters: 64
        kernel_sizes: 5
        kernel_sizes: 3
      }
    """
    return text_format.Merge(proto_txt,
                             center_net_pb2.CenterNet.ObjectDetection())

  def get_fake_mask_proto(self, customize_head_params=False):
    proto_txt = """
      task_loss_weight: 0.7
      classification_loss {
        weighted_softmax {}
      }
      mask_height: 8
      mask_width: 8
      score_threshold: 0.7
      heatmap_bias_init: -2.0
    """
    if customize_head_params:
      proto_txt += """
      mask_head_params {
        num_filters: 128
        num_filters: 64
        kernel_sizes: 5
        kernel_sizes: 3
      }
    """
    return text_format.Merge(proto_txt,
                             center_net_pb2.CenterNet.MaskEstimation())

  def get_fake_densepose_proto(self):
    proto_txt = """
      task_loss_weight: 0.5
      class_id: 0
      loss {
        classification_loss {
          weighted_softmax {}
        }
        localization_loss {
          l1_localization_loss {
          }
        }
      }
      num_parts: 24
      part_loss_weight: 1.0
      coordinate_loss_weight: 2.0
      upsample_to_input_res: true
      heatmap_bias_init: -2.0
    """
    return text_format.Merge(proto_txt,
                             center_net_pb2.CenterNet.DensePoseEstimation())

  @parameterized.parameters(
      {'customize_head_params': True},
      {'customize_head_params': False}
  )
  def test_create_center_net_model(self, customize_head_params):
    """Test building a CenterNet model from proto txt."""
    proto_txt = """
      center_net {
        num_classes: 10
        feature_extractor {
          type: "hourglass_52"
          channel_stds: [4, 5, 6]
          bgr_ordering: true
        }
        image_resizer {
          keep_aspect_ratio_resizer {
            min_dimension: 512
            max_dimension: 512
            pad_to_max_dimension: true
          }
        }
      }
    """
    # Set up the configuration proto.
    config = text_format.Merge(proto_txt, model_pb2.DetectionModel())
    config.center_net.object_center_params.CopyFrom(
        self.get_fake_object_center_proto(
            customize_head_params=customize_head_params))
    config.center_net.object_detection_task.CopyFrom(
        self.get_fake_object_detection_proto(
            customize_head_params=customize_head_params))
    config.center_net.keypoint_estimation_task.append(
        self.get_fake_keypoint_proto(
            customize_head_params=customize_head_params))
    config.center_net.keypoint_label_map_path = (
        self.get_fake_label_map_file_path())
    config.center_net.mask_estimation_task.CopyFrom(
        self.get_fake_mask_proto(
            customize_head_params=customize_head_params))
    config.center_net.densepose_estimation_task.CopyFrom(
        self.get_fake_densepose_proto())

    # Build the model from the configuration.
    model = model_builder.build(config, is_training=True)

    # Check object center related parameters.
    self.assertEqual(model._num_classes, 10)
    self.assertIsInstance(model._center_params.classification_loss,
                          losses.PenaltyReducedLogisticFocalLoss)
    self.assertEqual(model._center_params.classification_loss._alpha, 3.0)
    self.assertEqual(model._center_params.classification_loss._beta, 4.0)
    self.assertAlmostEqual(model._center_params.min_box_overlap_iou, 0.2)
    self.assertAlmostEqual(
        model._center_params.heatmap_bias_init, 3.14, places=4)
    self.assertEqual(model._center_params.max_box_predictions, 15)
    if customize_head_params:
      self.assertEqual(model._center_params.center_head_num_filters, [64, 32])
      self.assertEqual(model._center_params.center_head_kernel_sizes, [5, 3])
    else:
      self.assertEqual(model._center_params.center_head_num_filters, [256])
      self.assertEqual(model._center_params.center_head_kernel_sizes, [3])

    # Check object detection related parameters.
    self.assertAlmostEqual(model._od_params.offset_loss_weight, 0.1)
    self.assertAlmostEqual(model._od_params.scale_loss_weight, 0.2)
    self.assertAlmostEqual(model._od_params.task_loss_weight, 0.5)
    self.assertIsInstance(model._od_params.localization_loss,
                          losses.L1LocalizationLoss)
    self.assertEqual(model._od_params.offset_head_num_filters, [256])
    self.assertEqual(model._od_params.offset_head_kernel_sizes, [3])
    if customize_head_params:
      self.assertEqual(model._od_params.scale_head_num_filters, [128, 64])
      self.assertEqual(model._od_params.scale_head_kernel_sizes, [5, 3])
    else:
      self.assertEqual(model._od_params.scale_head_num_filters, [256])
      self.assertEqual(model._od_params.scale_head_kernel_sizes, [3])

    # Check keypoint estimation related parameters.
    kp_params = model._kp_params_dict['human_pose']
    self.assertAlmostEqual(kp_params.task_loss_weight, 0.9)
    self.assertAlmostEqual(kp_params.keypoint_regression_loss_weight, 1.0)
    self.assertAlmostEqual(kp_params.keypoint_offset_loss_weight, 0.5)
    self.assertAlmostEqual(kp_params.heatmap_bias_init, 2.14, places=4)
    self.assertEqual(kp_params.classification_loss._alpha, 3.0)
    self.assertEqual(kp_params.keypoint_indices, [0, 1, 2, 3])
    self.assertEqual(kp_params.keypoint_labels,
                     ['nose', 'left_shoulder', 'right_shoulder', 'hip'])
    self.assertAllClose(kp_params.keypoint_std_dev, [0.3, 1.0, 1.0, 0.0])
    self.assertEqual(kp_params.classification_loss._beta, 4.0)
    self.assertIsInstance(kp_params.localization_loss,
                          losses.L1LocalizationLoss)
    self.assertAlmostEqual(kp_params.keypoint_candidate_score_threshold, 0.3)
    self.assertEqual(kp_params.num_candidates_per_keypoint, 12)
    self.assertEqual(kp_params.peak_max_pool_kernel_size, 5)
    self.assertAlmostEqual(kp_params.unmatched_keypoint_score, 0.05)
    self.assertAlmostEqual(kp_params.box_scale, 1.7)
    self.assertAlmostEqual(kp_params.candidate_search_scale, 0.2)
    self.assertEqual(kp_params.candidate_ranking_mode, 'score_distance_ratio')
    self.assertEqual(kp_params.offset_peak_radius, 3)
    self.assertEqual(kp_params.per_keypoint_offset, True)
    self.assertEqual(kp_params.predict_depth, True)
    self.assertEqual(kp_params.per_keypoint_depth, True)
    self.assertAlmostEqual(kp_params.keypoint_depth_loss_weight, 0.3)
    if customize_head_params:
      # Set by the config.
      self.assertEqual(kp_params.heatmap_head_num_filters, [64, 32])
      self.assertEqual(kp_params.heatmap_head_kernel_sizes, [5, 3])
      self.assertEqual(kp_params.offset_head_num_filters, [128, 64])
      self.assertEqual(kp_params.offset_head_kernel_sizes, [5, 3])
    else:
      # Default values:
      self.assertEqual(kp_params.heatmap_head_num_filters, [256])
      self.assertEqual(kp_params.heatmap_head_kernel_sizes, [3])
      self.assertEqual(kp_params.offset_head_num_filters, [256])
      self.assertEqual(kp_params.offset_head_kernel_sizes, [3])

    # Check mask related parameters.
    self.assertAlmostEqual(model._mask_params.task_loss_weight, 0.7)
    self.assertIsInstance(model._mask_params.classification_loss,
                          losses.WeightedSoftmaxClassificationLoss)
    self.assertEqual(model._mask_params.mask_height, 8)
    self.assertEqual(model._mask_params.mask_width, 8)
    self.assertAlmostEqual(model._mask_params.score_threshold, 0.7)
    self.assertAlmostEqual(
        model._mask_params.heatmap_bias_init, -2.0, places=4)
    if customize_head_params:
      self.assertEqual(model._mask_params.mask_head_num_filters, [128, 64])
      self.assertEqual(model._mask_params.mask_head_kernel_sizes, [5, 3])
    else:
      self.assertEqual(model._mask_params.mask_head_num_filters, [256])
      self.assertEqual(model._mask_params.mask_head_kernel_sizes, [3])

    # Check DensePose related parameters.
    self.assertEqual(model._densepose_params.class_id, 0)
    self.assertIsInstance(model._densepose_params.classification_loss,
                          losses.WeightedSoftmaxClassificationLoss)
    self.assertIsInstance(model._densepose_params.localization_loss,
                          losses.L1LocalizationLoss)
    self.assertAlmostEqual(model._densepose_params.part_loss_weight, 1.0)
    self.assertAlmostEqual(model._densepose_params.coordinate_loss_weight, 2.0)
    self.assertEqual(model._densepose_params.num_parts, 24)
    self.assertAlmostEqual(model._densepose_params.task_loss_weight, 0.5)
    self.assertTrue(model._densepose_params.upsample_to_input_res)
    self.assertEqual(model._densepose_params.upsample_method, 'bilinear')
    self.assertAlmostEqual(
        model._densepose_params.heatmap_bias_init, -2.0, places=4)

    # Check feature extractor parameters.
    self.assertIsInstance(
        model._feature_extractor, center_net_hourglass_feature_extractor
        .CenterNetHourglassFeatureExtractor)
    self.assertAllClose(model._feature_extractor._channel_means, [0, 0, 0])
    self.assertAllClose(model._feature_extractor._channel_stds, [4, 5, 6])
    self.assertTrue(model._feature_extractor._bgr_ordering)
    backbone = model._feature_extractor._network
    self.assertIsInstance(backbone, hourglass_network.HourglassNetwork)
    self.assertTrue(backbone.num_hourglasses, 1)

  def test_create_center_net_model_from_keypoints(self):
    """Test building a CenterNet model from proto txt."""
    proto_txt = """
      center_net {
        num_classes: 10
        feature_extractor {
          type: "hourglass_52"
          channel_stds: [4, 5, 6]
          bgr_ordering: true
        }
        image_resizer {
          keep_aspect_ratio_resizer {
            min_dimension: 512
            max_dimension: 512
            pad_to_max_dimension: true
          }
        }
      }
    """
    # Set up the configuration proto.
    config = text_format.Parse(proto_txt, model_pb2.DetectionModel())
    # Only add object center and keypoint estimation configs here.
    config.center_net.object_center_params.CopyFrom(
        self.get_fake_object_center_from_keypoints_proto())
    config.center_net.keypoint_estimation_task.append(
        self.get_fake_keypoint_proto())
    config.center_net.keypoint_label_map_path = (
        self.get_fake_label_map_file_path())

    # Build the model from the configuration.
    model = model_builder.build(config, is_training=True)

    # Check object center related parameters.
    self.assertEqual(model._num_classes, 10)
    self.assertEqual(model._center_params.keypoint_weights_for_center,
                     [1.0, 0.0, 1.0, 0.0])

    # Check keypoint estimation related parameters.
    kp_params = model._kp_params_dict['human_pose']
    self.assertAlmostEqual(kp_params.task_loss_weight, 0.9)
    self.assertEqual(kp_params.keypoint_indices, [0, 1, 2, 3])
    self.assertEqual(kp_params.keypoint_labels,
                     ['nose', 'left_shoulder', 'right_shoulder', 'hip'])

  def test_create_center_net_model_mobilenet(self):
    """Test building a CenterNet model using bilinear interpolation."""
    proto_txt = """
      center_net {
        num_classes: 10
        feature_extractor {
          type: "mobilenet_v2_fpn"
          depth_multiplier: 1.0
          use_separable_conv: true
          upsampling_interpolation: "bilinear"
        }
        image_resizer {
          keep_aspect_ratio_resizer {
            min_dimension: 512
            max_dimension: 512
            pad_to_max_dimension: true
          }
        }
      }
    """
    # Set up the configuration proto.
    config = text_format.Parse(proto_txt, model_pb2.DetectionModel())
    # Only add object center and keypoint estimation configs here.
    config.center_net.object_center_params.CopyFrom(
        self.get_fake_object_center_from_keypoints_proto())
    config.center_net.keypoint_estimation_task.append(
        self.get_fake_keypoint_proto())
    config.center_net.keypoint_label_map_path = (
        self.get_fake_label_map_file_path())

    # Build the model from the configuration.
    model = model_builder.build(config, is_training=True)

    feature_extractor = model._feature_extractor
    # Verify the upsampling layers in the FPN use 'bilinear' interpolation.
    fpn = feature_extractor.get_layer('model_1')
    num_up_sampling2d_layers = 0
    for layer in fpn.layers:
      if 'up_sampling2d' in layer.name:
        num_up_sampling2d_layers += 1
        self.assertEqual('bilinear', layer.interpolation)
    # Verify that there are up_sampling2d layers.
    self.assertGreater(num_up_sampling2d_layers, 0)

  def test_create_center_net_deepmac(self):
    """Test building a CenterNet DeepMAC model."""

    proto_txt = """
      center_net {
        num_classes: 90
        feature_extractor {
          type: "hourglass_52"
        }
        image_resizer {
          keep_aspect_ratio_resizer {
            min_dimension: 512
            max_dimension: 512
            pad_to_max_dimension: true
          }
        }
        object_detection_task {
          task_loss_weight: 1.0
          offset_loss_weight: 1.0
          scale_loss_weight: 0.1
          localization_loss {
            l1_localization_loss {
            }
          }
        }
        object_center_params {
          object_center_loss_weight: 1.0
          min_box_overlap_iou: 0.7
          max_box_predictions: 100
          classification_loss {
            penalty_reduced_logistic_focal_loss {
              alpha: 2.0
              beta: 4.0
            }
          }
        }

        deepmac_mask_estimation {
          classification_loss {
            weighted_sigmoid {}
          }
        }
      }
    """
    # Set up the configuration proto.
    config = text_format.Parse(proto_txt, model_pb2.DetectionModel())

    # Build the model from the configuration.
    model = model_builder.build(config, is_training=True)
    self.assertIsInstance(model, deepmac_meta_arch.DeepMACMetaArch)


if __name__ == '__main__':
  tf.test.main()
