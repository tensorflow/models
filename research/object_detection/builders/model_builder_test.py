# Lint as: python2, python3
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
"""Tests for object_detection.models.model_builder."""

from absl.testing import parameterized

from google.protobuf import text_format
from object_detection.builders import model_builder
from object_detection.meta_architectures import faster_rcnn_meta_arch
from object_detection.meta_architectures import rfcn_meta_arch
from object_detection.meta_architectures import ssd_meta_arch
from object_detection.protos import hyperparams_pb2
from object_detection.protos import losses_pb2
from object_detection.protos import model_pb2
from object_detection.utils import test_case


class ModelBuilderTest(test_case.TestCase, parameterized.TestCase):

  def default_ssd_feature_extractor(self):
    raise NotImplementedError

  def default_faster_rcnn_feature_extractor(self):
    raise NotImplementedError

  def ssd_feature_extractors(self):
    raise NotImplementedError

  def get_override_base_feature_extractor_hyperparams(self, extractor_type):
    raise NotImplementedError

  def faster_rcnn_feature_extractors(self):
    raise NotImplementedError

  def create_model(self, model_config, is_training=True):
    """Builds a DetectionModel based on the model config.

    Args:
      model_config: A model.proto object containing the config for the desired
        DetectionModel.
      is_training: True if this model is being built for training purposes.

    Returns:
      DetectionModel based on the config.
    """
    return model_builder.build(model_config, is_training=is_training)

  def create_default_ssd_model_proto(self):
    """Creates a DetectionModel proto with ssd model fields populated."""
    model_text_proto = """
      ssd {
        feature_extractor {
          conv_hyperparams {
            regularizer {
                l2_regularizer {
                }
              }
              initializer {
                truncated_normal_initializer {
                }
              }
          }
        }
        box_coder {
          faster_rcnn_box_coder {
          }
        }
        matcher {
          argmax_matcher {
          }
        }
        similarity_calculator {
          iou_similarity {
          }
        }
        anchor_generator {
          ssd_anchor_generator {
            aspect_ratios: 1.0
          }
        }
        image_resizer {
          fixed_shape_resizer {
            height: 320
            width: 320
          }
        }
        box_predictor {
          convolutional_box_predictor {
            conv_hyperparams {
              regularizer {
                l2_regularizer {
                }
              }
              initializer {
                truncated_normal_initializer {
                }
              }
            }
          }
        }
        loss {
          classification_loss {
            weighted_softmax {
            }
          }
          localization_loss {
            weighted_smooth_l1 {
            }
          }
        }
      }"""
    model_proto = model_pb2.DetectionModel()
    text_format.Merge(model_text_proto, model_proto)
    model_proto.ssd.feature_extractor.type = (self.
                                              default_ssd_feature_extractor())
    return model_proto

  def create_default_faster_rcnn_model_proto(self):
    """Creates a DetectionModel proto with FasterRCNN model fields populated."""
    model_text_proto = """
      faster_rcnn {
        inplace_batchnorm_update: false
        num_classes: 3
        image_resizer {
          keep_aspect_ratio_resizer {
            min_dimension: 600
            max_dimension: 1024
          }
        }
        first_stage_anchor_generator {
          grid_anchor_generator {
            scales: [0.25, 0.5, 1.0, 2.0]
            aspect_ratios: [0.5, 1.0, 2.0]
            height_stride: 16
            width_stride: 16
          }
        }
        first_stage_box_predictor_conv_hyperparams {
          regularizer {
            l2_regularizer {
            }
          }
          initializer {
            truncated_normal_initializer {
            }
          }
        }
        initial_crop_size: 14
        maxpool_kernel_size: 2
        maxpool_stride: 2
        second_stage_box_predictor {
          mask_rcnn_box_predictor {
            conv_hyperparams {
              regularizer {
                l2_regularizer {
                }
              }
              initializer {
                truncated_normal_initializer {
                }
              }
            }
            fc_hyperparams {
              op: FC
              regularizer {
                l2_regularizer {
                }
              }
              initializer {
                truncated_normal_initializer {
                }
              }
            }
          }
        }
        second_stage_post_processing {
          batch_non_max_suppression {
            score_threshold: 0.01
            iou_threshold: 0.6
            max_detections_per_class: 100
            max_total_detections: 300
          }
          score_converter: SOFTMAX
        }
      }"""
    model_proto = model_pb2.DetectionModel()
    text_format.Merge(model_text_proto, model_proto)
    (model_proto.faster_rcnn.feature_extractor.type
    ) = self.default_faster_rcnn_feature_extractor()
    return model_proto

  def test_create_ssd_models_from_config(self):
    model_proto = self.create_default_ssd_model_proto()
    for extractor_type, extractor_class in self.ssd_feature_extractors().items(
    ):
      model_proto.ssd.feature_extractor.type = extractor_type
      model_proto.ssd.feature_extractor.override_base_feature_extractor_hyperparams = (
          self.get_override_base_feature_extractor_hyperparams(extractor_type))
      model = model_builder.build(model_proto, is_training=True)
      self.assertIsInstance(model, ssd_meta_arch.SSDMetaArch)
      self.assertIsInstance(model._feature_extractor, extractor_class)

  def test_create_ssd_fpn_model_from_config(self):
    model_proto = self.create_default_ssd_model_proto()
    model_proto.ssd.feature_extractor.fpn.min_level = 3
    model_proto.ssd.feature_extractor.fpn.max_level = 7
    model = model_builder.build(model_proto, is_training=True)
    self.assertEqual(model._feature_extractor._fpn_min_level, 3)
    self.assertEqual(model._feature_extractor._fpn_max_level, 7)


  @parameterized.named_parameters(
      {
          'testcase_name': 'mask_rcnn_with_matmul',
          'use_matmul_crop_and_resize': False,
          'enable_mask_prediction': True
      },
      {
          'testcase_name': 'mask_rcnn_without_matmul',
          'use_matmul_crop_and_resize': True,
          'enable_mask_prediction': True
      },
      {
          'testcase_name': 'faster_rcnn_with_matmul',
          'use_matmul_crop_and_resize': False,
          'enable_mask_prediction': False
      },
      {
          'testcase_name': 'faster_rcnn_without_matmul',
          'use_matmul_crop_and_resize': True,
          'enable_mask_prediction': False
      },
  )
  def test_create_faster_rcnn_models_from_config(self,
                                                 use_matmul_crop_and_resize,
                                                 enable_mask_prediction):
    model_proto = self.create_default_faster_rcnn_model_proto()
    faster_rcnn_config = model_proto.faster_rcnn
    faster_rcnn_config.use_matmul_crop_and_resize = use_matmul_crop_and_resize
    if enable_mask_prediction:
      faster_rcnn_config.second_stage_mask_prediction_loss_weight = 3.0
      mask_predictor_config = (
          faster_rcnn_config.second_stage_box_predictor.mask_rcnn_box_predictor)
      mask_predictor_config.predict_instance_masks = True

    for extractor_type, extractor_class in (
        self.faster_rcnn_feature_extractors().items()):
      faster_rcnn_config.feature_extractor.type = extractor_type
      model = model_builder.build(model_proto, is_training=True)
      self.assertIsInstance(model, faster_rcnn_meta_arch.FasterRCNNMetaArch)
      self.assertIsInstance(model._feature_extractor, extractor_class)
      if enable_mask_prediction:
        self.assertAlmostEqual(model._second_stage_mask_loss_weight, 3.0)

  def test_create_faster_rcnn_model_from_config_with_example_miner(self):
    model_proto = self.create_default_faster_rcnn_model_proto()
    model_proto.faster_rcnn.hard_example_miner.num_hard_examples = 64
    model = model_builder.build(model_proto, is_training=True)
    self.assertIsNotNone(model._hard_example_miner)

  def test_create_rfcn_model_from_config(self):
    model_proto = self.create_default_faster_rcnn_model_proto()
    rfcn_predictor_config = (
        model_proto.faster_rcnn.second_stage_box_predictor.rfcn_box_predictor)
    rfcn_predictor_config.conv_hyperparams.op = hyperparams_pb2.Hyperparams.CONV
    for extractor_type, extractor_class in (
        self.faster_rcnn_feature_extractors().items()):
      model_proto.faster_rcnn.feature_extractor.type = extractor_type
      model = model_builder.build(model_proto, is_training=True)
      self.assertIsInstance(model, rfcn_meta_arch.RFCNMetaArch)
      self.assertIsInstance(model._feature_extractor, extractor_class)

  @parameterized.parameters(True, False)
  def test_create_faster_rcnn_from_config_with_crop_feature(
      self, output_final_box_features):
    model_proto = self.create_default_faster_rcnn_model_proto()
    model_proto.faster_rcnn.output_final_box_features = (
        output_final_box_features)
    _ = model_builder.build(model_proto, is_training=True)

  def test_invalid_model_config_proto(self):
    model_proto = ''
    with self.assertRaisesRegex(
        ValueError, 'model_config not of type model_pb2.DetectionModel.'):
      model_builder.build(model_proto, is_training=True)

  def test_unknown_meta_architecture(self):
    model_proto = model_pb2.DetectionModel()
    with self.assertRaisesRegex(ValueError, 'Unknown meta architecture'):
      model_builder.build(model_proto, is_training=True)

  def test_unknown_ssd_feature_extractor(self):
    model_proto = self.create_default_ssd_model_proto()
    model_proto.ssd.feature_extractor.type = 'unknown_feature_extractor'
    with self.assertRaises(ValueError):
      model_builder.build(model_proto, is_training=True)

  def test_unknown_faster_rcnn_feature_extractor(self):
    model_proto = self.create_default_faster_rcnn_model_proto()
    model_proto.faster_rcnn.feature_extractor.type = 'unknown_feature_extractor'
    with self.assertRaises(ValueError):
      model_builder.build(model_proto, is_training=True)

  def test_invalid_first_stage_nms_iou_threshold(self):
    model_proto = self.create_default_faster_rcnn_model_proto()
    model_proto.faster_rcnn.first_stage_nms_iou_threshold = 1.1
    with self.assertRaisesRegex(ValueError,
                                r'iou_threshold not in \[0, 1\.0\]'):
      model_builder.build(model_proto, is_training=True)
    model_proto.faster_rcnn.first_stage_nms_iou_threshold = -0.1
    with self.assertRaisesRegex(ValueError,
                                r'iou_threshold not in \[0, 1\.0\]'):
      model_builder.build(model_proto, is_training=True)

  def test_invalid_second_stage_batch_size(self):
    model_proto = self.create_default_faster_rcnn_model_proto()
    model_proto.faster_rcnn.first_stage_max_proposals = 1
    model_proto.faster_rcnn.second_stage_batch_size = 2
    with self.assertRaisesRegex(
        ValueError, 'second_stage_batch_size should be no greater '
        'than first_stage_max_proposals.'):
      model_builder.build(model_proto, is_training=True)

  def test_invalid_faster_rcnn_batchnorm_update(self):
    model_proto = self.create_default_faster_rcnn_model_proto()
    model_proto.faster_rcnn.inplace_batchnorm_update = True
    with self.assertRaisesRegex(ValueError,
                                'inplace batchnorm updates not supported'):
      model_builder.build(model_proto, is_training=True)

  def test_create_experimental_model(self):

    model_text_proto = """
      experimental_model {
        name: 'model42'
      }"""

    build_func = lambda *args: 42
    model_builder.EXPERIMENTAL_META_ARCH_BUILDER_MAP['model42'] = build_func
    model_proto = model_pb2.DetectionModel()
    text_format.Merge(model_text_proto, model_proto)

    self.assertEqual(model_builder.build(model_proto, is_training=True), 42)
