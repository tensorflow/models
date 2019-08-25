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

"""Tests for box_predictor_builder."""

import mock
import tensorflow as tf

from google.protobuf import text_format
from object_detection.builders import box_predictor_builder
from object_detection.builders import hyperparams_builder
from object_detection.predictors import mask_rcnn_box_predictor
from object_detection.protos import box_predictor_pb2
from object_detection.protos import hyperparams_pb2


class ConvolutionalBoxPredictorBuilderTest(tf.test.TestCase):

  def test_box_predictor_calls_conv_argscope_fn(self):
    conv_hyperparams_text_proto = """
      regularizer {
        l1_regularizer {
          weight: 0.0003
        }
      }
      initializer {
        truncated_normal_initializer {
          mean: 0.0
          stddev: 0.3
        }
      }
      activation: RELU_6
    """
    hyperparams_proto = hyperparams_pb2.Hyperparams()
    text_format.Merge(conv_hyperparams_text_proto, hyperparams_proto)
    def mock_conv_argscope_builder(conv_hyperparams_arg, is_training):
      return (conv_hyperparams_arg, is_training)

    box_predictor_proto = box_predictor_pb2.BoxPredictor()
    box_predictor_proto.convolutional_box_predictor.conv_hyperparams.CopyFrom(
        hyperparams_proto)
    box_predictor = box_predictor_builder.build(
        argscope_fn=mock_conv_argscope_builder,
        box_predictor_config=box_predictor_proto,
        is_training=False,
        num_classes=10)
    (conv_hyperparams_actual, is_training) = box_predictor._conv_hyperparams_fn
    self.assertAlmostEqual((hyperparams_proto.regularizer.
                            l1_regularizer.weight),
                           (conv_hyperparams_actual.regularizer.l1_regularizer.
                            weight))
    self.assertAlmostEqual((hyperparams_proto.initializer.
                            truncated_normal_initializer.stddev),
                           (conv_hyperparams_actual.initializer.
                            truncated_normal_initializer.stddev))
    self.assertAlmostEqual((hyperparams_proto.initializer.
                            truncated_normal_initializer.mean),
                           (conv_hyperparams_actual.initializer.
                            truncated_normal_initializer.mean))
    self.assertEqual(hyperparams_proto.activation,
                     conv_hyperparams_actual.activation)
    self.assertFalse(is_training)

  def test_construct_non_default_conv_box_predictor(self):
    box_predictor_text_proto = """
      convolutional_box_predictor {
        min_depth: 2
        max_depth: 16
        num_layers_before_predictor: 2
        use_dropout: false
        dropout_keep_probability: 0.4
        kernel_size: 3
        box_code_size: 3
        apply_sigmoid_to_scores: true
        class_prediction_bias_init: 4.0
        use_depthwise: true
      }
    """
    conv_hyperparams_text_proto = """
      regularizer {
        l1_regularizer {
        }
      }
      initializer {
        truncated_normal_initializer {
        }
      }
    """
    hyperparams_proto = hyperparams_pb2.Hyperparams()
    text_format.Merge(conv_hyperparams_text_proto, hyperparams_proto)
    def mock_conv_argscope_builder(conv_hyperparams_arg, is_training):
      return (conv_hyperparams_arg, is_training)

    box_predictor_proto = box_predictor_pb2.BoxPredictor()
    text_format.Merge(box_predictor_text_proto, box_predictor_proto)
    box_predictor_proto.convolutional_box_predictor.conv_hyperparams.CopyFrom(
        hyperparams_proto)
    box_predictor = box_predictor_builder.build(
        argscope_fn=mock_conv_argscope_builder,
        box_predictor_config=box_predictor_proto,
        is_training=False,
        num_classes=10,
        add_background_class=False)
    class_head = box_predictor._class_prediction_head
    self.assertEqual(box_predictor._min_depth, 2)
    self.assertEqual(box_predictor._max_depth, 16)
    self.assertEqual(box_predictor._num_layers_before_predictor, 2)
    self.assertFalse(class_head._use_dropout)
    self.assertAlmostEqual(class_head._dropout_keep_prob, 0.4)
    self.assertTrue(class_head._apply_sigmoid_to_scores)
    self.assertAlmostEqual(class_head._class_prediction_bias_init, 4.0)
    self.assertEqual(class_head._num_class_slots, 10)
    self.assertEqual(box_predictor.num_classes, 10)
    self.assertFalse(box_predictor._is_training)
    self.assertTrue(class_head._use_depthwise)

  def test_construct_default_conv_box_predictor(self):
    box_predictor_text_proto = """
      convolutional_box_predictor {
        conv_hyperparams {
          regularizer {
            l1_regularizer {
            }
          }
          initializer {
            truncated_normal_initializer {
            }
          }
        }
      }"""
    box_predictor_proto = box_predictor_pb2.BoxPredictor()
    text_format.Merge(box_predictor_text_proto, box_predictor_proto)
    box_predictor = box_predictor_builder.build(
        argscope_fn=hyperparams_builder.build,
        box_predictor_config=box_predictor_proto,
        is_training=True,
        num_classes=90)
    class_head = box_predictor._class_prediction_head
    self.assertEqual(box_predictor._min_depth, 0)
    self.assertEqual(box_predictor._max_depth, 0)
    self.assertEqual(box_predictor._num_layers_before_predictor, 0)
    self.assertTrue(class_head._use_dropout)
    self.assertAlmostEqual(class_head._dropout_keep_prob, 0.8)
    self.assertFalse(class_head._apply_sigmoid_to_scores)
    self.assertEqual(class_head._num_class_slots, 91)
    self.assertEqual(box_predictor.num_classes, 90)
    self.assertTrue(box_predictor._is_training)
    self.assertFalse(class_head._use_depthwise)


class WeightSharedConvolutionalBoxPredictorBuilderTest(tf.test.TestCase):

  def test_box_predictor_calls_conv_argscope_fn(self):
    conv_hyperparams_text_proto = """
      regularizer {
        l1_regularizer {
          weight: 0.0003
        }
      }
      initializer {
        truncated_normal_initializer {
          mean: 0.0
          stddev: 0.3
        }
      }
      activation: RELU_6
    """
    hyperparams_proto = hyperparams_pb2.Hyperparams()
    text_format.Merge(conv_hyperparams_text_proto, hyperparams_proto)
    def mock_conv_argscope_builder(conv_hyperparams_arg, is_training):
      return (conv_hyperparams_arg, is_training)

    box_predictor_proto = box_predictor_pb2.BoxPredictor()
    (box_predictor_proto.weight_shared_convolutional_box_predictor
     .conv_hyperparams.CopyFrom(hyperparams_proto))
    box_predictor = box_predictor_builder.build(
        argscope_fn=mock_conv_argscope_builder,
        box_predictor_config=box_predictor_proto,
        is_training=False,
        num_classes=10)
    (conv_hyperparams_actual, is_training) = box_predictor._conv_hyperparams_fn
    self.assertAlmostEqual((hyperparams_proto.regularizer.
                            l1_regularizer.weight),
                           (conv_hyperparams_actual.regularizer.l1_regularizer.
                            weight))
    self.assertAlmostEqual((hyperparams_proto.initializer.
                            truncated_normal_initializer.stddev),
                           (conv_hyperparams_actual.initializer.
                            truncated_normal_initializer.stddev))
    self.assertAlmostEqual((hyperparams_proto.initializer.
                            truncated_normal_initializer.mean),
                           (conv_hyperparams_actual.initializer.
                            truncated_normal_initializer.mean))
    self.assertEqual(hyperparams_proto.activation,
                     conv_hyperparams_actual.activation)
    self.assertFalse(is_training)

  def test_construct_non_default_conv_box_predictor(self):
    box_predictor_text_proto = """
      weight_shared_convolutional_box_predictor {
        depth: 2
        num_layers_before_predictor: 2
        kernel_size: 7
        box_code_size: 3
        class_prediction_bias_init: 4.0
      }
    """
    conv_hyperparams_text_proto = """
      regularizer {
        l1_regularizer {
        }
      }
      initializer {
        truncated_normal_initializer {
        }
      }
    """
    hyperparams_proto = hyperparams_pb2.Hyperparams()
    text_format.Merge(conv_hyperparams_text_proto, hyperparams_proto)
    def mock_conv_argscope_builder(conv_hyperparams_arg, is_training):
      return (conv_hyperparams_arg, is_training)

    box_predictor_proto = box_predictor_pb2.BoxPredictor()
    text_format.Merge(box_predictor_text_proto, box_predictor_proto)
    (box_predictor_proto.weight_shared_convolutional_box_predictor.
     conv_hyperparams.CopyFrom(hyperparams_proto))
    box_predictor = box_predictor_builder.build(
        argscope_fn=mock_conv_argscope_builder,
        box_predictor_config=box_predictor_proto,
        is_training=False,
        num_classes=10,
        add_background_class=False)
    class_head = box_predictor._class_prediction_head
    self.assertEqual(box_predictor._depth, 2)
    self.assertEqual(box_predictor._num_layers_before_predictor, 2)
    self.assertAlmostEqual(class_head._class_prediction_bias_init, 4.0)
    self.assertEqual(box_predictor.num_classes, 10)
    self.assertFalse(box_predictor._is_training)
    self.assertEqual(box_predictor._apply_batch_norm, False)

  def test_construct_non_default_depthwise_conv_box_predictor(self):
    box_predictor_text_proto = """
      weight_shared_convolutional_box_predictor {
        depth: 2
        num_layers_before_predictor: 2
        kernel_size: 7
        box_code_size: 3
        class_prediction_bias_init: 4.0
        use_depthwise: true
      }
    """
    conv_hyperparams_text_proto = """
      regularizer {
        l1_regularizer {
        }
      }
      initializer {
        truncated_normal_initializer {
        }
      }
    """
    hyperparams_proto = hyperparams_pb2.Hyperparams()
    text_format.Merge(conv_hyperparams_text_proto, hyperparams_proto)
    def mock_conv_argscope_builder(conv_hyperparams_arg, is_training):
      return (conv_hyperparams_arg, is_training)

    box_predictor_proto = box_predictor_pb2.BoxPredictor()
    text_format.Merge(box_predictor_text_proto, box_predictor_proto)
    (box_predictor_proto.weight_shared_convolutional_box_predictor.
     conv_hyperparams.CopyFrom(hyperparams_proto))
    box_predictor = box_predictor_builder.build(
        argscope_fn=mock_conv_argscope_builder,
        box_predictor_config=box_predictor_proto,
        is_training=False,
        num_classes=10,
        add_background_class=False)
    class_head = box_predictor._class_prediction_head
    self.assertEqual(box_predictor._depth, 2)
    self.assertEqual(box_predictor._num_layers_before_predictor, 2)
    self.assertEqual(box_predictor._apply_batch_norm, False)
    self.assertEqual(box_predictor._use_depthwise, True)
    self.assertAlmostEqual(class_head._class_prediction_bias_init, 4.0)
    self.assertEqual(box_predictor.num_classes, 10)
    self.assertFalse(box_predictor._is_training)

  def test_construct_default_conv_box_predictor(self):
    box_predictor_text_proto = """
      weight_shared_convolutional_box_predictor {
        conv_hyperparams {
          regularizer {
            l1_regularizer {
            }
          }
          initializer {
            truncated_normal_initializer {
            }
          }
        }
      }"""
    box_predictor_proto = box_predictor_pb2.BoxPredictor()
    text_format.Merge(box_predictor_text_proto, box_predictor_proto)
    box_predictor = box_predictor_builder.build(
        argscope_fn=hyperparams_builder.build,
        box_predictor_config=box_predictor_proto,
        is_training=True,
        num_classes=90)
    self.assertEqual(box_predictor._depth, 0)
    self.assertEqual(box_predictor._num_layers_before_predictor, 0)
    self.assertEqual(box_predictor.num_classes, 90)
    self.assertTrue(box_predictor._is_training)
    self.assertEqual(box_predictor._apply_batch_norm, False)

  def test_construct_default_conv_box_predictor_with_batch_norm(self):
    box_predictor_text_proto = """
      weight_shared_convolutional_box_predictor {
        conv_hyperparams {
          regularizer {
            l1_regularizer {
            }
          }
          batch_norm {
            train: true
          }
          initializer {
            truncated_normal_initializer {
            }
          }
        }
      }"""
    box_predictor_proto = box_predictor_pb2.BoxPredictor()
    text_format.Merge(box_predictor_text_proto, box_predictor_proto)
    box_predictor = box_predictor_builder.build(
        argscope_fn=hyperparams_builder.build,
        box_predictor_config=box_predictor_proto,
        is_training=True,
        num_classes=90)
    self.assertEqual(box_predictor._depth, 0)
    self.assertEqual(box_predictor._num_layers_before_predictor, 0)
    self.assertEqual(box_predictor.num_classes, 90)
    self.assertTrue(box_predictor._is_training)
    self.assertEqual(box_predictor._apply_batch_norm, True)




class MaskRCNNBoxPredictorBuilderTest(tf.test.TestCase):

  def test_box_predictor_builder_calls_fc_argscope_fn(self):
    fc_hyperparams_text_proto = """
      regularizer {
        l1_regularizer {
          weight: 0.0003
        }
      }
      initializer {
        truncated_normal_initializer {
          mean: 0.0
          stddev: 0.3
        }
      }
      activation: RELU_6
      op: FC
    """
    hyperparams_proto = hyperparams_pb2.Hyperparams()
    text_format.Merge(fc_hyperparams_text_proto, hyperparams_proto)
    box_predictor_proto = box_predictor_pb2.BoxPredictor()
    box_predictor_proto.mask_rcnn_box_predictor.fc_hyperparams.CopyFrom(
        hyperparams_proto)
    mock_argscope_fn = mock.Mock(return_value='arg_scope')
    box_predictor = box_predictor_builder.build(
        argscope_fn=mock_argscope_fn,
        box_predictor_config=box_predictor_proto,
        is_training=False,
        num_classes=10)
    mock_argscope_fn.assert_called_with(hyperparams_proto, False)
    self.assertEqual(box_predictor._box_prediction_head._fc_hyperparams_fn,
                     'arg_scope')
    self.assertEqual(box_predictor._class_prediction_head._fc_hyperparams_fn,
                     'arg_scope')

  def test_non_default_mask_rcnn_box_predictor(self):
    fc_hyperparams_text_proto = """
      regularizer {
        l1_regularizer {
        }
      }
      initializer {
        truncated_normal_initializer {
        }
      }
      activation: RELU_6
      op: FC
    """
    box_predictor_text_proto = """
      mask_rcnn_box_predictor {
        use_dropout: true
        dropout_keep_probability: 0.8
        box_code_size: 3
        share_box_across_classes: true
      }
    """
    hyperparams_proto = hyperparams_pb2.Hyperparams()
    text_format.Merge(fc_hyperparams_text_proto, hyperparams_proto)
    def mock_fc_argscope_builder(fc_hyperparams_arg, is_training):
      return (fc_hyperparams_arg, is_training)

    box_predictor_proto = box_predictor_pb2.BoxPredictor()
    text_format.Merge(box_predictor_text_proto, box_predictor_proto)
    box_predictor_proto.mask_rcnn_box_predictor.fc_hyperparams.CopyFrom(
        hyperparams_proto)
    box_predictor = box_predictor_builder.build(
        argscope_fn=mock_fc_argscope_builder,
        box_predictor_config=box_predictor_proto,
        is_training=True,
        num_classes=90)
    box_head = box_predictor._box_prediction_head
    class_head = box_predictor._class_prediction_head
    self.assertTrue(box_head._use_dropout)
    self.assertTrue(class_head._use_dropout)
    self.assertAlmostEqual(box_head._dropout_keep_prob, 0.8)
    self.assertAlmostEqual(class_head._dropout_keep_prob, 0.8)
    self.assertEqual(box_predictor.num_classes, 90)
    self.assertTrue(box_predictor._is_training)
    self.assertEqual(box_head._box_code_size, 3)
    self.assertEqual(box_head._share_box_across_classes, True)

  def test_build_default_mask_rcnn_box_predictor(self):
    box_predictor_proto = box_predictor_pb2.BoxPredictor()
    box_predictor_proto.mask_rcnn_box_predictor.fc_hyperparams.op = (
        hyperparams_pb2.Hyperparams.FC)
    box_predictor = box_predictor_builder.build(
        argscope_fn=mock.Mock(return_value='arg_scope'),
        box_predictor_config=box_predictor_proto,
        is_training=True,
        num_classes=90)
    box_head = box_predictor._box_prediction_head
    class_head = box_predictor._class_prediction_head
    self.assertFalse(box_head._use_dropout)
    self.assertFalse(class_head._use_dropout)
    self.assertAlmostEqual(box_head._dropout_keep_prob, 0.5)
    self.assertEqual(box_predictor.num_classes, 90)
    self.assertTrue(box_predictor._is_training)
    self.assertEqual(box_head._box_code_size, 4)
    self.assertEqual(len(box_predictor._third_stage_heads.keys()), 0)

  def test_build_box_predictor_with_mask_branch(self):
    box_predictor_proto = box_predictor_pb2.BoxPredictor()
    box_predictor_proto.mask_rcnn_box_predictor.fc_hyperparams.op = (
        hyperparams_pb2.Hyperparams.FC)
    box_predictor_proto.mask_rcnn_box_predictor.conv_hyperparams.op = (
        hyperparams_pb2.Hyperparams.CONV)
    box_predictor_proto.mask_rcnn_box_predictor.predict_instance_masks = True
    box_predictor_proto.mask_rcnn_box_predictor.mask_prediction_conv_depth = 512
    box_predictor_proto.mask_rcnn_box_predictor.mask_height = 16
    box_predictor_proto.mask_rcnn_box_predictor.mask_width = 16
    mock_argscope_fn = mock.Mock(return_value='arg_scope')
    box_predictor = box_predictor_builder.build(
        argscope_fn=mock_argscope_fn,
        box_predictor_config=box_predictor_proto,
        is_training=True,
        num_classes=90)
    mock_argscope_fn.assert_has_calls(
        [mock.call(box_predictor_proto.mask_rcnn_box_predictor.fc_hyperparams,
                   True),
         mock.call(box_predictor_proto.mask_rcnn_box_predictor.conv_hyperparams,
                   True)], any_order=True)
    box_head = box_predictor._box_prediction_head
    class_head = box_predictor._class_prediction_head
    third_stage_heads = box_predictor._third_stage_heads
    self.assertFalse(box_head._use_dropout)
    self.assertFalse(class_head._use_dropout)
    self.assertAlmostEqual(box_head._dropout_keep_prob, 0.5)
    self.assertAlmostEqual(class_head._dropout_keep_prob, 0.5)
    self.assertEqual(box_predictor.num_classes, 90)
    self.assertTrue(box_predictor._is_training)
    self.assertEqual(box_head._box_code_size, 4)
    self.assertTrue(
        mask_rcnn_box_predictor.MASK_PREDICTIONS in third_stage_heads)
    self.assertEqual(
        third_stage_heads[mask_rcnn_box_predictor.MASK_PREDICTIONS]
        ._mask_prediction_conv_depth, 512)

  def test_build_box_predictor_with_convlve_then_upsample_masks(self):
    box_predictor_proto = box_predictor_pb2.BoxPredictor()
    box_predictor_proto.mask_rcnn_box_predictor.fc_hyperparams.op = (
        hyperparams_pb2.Hyperparams.FC)
    box_predictor_proto.mask_rcnn_box_predictor.conv_hyperparams.op = (
        hyperparams_pb2.Hyperparams.CONV)
    box_predictor_proto.mask_rcnn_box_predictor.predict_instance_masks = True
    box_predictor_proto.mask_rcnn_box_predictor.mask_prediction_conv_depth = 512
    box_predictor_proto.mask_rcnn_box_predictor.mask_height = 24
    box_predictor_proto.mask_rcnn_box_predictor.mask_width = 24
    box_predictor_proto.mask_rcnn_box_predictor.convolve_then_upsample_masks = (
        True)

    mock_argscope_fn = mock.Mock(return_value='arg_scope')
    box_predictor = box_predictor_builder.build(
        argscope_fn=mock_argscope_fn,
        box_predictor_config=box_predictor_proto,
        is_training=True,
        num_classes=90)
    mock_argscope_fn.assert_has_calls(
        [mock.call(box_predictor_proto.mask_rcnn_box_predictor.fc_hyperparams,
                   True),
         mock.call(box_predictor_proto.mask_rcnn_box_predictor.conv_hyperparams,
                   True)], any_order=True)
    box_head = box_predictor._box_prediction_head
    class_head = box_predictor._class_prediction_head
    third_stage_heads = box_predictor._third_stage_heads
    self.assertFalse(box_head._use_dropout)
    self.assertFalse(class_head._use_dropout)
    self.assertAlmostEqual(box_head._dropout_keep_prob, 0.5)
    self.assertAlmostEqual(class_head._dropout_keep_prob, 0.5)
    self.assertEqual(box_predictor.num_classes, 90)
    self.assertTrue(box_predictor._is_training)
    self.assertEqual(box_head._box_code_size, 4)
    self.assertTrue(
        mask_rcnn_box_predictor.MASK_PREDICTIONS in third_stage_heads)
    self.assertEqual(
        third_stage_heads[mask_rcnn_box_predictor.MASK_PREDICTIONS]
        ._mask_prediction_conv_depth, 512)
    self.assertTrue(third_stage_heads[mask_rcnn_box_predictor.MASK_PREDICTIONS]
                    ._convolve_then_upsample)


class RfcnBoxPredictorBuilderTest(tf.test.TestCase):

  def test_box_predictor_calls_fc_argscope_fn(self):
    conv_hyperparams_text_proto = """
      regularizer {
        l1_regularizer {
          weight: 0.0003
        }
      }
      initializer {
        truncated_normal_initializer {
          mean: 0.0
          stddev: 0.3
        }
      }
      activation: RELU_6
    """
    hyperparams_proto = hyperparams_pb2.Hyperparams()
    text_format.Merge(conv_hyperparams_text_proto, hyperparams_proto)
    def mock_conv_argscope_builder(conv_hyperparams_arg, is_training):
      return (conv_hyperparams_arg, is_training)

    box_predictor_proto = box_predictor_pb2.BoxPredictor()
    box_predictor_proto.rfcn_box_predictor.conv_hyperparams.CopyFrom(
        hyperparams_proto)
    box_predictor = box_predictor_builder.build(
        argscope_fn=mock_conv_argscope_builder,
        box_predictor_config=box_predictor_proto,
        is_training=False,
        num_classes=10)
    (conv_hyperparams_actual, is_training) = box_predictor._conv_hyperparams_fn
    self.assertAlmostEqual((hyperparams_proto.regularizer.
                            l1_regularizer.weight),
                           (conv_hyperparams_actual.regularizer.l1_regularizer.
                            weight))
    self.assertAlmostEqual((hyperparams_proto.initializer.
                            truncated_normal_initializer.stddev),
                           (conv_hyperparams_actual.initializer.
                            truncated_normal_initializer.stddev))
    self.assertAlmostEqual((hyperparams_proto.initializer.
                            truncated_normal_initializer.mean),
                           (conv_hyperparams_actual.initializer.
                            truncated_normal_initializer.mean))
    self.assertEqual(hyperparams_proto.activation,
                     conv_hyperparams_actual.activation)
    self.assertFalse(is_training)

  def test_non_default_rfcn_box_predictor(self):
    conv_hyperparams_text_proto = """
      regularizer {
        l1_regularizer {
        }
      }
      initializer {
        truncated_normal_initializer {
        }
      }
      activation: RELU_6
    """
    box_predictor_text_proto = """
      rfcn_box_predictor {
        num_spatial_bins_height: 4
        num_spatial_bins_width: 4
        depth: 4
        box_code_size: 3
        crop_height: 16
        crop_width: 16
      }
    """
    hyperparams_proto = hyperparams_pb2.Hyperparams()
    text_format.Merge(conv_hyperparams_text_proto, hyperparams_proto)
    def mock_conv_argscope_builder(conv_hyperparams_arg, is_training):
      return (conv_hyperparams_arg, is_training)

    box_predictor_proto = box_predictor_pb2.BoxPredictor()
    text_format.Merge(box_predictor_text_proto, box_predictor_proto)
    box_predictor_proto.rfcn_box_predictor.conv_hyperparams.CopyFrom(
        hyperparams_proto)
    box_predictor = box_predictor_builder.build(
        argscope_fn=mock_conv_argscope_builder,
        box_predictor_config=box_predictor_proto,
        is_training=True,
        num_classes=90)
    self.assertEqual(box_predictor.num_classes, 90)
    self.assertTrue(box_predictor._is_training)
    self.assertEqual(box_predictor._box_code_size, 3)
    self.assertEqual(box_predictor._num_spatial_bins, [4, 4])
    self.assertEqual(box_predictor._crop_size, [16, 16])

  def test_default_rfcn_box_predictor(self):
    conv_hyperparams_text_proto = """
      regularizer {
        l1_regularizer {
        }
      }
      initializer {
        truncated_normal_initializer {
        }
      }
      activation: RELU_6
    """
    hyperparams_proto = hyperparams_pb2.Hyperparams()
    text_format.Merge(conv_hyperparams_text_proto, hyperparams_proto)
    def mock_conv_argscope_builder(conv_hyperparams_arg, is_training):
      return (conv_hyperparams_arg, is_training)

    box_predictor_proto = box_predictor_pb2.BoxPredictor()
    box_predictor_proto.rfcn_box_predictor.conv_hyperparams.CopyFrom(
        hyperparams_proto)
    box_predictor = box_predictor_builder.build(
        argscope_fn=mock_conv_argscope_builder,
        box_predictor_config=box_predictor_proto,
        is_training=True,
        num_classes=90)
    self.assertEqual(box_predictor.num_classes, 90)
    self.assertTrue(box_predictor._is_training)
    self.assertEqual(box_predictor._box_code_size, 4)
    self.assertEqual(box_predictor._num_spatial_bins, [3, 3])
    self.assertEqual(box_predictor._crop_size, [12, 12])


if __name__ == '__main__':
  tf.test.main()
