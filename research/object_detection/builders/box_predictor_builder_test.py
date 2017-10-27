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
    (conv_hyperparams_actual, is_training) = box_predictor._conv_hyperparams
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
        num_classes=10)
    self.assertEqual(box_predictor._min_depth, 2)
    self.assertEqual(box_predictor._max_depth, 16)
    self.assertEqual(box_predictor._num_layers_before_predictor, 2)
    self.assertFalse(box_predictor._use_dropout)
    self.assertAlmostEqual(box_predictor._dropout_keep_prob, 0.4)
    self.assertTrue(box_predictor._apply_sigmoid_to_scores)
    self.assertAlmostEqual(box_predictor._class_prediction_bias_init, 4.0)
    self.assertEqual(box_predictor.num_classes, 10)
    self.assertFalse(box_predictor._is_training)

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
    self.assertEqual(box_predictor._min_depth, 0)
    self.assertEqual(box_predictor._max_depth, 0)
    self.assertEqual(box_predictor._num_layers_before_predictor, 0)
    self.assertTrue(box_predictor._use_dropout)
    self.assertAlmostEqual(box_predictor._dropout_keep_prob, 0.8)
    self.assertFalse(box_predictor._apply_sigmoid_to_scores)
    self.assertEqual(box_predictor.num_classes, 90)
    self.assertTrue(box_predictor._is_training)


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
    self.assertEqual(box_predictor._fc_hyperparams, 'arg_scope')

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
    self.assertTrue(box_predictor._use_dropout)
    self.assertAlmostEqual(box_predictor._dropout_keep_prob, 0.8)
    self.assertEqual(box_predictor.num_classes, 90)
    self.assertTrue(box_predictor._is_training)
    self.assertEqual(box_predictor._box_code_size, 3)

  def test_build_default_mask_rcnn_box_predictor(self):
    box_predictor_proto = box_predictor_pb2.BoxPredictor()
    box_predictor_proto.mask_rcnn_box_predictor.fc_hyperparams.op = (
        hyperparams_pb2.Hyperparams.FC)
    box_predictor = box_predictor_builder.build(
        argscope_fn=mock.Mock(return_value='arg_scope'),
        box_predictor_config=box_predictor_proto,
        is_training=True,
        num_classes=90)
    self.assertFalse(box_predictor._use_dropout)
    self.assertAlmostEqual(box_predictor._dropout_keep_prob, 0.5)
    self.assertEqual(box_predictor.num_classes, 90)
    self.assertTrue(box_predictor._is_training)
    self.assertEqual(box_predictor._box_code_size, 4)
    self.assertFalse(box_predictor._predict_instance_masks)
    self.assertFalse(box_predictor._predict_keypoints)

  def test_build_box_predictor_with_mask_branch(self):
    box_predictor_proto = box_predictor_pb2.BoxPredictor()
    box_predictor_proto.mask_rcnn_box_predictor.fc_hyperparams.op = (
        hyperparams_pb2.Hyperparams.FC)
    box_predictor_proto.mask_rcnn_box_predictor.conv_hyperparams.op = (
        hyperparams_pb2.Hyperparams.CONV)
    box_predictor_proto.mask_rcnn_box_predictor.predict_instance_masks = True
    box_predictor_proto.mask_rcnn_box_predictor.mask_prediction_conv_depth = 512
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
    self.assertFalse(box_predictor._use_dropout)
    self.assertAlmostEqual(box_predictor._dropout_keep_prob, 0.5)
    self.assertEqual(box_predictor.num_classes, 90)
    self.assertTrue(box_predictor._is_training)
    self.assertEqual(box_predictor._box_code_size, 4)
    self.assertTrue(box_predictor._predict_instance_masks)
    self.assertEqual(box_predictor._mask_prediction_conv_depth, 512)
    self.assertFalse(box_predictor._predict_keypoints)


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
    (conv_hyperparams_actual, is_training) = box_predictor._conv_hyperparams
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
