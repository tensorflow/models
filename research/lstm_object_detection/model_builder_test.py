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

"""Tests for video_object_detection.tensorflow.model_builder."""

import tensorflow as tf
from google.protobuf import text_format
from lstm_object_detection import model_builder
from lstm_object_detection.lstm import lstm_meta_arch
from lstm_object_detection.protos import pipeline_pb2 as internal_pipeline_pb2
from object_detection.protos import pipeline_pb2


class ModelBuilderTest(tf.test.TestCase):

  def create_model(self, model_config, lstm_config):
    """Builds a DetectionModel based on the model config.

    Args:
      model_config: A model.proto object containing the config for the desired
        DetectionModel.
      lstm_config: LstmModel config proto that specifies LSTM train/eval
        configs.

    Returns:
      DetectionModel based on the config.
    """
    return model_builder.build(model_config, lstm_config, is_training=True)

  def get_model_configs_from_proto(self):
    """Creates a model text proto for testing.

    Returns:
      A dictionary of model configs.
    """

    model_text_proto = """
    [object_detection.protos.lstm_model] {
      train_unroll_length: 4
      eval_unroll_length: 4
    }
    model {
      ssd {
        feature_extractor {
          type: 'lstm_mobilenet_v1'
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
        negative_class_weight: 2.0
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
        normalize_loc_loss_by_codesize: true
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
      }
    }"""

    pipeline_config = pipeline_pb2.TrainEvalPipelineConfig()
    text_format.Merge(model_text_proto, pipeline_config)

    configs = {}
    configs['model'] = pipeline_config.model
    configs['lstm_model'] = pipeline_config.Extensions[
        internal_pipeline_pb2.lstm_model]

    return configs

  def test_model_creation_from_valid_configs(self):
    configs = self.get_model_configs_from_proto()
    # Test model properties.
    self.assertEqual(configs['model'].ssd.negative_class_weight, 2.0)
    self.assertTrue(configs['model'].ssd.normalize_loc_loss_by_codesize)
    self.assertEqual(configs['model'].ssd.feature_extractor.type,
                     'lstm_mobilenet_v1')

    model = self.create_model(configs['model'], configs['lstm_model'])
    # Test architechture type.
    self.assertIsInstance(model, lstm_meta_arch.LSTMMetaArch)
    # Test LSTM unroll length.
    self.assertEqual(model.unroll_length, 4)

  def test_model_creation_from_invalid_configs(self):
    configs = self.get_model_configs_from_proto()
    # Test model build failure with wrong input configs.
    with self.assertRaises(AttributeError):
      _ = self.create_model(configs['model'], configs['model'])

    # Test model builder failure with missing configs.
    with self.assertRaises(TypeError):
      # pylint: disable=no-value-for-parameter
      _ = self.create_model(configs['lstm_model'])


if __name__ == '__main__':
  tf.test.main()
