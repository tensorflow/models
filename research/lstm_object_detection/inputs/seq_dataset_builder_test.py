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

"""Tests for dataset_builder."""

import os
import numpy as np
import tensorflow.compat.v1 as tf

from google.protobuf import text_format
from tensorflow.core.example import example_pb2
from tensorflow.core.example import feature_pb2
from lstm_object_detection.inputs import seq_dataset_builder
from lstm_object_detection.protos import pipeline_pb2 as internal_pipeline_pb2
from object_detection.builders import preprocessor_builder
from object_detection.core import standard_fields as fields
from object_detection.protos import input_reader_pb2
from object_detection.protos import pipeline_pb2
from object_detection.protos import preprocessor_pb2


class DatasetBuilderTest(tf.test.TestCase):

  def _create_tf_record(self):
    path = os.path.join(self.get_temp_dir(), 'tfrecord')
    writer = tf.python_io.TFRecordWriter(path)

    image_tensor = np.random.randint(255, size=(16, 16, 3)).astype(np.uint8)
    with self.test_session():
      encoded_jpeg = tf.image.encode_jpeg(tf.constant(image_tensor)).eval()

    sequence_example = example_pb2.SequenceExample(
        context=feature_pb2.Features(
            feature={
                'image/format':
                    feature_pb2.Feature(
                        bytes_list=feature_pb2.BytesList(
                            value=['jpeg'.encode('utf-8')])),
                'image/height':
                    feature_pb2.Feature(
                        int64_list=feature_pb2.Int64List(value=[16])),
                'image/width':
                    feature_pb2.Feature(
                        int64_list=feature_pb2.Int64List(value=[16])),
            }),
        feature_lists=feature_pb2.FeatureLists(
            feature_list={
                'image/encoded':
                    feature_pb2.FeatureList(feature=[
                        feature_pb2.Feature(
                            bytes_list=feature_pb2.BytesList(
                                value=[encoded_jpeg])),
                    ]),
                'image/object/bbox/xmin':
                    feature_pb2.FeatureList(feature=[
                        feature_pb2.Feature(
                            float_list=feature_pb2.FloatList(value=[0.0])),
                    ]),
                'image/object/bbox/xmax':
                    feature_pb2.FeatureList(feature=[
                        feature_pb2.Feature(
                            float_list=feature_pb2.FloatList(value=[1.0]))
                    ]),
                'image/object/bbox/ymin':
                    feature_pb2.FeatureList(feature=[
                        feature_pb2.Feature(
                            float_list=feature_pb2.FloatList(value=[0.0])),
                    ]),
                'image/object/bbox/ymax':
                    feature_pb2.FeatureList(feature=[
                        feature_pb2.Feature(
                            float_list=feature_pb2.FloatList(value=[1.0]))
                    ]),
                'image/object/class/label':
                    feature_pb2.FeatureList(feature=[
                        feature_pb2.Feature(
                            int64_list=feature_pb2.Int64List(value=[2]))
                    ]),
            }))

    writer.write(sequence_example.SerializeToString())
    writer.close()

    return path

  def _get_model_configs_from_proto(self):
    """Creates a model text proto for testing.

    Returns:
      A dictionary of model configs.
    """

    model_text_proto = """
    [lstm_object_detection.protos.lstm_model] {
      train_unroll_length: 4
      eval_unroll_length: 4
    }
    model {
      ssd {
        feature_extractor {
          type: 'lstm_mobilenet_v1_fpn'
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
            height: 32
            width: 32
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

  def _get_data_augmentation_preprocessor_proto(self):
    preprocessor_text_proto = """
    random_horizontal_flip {
    }
    """
    preprocessor_proto = preprocessor_pb2.PreprocessingStep()
    text_format.Merge(preprocessor_text_proto, preprocessor_proto)
    return preprocessor_proto

  def _create_training_dict(self, tensor_dict):
    image_dict = {}
    all_dict = {}
    all_dict['batch'] = tensor_dict.pop('batch')
    for i, _ in enumerate(tensor_dict[fields.InputDataFields.image]):
      for key, val in tensor_dict.items():
        image_dict[key] = val[i]

      image_dict[fields.InputDataFields.image] = tf.to_float(
          tf.expand_dims(image_dict[fields.InputDataFields.image], 0))
      suffix = str(i)
      for key, val in image_dict.items():
        all_dict[key + suffix] = val
    return all_dict

  def _get_input_proto(self, input_reader):
    return """
        external_input_reader {
          [lstm_object_detection.protos.GoogleInputReader.google_input_reader] {
            %s: {
              input_path: '{0}'
              data_type: TF_SEQUENCE_EXAMPLE
              video_length: 4
            }
          }
        }
      """ % input_reader

  def test_video_input_reader(self):
    input_reader_proto = input_reader_pb2.InputReader()
    text_format.Merge(
        self._get_input_proto('tf_record_video_input_reader'),
        input_reader_proto)

    configs = self._get_model_configs_from_proto()
    tensor_dict = seq_dataset_builder.build(
        input_reader_proto,
        configs['model'],
        configs['lstm_model'],
        unroll_length=1)

    all_dict = self._create_training_dict(tensor_dict)

    self.assertEqual((1, 32, 32, 3), all_dict['image0'].shape)
    self.assertEqual(4, all_dict['groundtruth_boxes0'].shape[1])

  def test_build_with_data_augmentation(self):
    input_reader_proto = input_reader_pb2.InputReader()
    text_format.Merge(
        self._get_input_proto('tf_record_video_input_reader'),
        input_reader_proto)

    configs = self._get_model_configs_from_proto()
    data_augmentation_options = [
        preprocessor_builder.build(
            self._get_data_augmentation_preprocessor_proto())
    ]
    tensor_dict = seq_dataset_builder.build(
        input_reader_proto,
        configs['model'],
        configs['lstm_model'],
        unroll_length=1,
        data_augmentation_options=data_augmentation_options)

    all_dict = self._create_training_dict(tensor_dict)
    self.assertEqual((1, 32, 32, 3), all_dict['image0'].shape)
    self.assertEqual(4, all_dict['groundtruth_boxes0'].shape[1])

  def test_raises_error_without_input_paths(self):
    input_reader_text_proto = """
      shuffle: false
      num_readers: 1
      load_instance_masks: true
    """
    input_reader_proto = input_reader_pb2.InputReader()
    text_format.Merge(input_reader_text_proto, input_reader_proto)

    configs = self._get_model_configs_from_proto()
    with self.assertRaises(ValueError):
      _ = seq_dataset_builder.build(
          input_reader_proto,
          configs['model'],
          configs['lstm_model'],
          unroll_length=1)


if __name__ == '__main__':
  tf.test.main()
