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

"""Added functionality to load from pipeline config for lstm framework."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import tensorflow as tf

from google.protobuf import text_format
from lstm_object_detection.protos import input_reader_google_pb2  # pylint: disable=unused-import
from lstm_object_detection.protos import pipeline_pb2 as internal_pipeline_pb2
from object_detection.protos import pipeline_pb2
from object_detection.utils import config_util


def get_configs_from_pipeline_file(pipeline_config_path):
  """Reads configuration from a pipeline_pb2.TrainEvalPipelineConfig.

  Args:
    pipeline_config_path: Path to pipeline_pb2.TrainEvalPipelineConfig text
      proto.

  Returns:
    Dictionary of configuration objects. Keys are `model`, `train_config`,
      `train_input_config`, `eval_config`, `eval_input_config`, `lstm_confg`.
      Value are the corresponding config objects.
  """
  pipeline_config = pipeline_pb2.TrainEvalPipelineConfig()
  with tf.gfile.GFile(pipeline_config_path, "r") as f:
    proto_str = f.read()
    text_format.Merge(proto_str, pipeline_config)
  configs = config_util.get_configs_from_pipeline_file(pipeline_config_path)
  if pipeline_config.HasExtension(internal_pipeline_pb2.lstm_model):
    configs["lstm_model"] = pipeline_config.Extensions[
        internal_pipeline_pb2.lstm_model]
  return configs


def create_pipeline_proto_from_configs(configs):
  """Creates a pipeline_pb2.TrainEvalPipelineConfig from configs dictionary.

  This function nearly performs the inverse operation of
  get_configs_from_pipeline_file(). Instead of returning a file path, it returns
  a `TrainEvalPipelineConfig` object.

  Args:
    configs: Dictionary of configs. See get_configs_from_pipeline_file().

  Returns:
    A fully populated pipeline_pb2.TrainEvalPipelineConfig.
  """
  pipeline_config = config_util.create_pipeline_proto_from_configs(configs)
  if "lstm_model" in configs:
    pipeline_config.Extensions[internal_pipeline_pb2.lstm_model].CopyFrom(
        configs["lstm_model"])
  return pipeline_config


def get_configs_from_multiple_files(model_config_path="",
                                    train_config_path="",
                                    train_input_config_path="",
                                    eval_config_path="",
                                    eval_input_config_path="",
                                    lstm_config_path=""):
  """Reads training configuration from multiple config files.

  Args:
    model_config_path: Path to model_pb2.DetectionModel.
    train_config_path: Path to train_pb2.TrainConfig.
    train_input_config_path: Path to input_reader_pb2.InputReader.
    eval_config_path: Path to eval_pb2.EvalConfig.
    eval_input_config_path: Path to input_reader_pb2.InputReader.
    lstm_config_path: Path to pipeline_pb2.LstmModel.

  Returns:
    Dictionary of configuration objects. Keys are `model`, `train_config`,
      `train_input_config`, `eval_config`, `eval_input_config`, `lstm_model`.
      Key/Values are returned only for valid (non-empty) strings.
  """
  configs = config_util.get_configs_from_multiple_files(
      model_config_path=model_config_path,
      train_config_path=train_config_path,
      train_input_config_path=train_input_config_path,
      eval_config_path=eval_config_path,
      eval_input_config_path=eval_input_config_path)
  if lstm_config_path:
    lstm_config = internal_pipeline_pb2.LstmModel()
    with tf.gfile.GFile(lstm_config_path, "r") as f:
      text_format.Merge(f.read(), lstm_config)
      configs["lstm_model"] = lstm_config
  return configs
