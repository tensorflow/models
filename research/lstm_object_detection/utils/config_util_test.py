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

"""Tests for object_detection.utils.config_util."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os
import tensorflow as tf

from google.protobuf import text_format
from lstm_object_detection.protos import pipeline_pb2 as internal_pipeline_pb2
from lstm_object_detection.utils import config_util
from object_detection.protos import pipeline_pb2


def _write_config(config, config_path):
  """Writes a config object to disk."""
  config_text = text_format.MessageToString(config)
  with tf.gfile.Open(config_path, "wb") as f:
    f.write(config_text)


class ConfigUtilTest(tf.test.TestCase):

  def test_get_configs_from_pipeline_file(self):
    """Test that proto configs can be read from pipeline config file."""
    pipeline_config_path = os.path.join(self.get_temp_dir(), "pipeline.config")
    pipeline_config = pipeline_pb2.TrainEvalPipelineConfig()
    pipeline_config.model.ssd.num_classes = 10
    pipeline_config.train_config.batch_size = 32
    pipeline_config.train_input_reader.label_map_path = "path/to/label_map"
    pipeline_config.eval_config.num_examples = 20
    pipeline_config.eval_input_reader.add().queue_capacity = 100

    pipeline_config.Extensions[
        internal_pipeline_pb2.lstm_model].train_unroll_length = 5
    pipeline_config.Extensions[
        internal_pipeline_pb2.lstm_model].eval_unroll_length = 10

    _write_config(pipeline_config, pipeline_config_path)

    configs = config_util.get_configs_from_pipeline_file(pipeline_config_path)
    self.assertProtoEquals(pipeline_config.model, configs["model"])
    self.assertProtoEquals(pipeline_config.train_config,
                           configs["train_config"])
    self.assertProtoEquals(pipeline_config.train_input_reader,
                           configs["train_input_config"])
    self.assertProtoEquals(pipeline_config.eval_config, configs["eval_config"])
    self.assertProtoEquals(pipeline_config.eval_input_reader,
                           configs["eval_input_configs"])
    self.assertProtoEquals(
        pipeline_config.Extensions[internal_pipeline_pb2.lstm_model],
        configs["lstm_model"])

  def test_create_pipeline_proto_from_configs(self):
    """Tests that proto can be reconstructed from configs dictionary."""
    pipeline_config_path = os.path.join(self.get_temp_dir(), "pipeline.config")

    pipeline_config = pipeline_pb2.TrainEvalPipelineConfig()
    pipeline_config.model.ssd.num_classes = 10
    pipeline_config.train_config.batch_size = 32
    pipeline_config.train_input_reader.label_map_path = "path/to/label_map"
    pipeline_config.eval_config.num_examples = 20
    pipeline_config.eval_input_reader.add().queue_capacity = 100

    pipeline_config.Extensions[
        internal_pipeline_pb2.lstm_model].train_unroll_length = 5
    pipeline_config.Extensions[
        internal_pipeline_pb2.lstm_model].eval_unroll_length = 10
    _write_config(pipeline_config, pipeline_config_path)

    configs = config_util.get_configs_from_pipeline_file(pipeline_config_path)
    pipeline_config_reconstructed = (
        config_util.create_pipeline_proto_from_configs(configs))
    self.assertEqual(pipeline_config, pipeline_config_reconstructed)


if __name__ == "__main__":
  tf.test.main()
