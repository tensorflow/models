# Copyright 2019 The TensorFlow Authors. All Rights Reserved.
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

"""Export a LSTD model in tflite format."""

import os
from absl import flags
import tensorflow.compat.v1 as tf

from lstm_object_detection.utils import config_util

flags.DEFINE_string('export_path', None, 'Path to export model.')
flags.DEFINE_string('frozen_graph_path', None, 'Path to frozen graph.')
flags.DEFINE_string(
    'pipeline_config_path', '',
    'Path to a pipeline_pb2.TrainEvalPipelineConfig config file.')

FLAGS = flags.FLAGS


def main(_):
  flags.mark_flag_as_required('export_path')
  flags.mark_flag_as_required('frozen_graph_path')
  flags.mark_flag_as_required('pipeline_config_path')

  configs = config_util.get_configs_from_pipeline_file(
      FLAGS.pipeline_config_path)
  lstm_config = configs['lstm_model']

  input_arrays = ['input_video_tensor']
  output_arrays = [
      'TFLite_Detection_PostProcess',
      'TFLite_Detection_PostProcess:1',
      'TFLite_Detection_PostProcess:2',
      'TFLite_Detection_PostProcess:3',
  ]
  input_shapes = {
      'input_video_tensor': [lstm_config.eval_unroll_length, 320, 320, 3],
  }

  converter = tf.lite.TFLiteConverter.from_frozen_graph(
      FLAGS.frozen_graph_path,
      input_arrays,
      output_arrays,
      input_shapes=input_shapes)
  converter.allow_custom_ops = True
  tflite_model = converter.convert()
  ofilename = os.path.join(FLAGS.export_path)
  open(ofilename, 'wb').write(tflite_model)


if __name__ == '__main__':
  tf.app.run()
