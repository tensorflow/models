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

r"""Evaluation executable for detection models.

This executable is used to evaluate DetectionModels. Example usage:
    ./eval \
        --logtostderr \
        --checkpoint_dir=path/to/checkpoint_dir \
        --eval_dir=path/to/eval_dir \
        --pipeline_config_path=pipeline_config.pbtxt
"""

import functools
import os
import tensorflow as tf
from google.protobuf import text_format
from google3.pyglib import app
from google3.pyglib import flags
from lstm_object_detection import evaluator
from lstm_object_detection import model_builder
from lstm_object_detection.inputs import seq_dataset_builder
from lstm_object_detection.utils import config_util
from object_detection.utils import label_map_util

tf.logging.set_verbosity(tf.logging.INFO)
flags = tf.app.flags
flags.DEFINE_boolean('eval_training_data', False,
                     'If training data should be evaluated for this job.')
flags.DEFINE_string('checkpoint_dir', '',
                    'Directory containing checkpoints to evaluate, typically '
                    'set to `train_dir` used in the training job.')
flags.DEFINE_string('eval_dir', '', 'Directory to write eval summaries to.')
flags.DEFINE_string('pipeline_config_path', '',
                    'Path to a pipeline_pb2.TrainEvalPipelineConfig config '
                    'file. If provided, other configs are ignored')
flags.DEFINE_boolean('run_once', False, 'Option to only run a single pass of '
                     'evaluation. Overrides the `max_evals` parameter in the '
                     'provided config.')
FLAGS = flags.FLAGS


def main(unused_argv):
  assert FLAGS.checkpoint_dir, '`checkpoint_dir` is missing.'
  assert FLAGS.eval_dir, '`eval_dir` is missing.'
  if FLAGS.pipeline_config_path:
    configs = config_util.get_configs_from_pipeline_file(
        FLAGS.pipeline_config_path)
  else:
    configs = config_util.get_configs_from_multiple_files(
        model_config_path=FLAGS.model_config_path,
        eval_config_path=FLAGS.eval_config_path,
        eval_input_config_path=FLAGS.input_config_path)

  pipeline_proto = config_util.create_pipeline_proto_from_configs(configs)
  config_text = text_format.MessageToString(pipeline_proto)
  tf.gfile.MakeDirs(FLAGS.eval_dir)
  with tf.gfile.Open(os.path.join(FLAGS.eval_dir, 'pipeline.config'),
                     'wb') as f:
    f.write(config_text)

  model_config = configs['model']
  lstm_config = configs['lstm_model']
  eval_config = configs['eval_config']
  input_config = configs['eval_input_config']

  if FLAGS.eval_training_data:
    input_config.external_input_reader.CopyFrom(
        configs['train_input_config'].external_input_reader)
    lstm_config.eval_unroll_length = lstm_config.train_unroll_length

  model_fn = functools.partial(
      model_builder.build,
      model_config=model_config,
      lstm_config=lstm_config,
      is_training=False)

  def get_next(config, model_config, lstm_config, unroll_length):
    return seq_dataset_builder.build(config, model_config, lstm_config,
                                     unroll_length)

  create_input_dict_fn = functools.partial(get_next, input_config, model_config,
                                           lstm_config,
                                           lstm_config.eval_unroll_length)

  label_map = label_map_util.load_labelmap(input_config.label_map_path)
  max_num_classes = max([item.id for item in label_map.item])
  categories = label_map_util.convert_label_map_to_categories(
      label_map, max_num_classes)

  if FLAGS.run_once:
    eval_config.max_evals = 1

  evaluator.evaluate(create_input_dict_fn, model_fn, eval_config, categories,
                     FLAGS.checkpoint_dir, FLAGS.eval_dir)

if __name__ == '__main__':
  app.run()
