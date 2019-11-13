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
r"""Evaluation executable for detection models.

This executable is used to evaluate DetectionModels. There are two ways of
configuring the eval job.

1) A single pipeline_pb2.TrainEvalPipelineConfig file maybe specified instead.
In this mode, the --eval_training_data flag may be given to force the pipeline
to evaluate on training data instead.

Example usage:
    ./eval \
        --logtostderr \
        --checkpoint_dir=path/to/checkpoint_dir \
        --eval_dir=path/to/eval_dir \
        --pipeline_config_path=pipeline_config.pbtxt

2) Three configuration files may be provided: a model_pb2.DetectionModel
configuration file to define what type of DetectionModel is being evaluated, an
input_reader_pb2.InputReader file to specify what data the model is evaluating
and an eval_pb2.EvalConfig file to configure evaluation parameters.

Example usage:
    ./eval \
        --logtostderr \
        --checkpoint_dir=path/to/checkpoint_dir \
        --eval_dir=path/to/eval_dir \
        --eval_config_path=eval_config.pbtxt \
        --model_config_path=model_config.pbtxt \
        --input_config_path=eval_input_config.pbtxt
"""
import functools
import os
import tensorflow as tf
from tensorflow.contrib import framework as contrib_framework

from object_detection.builders import dataset_builder
from object_detection.builders import graph_rewriter_builder
from object_detection.builders import model_builder
from object_detection.legacy import evaluator
from object_detection.utils import config_util
from object_detection.utils import label_map_util

tf.logging.set_verbosity(tf.logging.INFO)

flags = tf.app.flags
flags.DEFINE_boolean('eval_training_data', False,
                     'If training data should be evaluated for this job.')
flags.DEFINE_string(
    'checkpoint_dir', '',
    'Directory containing checkpoints to evaluate, typically '
    'set to `train_dir` used in the training job.')
flags.DEFINE_string('eval_dir', '', 'Directory to write eval summaries to.')
flags.DEFINE_string(
    'pipeline_config_path', '',
    'Path to a pipeline_pb2.TrainEvalPipelineConfig config '
    'file. If provided, other configs are ignored')
flags.DEFINE_string('eval_config_path', '',
                    'Path to an eval_pb2.EvalConfig config file.')
flags.DEFINE_string('input_config_path', '',
                    'Path to an input_reader_pb2.InputReader config file.')
flags.DEFINE_string('model_config_path', '',
                    'Path to a model_pb2.DetectionModel config file.')
flags.DEFINE_boolean(
    'run_once', False, 'Option to only run a single pass of '
    'evaluation. Overrides the `max_evals` parameter in the '
    'provided config.')
FLAGS = flags.FLAGS


@contrib_framework.deprecated(None, 'Use object_detection/model_main.py.')
def main(unused_argv):
  assert FLAGS.checkpoint_dir, '`checkpoint_dir` is missing.'
  assert FLAGS.eval_dir, '`eval_dir` is missing.'
  tf.gfile.MakeDirs(FLAGS.eval_dir)
  if FLAGS.pipeline_config_path:
    configs = config_util.get_configs_from_pipeline_file(
        FLAGS.pipeline_config_path)
    tf.gfile.Copy(
        FLAGS.pipeline_config_path,
        os.path.join(FLAGS.eval_dir, 'pipeline.config'),
        overwrite=True)
  else:
    configs = config_util.get_configs_from_multiple_files(
        model_config_path=FLAGS.model_config_path,
        eval_config_path=FLAGS.eval_config_path,
        eval_input_config_path=FLAGS.input_config_path)
    for name, config in [('model.config', FLAGS.model_config_path),
                         ('eval.config', FLAGS.eval_config_path),
                         ('input.config', FLAGS.input_config_path)]:
      tf.gfile.Copy(config, os.path.join(FLAGS.eval_dir, name), overwrite=True)

  model_config = configs['model']
  eval_config = configs['eval_config']
  input_config = configs['eval_input_config']
  if FLAGS.eval_training_data:
    input_config = configs['train_input_config']

  model_fn = functools.partial(
      model_builder.build, model_config=model_config, is_training=False)

  def get_next(config):
    return dataset_builder.make_initializable_iterator(
        dataset_builder.build(config)).get_next()

  create_input_dict_fn = functools.partial(get_next, input_config)

  categories = label_map_util.create_categories_from_labelmap(
      input_config.label_map_path)

  if FLAGS.run_once:
    eval_config.max_evals = 1

  graph_rewriter_fn = None
  if 'graph_rewriter_config' in configs:
    graph_rewriter_fn = graph_rewriter_builder.build(
        configs['graph_rewriter_config'], is_training=False)

  evaluator.evaluate(
      create_input_dict_fn,
      model_fn,
      eval_config,
      categories,
      FLAGS.checkpoint_dir,
      FLAGS.eval_dir,
      graph_hook_fn=graph_rewriter_fn)


if __name__ == '__main__':
  tf.app.run()
