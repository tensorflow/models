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
r"""Creates and runs `Estimator` for object detection model on TPUs.

This uses the TPUEstimator API to define and run a model in TRAIN/EVAL modes.
"""
# pylint: enable=line-too-long

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import functools
import os

import tensorflow as tf

from tensorflow.contrib.tpu.python.tpu import tpu_config
from tensorflow.contrib.tpu.python.tpu import tpu_estimator
from tensorflow.contrib.training.python.training import evaluation

from object_detection import inputs
from object_detection import model
from object_detection import model_hparams
from object_detection.builders import model_builder
from object_detection.utils import config_util

tf.flags.DEFINE_bool('use_tpu', True, 'Use TPUs rather than plain CPUs')

# Cloud TPU Cluster Resolvers
tf.flags.DEFINE_string(
    'gcp_project',
    default=None,
    help='Project name for the Cloud TPU-enabled project. If not specified, we '
    'will attempt to automatically detect the GCE project from metadata.')
tf.flags.DEFINE_string(
    'tpu_zone',
    default=None,
    help='GCE zone where the Cloud TPU is located in. If not specified, we '
    'will attempt to automatically detect the GCE project from metadata.')
tf.flags.DEFINE_string(
    'tpu_name',
    default=None,
    help='Name of the Cloud TPU for Cluster Resolvers. You must specify either '
    'this flag or --master.')

tf.flags.DEFINE_string(
    'master', default=None,
    help='GRPC URL of the master (e.g. grpc://ip.address.of.tpu:8470). You '
    'must specify either this flag or --tpu_name.')

tf.flags.DEFINE_integer('num_shards', 8, 'Number of shards (TPU cores).')
tf.flags.DEFINE_integer('iterations_per_loop', 100,
                        'Number of iterations per TPU training loop.')
# For mode=train_and_eval, evaluation occurs after training is finished.
# Note: independently of steps_per_checkpoint, estimator will save the most
# recent checkpoint every 10 minutes by default for train_and_eval
tf.flags.DEFINE_string('mode', 'train_and_eval',
                       'Mode to run: train, eval, train_and_eval')
tf.flags.DEFINE_integer('train_batch_size', 32 * 8, 'Batch size for training.')

# For EVAL.
tf.flags.DEFINE_integer('min_eval_interval_secs', 180,
                        'Minimum seconds between evaluations.')
tf.flags.DEFINE_integer(
    'eval_timeout_secs', None,
    'Maximum seconds between checkpoints before evaluation terminates.')
tf.flags.DEFINE_string('hparams_overrides', None, 'Comma-separated list of '
                       'hyperparameters to override defaults.')
tf.flags.DEFINE_boolean('eval_training_data', False,
                        'If training data should be evaluated for this job.')

FLAGS = tf.flags.FLAGS


def create_estimator(run_config,
                     hparams,
                     pipeline_config_path,
                     train_steps=None,
                     eval_steps=None,
                     train_batch_size=None,
                     model_fn_creator=model.create_model_fn,
                     use_tpu=False,
                     num_shards=1,
                     params=None,
                     **kwargs):
  """Creates an `Estimator` object.

  Args:
    run_config: A `RunConfig`.
    hparams: A `HParams`.
    pipeline_config_path: A path to a pipeline config file.
    train_steps: Number of training steps. If None, the number of training steps
      is set from the `TrainConfig` proto.
    eval_steps: Number of evaluation steps per evaluation cycle. If None, the
      number of evaluation steps is set from the `EvalConfig` proto.
    train_batch_size: Training batch size. If none, use batch size from
      `TrainConfig` proto.
    model_fn_creator: A function that creates a `model_fn` for `Estimator`.
      Follows the signature:

      * Args:
        * `detection_model_fn`: Function that returns `DetectionModel` instance.
        * `configs`: Dictionary of pipeline config objects.
        * `hparams`: `HParams` object.
      * Returns:
        `model_fn` for `Estimator`.

    use_tpu: Boolean, whether training and evaluation should run on TPU.
    num_shards: Number of shards (TPU cores).
    params: Parameter dictionary passed from the estimator.
    **kwargs: Additional keyword arguments for configuration override.

  Returns:
    Estimator: A estimator object used for training and evaluation
    train_input_fn: Input function for the training loop
    eval_validation_input_fn: Input function to run for evaluation on
      validation data.
    eval_training_input_fn: Input function to run for evaluation on
      training data.
    train_steps: Number of training steps either from arg `train_steps` or
      `TrainConfig` proto
    eval_steps: Number of evaluation steps either from arg `eval_steps` or
      `EvalConfig` proto
  """
  configs = config_util.get_configs_from_pipeline_file(pipeline_config_path)
  configs = config_util.merge_external_params_with_configs(
      configs,
      hparams,
      train_steps=train_steps,
      eval_steps=eval_steps,
      batch_size=train_batch_size,
      **kwargs)
  model_config = configs['model']
  train_config = configs['train_config']
  train_input_config = configs['train_input_config']
  eval_config = configs['eval_config']
  eval_input_config = configs['eval_input_config']
  if FLAGS.eval_training_data:
    eval_input_config = configs['train_input_config']

  if params is None:
    params = {}

  if train_steps is None and train_config.num_steps:
    train_steps = train_config.num_steps

  if eval_steps is None and eval_config.num_examples:
    eval_steps = eval_config.num_examples

  detection_model_fn = functools.partial(
      model_builder.build, model_config=model_config)

  # Create the input functions for TRAIN/EVAL.
  train_input_fn = inputs.create_train_input_fn(
      train_config=train_config,
      train_input_config=train_input_config,
      model_config=model_config)
  eval_validation_input_fn = inputs.create_eval_input_fn(
      eval_config=eval_config,
      eval_input_config=eval_input_config,
      model_config=model_config)
  eval_training_input_fn = inputs.create_eval_input_fn(
      eval_config=eval_config,
      eval_input_config=train_input_config,
      model_config=model_config)

  estimator = tpu_estimator.TPUEstimator(
      model_fn=model_fn_creator(detection_model_fn, configs, hparams,
                                use_tpu),
      train_batch_size=train_config.batch_size,
      # For each core, only batch size 1 is supported for eval.
      eval_batch_size=num_shards * 1 if use_tpu else 1,
      use_tpu=use_tpu,
      config=run_config,
      params=params)
  return (estimator, train_input_fn, eval_validation_input_fn,
          eval_training_input_fn, train_steps, eval_steps)


def main(unused_argv):
  tf.flags.mark_flag_as_required('model_dir')
  tf.flags.mark_flag_as_required('pipeline_config_path')

  if FLAGS.master is None and FLAGS.tpu_name is None:
    raise RuntimeError('You must specify either --master or --tpu_name.')

  if FLAGS.master is not None:
    if FLAGS.tpu_name is not None:
      tf.logging.warn('Both --master and --tpu_name are set. Ignoring '
                      '--tpu_name and using --master.')
    tpu_grpc_url = FLAGS.master
  else:
    tpu_cluster_resolver = (
        tf.contrib.cluster_resolver.python.training.TPUClusterResolver(
            tpu_names=[FLAGS.tpu_name],
            zone=FLAGS.tpu_zone,
            project=FLAGS.gcp_project))
    tpu_grpc_url = tpu_cluster_resolver.get_master()

  config = tpu_config.RunConfig(
      master=tpu_grpc_url,
      evaluation_master=tpu_grpc_url,
      model_dir=FLAGS.model_dir,
      tpu_config=tpu_config.TPUConfig(
          iterations_per_loop=FLAGS.iterations_per_loop,
          num_shards=FLAGS.num_shards))
  params = {}
  (estimator, train_input_fn, eval_validation_input_fn, eval_training_input_fn,
   train_steps, eval_steps) = (
       create_estimator(
           config,
           model_hparams.create_hparams(
               hparams_overrides=FLAGS.hparams_overrides),
           FLAGS.pipeline_config_path,
           train_steps=FLAGS.num_train_steps,
           eval_steps=FLAGS.num_eval_steps,
           train_batch_size=FLAGS.train_batch_size,
           use_tpu=FLAGS.use_tpu,
           num_shards=FLAGS.num_shards,
           params=params))

  if FLAGS.mode in ['train', 'train_and_eval']:
    estimator.train(input_fn=train_input_fn, max_steps=train_steps)

  if FLAGS.mode == 'train_and_eval':
    # Eval one time.
    eval_results = estimator.evaluate(
        input_fn=eval_validation_input_fn, steps=eval_steps)
    tf.logging.info('Eval results: %s' % eval_results)

  # Continuously evaluating.
  if FLAGS.mode == 'eval':
    def terminate_eval():
      tf.logging.info('Terminating eval after %d seconds of no checkpoints' %
                      FLAGS.eval_timeout_secs)
      return True

    # Run evaluation when there's a new checkpoint.
    for ckpt in evaluation.checkpoints_iterator(
        FLAGS.model_dir,
        min_interval_secs=FLAGS.min_eval_interval_secs,
        timeout=FLAGS.eval_timeout_secs,
        timeout_fn=terminate_eval):

      tf.logging.info('Starting to evaluate.')
      if FLAGS.eval_training_data:
        name = 'training_data'
        input_fn = eval_training_input_fn
      else:
        name = 'validation_data'
        input_fn = eval_validation_input_fn
      try:
        eval_results = estimator.evaluate(
            input_fn=input_fn,
            steps=eval_steps,
            checkpoint_path=ckpt,
            name=name)
        tf.logging.info('Eval results: %s' % eval_results)

        # Terminate eval job when final checkpoint is reached
        current_step = int(os.path.basename(ckpt).split('-')[1])
        if current_step >= train_steps:
          tf.logging.info(
              'Evaluation finished after training step %d' % current_step)
          break

      except tf.errors.NotFoundError:
        tf.logging.info(
            'Checkpoint %s no longer exists, skipping checkpoint' % ckpt)


if __name__ == '__main__':
  tf.app.run()
