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

import os
from absl import flags
import tensorflow as tf

from tensorflow.contrib.tpu.python.tpu import tpu_config
from tensorflow.contrib.training.python.training import evaluation

from object_detection import model_hparams
from object_detection import model_lib

tf.flags.DEFINE_bool('use_tpu', True, 'Use TPUs rather than plain CPUs')

# Cloud TPU Cluster Resolvers
flags.DEFINE_string(
    'gcp_project',
    default=None,
    help='Project name for the Cloud TPU-enabled project. If not specified, we '
    'will attempt to automatically detect the GCE project from metadata.')
flags.DEFINE_string(
    'tpu_zone',
    default=None,
    help='GCE zone where the Cloud TPU is located in. If not specified, we '
    'will attempt to automatically detect the GCE project from metadata.')
flags.DEFINE_string(
    'tpu_name',
    default=None,
    help='Name of the Cloud TPU for Cluster Resolvers. You must specify either '
    'this flag or --master.')

flags.DEFINE_string(
    'master',
    default=None,
    help='GRPC URL of the master (e.g. grpc://ip.address.of.tpu:8470). You '
    'must specify either this flag or --tpu_name.')

flags.DEFINE_integer('num_shards', 8, 'Number of shards (TPU cores).')
flags.DEFINE_integer('iterations_per_loop', 100,
                     'Number of iterations per TPU training loop.')
# For mode=train_and_eval, evaluation occurs after training is finished.
# Note: independently of steps_per_checkpoint, estimator will save the most
# recent checkpoint every 10 minutes by default for train_and_eval
flags.DEFINE_string('mode', 'train_and_eval',
                    'Mode to run: train, eval, train_and_eval')
flags.DEFINE_integer('train_batch_size', 32 * 8, 'Batch size for training.')

# For EVAL.
flags.DEFINE_integer('min_eval_interval_secs', 180,
                     'Minimum seconds between evaluations.')
flags.DEFINE_integer(
    'eval_timeout_secs', None,
    'Maximum seconds between checkpoints before evaluation terminates.')
flags.DEFINE_string(
    'hparams_overrides', None, 'Comma-separated list of '
    'hyperparameters to override defaults.')
flags.DEFINE_boolean('eval_training_data', False,
                     'If training data should be evaluated for this job.')
flags.DEFINE_string(
    'model_dir', None, 'Path to output model directory '
    'where event and checkpoint files will be written.')
flags.DEFINE_string('pipeline_config_path', None, 'Path to pipeline config '
                    'file.')
flags.DEFINE_integer('num_train_steps', None, 'Number of train steps.')
flags.DEFINE_integer('num_eval_steps', None, 'Number of train steps.')

FLAGS = tf.flags.FLAGS


def main(unused_argv):
  flags.mark_flag_as_required('model_dir')
  flags.mark_flag_as_required('pipeline_config_path')

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

  train_and_eval_dict = model_lib.create_estimator_and_inputs(
      run_config=config,
      hparams=model_hparams.create_hparams(FLAGS.hparams_overrides),
      pipeline_config_path=FLAGS.pipeline_config_path,
      train_steps=FLAGS.num_train_steps,
      eval_steps=FLAGS.num_eval_steps,
      use_tpu_estimator=True,
      use_tpu=FLAGS.use_tpu,
      num_shards=FLAGS.num_shards,
      batch_size=FLAGS.train_batch_size)
  estimator = train_and_eval_dict['estimator']
  train_input_fn = train_and_eval_dict['train_input_fn']
  eval_input_fn = train_and_eval_dict['eval_input_fn']
  train_steps = train_and_eval_dict['train_steps']
  eval_steps = train_and_eval_dict['eval_steps']

  if FLAGS.mode in ['train', 'train_and_eval']:
    estimator.train(input_fn=train_input_fn, max_steps=train_steps)

  if FLAGS.mode == 'train_and_eval':
    # Eval one time.
    eval_results = estimator.evaluate(input_fn=eval_input_fn, steps=eval_steps)
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
        input_fn = train_input_fn
      else:
        name = 'validation_data'
        input_fn = eval_input_fn
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
