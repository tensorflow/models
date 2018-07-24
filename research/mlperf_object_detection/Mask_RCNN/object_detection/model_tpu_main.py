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

from absl import flags
import tensorflow as tf

from tensorflow.contrib.tpu.python.tpu import tpu_config

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
    help='Name of the Cloud TPU for Cluster Resolvers.')

flags.DEFINE_integer('num_shards', 8, 'Number of shards (TPU cores).')
flags.DEFINE_integer('iterations_per_loop', 100,
                     'Number of iterations per TPU training loop.')
# For mode=train_and_eval, evaluation occurs after training is finished.
# Note: independently of steps_per_checkpoint, estimator will save the most
# recent checkpoint every 10 minutes by default for train_and_eval
flags.DEFINE_string('mode', 'train',
                    'Mode to run: train, eval')
flags.DEFINE_integer('train_batch_size', None, 'Batch size for training. If '
                     'this is not provided, batch size is read from training '
                     'config.')

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

  kwargs = {}
  if FLAGS.train_batch_size:
    kwargs['batch_size'] = FLAGS.train_batch_size

  train_and_eval_dict = model_lib.create_estimator_and_inputs(
      run_config=config,
      hparams=model_hparams.create_hparams(FLAGS.hparams_overrides),
      pipeline_config_path=FLAGS.pipeline_config_path,
      train_steps=FLAGS.num_train_steps,
      eval_steps=FLAGS.num_eval_steps,
      use_tpu_estimator=True,
      use_tpu=FLAGS.use_tpu,
      num_shards=FLAGS.num_shards,
      **kwargs)
  estimator = train_and_eval_dict['estimator']
  train_input_fn = train_and_eval_dict['train_input_fn']
  eval_input_fn = train_and_eval_dict['eval_input_fn']
  eval_on_train_input_fn = train_and_eval_dict['eval_on_train_input_fn']
  train_steps = train_and_eval_dict['train_steps']
  eval_steps = train_and_eval_dict['eval_steps']

  if FLAGS.mode == 'train':
    estimator.train(input_fn=train_input_fn, max_steps=train_steps)

  # Continuously evaluating.
  if FLAGS.mode == 'eval':
    if FLAGS.eval_training_data:
      name = 'training_data'
      input_fn = eval_on_train_input_fn
    else:
      name = 'validation_data'
      input_fn = eval_input_fn
    model_lib.continuous_eval(estimator, FLAGS.model_dir, input_fn, eval_steps,
                              train_steps, name)


if __name__ == '__main__':
  tf.app.run()
