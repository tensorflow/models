# Copyright 2021 The TensorFlow Authors. All Rights Reserved.
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

"""Defining common flags used across all BERT models/applications."""

from absl import flags
import tensorflow as tf

from official.utils import hyperparams_flags
from official.utils.flags import core as flags_core


def define_common_bert_flags():
  """Define common flags for BERT tasks."""
  flags_core.define_base(
      data_dir=False,
      model_dir=True,
      clean=False,
      train_epochs=False,
      epochs_between_evals=False,
      stop_threshold=False,
      batch_size=False,
      num_gpu=True,
      export_dir=False,
      distribution_strategy=True,
      run_eagerly=True)
  flags_core.define_distribution()
  flags.DEFINE_string('bert_config_file', None,
                      'Bert configuration file to define core bert layers.')
  flags.DEFINE_string(
      'model_export_path', None,
      'Path to the directory, where trainined model will be '
      'exported.')
  flags.DEFINE_string('tpu', '', 'TPU address to connect to.')
  flags.DEFINE_string(
      'init_checkpoint', None,
      'Initial checkpoint (usually from a pre-trained BERT model).')
  flags.DEFINE_integer('num_train_epochs', 3,
                       'Total number of training epochs to perform.')
  flags.DEFINE_integer(
      'steps_per_loop', None,
      'Number of steps per graph-mode loop. Only training step '
      'happens inside the loop. Callbacks will not be called '
      'inside. If not set the value will be configured depending on the '
      'devices available.')
  flags.DEFINE_float('learning_rate', 5e-5,
                     'The initial learning rate for Adam.')
  flags.DEFINE_float('end_lr', 0.0,
                     'The end learning rate for learning rate decay.')
  flags.DEFINE_string('optimizer_type', 'adamw',
                      'The type of optimizer to use for training (adamw|lamb)')
  flags.DEFINE_boolean(
      'scale_loss', False,
      'Whether to divide the loss by number of replica inside the per-replica '
      'loss function.')
  flags.DEFINE_boolean(
      'use_keras_compile_fit', False,
      'If True, uses Keras compile/fit() API for training logic. Otherwise '
      'use custom training loop.')
  flags.DEFINE_string(
      'hub_module_url', None, 'TF-Hub path/url to Bert module. '
      'If specified, init_checkpoint flag should not be used.')
  flags.DEFINE_bool('hub_module_trainable', True,
                    'True to make keras layers in the hub module trainable.')
  flags.DEFINE_string(
      'sub_model_export_name', None,
      'If set, `sub_model` checkpoints are exported into '
      'FLAGS.model_dir/FLAGS.sub_model_export_name.')
  flags.DEFINE_bool('explicit_allreduce', False,
                    'True to use explicit allreduce instead of the implicit '
                    'allreduce in optimizer.apply_gradients(). If fp16 mixed '
                    'precision training is used, this also enables allreduce '
                    'gradients in fp16.')
  flags.DEFINE_integer('allreduce_bytes_per_pack', 0,
                       'Number of bytes of a gradient pack for allreduce. '
                       'Should be positive integer, if set to 0, all '
                       'gradients are in one pack. Breaking gradient into '
                       'packs could enable overlap between allreduce and '
                       'backprop computation. This flag only takes effect '
                       'when explicit_allreduce is set to True.')

  flags_core.define_log_steps()

  # Adds flags for mixed precision and multi-worker training.
  flags_core.define_performance(
      num_parallel_calls=False,
      inter_op=False,
      intra_op=False,
      synthetic_data=False,
      max_train_steps=False,
      dtype=True,
      loss_scale=True,
      all_reduce_alg=True,
      num_packs=False,
      tf_gpu_thread_mode=True,
      datasets_num_private_threads=True,
      enable_xla=True,
      fp16_implementation=True,
  )

  # Adds gin configuration flags.
  hyperparams_flags.define_gin_flags()


def dtype():
  return flags_core.get_tf_dtype(flags.FLAGS)


def use_float16():
  return flags_core.get_tf_dtype(flags.FLAGS) == tf.float16


def use_graph_rewrite():
  return flags.FLAGS.fp16_implementation == 'graph_rewrite'


def get_loss_scale():
  return flags_core.get_loss_scale(flags.FLAGS, default_for_fp16='dynamic')
