# Copyright 2024 The TensorFlow Authors. All Rights Reserved.
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

"""TensorFlow Model Garden Vision training driver."""

from absl import app
from absl import flags
from absl import logging
import gin
import tensorflow as tf, tf_keras

from official.common import distribute_utils
from official.common import flags as tfm_flags
from official.core import task_factory
from official.core import train_lib
from official.core import train_utils
from official.modeling import performance
from official.vision import registry_imports  # pylint: disable=unused-import
from official.vision.utils import summary_manager


FLAGS = flags.FLAGS

flags.DEFINE_bool(
    'enable_async_checkpointing',
    default=True,
    help='A boolean indicating whether to enable async checkpoint saving')


def _run_experiment_with_preemption_recovery(params, model_dir):
  """Runs experiment and tries to reconnect when encounting a preemption."""
  keep_training = True
  while keep_training:
    preemption_watcher = None
    try:
      distribution_strategy = distribute_utils.get_distribution_strategy(
          distribution_strategy=params.runtime.distribution_strategy,
          all_reduce_alg=params.runtime.all_reduce_alg,
          num_gpus=params.runtime.num_gpus,
          tpu_address=params.runtime.tpu)
      with distribution_strategy.scope():
        task = task_factory.get_task(params.task, logging_dir=model_dir)
      # pylint: disable=line-too-long
      preemption_watcher = None  # copybara-replace
      # pylint: enable=line-too-long

      train_lib.run_experiment(
          distribution_strategy=distribution_strategy,
          task=task,
          mode=FLAGS.mode,
          params=params,
          model_dir=model_dir,
          summary_manager=None,
          eval_summary_manager=summary_manager.maybe_build_eval_summary_manager(
              params=params, model_dir=model_dir
          ),
          enable_async_checkpointing=FLAGS.enable_async_checkpointing,
      )

      keep_training = False
    except tf.errors.OpError as e:
      if preemption_watcher and preemption_watcher.preemption_message:
        preemption_watcher.block_until_worker_exit()
        logging.info(
            'Some TPU workers had been preempted (message: %s), '
            'retarting training from the last checkpoint...',
            preemption_watcher.preemption_message)
        keep_training = True
      else:
        raise e from None


def main(_):
  gin.parse_config_files_and_bindings(FLAGS.gin_file, FLAGS.gin_params)
  params = train_utils.parse_configuration(FLAGS)
  model_dir = FLAGS.model_dir
  if 'train' in FLAGS.mode:
    # Pure eval modes do not output yaml files. Otherwise continuous eval job
    # may race against the train job for writing the same file.
    train_utils.serialize_config(params, model_dir)

  # Sets mixed_precision policy. Using 'mixed_float16' or 'mixed_bfloat16'
  # can have significant impact on model speeds by utilizing float16 in case of
  # GPUs, and bfloat16 in the case of TPUs. loss_scale takes effect only when
  # dtype is float16
  if params.runtime.mixed_precision_dtype:
    performance.set_mixed_precision_policy(params.runtime.mixed_precision_dtype)

  _run_experiment_with_preemption_recovery(params, model_dir)
  train_utils.save_gin_config(FLAGS.mode, model_dir)

if __name__ == '__main__':
  tfm_flags.define_flags()
  flags.mark_flags_as_required(['experiment', 'mode', 'model_dir'])
  app.run(main)
