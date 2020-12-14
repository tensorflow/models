# Lint as: python3
# Copyright 2020 The TensorFlow Authors. All Rights Reserved.
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
"""TFM continuous finetuning+eval training driver."""
import gc
import os
import time
from typing import Any, Mapping, Optional

from absl import app
from absl import flags
from absl import logging
import gin
import tensorflow as tf

# pylint: disable=unused-import
from official.common import registry_imports
# pylint: enable=unused-import
from official.common import distribute_utils
from official.common import flags as tfm_flags
from official.core import config_definitions
from official.core import task_factory
from official.core import train_lib
from official.core import train_utils
from official.modeling import performance

FLAGS = flags.FLAGS

flags.DEFINE_integer(
    'pretrain_steps',
    default=None,
    help='The number of total training steps for the pretraining job.')


def run_continuous_finetune(
    mode: str,
    params: config_definitions.ExperimentConfig,
    model_dir: str,
    run_post_eval: bool = False,
    pretrain_steps: Optional[int] = None,
) -> Mapping[str, Any]:
  """Run modes with continuous training.

  Currently only supports continuous_train_and_eval.

  Args:
    mode: A 'str', specifying the mode. continuous_train_and_eval - monitors a
      checkpoint directory. Once a new checkpoint is discovered, loads the
      checkpoint, finetune the model by training it (probably on another dataset
      or with another task), then evaluate the finetuned model.
    params: ExperimentConfig instance.
    model_dir: A 'str', a path to store model checkpoints and summaries.
    run_post_eval: Whether to run post eval once after training, metrics logs
      are returned.
    pretrain_steps: Optional, the number of total training steps for the
      pretraining job.

  Returns:
    eval logs: returns eval metrics logs when run_post_eval is set to True,
      othewise, returns {}.
  """

  assert mode == 'continuous_train_and_eval', (
      'Only continuous_train_and_eval is supported by continuous_finetune. '
      'Got mode: {}'.format(mode))

  # Sets mixed_precision policy. Using 'mixed_float16' or 'mixed_bfloat16'
  # can have significant impact on model speeds by utilizing float16 in case of
  # GPUs, and bfloat16 in the case of TPUs. loss_scale takes effect only when
  # dtype is float16
  if params.runtime.mixed_precision_dtype:
    performance.set_mixed_precision_policy(params.runtime.mixed_precision_dtype,
                                           params.runtime.loss_scale)
  distribution_strategy = distribute_utils.get_distribution_strategy(
      distribution_strategy=params.runtime.distribution_strategy,
      all_reduce_alg=params.runtime.all_reduce_alg,
      num_gpus=params.runtime.num_gpus,
      tpu_address=params.runtime.tpu)

  retry_times = 0
  while not tf.io.gfile.isdir(params.task.init_checkpoint):
    # Wait for the init_checkpoint directory to be created.
    if retry_times >= 60:
      raise ValueError(
          'ExperimentConfig.task.init_checkpoint must be a directory for '
          'continuous_train_and_eval mode.')
    retry_times += 1
    time.sleep(60)

  summary_writer = tf.summary.create_file_writer(
      os.path.join(model_dir, 'eval'))

  global_step = 0

  def timeout_fn():
    if pretrain_steps and global_step < pretrain_steps:
      # Keeps waiting for another timeout period.
      logging.info(
          'Continue waiting for new checkpoint as current pretrain '
          'global_step=%d and target is %d.', global_step, pretrain_steps)
      return False
    # Quits the loop.
    return True

  for pretrain_ckpt in tf.train.checkpoints_iterator(
      checkpoint_dir=params.task.init_checkpoint,
      min_interval_secs=10,
      timeout=params.trainer.continuous_eval_timeout,
      timeout_fn=timeout_fn):
    with distribution_strategy.scope():
      global_step = train_utils.read_global_step_from_checkpoint(pretrain_ckpt)

    if params.trainer.best_checkpoint_export_subdir:
      best_ckpt_subdir = '{}_{}'.format(
          params.trainer.best_checkpoint_export_subdir, global_step)
      params_replaced = params.replace(
          task={'init_checkpoint': pretrain_ckpt},
          trainer={'best_checkpoint_export_subdir': best_ckpt_subdir})
    else:
      params_replaced = params.replace(task={'init_checkpoint': pretrain_ckpt})
    params_replaced.lock()
    logging.info('Running finetuning with params: %s', params_replaced)

    with distribution_strategy.scope():
      task = task_factory.get_task(params_replaced.task, logging_dir=model_dir)

    _, eval_metrics = train_lib.run_experiment(
        distribution_strategy=distribution_strategy,
        task=task,
        mode='train_and_eval',
        # replace params.task.init_checkpoint to make sure that we load
        # exactly this pretrain checkpoint.
        params=params_replaced,
        model_dir=model_dir,
        run_post_eval=True,
        save_summary=False)
    logging.info('Evaluation finished. Pretrain global_step: %d', global_step)
    train_utils.write_json_summary(model_dir, global_step, eval_metrics)

    if not os.path.basename(model_dir):  # if model_dir.endswith('/')
      summary_grp = os.path.dirname(model_dir) + '_' + task.name
    else:
      summary_grp = os.path.basename(model_dir) + '_' + task.name
    summaries = {}
    for name, value in eval_metrics.items():
      summaries[summary_grp + '/' + name] = value
    train_utils.write_summary(summary_writer, global_step, summaries)

    train_utils.remove_ckpts(model_dir)
    # In TF2, the resource life cycle is bound with the python object life
    # cycle. Force trigger python garbage collection here so those resources
    # can be deallocated in time, so it doesn't cause OOM when allocating new
    # objects.
    # TODO(b/169178664): Fix cycle reference in Keras model and revisit to see
    # if we need gc here.
    gc.collect()

  if run_post_eval:
    return eval_metrics
  return {}


def main(_):
  gin.parse_config_files_and_bindings(FLAGS.gin_file, FLAGS.gin_params)
  params = train_utils.parse_configuration(FLAGS)
  model_dir = FLAGS.model_dir
  train_utils.serialize_config(params, model_dir)
  run_continuous_finetune(FLAGS.mode, params, model_dir, FLAGS.pretrain_steps)


if __name__ == '__main__':
  tfm_flags.define_flags()
  app.run(main)
