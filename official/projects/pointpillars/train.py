# Copyright 2023 The TensorFlow Authors. All Rights Reserved.
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

"""PointPillars trainer."""

import os

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
from official.projects.pointpillars import registry_imports  # pylint: disable=unused-import
from official.projects.pointpillars.utils import model_exporter

FLAGS = flags.FLAGS


def _check_if_resumed_job(model_dir: str, manual_checkpoint_path: str):
  """Check if the job is a resumed job."""
  logging.info('Check if the job is resumed from %s', model_dir)
  if not tf.io.gfile.exists(model_dir):
    logging.info('%s not found, this is a new job.', model_dir)
    return
  try:
    tf.train.load_checkpoint(model_dir)
  except ValueError:
    logging.info('No checkpoints found in %s, this is a new job.', model_dir)
    return
  else:
    logging.info('The job is resuming from %s', model_dir)
    if manual_checkpoint_path:
      logging.warning('Found manually indicated checkpoint path %s for a '
                      'resuming job, the manual checkpoint path will be '
                      'ignored because the model must restore from '
                      'checkpoints in %s.', manual_checkpoint_path, model_dir)


def main(_):
  gin.parse_config_files_and_bindings(FLAGS.gin_file, FLAGS.gin_params)
  params = train_utils.parse_configuration(FLAGS)
  model_dir = FLAGS.model_dir

  # A training job could be terminated and resumed at any time by machine
  # scheduler. A resuming job will automatically restore states from the
  # model_dir, like loading checkpoints. It will skip checkpointed training
  # steps and start from there for subsequent training. This function simply
  # checks if the job is a resumed job or not and logs info for that.
  _check_if_resumed_job(model_dir, params.task.init_checkpoint)

  if 'train' in FLAGS.mode:
    # Pure eval modes do not output yaml files. Otherwise continuous eval job
    # may race against the train job for writing the same file.
    train_utils.serialize_config(params, model_dir)

  # Sets mixed_precision policy. Using 'mixed_float16' or 'mixed_bfloat16'
  # can have significant impact on model speeds by utilizing float16 in case of
  # GPUs, and bfloat16 in the case of TPUs. 'loss_scale' takes effect only when
  # dtype is float16.
  if params.runtime.mixed_precision_dtype:
    performance.set_mixed_precision_policy(params.runtime.mixed_precision_dtype)
  distribution_strategy = distribute_utils.get_distribution_strategy(
      distribution_strategy=params.runtime.distribution_strategy,
      all_reduce_alg=params.runtime.all_reduce_alg,
      num_gpus=params.runtime.num_gpus,
      tpu_address=params.runtime.tpu)
  with distribution_strategy.scope():
    task = task_factory.get_task(params.task, logging_dir=model_dir)

  train_lib.run_experiment(
      distribution_strategy=distribution_strategy,
      task=task,
      mode=FLAGS.mode,
      params=params,
      model_dir=model_dir)

  model_exporter.export_inference_graph(
      batch_size=1,
      params=params,
      checkpoint_path=model_dir,
      export_dir=os.path.join(model_dir, 'saved_model'))


if __name__ == '__main__':
  tfm_flags.define_flags()
  app.run(main)
