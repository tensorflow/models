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

"""Training driver for MOSAIC models."""

from absl import app
from absl import flags
import gin

from official.common import distribute_utils
from official.common import flags as tfm_flags
from official.core import base_trainer
from official.core import config_definitions
from official.core import task_factory
from official.core import train_lib
from official.core import train_utils
from official.modeling import performance

# Import MOSAIC libraries to register the model into tf.vision
# model garden factory.
# pylint: disable=unused-import
from official.projects.mosaic import mosaic_tasks
from official.projects.mosaic import registry_imports as mosaic_registry_imports
from official.vision import registry_imports
from official.vision.utils import summary_manager
# pylint: enable=unused-import

FLAGS = flags.FLAGS


# Note: we overrided the `build_trainer` due to the customized `build_model`
# methods in `MosaicSemanticSegmentationTask.
def _build_mosaic_trainer(params: config_definitions.ExperimentConfig,
                          task: mosaic_tasks.MosaicSemanticSegmentationTask,
                          model_dir: str, train: bool,
                          evaluate: bool) -> base_trainer.Trainer:
  """Creates custom trainer."""
  checkpoint_exporter = train_lib.maybe_create_best_ckpt_exporter(
      params, model_dir)
  model = task.build_model(train)
  optimizer = train_utils.create_optimizer(task, params)
  trainer = base_trainer.Trainer(
      params,
      task,
      model=model,
      optimizer=optimizer,
      train=train,
      evaluate=evaluate,
      checkpoint_exporter=checkpoint_exporter)
  return trainer


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
  distribution_strategy = distribute_utils.get_distribution_strategy(
      distribution_strategy=params.runtime.distribution_strategy,
      all_reduce_alg=params.runtime.all_reduce_alg,
      num_gpus=params.runtime.num_gpus,
      tpu_address=params.runtime.tpu)
  with distribution_strategy.scope():
    task = task_factory.get_task(params.task, logging_dir=model_dir)
    mosaic_trainer = _build_mosaic_trainer(
        task=task,
        params=params,
        model_dir=model_dir,
        train='train' in FLAGS.mode,
        evaluate='eval' in FLAGS.mode)
  train_lib.run_experiment(
      distribution_strategy=distribution_strategy,
      task=task,
      mode=FLAGS.mode,
      params=params,
      model_dir=model_dir,
      trainer=mosaic_trainer,
      eval_summary_manager=summary_manager.maybe_build_eval_summary_manager(
          params=params, model_dir=model_dir
      ),
  )

  train_utils.save_gin_config(FLAGS.mode, model_dir)

if __name__ == '__main__':
  tfm_flags.define_flags()
  flags.mark_flags_as_required(['experiment', 'mode', 'model_dir'])
  app.run(main)
