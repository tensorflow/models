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

"""MobileBERT-EdgeTPU model runner."""
import os

from absl import app
from absl import flags
from absl import logging
import orbit
import tensorflow as tf

from official.common import distribute_utils
from official.common import flags as tfm_flags
from official.projects.edgetpu.nlp import mobilebert_edgetpu_trainer
from official.projects.edgetpu.nlp.configs import params
from official.projects.edgetpu.nlp.modeling import model_builder
from official.projects.edgetpu.nlp.utils import utils


FLAGS = flags.FLAGS


def main(_):

  # Set up experiment params and load the configs from file/files.
  experiment_params = params.EdgeTPUBERTCustomParams()
  experiment_params = utils.config_override(experiment_params, FLAGS)
  model_dir = utils.get_model_dir(experiment_params, FLAGS)

  distribution_strategy = distribute_utils.get_distribution_strategy(
      distribution_strategy=experiment_params.runtime.distribution_strategy,
      all_reduce_alg=experiment_params.runtime.all_reduce_alg,
      num_gpus=experiment_params.runtime.num_gpus,
      tpu_address=experiment_params.runtime.tpu_address)

  with distribution_strategy.scope():
    teacher_model = model_builder.build_bert_pretrainer(
        pretrainer_cfg=experiment_params.teacher_model,
        quantization_friendly=False,
        name='teacher')
    student_model = model_builder.build_bert_pretrainer(
        pretrainer_cfg=experiment_params.student_model,
        quantization_friendly=True,
        name='student')

    # Load model weights.
    teacher_ckpt_dir_or_file = experiment_params.teacher_model_init_checkpoint
    if not teacher_ckpt_dir_or_file:
      raise ValueError('`teacher_model_init_checkpoint` is not specified.')
    utils.load_checkpoint(teacher_model, teacher_ckpt_dir_or_file)

    student_ckpt_dir_or_file = experiment_params.student_model_init_checkpoint
    if not student_ckpt_dir_or_file:
      # Makes sure the pretrainer variables are created.
      _ = student_model(student_model.inputs)
      logging.warn('No student checkpoint is provided, training might take '
                   'much longer before converging.')
    else:
      utils.load_checkpoint(student_model, student_ckpt_dir_or_file)

    runner = mobilebert_edgetpu_trainer.MobileBERTEdgeTPUDistillationTrainer(
        teacher_model=teacher_model,
        student_model=student_model,
        strategy=distribution_strategy,
        experiment_params=experiment_params,
        export_ckpt_path=model_dir)

    # Save checkpoint for preemption handling.
    # Checkpoint for downstreaming tasks are saved separately inside the
    # runner's train_loop_end() function.
    checkpoint = tf.train.Checkpoint(
        teacher_model=runner.teacher_model,
        student_model=runner.student_model,
        layer_wise_optimizer=runner.layer_wise_optimizer,
        e2e_optimizer=runner.e2e_optimizer,
        current_step=runner.current_step)
    checkpoint_manager = tf.train.CheckpointManager(
        checkpoint,
        directory=model_dir,
        max_to_keep=5,
        step_counter=runner.current_step,
        checkpoint_interval=20000,
        init_fn=None)

  controller = orbit.Controller(
      trainer=runner,
      evaluator=runner,
      global_step=runner.current_step,
      strategy=distribution_strategy,
      steps_per_loop=experiment_params.orbit_config.steps_per_loop,
      summary_dir=os.path.join(model_dir, 'train'),
      eval_summary_dir=os.path.join(model_dir, 'eval'),
      checkpoint_manager=checkpoint_manager)

  if FLAGS.mode == 'train':
    controller.train(steps=experiment_params.orbit_config.total_steps)
  else:
    raise ValueError('Unsupported mode, only support `train`')

if __name__ == '__main__':
  tfm_flags.define_flags()
  app.run(main)
