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

"""Tests for official.nlp.projects.mobilebert.distillation."""
import os

from absl import logging
from absl.testing import parameterized
import tensorflow as tf

from official.core import config_definitions as cfg
from official.modeling import optimization
from official.modeling import tf_utils
from official.modeling.progressive import trainer as prog_trainer_lib
from official.nlp.configs import bert
from official.nlp.configs import encoders
from official.nlp.data import pretrain_dataloader
from official.nlp.modeling import layers
from official.nlp.modeling import models
from official.nlp.projects.mobilebert import distillation


class DistillationTest(tf.test.TestCase, parameterized.TestCase):

  def prepare_config(self, teacher_block_num, student_block_num,
                     transfer_teacher_layers):
    # using small model for testing
    task_config = distillation.BertDistillationTaskConfig(
        teacher_model=bert.PretrainerConfig(
            encoder=encoders.EncoderConfig(
                type='mobilebert',
                mobilebert=encoders.MobileBertEncoderConfig(
                    num_blocks=teacher_block_num)),
            cls_heads=[
                bert.ClsHeadConfig(
                    inner_dim=256,
                    num_classes=2,
                    dropout_rate=0.1,
                    name='next_sentence')
            ],
            mlm_activation='gelu'),
        student_model=bert.PretrainerConfig(
            encoder=encoders.EncoderConfig(
                type='mobilebert',
                mobilebert=encoders.MobileBertEncoderConfig(
                    num_blocks=student_block_num)),
            cls_heads=[
                bert.ClsHeadConfig(
                    inner_dim=256,
                    num_classes=2,
                    dropout_rate=0.1,
                    name='next_sentence')
            ],
            mlm_activation='relu'),
        train_data=pretrain_dataloader.BertPretrainDataConfig(
            input_path='dummy',
            max_predictions_per_seq=76,
            seq_length=512,
            global_batch_size=10),
        validation_data=pretrain_dataloader.BertPretrainDataConfig(
            input_path='dummy',
            max_predictions_per_seq=76,
            seq_length=512,
            global_batch_size=10))

    # set only 1 step for each stage
    progressive_config = distillation.BertDistillationProgressiveConfig()
    progressive_config.layer_wise_distill_config.transfer_teacher_layers = (
        transfer_teacher_layers)
    progressive_config.layer_wise_distill_config.num_steps = 1
    progressive_config.pretrain_distill_config.num_steps = 1

    optimization_config = optimization.OptimizationConfig(
        optimizer=optimization.OptimizerConfig(
            type='lamb',
            lamb=optimization.LAMBConfig(
                weight_decay_rate=0.0001,
                exclude_from_weight_decay=[
                    'LayerNorm', 'layer_norm', 'bias', 'no_norm'
                ])),
        learning_rate=optimization.LrConfig(
            type='polynomial',
            polynomial=optimization.PolynomialLrConfig(
                initial_learning_rate=1.5e-3,
                decay_steps=10000,
                end_learning_rate=1.5e-3)),
        warmup=optimization.WarmupConfig(
            type='linear',
            linear=optimization.LinearWarmupConfig(warmup_learning_rate=0)))

    exp_config = cfg.ExperimentConfig(
        task=task_config,
        trainer=prog_trainer_lib.ProgressiveTrainerConfig(
            progressive=progressive_config,
            optimizer_config=optimization_config))

    # Create a teacher model checkpoint.
    teacher_encoder = encoders.build_encoder(task_config.teacher_model.encoder)
    pretrainer_config = task_config.teacher_model
    if pretrainer_config.cls_heads:
      teacher_cls_heads = [
          layers.ClassificationHead(**cfg.as_dict())
          for cfg in pretrainer_config.cls_heads
      ]
    else:
      teacher_cls_heads = []

    masked_lm = layers.MobileBertMaskedLM(
        embedding_table=teacher_encoder.get_embedding_table(),
        activation=tf_utils.get_activation(pretrainer_config.mlm_activation),
        initializer=tf.keras.initializers.TruncatedNormal(
            stddev=pretrainer_config.mlm_initializer_range),
        name='cls/predictions')
    teacher_pretrainer = models.BertPretrainerV2(
        encoder_network=teacher_encoder,
        classification_heads=teacher_cls_heads,
        customized_masked_lm=masked_lm)

    # The model variables will be created after the forward call.
    _ = teacher_pretrainer(teacher_pretrainer.inputs)
    teacher_pretrainer_ckpt = tf.train.Checkpoint(
        **teacher_pretrainer.checkpoint_items)
    teacher_ckpt_path = os.path.join(self.get_temp_dir(), 'teacher_model.ckpt')
    teacher_pretrainer_ckpt.save(teacher_ckpt_path)
    exp_config.task.teacher_model_init_checkpoint = self.get_temp_dir()

    return exp_config

  @parameterized.parameters((2, 2, None), (4, 2, [1, 3]))
  def test_task(self, teacher_block_num, student_block_num,
                transfer_teacher_layers):
    exp_config = self.prepare_config(teacher_block_num, student_block_num,
                                     transfer_teacher_layers)
    bert_distillation_task = distillation.BertDistillationTask(
        strategy=tf.distribute.get_strategy(),
        progressive=exp_config.trainer.progressive,
        optimizer_config=exp_config.trainer.optimizer_config,
        task_config=exp_config.task)
    metrics = bert_distillation_task.build_metrics()
    train_dataset = bert_distillation_task.get_train_dataset(stage_id=0)
    train_iterator = iter(train_dataset)

    eval_dataset = bert_distillation_task.get_eval_dataset(stage_id=0)
    eval_iterator = iter(eval_dataset)
    optimizer = tf.keras.optimizers.SGD(lr=0.1)

    # test train/val step for all stages, including the last pretraining stage
    for stage in range(student_block_num + 1):
      step = stage
      bert_distillation_task.update_pt_stage(step)
      model = bert_distillation_task.get_model(stage, None)
      bert_distillation_task.initialize(model)
      bert_distillation_task.train_step(next(train_iterator), model, optimizer,
                                        metrics=metrics)
      bert_distillation_task.validation_step(next(eval_iterator), model,
                                             metrics=metrics)

    logging.info('begin to save and load model checkpoint')
    ckpt = tf.train.Checkpoint(model=model)
    ckpt.save(self.get_temp_dir())

if __name__ == '__main__':
  tf.test.main()
