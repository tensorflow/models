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
"""Tests for google.nlp.progressive_masked_lm."""

# Import libraries
from absl.testing import parameterized
import gin
import tensorflow as tf

from tensorflow.python.distribute import combinations
from tensorflow.python.distribute import strategy_combinations
from official.core import config_definitions as cfg
from official.modeling.progressive import trainer as prog_trainer_lib
from official.nlp.configs import bert
from official.nlp.configs import encoders
from official.nlp.data import pretrain_dataloader
from official.nlp.tasks import progressive_masked_lm


def all_strategy_combinations():
  return combinations.combine(
      distribution=[
          strategy_combinations.default_strategy,
          strategy_combinations.cloud_tpu_strategy,
          strategy_combinations.one_device_strategy_gpu,
      ],
      mode="eager",
  )


class ProgressiveMaskedLMTest(tf.test.TestCase, parameterized.TestCase):

  def setUp(self):
    super(ProgressiveMaskedLMTest, self).setUp()
    self.task_config = progressive_masked_lm.ProgMaskedLMConfig(
        model=bert.PretrainerConfig(
            encoder=encoders.EncoderConfig(
                bert=encoders.BertEncoderConfig(vocab_size=30522,
                                                num_layers=2)),
            cls_heads=[
                bert.ClsHeadConfig(
                    inner_dim=10, num_classes=2, name="next_sentence")
            ]),
        train_data=pretrain_dataloader.BertPretrainDataConfig(
            input_path="dummy",
            max_predictions_per_seq=20,
            seq_length=128,
            global_batch_size=1),
        validation_data=pretrain_dataloader.BertPretrainDataConfig(
            input_path="dummy",
            max_predictions_per_seq=20,
            seq_length=128,
            global_batch_size=1),
        stage_list=[
            progressive_masked_lm.StackingStageConfig(
                num_layers=1, num_steps=4),
            progressive_masked_lm.StackingStageConfig(
                num_layers=2, num_steps=8),
            ],
        )
    self.exp_config = cfg.ExperimentConfig(
        task=self.task_config,
        trainer=prog_trainer_lib.ProgressiveTrainerConfig())

  @combinations.generate(all_strategy_combinations())
  def test_num_stages(self, distribution):
    with distribution.scope():
      prog_masked_lm = progressive_masked_lm.ProgressiveMaskedLM(
          self.task_config)
      self.assertEqual(prog_masked_lm.num_stages(), 2)
      self.assertEqual(prog_masked_lm.num_steps(0), 4)
      self.assertEqual(prog_masked_lm.num_steps(1), 8)

  @combinations.generate(all_strategy_combinations())
  def test_weight_copying(self, distribution):
    with distribution.scope():
      prog_masked_lm = progressive_masked_lm.ProgressiveMaskedLM(
          self.task_config)
      old_model = prog_masked_lm.get_model(stage_id=0)
      for w in old_model.trainable_weights:
        w.assign(tf.zeros_like(w) + 0.12345)
      new_model = prog_masked_lm.get_model(stage_id=1, old_model=old_model)
      for w in new_model.trainable_weights:
        self.assertAllClose(w, tf.zeros_like(w) + 0.12345)

    gin.parse_config_files_and_bindings(
        None, "encoders.build_encoder.encoder_cls = @EncoderScaffold")
    with distribution.scope():
      prog_masked_lm = progressive_masked_lm.ProgressiveMaskedLM(
          self.task_config)
      old_model = prog_masked_lm.get_model(stage_id=0)
      for w in old_model.trainable_weights:
        w.assign(tf.zeros_like(w) + 0.12345)
      new_model = prog_masked_lm.get_model(stage_id=1, old_model=old_model)
      for w in new_model.trainable_weights:
        self.assertAllClose(w, tf.zeros_like(w) + 0.12345)


if __name__ == "__main__":
  tf.test.main()
