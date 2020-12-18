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
"""Tests for google.nlp.progressive_translation."""

import os
from absl.testing import parameterized
import tensorflow as tf

from sentencepiece import SentencePieceTrainer
from tensorflow.python.distribute import combinations
from tensorflow.python.distribute import strategy_combinations
from official.core import config_definitions as cfg
from official.modeling.progressive import trainer as prog_trainer_lib
from official.nlp.data import wmt_dataloader
from official.nlp.tasks import progressive_translation
from official.nlp.tasks import translation


def all_strategy_combinations():
  return combinations.combine(
      distribution=[
          strategy_combinations.default_strategy,
          strategy_combinations.cloud_tpu_strategy,
          strategy_combinations.one_device_strategy_gpu,
      ],
      mode="eager",
  )


def _generate_line_file(filepath, lines):
  with tf.io.gfile.GFile(filepath, "w") as f:
    for l in lines:
      f.write("{}\n".format(l))


def _generate_record_file(filepath, src_lines, tgt_lines):
  writer = tf.io.TFRecordWriter(filepath)
  for src, tgt in zip(src_lines, tgt_lines):
    example = tf.train.Example(
        features=tf.train.Features(
            feature={
                "en": tf.train.Feature(
                    bytes_list=tf.train.BytesList(
                        value=[src.encode()])),
                "reverse_en": tf.train.Feature(
                    bytes_list=tf.train.BytesList(
                        value=[tgt.encode()])),
            }))
    writer.write(example.SerializeToString())
  writer.close()


def _train_sentencepiece(input_path, vocab_size, model_path, eos_id=1):
  argstr = " ".join([
      f"--input={input_path}", f"--vocab_size={vocab_size}",
      "--character_coverage=0.995",
      f"--model_prefix={model_path}", "--model_type=bpe",
      "--bos_id=-1", "--pad_id=0", f"--eos_id={eos_id}", "--unk_id=2"
  ])
  SentencePieceTrainer.Train(argstr)


class ProgressiveTranslationTest(tf.test.TestCase, parameterized.TestCase):

  def setUp(self):
    super(ProgressiveTranslationTest, self).setUp()
    self._temp_dir = self.get_temp_dir()
    src_lines = ["abc ede fg", "bbcd ef a g", "de f a a g"]
    tgt_lines = ["dd cc a ef  g", "bcd ef a g", "gef cd ba"]
    self._record_input_path = os.path.join(self._temp_dir, "train.record")
    _generate_record_file(self._record_input_path, src_lines, tgt_lines)
    self._sentencepeice_input_path = os.path.join(self._temp_dir, "inputs.txt")
    _generate_line_file(self._sentencepeice_input_path, src_lines + tgt_lines)
    sentencepeice_model_prefix = os.path.join(self._temp_dir, "sp")
    _train_sentencepiece(self._sentencepeice_input_path, 11,
                         sentencepeice_model_prefix)
    self._sentencepeice_model_path = "{}.model".format(
        sentencepeice_model_prefix)
    encdecoder = translation.EncDecoder(
        num_attention_heads=2, intermediate_size=8)
    self.task_config = progressive_translation.ProgTranslationConfig(
        model=translation.ModelConfig(
            encoder=encdecoder,
            decoder=encdecoder,
            embedding_width=8,
            padded_decode=True,
            decode_max_length=100),
        train_data=wmt_dataloader.WMTDataConfig(
            input_path=self._record_input_path,
            is_training=True,
            global_batch_size=24,
            static_batch=True,
            src_lang="en",
            tgt_lang="reverse_en",
            max_seq_length=12),
        validation_data=wmt_dataloader.WMTDataConfig(
            input_path=self._record_input_path,
            is_training=False,
            global_batch_size=2,
            static_batch=True,
            src_lang="en",
            tgt_lang="reverse_en",
            max_seq_length=12),
        sentencepiece_model_path=self._sentencepeice_model_path,
        stage_list=[
            progressive_translation.StackingStageConfig(
                num_encoder_layers=1, num_decoder_layers=1, num_steps=4),
            progressive_translation.StackingStageConfig(
                num_encoder_layers=2, num_decoder_layers=1, num_steps=8),
            ],
        )
    self.exp_config = cfg.ExperimentConfig(
        task=self.task_config,
        trainer=prog_trainer_lib.ProgressiveTrainerConfig())

  @combinations.generate(all_strategy_combinations())
  def test_num_stages(self, distribution):
    with distribution.scope():
      prog_translation = progressive_translation.ProgressiveTranslationTask(
          self.task_config)
      self.assertEqual(prog_translation.num_stages(), 2)
      self.assertEqual(prog_translation.num_steps(0), 4)
      self.assertEqual(prog_translation.num_steps(1), 8)

  @combinations.generate(all_strategy_combinations())
  def test_weight_copying(self, distribution):
    with distribution.scope():
      prog_translation = progressive_translation.ProgressiveTranslationTask(
          self.task_config)
      old_model = prog_translation.get_model(stage_id=0)
      for w in old_model.trainable_weights:
        w.assign(tf.zeros_like(w) + 0.12345)
      new_model = prog_translation.get_model(stage_id=1, old_model=old_model)
      for w in new_model.trainable_weights:
        self.assertAllClose(w, tf.zeros_like(w) + 0.12345)


if __name__ == "__main__":
  tf.test.main()
