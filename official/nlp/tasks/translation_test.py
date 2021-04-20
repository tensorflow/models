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

"""Tests for official.nlp.tasks.translation."""
import functools
import os

import orbit
import tensorflow as tf

from sentencepiece import SentencePieceTrainer
from official.nlp.data import wmt_dataloader
from official.nlp.tasks import translation


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


class TranslationTaskTest(tf.test.TestCase):

  def setUp(self):
    super(TranslationTaskTest, self).setUp()
    self._temp_dir = self.get_temp_dir()
    src_lines = [
        "abc ede fg",
        "bbcd ef a g",
        "de f a a g"
    ]
    tgt_lines = [
        "dd cc a ef  g",
        "bcd ef a g",
        "gef cd ba"
    ]
    self._record_input_path = os.path.join(self._temp_dir, "inputs.record")
    _generate_record_file(self._record_input_path, src_lines, tgt_lines)
    self._sentencepeice_input_path = os.path.join(self._temp_dir, "inputs.txt")
    _generate_line_file(self._sentencepeice_input_path, src_lines + tgt_lines)
    sentencepeice_model_prefix = os.path.join(self._temp_dir, "sp")
    _train_sentencepiece(self._sentencepeice_input_path, 11,
                         sentencepeice_model_prefix)
    self._sentencepeice_model_path = "{}.model".format(
        sentencepeice_model_prefix)

  def test_task(self):
    config = translation.TranslationConfig(
        model=translation.ModelConfig(
            encoder=translation.EncDecoder(), decoder=translation.EncDecoder()),
        train_data=wmt_dataloader.WMTDataConfig(
            input_path=self._record_input_path,
            src_lang="en", tgt_lang="reverse_en",
            is_training=True, static_batch=True, global_batch_size=24,
            max_seq_length=12),
        sentencepiece_model_path=self._sentencepeice_model_path)
    task = translation.TranslationTask(config)
    model = task.build_model()
    dataset = task.build_inputs(config.train_data)
    iterator = iter(dataset)
    optimizer = tf.keras.optimizers.SGD(lr=0.1)
    task.train_step(next(iterator), model, optimizer)

  def test_no_sentencepiece_path(self):
    config = translation.TranslationConfig(
        model=translation.ModelConfig(
            encoder=translation.EncDecoder(), decoder=translation.EncDecoder()),
        train_data=wmt_dataloader.WMTDataConfig(
            input_path=self._record_input_path,
            src_lang="en", tgt_lang="reverse_en",
            is_training=True, static_batch=True, global_batch_size=4,
            max_seq_length=4),
        sentencepiece_model_path=None)
    with self.assertRaisesRegex(
        ValueError,
        "Setencepiece model path not provided."):
      translation.TranslationTask(config)

  def test_sentencepiece_no_eos(self):
    sentencepeice_model_prefix = os.path.join(self._temp_dir, "sp_no_eos")
    _train_sentencepiece(self._sentencepeice_input_path, 20,
                         sentencepeice_model_prefix, eos_id=-1)
    sentencepeice_model_path = "{}.model".format(
        sentencepeice_model_prefix)
    config = translation.TranslationConfig(
        model=translation.ModelConfig(
            encoder=translation.EncDecoder(), decoder=translation.EncDecoder()),
        train_data=wmt_dataloader.WMTDataConfig(
            input_path=self._record_input_path,
            src_lang="en", tgt_lang="reverse_en",
            is_training=True, static_batch=True, global_batch_size=4,
            max_seq_length=4),
        sentencepiece_model_path=sentencepeice_model_path)
    with self.assertRaisesRegex(
        ValueError,
        "EOS token not in tokenizer vocab.*"):
      translation.TranslationTask(config)

  def test_evaluation(self):
    config = translation.TranslationConfig(
        model=translation.ModelConfig(
            encoder=translation.EncDecoder(), decoder=translation.EncDecoder(),
            padded_decode=False,
            decode_max_length=64),
        validation_data=wmt_dataloader.WMTDataConfig(
            input_path=self._record_input_path, src_lang="en",
            tgt_lang="reverse_en", static_batch=True, global_batch_size=4),
        sentencepiece_model_path=self._sentencepeice_model_path)
    logging_dir = self.get_temp_dir()
    task = translation.TranslationTask(config, logging_dir=logging_dir)
    dataset = orbit.utils.make_distributed_dataset(tf.distribute.get_strategy(),
                                                   task.build_inputs,
                                                   config.validation_data)
    model = task.build_model()
    strategy = tf.distribute.get_strategy()
    aggregated = None
    for data in dataset:
      distributed_outputs = strategy.run(
          functools.partial(task.validation_step, model=model),
          args=(data,))
      outputs = tf.nest.map_structure(strategy.experimental_local_results,
                                      distributed_outputs)
      aggregated = task.aggregate_logs(state=aggregated, step_outputs=outputs)
    metrics = task.reduce_aggregated_logs(aggregated)
    self.assertIn("sacrebleu_score", metrics)
    self.assertIn("bleu_score", metrics)

if __name__ == "__main__":
  tf.test.main()
