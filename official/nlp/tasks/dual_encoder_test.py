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

"""Tests for official.nlp.tasks.sentence_prediction."""
import functools
import os

from absl.testing import parameterized
import tensorflow as tf

from official.legacy.bert import configs
from official.nlp.configs import bert
from official.nlp.configs import encoders
from official.nlp.data import dual_encoder_dataloader
from official.nlp.tasks import dual_encoder
from official.nlp.tasks import masked_lm
from official.nlp.tools import export_tfhub_lib


class DualEncoderTaskTest(tf.test.TestCase, parameterized.TestCase):

  def setUp(self):
    super(DualEncoderTaskTest, self).setUp()
    self._train_data_config = (
        dual_encoder_dataloader.DualEncoderDataConfig(
            input_path="dummy", seq_length=32))

  def get_model_config(self):
    return dual_encoder.ModelConfig(
        max_sequence_length=32,
        encoder=encoders.EncoderConfig(
            bert=encoders.BertEncoderConfig(vocab_size=30522, num_layers=1)))

  def _run_task(self, config):
    task = dual_encoder.DualEncoderTask(config)
    model = task.build_model()
    metrics = task.build_metrics()

    strategy = tf.distribute.get_strategy()
    dataset = strategy.distribute_datasets_from_function(
        functools.partial(task.build_inputs, config.train_data))

    dataset.batch(10)
    iterator = iter(dataset)
    optimizer = tf.keras.optimizers.SGD(lr=0.1)
    task.train_step(next(iterator), model, optimizer, metrics=metrics)
    task.validation_step(next(iterator), model, metrics=metrics)
    model.save(os.path.join(self.get_temp_dir(), "saved_model"))

  def test_task(self):
    config = dual_encoder.DualEncoderConfig(
        init_checkpoint=self.get_temp_dir(),
        model=self.get_model_config(),
        train_data=self._train_data_config)
    task = dual_encoder.DualEncoderTask(config)
    model = task.build_model()
    metrics = task.build_metrics()
    dataset = task.build_inputs(config.train_data)

    iterator = iter(dataset)
    optimizer = tf.keras.optimizers.SGD(lr=0.1)
    task.train_step(next(iterator), model, optimizer, metrics=metrics)
    task.validation_step(next(iterator), model, metrics=metrics)

    # Saves a checkpoint.
    pretrain_cfg = bert.PretrainerConfig(
        encoder=encoders.EncoderConfig(
            bert=encoders.BertEncoderConfig(vocab_size=30522, num_layers=1)))
    pretrain_model = masked_lm.MaskedLMTask(None).build_model(pretrain_cfg)
    ckpt = tf.train.Checkpoint(
        model=pretrain_model, **pretrain_model.checkpoint_items)
    ckpt.save(config.init_checkpoint)
    task.initialize(model)

  def _export_bert_tfhub(self):
    bert_config = configs.BertConfig(
        vocab_size=30522,
        hidden_size=16,
        intermediate_size=32,
        max_position_embeddings=128,
        num_attention_heads=2,
        num_hidden_layers=4)
    encoder = export_tfhub_lib.get_bert_encoder(bert_config)
    model_checkpoint_dir = os.path.join(self.get_temp_dir(), "checkpoint")

    checkpoint = tf.train.Checkpoint(encoder=encoder)
    checkpoint.save(os.path.join(model_checkpoint_dir, "test"))
    model_checkpoint_path = tf.train.latest_checkpoint(model_checkpoint_dir)

    vocab_file = os.path.join(self.get_temp_dir(), "uncased_vocab.txt")
    with tf.io.gfile.GFile(vocab_file, "w") as f:
      f.write("dummy content")

    export_path = os.path.join(self.get_temp_dir(), "hub")
    export_tfhub_lib.export_model(
        export_path,
        bert_config=bert_config,
        encoder_config=None,
        model_checkpoint_path=model_checkpoint_path,
        vocab_file=vocab_file,
        do_lower_case=True,
        with_mlm=False)
    return export_path

  def test_task_with_hub(self):
    hub_module_url = self._export_bert_tfhub()
    config = dual_encoder.DualEncoderConfig(
        hub_module_url=hub_module_url,
        model=self.get_model_config(),
        train_data=self._train_data_config)
    self._run_task(config)


if __name__ == "__main__":
  tf.test.main()
