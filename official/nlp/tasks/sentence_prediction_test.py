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
"""Tests for official.nlp.tasks.sentence_prediction."""
import functools
import os
import tensorflow as tf

from official.nlp.bert import configs
from official.nlp.bert import export_tfhub
from official.nlp.configs import bert
from official.nlp.configs import encoders
from official.nlp.tasks import sentence_prediction


class SentencePredictionTaskTest(tf.test.TestCase):

  def _run_task(self, config):
    task = sentence_prediction.SentencePredictionTask(config)
    model = task.build_model()
    metrics = task.build_metrics()

    strategy = tf.distribute.get_strategy()
    dataset = strategy.experimental_distribute_datasets_from_function(
        functools.partial(task.build_inputs, config.train_data))

    iterator = iter(dataset)
    optimizer = tf.keras.optimizers.SGD(lr=0.1)
    task.train_step(next(iterator), model, optimizer, metrics=metrics)
    task.validation_step(next(iterator), model, metrics=metrics)

  def test_task(self):
    config = sentence_prediction.SentencePredictionConfig(
        network=bert.BertPretrainerConfig(
            encoders.TransformerEncoderConfig(vocab_size=30522, num_layers=1),
            num_masked_tokens=0,
            cls_heads=[
                bert.ClsHeadConfig(
                    inner_dim=10, num_classes=3, name="sentence_prediction")
            ]),
        train_data=bert.BertSentencePredictionDataConfig(
            input_path="dummy", seq_length=128, global_batch_size=1))
    task = sentence_prediction.SentencePredictionTask(config)
    model = task.build_model()
    metrics = task.build_metrics()
    dataset = task.build_inputs(config.train_data)

    iterator = iter(dataset)
    optimizer = tf.keras.optimizers.SGD(lr=0.1)
    task.train_step(next(iterator), model, optimizer, metrics=metrics)
    task.validation_step(next(iterator), model, metrics=metrics)

  def _export_bert_tfhub(self):
    bert_config = configs.BertConfig(
        vocab_size=30522,
        hidden_size=16,
        intermediate_size=32,
        max_position_embeddings=128,
        num_attention_heads=2,
        num_hidden_layers=1)
    _, encoder = export_tfhub.create_bert_model(bert_config)
    model_checkpoint_dir = os.path.join(self.get_temp_dir(), "checkpoint")
    checkpoint = tf.train.Checkpoint(model=encoder)
    checkpoint.save(os.path.join(model_checkpoint_dir, "test"))
    model_checkpoint_path = tf.train.latest_checkpoint(model_checkpoint_dir)

    vocab_file = os.path.join(self.get_temp_dir(), "uncased_vocab.txt")
    with tf.io.gfile.GFile(vocab_file, "w") as f:
      f.write("dummy content")

    hub_destination = os.path.join(self.get_temp_dir(), "hub")
    export_tfhub.export_bert_tfhub(bert_config, model_checkpoint_path,
                                   hub_destination, vocab_file)
    return hub_destination

  def test_task_with_hub(self):
    hub_module_url = self._export_bert_tfhub()
    config = sentence_prediction.SentencePredictionConfig(
        hub_module_url=hub_module_url,
        network=bert.BertPretrainerConfig(
            encoders.TransformerEncoderConfig(vocab_size=30522, num_layers=1),
            num_masked_tokens=0,
            cls_heads=[
                bert.ClsHeadConfig(
                    inner_dim=10, num_classes=3, name="sentence_prediction")
            ]),
        train_data=bert.BertSentencePredictionDataConfig(
            input_path="dummy", seq_length=128, global_batch_size=10))
    self._run_task(config)


if __name__ == "__main__":
  tf.test.main()
