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
"""Tests for official.nlp.tasks.tagging."""
import functools
import os
import tensorflow as tf

from official.nlp.bert import configs
from official.nlp.bert import export_tfhub
from official.nlp.configs import bert
from official.nlp.configs import encoders
from official.nlp.tasks import tagging


class TaggingTest(tf.test.TestCase):

  def setUp(self):
    super(TaggingTest, self).setUp()
    self._encoder_config = encoders.TransformerEncoderConfig(
        vocab_size=30522, num_layers=1)
    self._train_data_config = bert.TaggingDataConfig(
        input_path="dummy", seq_length=128, global_batch_size=1)

  def _run_task(self, config):
    task = tagging.TaggingTask(config)
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
    # Saves a checkpoint.
    encoder = encoders.instantiate_encoder_from_cfg(self._encoder_config)
    ckpt = tf.train.Checkpoint(encoder=encoder)
    saved_path = ckpt.save(self.get_temp_dir())

    config = tagging.TaggingConfig(
        init_checkpoint=saved_path,
        model=self._encoder_config,
        train_data=self._train_data_config,
        num_classes=3)
    task = tagging.TaggingTask(config)
    model = task.build_model()
    metrics = task.build_metrics()
    dataset = task.build_inputs(config.train_data)

    iterator = iter(dataset)
    optimizer = tf.keras.optimizers.SGD(lr=0.1)
    task.train_step(next(iterator), model, optimizer, metrics=metrics)
    task.validation_step(next(iterator), model, metrics=metrics)
    task.initialize(model)

  def test_task_with_fit(self):
    config = tagging.TaggingConfig(
        model=self._encoder_config,
        train_data=self._train_data_config,
        num_classes=3)

    task = tagging.TaggingTask(config)
    model = task.build_model()
    model = task.compile_model(
        model,
        optimizer=tf.keras.optimizers.SGD(lr=0.1),
        train_step=task.train_step,
        metrics=[tf.keras.metrics.SparseCategoricalAccuracy(name="accuracy")])
    dataset = task.build_inputs(config.train_data)
    logs = model.fit(dataset, epochs=1, steps_per_epoch=2)
    self.assertIn("loss", logs.history)
    self.assertIn("accuracy", logs.history)

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
    config = tagging.TaggingConfig(
        hub_module_url=hub_module_url,
        model=self._encoder_config,
        num_classes=4,
        train_data=self._train_data_config)
    self._run_task(config)


if __name__ == "__main__":
  tf.test.main()
