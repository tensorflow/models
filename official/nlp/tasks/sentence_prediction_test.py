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

from absl.testing import parameterized
import numpy as np
import tensorflow as tf

from official.nlp.bert import configs
from official.nlp.bert import export_tfhub
from official.nlp.configs import bert
from official.nlp.configs import encoders
from official.nlp.data import sentence_prediction_dataloader
from official.nlp.tasks import sentence_prediction


def _create_fake_dataset(output_path, seq_length, num_classes, num_examples):
  """Creates a fake dataset."""
  writer = tf.io.TFRecordWriter(output_path)

  def create_int_feature(values):
    return tf.train.Feature(int64_list=tf.train.Int64List(value=list(values)))

  def create_float_feature(values):
    return tf.train.Feature(float_list=tf.train.FloatList(value=list(values)))

  for _ in range(num_examples):
    features = {}
    input_ids = np.random.randint(100, size=(seq_length))
    features["input_ids"] = create_int_feature(input_ids)
    features["input_mask"] = create_int_feature(np.ones_like(input_ids))
    features["segment_ids"] = create_int_feature(np.ones_like(input_ids))
    features["segment_ids"] = create_int_feature(np.ones_like(input_ids))

    if num_classes == 1:
      features["label_ids"] = create_float_feature([np.random.random()])
    else:
      features["label_ids"] = create_int_feature(
          [np.random.random_integers(0, num_classes - 1, size=())])

    tf_example = tf.train.Example(features=tf.train.Features(feature=features))
    writer.write(tf_example.SerializeToString())
  writer.close()


class SentencePredictionTaskTest(tf.test.TestCase, parameterized.TestCase):

  def setUp(self):
    super(SentencePredictionTaskTest, self).setUp()
    self._train_data_config = (
        sentence_prediction_dataloader.SentencePredictionDataConfig(
            input_path="dummy", seq_length=128, global_batch_size=1))

  def get_model_config(self, num_classes):
    return sentence_prediction.ModelConfig(
        encoder=encoders.TransformerEncoderConfig(
            vocab_size=30522, num_layers=1),
        num_classes=num_classes)

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
        init_checkpoint=self.get_temp_dir(),
        model=self.get_model_config(2),
        train_data=self._train_data_config)
    task = sentence_prediction.SentencePredictionTask(config)
    model = task.build_model()
    metrics = task.build_metrics()
    dataset = task.build_inputs(config.train_data)

    iterator = iter(dataset)
    optimizer = tf.keras.optimizers.SGD(lr=0.1)
    task.train_step(next(iterator), model, optimizer, metrics=metrics)
    task.validation_step(next(iterator), model, metrics=metrics)

    # Saves a checkpoint.
    pretrain_cfg = bert.BertPretrainerConfig(
        encoder=encoders.TransformerEncoderConfig(
            vocab_size=30522, num_layers=1),
        cls_heads=[
            bert.ClsHeadConfig(
                inner_dim=10, num_classes=3, name="next_sentence")
        ])
    pretrain_model = bert.instantiate_pretrainer_from_cfg(pretrain_cfg)
    ckpt = tf.train.Checkpoint(
        model=pretrain_model, **pretrain_model.checkpoint_items)
    ckpt.save(config.init_checkpoint)
    task.initialize(model)

  @parameterized.named_parameters(
      {
          "testcase_name": "regression",
          "num_classes": 1,
      },
      {
          "testcase_name": "classification",
          "num_classes": 2,
      },
  )
  def test_metrics_and_losses(self, num_classes):
    config = sentence_prediction.SentencePredictionConfig(
        init_checkpoint=self.get_temp_dir(),
        model=self.get_model_config(num_classes),
        train_data=self._train_data_config)
    task = sentence_prediction.SentencePredictionTask(config)
    model = task.build_model()
    metrics = task.build_metrics()
    if num_classes == 1:
      self.assertIsInstance(metrics[0], tf.keras.metrics.MeanSquaredError)
    else:
      self.assertIsInstance(
          metrics[0], tf.keras.metrics.SparseCategoricalAccuracy)

    dataset = task.build_inputs(config.train_data)
    iterator = iter(dataset)
    optimizer = tf.keras.optimizers.SGD(lr=0.1)
    task.train_step(next(iterator), model, optimizer, metrics=metrics)

    logs = task.validation_step(next(iterator), model, metrics=metrics)
    loss = logs["loss"].numpy()
    if num_classes == 1:
      self.assertAlmostEqual(loss, 42.77483, places=3)
    else:
      self.assertAlmostEqual(loss, 3.57627e-6, places=3)

  @parameterized.parameters(("matthews_corrcoef", 2),
                            ("pearson_spearman_corr", 1))
  def test_np_metrics(self, metric_type, num_classes):
    config = sentence_prediction.SentencePredictionConfig(
        metric_type=metric_type,
        init_checkpoint=self.get_temp_dir(),
        model=self.get_model_config(num_classes),
        train_data=self._train_data_config)
    task = sentence_prediction.SentencePredictionTask(config)
    model = task.build_model()
    dataset = task.build_inputs(config.train_data)

    iterator = iter(dataset)
    strategy = tf.distribute.get_strategy()
    distributed_outputs = strategy.run(
        functools.partial(task.validation_step, model=model),
        args=(next(iterator),))
    outputs = tf.nest.map_structure(strategy.experimental_local_results,
                                    distributed_outputs)
    aggregated = task.aggregate_logs(step_outputs=outputs)
    aggregated = task.aggregate_logs(state=aggregated, step_outputs=outputs)
    self.assertIn(metric_type, task.reduce_aggregated_logs(aggregated))

  def test_task_with_fit(self):
    config = sentence_prediction.SentencePredictionConfig(
        model=self.get_model_config(2), train_data=self._train_data_config)
    task = sentence_prediction.SentencePredictionTask(config)
    model = task.build_model()
    model = task.compile_model(
        model,
        optimizer=tf.keras.optimizers.SGD(lr=0.1),
        train_step=task.train_step,
        metrics=task.build_metrics())
    dataset = task.build_inputs(config.train_data)
    logs = model.fit(dataset, epochs=1, steps_per_epoch=2)
    self.assertIn("loss", logs.history)

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
        model=self.get_model_config(2),
        train_data=self._train_data_config)
    self._run_task(config)

  @parameterized.named_parameters(("classification", 5), ("regression", 1))
  def test_prediction(self, num_classes):
    task_config = sentence_prediction.SentencePredictionConfig(
        model=self.get_model_config(num_classes=num_classes),
        train_data=self._train_data_config)
    task = sentence_prediction.SentencePredictionTask(task_config)
    model = task.build_model()

    test_data_path = os.path.join(self.get_temp_dir(), "test.tf_record")
    seq_length = 16
    num_examples = 100
    _create_fake_dataset(
        test_data_path,
        seq_length=seq_length,
        num_classes=num_classes,
        num_examples=num_examples)

    test_data_config = (
        sentence_prediction_dataloader.SentencePredictionDataConfig(
            input_path=test_data_path,
            seq_length=seq_length,
            is_training=False,
            label_type="int" if num_classes > 1 else "float",
            global_batch_size=16,
            drop_remainder=False))

    predictions = sentence_prediction.predict(task, test_data_config, model)
    self.assertLen(predictions, num_examples)


if __name__ == "__main__":
  tf.test.main()
