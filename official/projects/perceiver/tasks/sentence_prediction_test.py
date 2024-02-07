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

"""Tests for official.nlp.tasks.sentence_prediction."""
import functools
import os

from absl.testing import parameterized
import numpy as np
import tensorflow as tf, tf_keras

from official.nlp.data import sentence_prediction_dataloader
from official.nlp.tasks import sentence_prediction
from official.projects.perceiver.configs import perceiver
from official.projects.perceiver.tasks import pretrain
from official.projects.perceiver.tasks import sentence_prediction as perceiver_pred


def _create_fake_dataset(output_path, seq_length, num_classes, num_examples):
  """Creates a fake dataset.

  Args:
    output_path:
      output path for the writer to serialize the dataset.
    seq_length:
      sequence length of the data.
    num_classes:
      Number of classes in the sentence prediction output. This is used to
      determine if the label id feature should be for regression or
      classification.
    num_examples:
      number of fake examples to create.
  """

  with tf.io.TFRecordWriter(output_path) as writer:
    def create_int_feature(values):
      return tf.train.Feature(
          int64_list=tf.train.Int64List(value=np.ravel(values)))

    def create_float_feature(values):
      return tf.train.Feature(
          float_list=tf.train.FloatList(value=np.ravel(values)))

    for i in range(num_examples):
      features = {}
      input_ids = np.random.randint(100, size=(seq_length))
      features["input_ids"] = create_int_feature(input_ids)
      features["input_mask"] = create_int_feature(np.ones_like(input_ids))
      features["segment_ids"] = create_int_feature(np.ones_like(input_ids))
      features["segment_ids"] = create_int_feature(np.ones_like(input_ids))
      features["example_id"] = create_int_feature([i])

      if num_classes == 1:
        features["label_ids"] = create_float_feature([np.random.random()])
      else:
        features["label_ids"] = create_int_feature(
            [np.random.random_integers(0, num_classes - 1, size=())])

      tf_example = tf.train.Example(
          features=tf.train.Features(feature=features))
      writer.write(tf_example.SerializeToString())


class SentencePredictionTaskTest(tf.test.TestCase, parameterized.TestCase):

  def setUp(self):
    super().setUp()
    self._train_data_config = (
        sentence_prediction_dataloader.SentencePredictionDataConfig(
            input_path="dummy", seq_length=128, global_batch_size=1))

  def get_model_config(self, num_classes):
    return perceiver.ClassificationConfig(
        num_classes=num_classes,
        encoder=perceiver.SequenceEncoderConfig(
            vocab_size=30_522,
            encoder=perceiver.EncoderConfig(
                num_self_attends_per_block=2)))

  def _run_task(self, config):
    task = perceiver_pred.SentencePredictionTask(config)
    model = task.build_model()
    metrics = task.build_metrics()

    strategy = tf.distribute.get_strategy()
    dataset = strategy.distribute_datasets_from_function(
        functools.partial(task.build_inputs, config.train_data))

    iterator = iter(dataset)
    optimizer = tf_keras.optimizers.SGD(lr=0.1)
    task.train_step(next(iterator), model, optimizer, metrics=metrics)
    # model.save(os.path.join(self.get_temp_dir(), "saved_model"))
    # TODO(b/222634115) fix save
    return task.validation_step(next(iterator), model, metrics=metrics)

  def test_task(self):
    # Saves a checkpoint.
    pretrain_cfg = perceiver.PretrainerConfig(
        encoder=perceiver.SequenceEncoderConfig(
            vocab_size=30_522,
            encoder=perceiver.EncoderConfig(
                num_self_attends_per_block=2)))
    pretrain_model = pretrain.PretrainTask(
        None).build_model(pretrain_cfg)

    # The model variables will be created after the forward call.
    _ = pretrain_model(pretrain_model.inputs)
    ckpt = tf.train.Checkpoint(
        model=pretrain_model, **pretrain_model.checkpoint_items)
    init_path = ckpt.save(self.get_temp_dir())

    # Creates the task.
    config = perceiver.SentencePredictionConfig(
        init_checkpoint=init_path,
        model=self.get_model_config(num_classes=2),
        train_data=self._train_data_config)
    task = perceiver_pred.SentencePredictionTask(config)
    model = task.build_model()
    metrics = task.build_metrics()
    dataset = task.build_inputs(config.train_data)

    iterator = iter(dataset)
    optimizer = tf_keras.optimizers.SGD(lr=0.1)
    task.initialize(model)
    task.train_step(next(iterator), model, optimizer, metrics=metrics)
    task.validation_step(next(iterator), model, metrics=metrics)

  @parameterized.named_parameters(
      {
          "testcase_name":
              "regression",
          "num_classes":
              1,
          "expected_loss_predicate":
              lambda loss: loss > 1.0,
          "metric": tf_keras.metrics.MeanSquaredError,
      },
      {
          "testcase_name":
              "classification",
          "num_classes":
              2,
          "expected_loss_predicate":
              lambda loss: loss < 1.0,
          "metric": tf_keras.metrics.SparseCategoricalAccuracy
      },
  )
  def test_metrics_and_losses(self, num_classes, expected_loss_predicate,
                              metric):
    config = perceiver.SentencePredictionConfig(
        init_checkpoint=self.get_temp_dir(),
        model=self.get_model_config(num_classes),
        train_data=self._train_data_config)
    task = perceiver_pred.SentencePredictionTask(config)
    model = task.build_model()
    metrics = task.build_metrics()
    self.assertIsInstance(metrics[0], metric)

    dataset = task.build_inputs(config.train_data)
    iterator = iter(dataset)
    optimizer = tf_keras.optimizers.SGD(lr=0.1)
    task.train_step(next(iterator), model, optimizer, metrics=metrics)

    logs = task.validation_step(next(iterator), model, metrics=metrics)
    loss = logs["loss"].numpy()
    self.assertTrue(expected_loss_predicate(loss))

  @parameterized.named_parameters(
      {
          "testcase_name": "matthews_corrcoef",
          "num_classes": 2,
          "metric_type": "matthews_corrcoef"
      }, {
          "testcase_name": "pearson_spearman_corr",
          "num_classes": 1,
          "metric_type": "pearson_spearman_corr"
      })
  def test_np_metrics(self, metric_type, num_classes):
    config = perceiver.SentencePredictionConfig(
        metric_type=metric_type,
        init_checkpoint=self.get_temp_dir(),
        model=self.get_model_config(num_classes),
        train_data=self._train_data_config)
    task = perceiver_pred.SentencePredictionTask(config)
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

  def test_np_metrics_cola_partial_batch(self):
    train_data_path = os.path.join(self.get_temp_dir(), "train.tf_record")
    num_examples = 5
    global_batch_size = 8
    seq_length = 16
    _create_fake_dataset(
        train_data_path,
        seq_length=seq_length,
        num_classes=2,
        num_examples=num_examples)

    train_data_config = (
        sentence_prediction_dataloader.SentencePredictionDataConfig(
            input_path=train_data_path,
            seq_length=seq_length,
            is_training=True,
            label_type="int",
            global_batch_size=global_batch_size,
            drop_remainder=False,
            include_example_id=True))

    config = perceiver.SentencePredictionConfig(
        metric_type="matthews_corrcoef",
        model=self.get_model_config(2),
        train_data=train_data_config)
    outputs = self._run_task(config)
    self.assertEqual(outputs["sentence_prediction"].shape.as_list(), [8, 1])

  @parameterized.named_parameters(
      {
          "testcase_name": "classification",
          "num_classes": 5,
      }, {
          "testcase_name": "regression",
          "num_classes": 1,
      })
  def test_prediction(self, num_classes):
    config = perceiver.SentencePredictionConfig(
        model=self.get_model_config(num_classes=num_classes),
        train_data=self._train_data_config)
    task = perceiver_pred.SentencePredictionTask(config)
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
            drop_remainder=False,
            include_example_id=True))

    predictions = sentence_prediction.predict(task, test_data_config, model)
    self.assertLen(predictions, num_examples)
    for prediction in predictions:
      self.assertEqual(prediction.dtype,
                       tf.int64 if num_classes > 1 else tf.float32)


if __name__ == "__main__":
  tf.test.main()
