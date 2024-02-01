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

"""Tests for nlp.projects.example.classification_example."""

import tensorflow as tf, tf_keras

from official.core import config_definitions as cfg
from official.nlp.configs import encoders
from official.projects.text_classification_example import classification_data_loader
from official.projects.text_classification_example import classification_example


class ClassificationExampleTest(tf.test.TestCase):

  def get_model_config(self):
    return classification_example.ModelConfig(
        encoder=encoders.EncoderConfig(
            bert=encoders.BertEncoderConfig(vocab_size=30522, num_layers=2)))

  def get_dummy_dataset(self, params: cfg.DataConfig):

    def dummy_data(_):
      dummy_ids = tf.zeros((1, params.seq_length), dtype=tf.int32)
      x = dict(
          input_word_ids=dummy_ids,
          input_mask=dummy_ids,
          input_type_ids=dummy_ids)

      y = tf.zeros((1, 1), dtype=tf.int32)
      return (x, y)

    dataset = tf.data.Dataset.range(1)
    dataset = dataset.repeat()
    dataset = dataset.map(
        dummy_data, num_parallel_calls=tf.data.experimental.AUTOTUNE)
    return dataset

  def test_task_with_dummy_data(self):
    train_data_config = (
        classification_data_loader.ClassificationExampleDataConfig(
            input_path='dummy', seq_length=128, global_batch_size=1))
    task_config = classification_example.ClassificationExampleConfig(
        model=self.get_model_config(),)
    task = classification_example.ClassificationExampleTask(task_config)
    task.build_inputs = self.get_dummy_dataset
    model = task.build_model()
    metrics = task.build_metrics()
    dataset = task.build_inputs(train_data_config)

    iterator = iter(dataset)
    optimizer = tf_keras.optimizers.SGD(lr=0.1)
    task.initialize(model)
    task.train_step(next(iterator), model, optimizer, metrics=metrics)


if __name__ == '__main__':
  tf.test.main()
