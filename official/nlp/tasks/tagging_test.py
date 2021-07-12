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

"""Tests for official.nlp.tasks.tagging."""
import functools
import os

import numpy as np
import tensorflow as tf

from official.nlp.configs import encoders
from official.nlp.data import tagging_dataloader
from official.nlp.tasks import tagging


def _create_fake_dataset(output_path, seq_length, num_labels, num_examples):
  """Creates a fake dataset."""
  writer = tf.io.TFRecordWriter(output_path)

  def create_int_feature(values):
    f = tf.train.Feature(int64_list=tf.train.Int64List(value=list(values)))
    return f

  for i in range(num_examples):
    features = {}
    input_ids = np.random.randint(100, size=(seq_length))
    features["input_ids"] = create_int_feature(input_ids)
    features["input_mask"] = create_int_feature(np.ones_like(input_ids))
    features["segment_ids"] = create_int_feature(np.ones_like(input_ids))
    features["label_ids"] = create_int_feature(
        np.random.random_integers(-1, num_labels - 1, size=(seq_length)))
    features["sentence_id"] = create_int_feature([i])
    features["sub_sentence_id"] = create_int_feature([0])

    tf_example = tf.train.Example(features=tf.train.Features(feature=features))
    writer.write(tf_example.SerializeToString())
  writer.close()


class TaggingTest(tf.test.TestCase):

  def setUp(self):
    super(TaggingTest, self).setUp()
    self._encoder_config = encoders.EncoderConfig(
        bert=encoders.BertEncoderConfig(vocab_size=30522, num_layers=1))
    self._train_data_config = tagging_dataloader.TaggingDataConfig(
        input_path="dummy", seq_length=128, global_batch_size=1)

  def _run_task(self, config):
    task = tagging.TaggingTask(config)
    model = task.build_model()
    metrics = task.build_metrics()

    strategy = tf.distribute.get_strategy()
    dataset = strategy.distribute_datasets_from_function(
        functools.partial(task.build_inputs, config.train_data))

    iterator = iter(dataset)
    optimizer = tf.keras.optimizers.SGD(lr=0.1)
    task.train_step(next(iterator), model, optimizer, metrics=metrics)
    task.validation_step(next(iterator), model, metrics=metrics)
    model.save(os.path.join(self.get_temp_dir(), "saved_model"))

  def test_task(self):
    # Saves a checkpoint.
    encoder = encoders.build_encoder(self._encoder_config)
    ckpt = tf.train.Checkpoint(encoder=encoder)
    saved_path = ckpt.save(self.get_temp_dir())

    config = tagging.TaggingConfig(
        init_checkpoint=saved_path,
        model=tagging.ModelConfig(encoder=self._encoder_config),
        train_data=self._train_data_config,
        class_names=["O", "B-PER", "I-PER"])
    task = tagging.TaggingTask(config)
    model = task.build_model()
    metrics = task.build_metrics()
    dataset = task.build_inputs(config.train_data)

    iterator = iter(dataset)
    optimizer = tf.keras.optimizers.SGD(lr=0.1)
    task.train_step(next(iterator), model, optimizer, metrics=metrics)
    task.validation_step(next(iterator), model, metrics=metrics)
    task.initialize(model)

  def _export_bert_tfhub(self):
    encoder = encoders.build_encoder(
        encoders.EncoderConfig(
            bert=encoders.BertEncoderConfig(vocab_size=30522, num_layers=1)))
    encoder_inputs_dict = {x.name: x for x in encoder.inputs}
    encoder_output_dict = encoder(encoder_inputs_dict)
    core_model = tf.keras.Model(
        inputs=encoder_inputs_dict, outputs=encoder_output_dict)
    hub_destination = os.path.join(self.get_temp_dir(), "hub")
    core_model.save(hub_destination, include_optimizer=False, save_format="tf")
    return hub_destination

  def test_task_with_hub(self):
    hub_module_url = self._export_bert_tfhub()
    config = tagging.TaggingConfig(
        hub_module_url=hub_module_url,
        class_names=["O", "B-PER", "I-PER"],
        train_data=self._train_data_config)
    self._run_task(config)

  def test_seqeval_metrics(self):
    config = tagging.TaggingConfig(
        model=tagging.ModelConfig(encoder=self._encoder_config),
        train_data=self._train_data_config,
        class_names=["O", "B-PER", "I-PER"])
    task = tagging.TaggingTask(config)
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
    self.assertCountEqual({"f1", "precision", "recall", "accuracy"},
                          task.reduce_aggregated_logs(aggregated).keys())

  def test_predict(self):
    task_config = tagging.TaggingConfig(
        model=tagging.ModelConfig(encoder=self._encoder_config),
        train_data=self._train_data_config,
        class_names=["O", "B-PER", "I-PER"])
    task = tagging.TaggingTask(task_config)
    model = task.build_model()

    test_data_path = os.path.join(self.get_temp_dir(), "test.tf_record")
    seq_length = 16
    num_examples = 100
    _create_fake_dataset(
        test_data_path,
        seq_length=seq_length,
        num_labels=len(task_config.class_names),
        num_examples=num_examples)
    test_data_config = tagging_dataloader.TaggingDataConfig(
        input_path=test_data_path,
        seq_length=seq_length,
        is_training=False,
        global_batch_size=16,
        drop_remainder=False,
        include_sentence_id=True)

    results = tagging.predict(task, test_data_config, model)
    self.assertLen(results, num_examples)
    self.assertLen(results[0], 3)


if __name__ == "__main__":
  tf.test.main()
