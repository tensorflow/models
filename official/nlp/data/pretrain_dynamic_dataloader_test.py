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

"""Tests for nlp.data.pretrain_dynamic_dataloader."""
import os

from absl import logging
from absl.testing import parameterized
import numpy as np
import orbit
import tensorflow as tf

from tensorflow.python.distribute import combinations
from tensorflow.python.distribute import strategy_combinations
from official.nlp.configs import bert
from official.nlp.configs import encoders
from official.nlp.data import pretrain_dataloader
from official.nlp.data import pretrain_dynamic_dataloader
from official.nlp.tasks import masked_lm


def _create_fake_dataset(output_path, seq_length, num_masked_tokens,
                         max_seq_length, num_examples):
  """Creates a fake dataset."""
  writer = tf.io.TFRecordWriter(output_path)

  def create_int_feature(values):
    f = tf.train.Feature(int64_list=tf.train.Int64List(value=list(values)))
    return f

  def create_float_feature(values):
    f = tf.train.Feature(float_list=tf.train.FloatList(value=list(values)))
    return f

  for _ in range(num_examples):
    features = {}
    padding = np.zeros(shape=(max_seq_length - seq_length), dtype=np.int32)
    input_ids = np.random.randint(low=1, high=100, size=(seq_length))
    features['input_ids'] = create_int_feature(
        np.concatenate((input_ids, padding)))
    features['input_mask'] = create_int_feature(
        np.concatenate((np.ones_like(input_ids), padding)))
    features['segment_ids'] = create_int_feature(
        np.concatenate((np.ones_like(input_ids), padding)))
    features['position_ids'] = create_int_feature(
        np.concatenate((np.ones_like(input_ids), padding)))
    features['masked_lm_positions'] = create_int_feature(
        np.random.randint(60, size=(num_masked_tokens), dtype=np.int64))
    features['masked_lm_ids'] = create_int_feature(
        np.random.randint(100, size=(num_masked_tokens), dtype=np.int64))
    features['masked_lm_weights'] = create_float_feature(
        np.ones((num_masked_tokens,), dtype=np.float32))
    features['next_sentence_labels'] = create_int_feature(np.array([0]))

    tf_example = tf.train.Example(features=tf.train.Features(feature=features))
    writer.write(tf_example.SerializeToString())
  writer.close()


class PretrainDynamicDataLoaderTest(tf.test.TestCase, parameterized.TestCase):

  @combinations.generate(
      combinations.combine(
          distribution_strategy=[
              strategy_combinations.cloud_tpu_strategy,
          ],
          mode='eager'))
  def test_distribution_strategy(self, distribution_strategy):
    max_seq_length = 128
    batch_size = 8
    input_path = os.path.join(self.get_temp_dir(), 'train.tf_record')
    _create_fake_dataset(
        input_path,
        seq_length=60,
        num_masked_tokens=20,
        max_seq_length=max_seq_length,
        num_examples=batch_size)
    data_config = pretrain_dynamic_dataloader.BertPretrainDataConfig(
        is_training=False,
        input_path=input_path,
        seq_bucket_lengths=[64, 128],
        global_batch_size=batch_size)
    dataloader = pretrain_dynamic_dataloader.PretrainingDynamicDataLoader(
        data_config)
    distributed_ds = orbit.utils.make_distributed_dataset(
        distribution_strategy, dataloader.load)
    train_iter = iter(distributed_ds)
    with distribution_strategy.scope():
      config = masked_lm.MaskedLMConfig(
          init_checkpoint=self.get_temp_dir(),
          model=bert.PretrainerConfig(
              encoders.EncoderConfig(
                  bert=encoders.BertEncoderConfig(
                      vocab_size=30522, num_layers=1)),
              cls_heads=[
                  bert.ClsHeadConfig(
                      inner_dim=10, num_classes=2, name='next_sentence')
              ]),
          train_data=data_config)
      task = masked_lm.MaskedLMTask(config)
      model = task.build_model()
      metrics = task.build_metrics()

    @tf.function
    def step_fn(features):
      return task.validation_step(features, model, metrics=metrics)

    distributed_outputs = distribution_strategy.run(
        step_fn, args=(next(train_iter),))
    local_results = tf.nest.map_structure(
        distribution_strategy.experimental_local_results, distributed_outputs)
    logging.info('Dynamic padding:  local_results= %s', str(local_results))
    dynamic_metrics = {}
    for metric in metrics:
      dynamic_metrics[metric.name] = metric.result()

    data_config = pretrain_dataloader.BertPretrainDataConfig(
        is_training=False,
        input_path=input_path,
        seq_length=max_seq_length,
        max_predictions_per_seq=20,
        global_batch_size=batch_size)
    dataloader = pretrain_dataloader.BertPretrainDataLoader(data_config)
    distributed_ds = orbit.utils.make_distributed_dataset(
        distribution_strategy, dataloader.load)
    train_iter = iter(distributed_ds)
    with distribution_strategy.scope():
      metrics = task.build_metrics()

    @tf.function
    def step_fn_b(features):
      return task.validation_step(features, model, metrics=metrics)

    distributed_outputs = distribution_strategy.run(
        step_fn_b, args=(next(train_iter),))
    local_results = tf.nest.map_structure(
        distribution_strategy.experimental_local_results, distributed_outputs)
    logging.info('Static padding:  local_results= %s', str(local_results))
    static_metrics = {}
    for metric in metrics:
      static_metrics[metric.name] = metric.result()
    for key in static_metrics:
      # We need to investigate the differences on losses.
      if key != 'next_sentence_loss':
        self.assertEqual(dynamic_metrics[key], static_metrics[key])

  def test_load_dataset(self):
    max_seq_length = 128
    batch_size = 2
    input_path_1 = os.path.join(self.get_temp_dir(), 'train_1.tf_record')
    _create_fake_dataset(
        input_path_1,
        seq_length=60,
        num_masked_tokens=20,
        max_seq_length=max_seq_length,
        num_examples=batch_size)
    input_path_2 = os.path.join(self.get_temp_dir(), 'train_2.tf_record')
    _create_fake_dataset(
        input_path_2,
        seq_length=100,
        num_masked_tokens=70,
        max_seq_length=max_seq_length,
        num_examples=batch_size)
    input_paths = ','.join([input_path_1, input_path_2])
    data_config = pretrain_dynamic_dataloader.BertPretrainDataConfig(
        is_training=False,
        input_path=input_paths,
        seq_bucket_lengths=[64, 128],
        use_position_id=True,
        global_batch_size=batch_size)
    dataset = pretrain_dynamic_dataloader.PretrainingDynamicDataLoader(
        data_config).load()
    dataset_it = iter(dataset)
    features = next(dataset_it)
    self.assertCountEqual([
        'input_word_ids',
        'input_mask',
        'input_type_ids',
        'next_sentence_labels',
        'masked_lm_positions',
        'masked_lm_ids',
        'masked_lm_weights',
        'position_ids',
    ], features.keys())
    # Sequence length dimension should be bucketized and pad to 64.
    self.assertEqual(features['input_word_ids'].shape, (batch_size, 64))
    self.assertEqual(features['input_mask'].shape, (batch_size, 64))
    self.assertEqual(features['input_type_ids'].shape, (batch_size, 64))
    self.assertEqual(features['position_ids'].shape, (batch_size, 64))
    self.assertEqual(features['masked_lm_positions'].shape, (batch_size, 20))
    features = next(dataset_it)
    self.assertEqual(features['input_word_ids'].shape, (batch_size, 128))
    self.assertEqual(features['input_mask'].shape, (batch_size, 128))
    self.assertEqual(features['input_type_ids'].shape, (batch_size, 128))
    self.assertEqual(features['position_ids'].shape, (batch_size, 128))
    self.assertEqual(features['masked_lm_positions'].shape, (batch_size, 70))

  def test_load_dataset_not_same_masks(self):
    max_seq_length = 128
    batch_size = 2
    input_path_1 = os.path.join(self.get_temp_dir(), 'train_3.tf_record')
    _create_fake_dataset(
        input_path_1,
        seq_length=60,
        num_masked_tokens=20,
        max_seq_length=max_seq_length,
        num_examples=batch_size)
    input_path_2 = os.path.join(self.get_temp_dir(), 'train_4.tf_record')
    _create_fake_dataset(
        input_path_2,
        seq_length=60,
        num_masked_tokens=15,
        max_seq_length=max_seq_length,
        num_examples=batch_size)
    input_paths = ','.join([input_path_1, input_path_2])
    data_config = pretrain_dynamic_dataloader.BertPretrainDataConfig(
        is_training=False,
        input_path=input_paths,
        seq_bucket_lengths=[64, 128],
        use_position_id=True,
        global_batch_size=batch_size * 2)
    dataset = pretrain_dynamic_dataloader.PretrainingDynamicDataLoader(
        data_config).load()
    dataset_it = iter(dataset)
    with self.assertRaisesRegex(
        tf.errors.InvalidArgumentError, '.*Number of non padded mask tokens.*'):
      next(dataset_it)


if __name__ == '__main__':
  tf.test.main()
