# Copyright 2022 The TensorFlow Authors. All Rights Reserved.
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

"""Tests for projects.nhnet.models."""

import os

from absl import logging
from absl.testing import parameterized
import numpy as np
import tensorflow as tf

# pylint: disable=g-direct-tensorflow-import
from tensorflow.python.distribute import combinations
from tensorflow.python.distribute import strategy_combinations
# pylint: enable=g-direct-tensorflow-import
from official.projects.nhnet import configs
from official.projects.nhnet import models
from official.projects.nhnet import utils


def all_strategy_combinations():
  return combinations.combine(
      distribution=[
          strategy_combinations.default_strategy,
          strategy_combinations.cloud_tpu_strategy,
          strategy_combinations.one_device_strategy_gpu,
          strategy_combinations.mirrored_strategy_with_gpu_and_cpu,
          strategy_combinations.mirrored_strategy_with_two_gpus,
      ],)


def distribution_forward_path(strategy,
                              model,
                              inputs,
                              batch_size,
                              mode="train"):
  dataset = tf.data.Dataset.from_tensor_slices((inputs))
  dataset = dataset.batch(batch_size)
  dataset = strategy.experimental_distribute_dataset(dataset)

  @tf.function
  def test_step(inputs):
    """Calculates evaluation metrics on distributed devices."""

    def _test_step_fn(inputs):
      """Replicated accuracy calculation."""
      return model(inputs, mode=mode, training=False)

    outputs = strategy.run(_test_step_fn, args=(inputs,))
    return tf.nest.map_structure(strategy.experimental_local_results, outputs)

  return [test_step(inputs) for inputs in dataset]


def process_decoded_ids(predictions, end_token_id):
  """Transforms decoded tensors to lists ending with END_TOKEN_ID."""
  if isinstance(predictions, tf.Tensor):
    predictions = predictions.numpy()
  flatten_ids = predictions.reshape((-1, predictions.shape[-1]))
  results = []
  for ids in flatten_ids:
    ids = list(ids)
    if end_token_id in ids:
      ids = ids[:ids.index(end_token_id)]
      results.append(ids)
  return results


class Bert2BertTest(tf.test.TestCase, parameterized.TestCase):

  def setUp(self):
    super(Bert2BertTest, self).setUp()
    self._config = utils.get_test_params()

  def test_model_creation(self):
    model = models.create_bert2bert_model(params=self._config)
    fake_ids = np.zeros((2, 10), dtype=np.int32)
    fake_inputs = {
        "input_ids": fake_ids,
        "input_mask": fake_ids,
        "segment_ids": fake_ids,
        "target_ids": fake_ids,
    }
    model(fake_inputs)

  @combinations.generate(all_strategy_combinations())
  def test_bert2bert_train_forward(self, distribution):
    seq_length = 10
    # Defines the model inside distribution strategy scope.
    with distribution.scope():
      # Forward path.
      batch_size = 2
      batches = 4
      fake_ids = np.zeros((batch_size * batches, seq_length), dtype=np.int32)
      fake_inputs = {
          "input_ids": fake_ids,
          "input_mask": fake_ids,
          "segment_ids": fake_ids,
          "target_ids": fake_ids,
      }
      model = models.create_bert2bert_model(params=self._config)
      results = distribution_forward_path(distribution, model, fake_inputs,
                                          batch_size)
      logging.info("Forward path results: %s", str(results))
      self.assertLen(results, batches)

  def test_bert2bert_decoding(self):
    seq_length = 10
    self._config.override(
        {
            "beam_size": 3,
            "len_title": seq_length,
            "alpha": 0.6,
        },
        is_strict=False)

    batch_size = 2
    fake_ids = np.zeros((batch_size, seq_length), dtype=np.int32)
    fake_inputs = {
        "input_ids": fake_ids,
        "input_mask": fake_ids,
        "segment_ids": fake_ids,
    }
    self._config.override({
        "padded_decode": False,
        "use_cache": False,
    },
                          is_strict=False)
    model = models.create_bert2bert_model(params=self._config)
    ckpt = tf.train.Checkpoint(model=model)

    # Initializes variables from checkpoint to keep outputs deterministic.
    init_checkpoint = ckpt.save(os.path.join(self.get_temp_dir(), "ckpt"))
    ckpt.restore(init_checkpoint).assert_existing_objects_matched()
    top_ids, scores = model(fake_inputs, mode="predict")

    self._config.override({
        "padded_decode": False,
        "use_cache": True,
    },
                          is_strict=False)
    model = models.create_bert2bert_model(params=self._config)
    ckpt = tf.train.Checkpoint(model=model)
    ckpt.restore(init_checkpoint).assert_existing_objects_matched()
    cached_top_ids, cached_scores = model(fake_inputs, mode="predict")
    self.assertEqual(
        process_decoded_ids(top_ids, self._config.end_token_id),
        process_decoded_ids(cached_top_ids, self._config.end_token_id))
    self.assertAllClose(scores, cached_scores)

    self._config.override({
        "padded_decode": True,
        "use_cache": True,
    },
                          is_strict=False)
    model = models.create_bert2bert_model(params=self._config)
    ckpt = tf.train.Checkpoint(model=model)
    ckpt.restore(init_checkpoint).assert_existing_objects_matched()
    padded_top_ids, padded_scores = model(fake_inputs, mode="predict")
    self.assertEqual(
        process_decoded_ids(top_ids, self._config.end_token_id),
        process_decoded_ids(padded_top_ids, self._config.end_token_id))
    self.assertAllClose(scores, padded_scores)

  @combinations.generate(all_strategy_combinations())
  def test_bert2bert_eval(self, distribution):
    seq_length = 10
    padded_decode = isinstance(
        distribution,
        (tf.distribute.TPUStrategy, tf.distribute.experimental.TPUStrategy))
    self._config.override(
        {
            "beam_size": 3,
            "len_title": seq_length,
            "alpha": 0.6,
            "padded_decode": padded_decode,
        },
        is_strict=False)
    # Defines the model inside distribution strategy scope.
    with distribution.scope():
      # Forward path.
      batch_size = 2
      batches = 4
      fake_ids = np.zeros((batch_size * batches, seq_length), dtype=np.int32)
      fake_inputs = {
          "input_ids": fake_ids,
          "input_mask": fake_ids,
          "segment_ids": fake_ids,
      }
      model = models.create_bert2bert_model(params=self._config)
      results = distribution_forward_path(
          distribution, model, fake_inputs, batch_size, mode="predict")
      self.assertLen(results, batches)
      results = distribution_forward_path(
          distribution, model, fake_inputs, batch_size, mode="eval")
      self.assertLen(results, batches)


class NHNetTest(tf.test.TestCase, parameterized.TestCase):

  def setUp(self):
    super(NHNetTest, self).setUp()
    self._nhnet_config = configs.NHNetConfig()
    self._nhnet_config.override(utils.get_test_params().as_dict())
    self._bert2bert_config = configs.BERT2BERTConfig()
    self._bert2bert_config.override(utils.get_test_params().as_dict())

  def _count_params(self, layer, trainable_only=True):
    """Returns the count of all model parameters, or just trainable ones."""
    if not trainable_only:
      return layer.count_params()
    else:
      return int(
          np.sum([
              tf.keras.backend.count_params(p) for p in layer.trainable_weights
          ]))

  def test_create_nhnet_layers(self):
    single_doc_bert, single_doc_decoder = models.get_bert2bert_layers(
        self._bert2bert_config)
    multi_doc_bert, multi_doc_decoder = models.get_nhnet_layers(
        self._nhnet_config)
    # Expects multi-doc encoder/decoder have the same number of parameters as
    # single-doc encoder/decoder.
    self.assertEqual(
        self._count_params(multi_doc_bert), self._count_params(single_doc_bert))
    self.assertEqual(
        self._count_params(multi_doc_decoder),
        self._count_params(single_doc_decoder))

  def test_checkpoint_restore(self):
    bert2bert_model = models.create_bert2bert_model(self._bert2bert_config)
    ckpt = tf.train.Checkpoint(model=bert2bert_model)
    init_checkpoint = ckpt.save(os.path.join(self.get_temp_dir(), "ckpt"))
    nhnet_model = models.create_nhnet_model(
        params=self._nhnet_config, init_checkpoint=init_checkpoint)
    source_weights = (
        bert2bert_model.bert_layer.trainable_weights +
        bert2bert_model.decoder_layer.trainable_weights)
    dest_weights = (
        nhnet_model.bert_layer.trainable_weights +
        nhnet_model.decoder_layer.trainable_weights)
    for source_weight, dest_weight in zip(source_weights, dest_weights):
      self.assertAllClose(source_weight.numpy(), dest_weight.numpy())

  @combinations.generate(all_strategy_combinations())
  def test_nhnet_train_forward(self, distribution):
    seq_length = 10
    # Defines the model inside distribution strategy scope.
    with distribution.scope():
      # Forward path.
      batch_size = 2
      num_docs = 2
      batches = 4
      fake_ids = np.zeros((batch_size * batches, num_docs, seq_length),
                          dtype=np.int32)
      fake_inputs = {
          "input_ids":
              fake_ids,
          "input_mask":
              fake_ids,
          "segment_ids":
              fake_ids,
          "target_ids":
              np.zeros((batch_size * batches, seq_length * 2), dtype=np.int32),
      }
      model = models.create_nhnet_model(params=self._nhnet_config)
      results = distribution_forward_path(distribution, model, fake_inputs,
                                          batch_size)
      logging.info("Forward path results: %s", str(results))
      self.assertLen(results, batches)

  @combinations.generate(all_strategy_combinations())
  def test_nhnet_eval(self, distribution):
    seq_length = 10
    padded_decode = isinstance(
        distribution,
        (tf.distribute.TPUStrategy, tf.distribute.experimental.TPUStrategy))
    self._nhnet_config.override(
        {
            "beam_size": 4,
            "len_title": seq_length,
            "alpha": 0.6,
            "multi_channel_cross_attention": True,
            "padded_decode": padded_decode,
        },
        is_strict=False)
    # Defines the model inside distribution strategy scope.
    with distribution.scope():
      # Forward path.
      batch_size = 2
      num_docs = 2
      batches = 4
      fake_ids = np.zeros((batch_size * batches, num_docs, seq_length),
                          dtype=np.int32)
      fake_inputs = {
          "input_ids": fake_ids,
          "input_mask": fake_ids,
          "segment_ids": fake_ids,
          "target_ids": np.zeros((batch_size * batches, 5), dtype=np.int32),
      }
      model = models.create_nhnet_model(params=self._nhnet_config)
      results = distribution_forward_path(
          distribution, model, fake_inputs, batch_size, mode="predict")
      self.assertLen(results, batches)
      results = distribution_forward_path(
          distribution, model, fake_inputs, batch_size, mode="eval")
      self.assertLen(results, batches)


if __name__ == "__main__":
  tf.test.main()
