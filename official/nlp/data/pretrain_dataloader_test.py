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

"""Tests for official.nlp.data.pretrain_dataloader."""
import itertools
import os

from absl.testing import parameterized
import numpy as np
import tensorflow as tf, tf_keras

from official.nlp.data import pretrain_dataloader


def create_int_feature(values):
  f = tf.train.Feature(int64_list=tf.train.Int64List(value=list(values)))
  return f


def _create_fake_bert_dataset(
    output_path,
    seq_length,
    max_predictions_per_seq,
    use_position_id,
    use_next_sentence_label,
    use_v2_feature_names=False):
  """Creates a fake dataset."""
  writer = tf.io.TFRecordWriter(output_path)

  def create_float_feature(values):
    f = tf.train.Feature(float_list=tf.train.FloatList(value=list(values)))
    return f

  for _ in range(100):
    features = {}
    input_ids = np.random.randint(100, size=(seq_length))
    features["input_mask"] = create_int_feature(np.ones_like(input_ids))
    if use_v2_feature_names:
      features["input_word_ids"] = create_int_feature(input_ids)
      features["input_type_ids"] = create_int_feature(np.ones_like(input_ids))
    else:
      features["input_ids"] = create_int_feature(input_ids)
      features["segment_ids"] = create_int_feature(np.ones_like(input_ids))

    features["masked_lm_positions"] = create_int_feature(
        np.random.randint(100, size=(max_predictions_per_seq)))
    features["masked_lm_ids"] = create_int_feature(
        np.random.randint(100, size=(max_predictions_per_seq)))
    features["masked_lm_weights"] = create_float_feature(
        [1.0] * max_predictions_per_seq)

    if use_next_sentence_label:
      features["next_sentence_labels"] = create_int_feature([1])

    if use_position_id:
      features["position_ids"] = create_int_feature(range(0, seq_length))

    tf_example = tf.train.Example(features=tf.train.Features(feature=features))
    writer.write(tf_example.SerializeToString())
  writer.close()


def _create_fake_xlnet_dataset(
    output_path, seq_length, max_predictions_per_seq):
  """Creates a fake dataset."""
  writer = tf.io.TFRecordWriter(output_path)
  for _ in range(100):
    features = {}
    input_ids = np.random.randint(100, size=(seq_length))
    num_boundary_indices = np.random.randint(1, seq_length)

    if max_predictions_per_seq is not None:
      input_mask = np.zeros_like(input_ids)
      input_mask[:max_predictions_per_seq] = 1
      np.random.shuffle(input_mask)
    else:
      input_mask = np.ones_like(input_ids)

    features["input_mask"] = create_int_feature(input_mask)
    features["input_word_ids"] = create_int_feature(input_ids)
    features["input_type_ids"] = create_int_feature(np.ones_like(input_ids))
    features["boundary_indices"] = create_int_feature(
        sorted(np.random.randint(seq_length, size=(num_boundary_indices))))
    features["target"] = create_int_feature(input_ids + 1)
    features["label"] = create_int_feature([1])
    tf_example = tf.train.Example(features=tf.train.Features(feature=features))
    writer.write(tf_example.SerializeToString())
  writer.close()


class BertPretrainDataTest(tf.test.TestCase, parameterized.TestCase):

  @parameterized.parameters(itertools.product(
      (False, True),
      (False, True),
  ))
  def test_load_data(self, use_next_sentence_label, use_position_id):
    train_data_path = os.path.join(self.get_temp_dir(), "train.tf_record")
    seq_length = 128
    max_predictions_per_seq = 20
    _create_fake_bert_dataset(
        train_data_path,
        seq_length,
        max_predictions_per_seq,
        use_next_sentence_label=use_next_sentence_label,
        use_position_id=use_position_id)
    data_config = pretrain_dataloader.BertPretrainDataConfig(
        input_path=train_data_path,
        max_predictions_per_seq=max_predictions_per_seq,
        seq_length=seq_length,
        global_batch_size=10,
        is_training=True,
        use_next_sentence_label=use_next_sentence_label,
        use_position_id=use_position_id)

    dataset = pretrain_dataloader.BertPretrainDataLoader(data_config).load()
    features = next(iter(dataset))
    self.assertLen(features,
                   6 + int(use_next_sentence_label) + int(use_position_id))
    self.assertIn("input_word_ids", features)
    self.assertIn("input_mask", features)
    self.assertIn("input_type_ids", features)
    self.assertIn("masked_lm_positions", features)
    self.assertIn("masked_lm_ids", features)
    self.assertIn("masked_lm_weights", features)

    self.assertEqual("next_sentence_labels" in features,
                     use_next_sentence_label)
    self.assertEqual("position_ids" in features, use_position_id)

  def test_v2_feature_names(self):
    train_data_path = os.path.join(self.get_temp_dir(), "train.tf_record")
    seq_length = 128
    max_predictions_per_seq = 20
    _create_fake_bert_dataset(
        train_data_path,
        seq_length,
        max_predictions_per_seq,
        use_next_sentence_label=True,
        use_position_id=False,
        use_v2_feature_names=True)
    data_config = pretrain_dataloader.BertPretrainDataConfig(
        input_path=train_data_path,
        max_predictions_per_seq=max_predictions_per_seq,
        seq_length=seq_length,
        global_batch_size=10,
        is_training=True,
        use_next_sentence_label=True,
        use_position_id=False,
        use_v2_feature_names=True)

    dataset = pretrain_dataloader.BertPretrainDataLoader(data_config).load()
    features = next(iter(dataset))
    self.assertIn("input_word_ids", features)
    self.assertIn("input_mask", features)
    self.assertIn("input_type_ids", features)
    self.assertIn("masked_lm_positions", features)
    self.assertIn("masked_lm_ids", features)
    self.assertIn("masked_lm_weights", features)


class XLNetPretrainDataTest(parameterized.TestCase, tf.test.TestCase):

  @parameterized.parameters(itertools.product(
      ("single_token", "whole_word", "token_span"),
      (0, 64),
      (20, None),
      ))
  def test_load_data(
      self, sample_strategy, reuse_length, max_predictions_per_seq):
    train_data_path = os.path.join(self.get_temp_dir(), "train.tf_record")
    seq_length = 128
    batch_size = 5

    _create_fake_xlnet_dataset(
        train_data_path, seq_length, max_predictions_per_seq)

    data_config = pretrain_dataloader.XLNetPretrainDataConfig(
        input_path=train_data_path,
        max_predictions_per_seq=max_predictions_per_seq,
        seq_length=seq_length,
        global_batch_size=batch_size,
        is_training=True,
        reuse_length=reuse_length,
        sample_strategy=sample_strategy,
        min_num_tokens=1,
        max_num_tokens=2,
        permutation_size=seq_length // 2,
        leak_ratio=0.1)

    if max_predictions_per_seq is None:
      with self.assertRaises(ValueError):
        dataset = pretrain_dataloader.XLNetPretrainDataLoader(
            data_config).load()
        features = next(iter(dataset))
    else:
      dataset = pretrain_dataloader.XLNetPretrainDataLoader(data_config).load()
      features = next(iter(dataset))

      self.assertIn("input_word_ids", features)
      self.assertIn("input_type_ids", features)
      self.assertIn("permutation_mask", features)
      self.assertIn("masked_tokens", features)
      self.assertIn("target", features)
      self.assertIn("target_mask", features)

      self.assertAllClose(features["input_word_ids"].shape,
                          (batch_size, seq_length))
      self.assertAllClose(features["input_type_ids"].shape,
                          (batch_size, seq_length))
      self.assertAllClose(features["permutation_mask"].shape,
                          (batch_size, seq_length, seq_length))
      self.assertAllClose(features["masked_tokens"].shape,
                          (batch_size, seq_length,))
      if max_predictions_per_seq is not None:
        self.assertIn("target_mapping", features)
        self.assertAllClose(features["target_mapping"].shape,
                            (batch_size, max_predictions_per_seq, seq_length))
        self.assertAllClose(features["target_mask"].shape,
                            (batch_size, max_predictions_per_seq))
        self.assertAllClose(features["target"].shape,
                            (batch_size, max_predictions_per_seq))
      else:
        self.assertAllClose(features["target_mask"].shape,
                            (batch_size, seq_length))
        self.assertAllClose(features["target"].shape,
                            (batch_size, seq_length))


if __name__ == "__main__":
  tf.test.main()
