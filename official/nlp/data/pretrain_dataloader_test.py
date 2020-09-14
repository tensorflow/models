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
"""Tests for official.nlp.data.pretrain_dataloader."""
import itertools
import os

from absl.testing import parameterized
import numpy as np
import tensorflow as tf

from official.nlp.data import pretrain_dataloader


def _create_fake_dataset(output_path,
                         seq_length,
                         max_predictions_per_seq,
                         use_position_id,
                         use_next_sentence_label,
                         use_v2_feature_names=False):
  """Creates a fake dataset."""
  writer = tf.io.TFRecordWriter(output_path)

  def create_int_feature(values):
    f = tf.train.Feature(int64_list=tf.train.Int64List(value=list(values)))
    return f

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


class BertPretrainDataTest(tf.test.TestCase, parameterized.TestCase):

  @parameterized.parameters(itertools.product(
      (False, True),
      (False, True),
  ))
  def test_load_data(self, use_next_sentence_label, use_position_id):
    train_data_path = os.path.join(self.get_temp_dir(), "train.tf_record")
    seq_length = 128
    max_predictions_per_seq = 20
    _create_fake_dataset(
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
    _create_fake_dataset(
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


if __name__ == "__main__":
  tf.test.main()
