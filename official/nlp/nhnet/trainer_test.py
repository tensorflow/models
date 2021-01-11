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
"""Tests for official.nlp.nhnet.trainer."""

import os

from absl import flags
from absl.testing import parameterized
import tensorflow as tf

# pylint: disable=g-direct-tensorflow-import
from tensorflow.python.distribute import combinations
from tensorflow.python.distribute import strategy_combinations
# pylint: enable=g-direct-tensorflow-import
from official.nlp.nhnet import trainer
from official.nlp.nhnet import utils

FLAGS = flags.FLAGS
trainer.define_flags()


def all_strategy_combinations():
  return combinations.combine(
      distribution=[
          strategy_combinations.one_device_strategy,
          strategy_combinations.one_device_strategy_gpu,
          strategy_combinations.mirrored_strategy_with_gpu_and_cpu,
          strategy_combinations.cloud_tpu_strategy,
      ],)


def get_trivial_data(config) -> tf.data.Dataset:
  """Gets trivial data in the ImageNet size."""
  batch_size, num_docs = 2, len(config.passage_list),
  len_passage = config.len_passage
  len_title = config.len_title

  def generate_data(_) -> tf.data.Dataset:
    fake_ids = tf.zeros((num_docs, len_passage), dtype=tf.int32)
    title = tf.zeros((len_title), dtype=tf.int32)
    return dict(
        input_ids=fake_ids,
        input_mask=fake_ids,
        segment_ids=fake_ids,
        target_ids=title)

  dataset = tf.data.Dataset.range(1)
  dataset = dataset.repeat()
  dataset = dataset.map(
      generate_data, num_parallel_calls=tf.data.experimental.AUTOTUNE)
  dataset = dataset.prefetch(buffer_size=1).batch(batch_size)
  return dataset


class TrainerTest(tf.test.TestCase, parameterized.TestCase):

  def setUp(self):
    super(TrainerTest, self).setUp()
    self._config = utils.get_test_params()
    self._config.override(
        {
            "vocab_size": 49911,
            "max_position_embeddings": 200,
            "len_title": 15,
            "len_passage": 20,
            "beam_size": 5,
            "alpha": 0.6,
            "learning_rate": 0.0,
            "learning_rate_warmup_steps": 0,
            "multi_channel_cross_attention": True,
            "passage_list": ["a", "b"],
        },
        is_strict=False)

  @combinations.generate(all_strategy_combinations())
  def test_train(self, distribution):
    FLAGS.train_steps = 10
    FLAGS.checkpoint_interval = 5
    FLAGS.model_dir = self.get_temp_dir()
    FLAGS.model_type = "nhnet"
    stats = trainer.train(self._config, distribution,
                          get_trivial_data(self._config))
    self.assertIn("training_loss", stats)
    self.assertLen(
        tf.io.gfile.glob(os.path.join(FLAGS.model_dir, "ckpt*.index")), 2)


if __name__ == "__main__":
  tf.test.main()
