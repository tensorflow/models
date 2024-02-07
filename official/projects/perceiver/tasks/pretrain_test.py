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

"""Tests for official.nlp.tasks.masked_lm."""

import tensorflow as tf, tf_keras
import tensorflow_datasets as tfds

from official.nlp.data import pretrain_dataloader
from official.projects.perceiver.configs import perceiver
from official.projects.perceiver.tasks import pretrain as tasks


_NUM_EXAMPLES = 10


def _gen_fn():
  word_ids = tf.constant([1, 1], dtype=tf.int32)
  mask = tf.constant([1, 1], dtype=tf.int32)
  lm_mask = tf.constant([1, 1], dtype=tf.int32)
  return {
      'file_name': 'test',
      'masked_lm_positions': lm_mask,
      'input_word_ids': word_ids,
      'input_mask': mask,
  }


def _as_dataset(self, *args, **kwargs):
  del args
  del kwargs
  return tf.data.Dataset.from_generator(
      lambda: (_gen_fn() for i in range(_NUM_EXAMPLES)),
      output_types=self.info.features.dtype,
      output_shapes=self.info.features.shape,
  )


def _fake_build_inputs(self, params, input_context=None):  # pylint: disable=unused-argument
  def dummy_data(_):
    dummy_ids = tf.zeros((1, params.seq_length), dtype=tf.int32)
    dummy_lm = tf.zeros((1, params.max_predictions_per_seq), dtype=tf.int32)
    return dict(
        input_word_ids=dummy_ids,
        input_mask=dummy_ids,
        masked_lm_positions=dummy_lm,
        masked_lm_ids=dummy_lm,
        masked_lm_weights=tf.cast(dummy_lm, dtype=tf.float32))

  dataset = tf.data.Dataset.range(1)
  dataset = dataset.repeat()
  dataset = dataset.map(
      dummy_data, num_parallel_calls=tf.data.experimental.AUTOTUNE)
  return dataset


class PretrainTaskTest(tf.test.TestCase):

  def setUp(self):
    super().setUp()
    tasks.PretrainTask.build_inputs = _fake_build_inputs

  def test_task(self):
    config = perceiver.PretrainConfig(
        train_data=pretrain_dataloader.BertPretrainDataConfig(
            input_path='dummy',
            global_batch_size=512,
            use_next_sentence_label=False,
            use_v2_feature_names=True),
        validation_data=pretrain_dataloader.BertPretrainDataConfig(
            input_path='dummy',
            global_batch_size=512,
            is_training=False,
            use_next_sentence_label=False,
            use_v2_feature_names=True))
    task = tasks.PretrainTask(config)

    model = task.build_model()
    metrics = task.build_metrics()
    dataset = task.build_inputs(config.train_data)

    iterator = iter(dataset)
    optimizer = tf_keras.optimizers.SGD(lr=0.1)
    task.train_step(next(iterator), model, optimizer, metrics=metrics)
    task.validation_step(next(iterator), model, metrics=metrics)

    # Saves a checkpoint.
    _ = tf.train.Checkpoint(model=model, **model.checkpoint_items)
    # ckpt.save(config.init_checkpoint)
    # TODO(b/222634115) fix ckpt.save
    task.initialize(model)

  def test_train_step(self):
    config = perceiver.PretrainConfig(
        train_data=pretrain_dataloader.BertPretrainDataConfig(
            input_path='dummy',
            global_batch_size=512,
            use_next_sentence_label=False,
            use_v2_feature_names=True),
        validation_data=pretrain_dataloader.BertPretrainDataConfig(
            input_path='dummy',
            global_batch_size=512,
            is_training=False,
            use_next_sentence_label=False,
            use_v2_feature_names=True))

    with tfds.testing.mock_data(as_dataset_fn=_as_dataset):
      task = tasks.PretrainTask(config)
      model = task.build_model()
      dataset = task.build_inputs(config.train_data)
      metrics = task.build_metrics()

      iterator = iter(dataset)
      opt_cfg = perceiver._MLM_WORDPIECE_TRAINER.optimizer_config
      optimizer = tasks.PretrainTask.create_optimizer(opt_cfg)
      task.train_step(next(iterator), model, optimizer, metrics=metrics)

# TODO(b/222634115) add test coverage.

if __name__ == '__main__':
  tf.test.main()
