# Copyright 2025 The TensorFlow Authors. All Rights Reserved.
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

"""Tests that masked LM models are deterministic when determinism is enabled."""

import tensorflow as tf, tf_keras

from official.nlp.configs import bert
from official.nlp.configs import encoders
from official.nlp.data import pretrain_dataloader
from official.nlp.tasks import masked_lm


class MLMTaskTest(tf.test.TestCase):

  def _build_dataset(self, params, vocab_size):
    def dummy_data(_):
      dummy_ids = tf.random.uniform((1, params.seq_length), maxval=vocab_size,
                                    dtype=tf.int32)
      dummy_mask = tf.ones((1, params.seq_length), dtype=tf.int32)
      dummy_type_ids = tf.zeros((1, params.seq_length), dtype=tf.int32)
      dummy_lm = tf.zeros((1, params.max_predictions_per_seq), dtype=tf.int32)
      return dict(
          input_word_ids=dummy_ids,
          input_mask=dummy_mask,
          input_type_ids=dummy_type_ids,
          masked_lm_positions=dummy_lm,
          masked_lm_ids=dummy_lm,
          masked_lm_weights=tf.cast(dummy_lm, dtype=tf.float32),
          next_sentence_labels=tf.zeros((1, 1), dtype=tf.int32))

    dataset = tf.data.Dataset.range(1)
    dataset = dataset.repeat()
    dataset = dataset.map(
        dummy_data, num_parallel_calls=tf.data.experimental.AUTOTUNE)
    return dataset

  def _build_and_run_model(self, config, num_steps=5):
    task = masked_lm.MaskedLMTask(config)
    model = task.build_model()
    metrics = task.build_metrics()
    dataset = self._build_dataset(config.train_data,
                                  config.model.encoder.get().vocab_size)

    iterator = iter(dataset)
    optimizer = tf_keras.optimizers.SGD(lr=0.1)

    # Run training
    for _ in range(num_steps):
      logs = task.train_step(next(iterator), model, optimizer, metrics=metrics)
    for metric in metrics:
      logs[metric.name] = metric.result()

    # Run validation
    validation_logs = task.validation_step(next(iterator), model,
                                           metrics=metrics)
    for metric in metrics:
      validation_logs[metric.name] = metric.result()

    return logs, validation_logs, model.weights

  def test_task_determinism(self):
    config = masked_lm.MaskedLMConfig(
        init_checkpoint=self.get_temp_dir(),
        scale_loss=True,
        model=bert.PretrainerConfig(
            encoder=encoders.EncoderConfig(
                bert=encoders.BertEncoderConfig(vocab_size=30522,
                                                num_layers=1)),
            cls_heads=[
                bert.ClsHeadConfig(
                    inner_dim=10, num_classes=2, name="next_sentence")
            ]),
        train_data=pretrain_dataloader.BertPretrainDataConfig(
            max_predictions_per_seq=20,
            seq_length=128,
            global_batch_size=1))

    tf_keras.utils.set_random_seed(1)
    logs1, validation_logs1, weights1 = self._build_and_run_model(config)
    tf_keras.utils.set_random_seed(1)
    logs2, validation_logs2, weights2 = self._build_and_run_model(config)

    self.assertEqual(logs1["loss"], logs2["loss"])
    self.assertEqual(validation_logs1["loss"], validation_logs2["loss"])
    for weight1, weight2 in zip(weights1, weights2):
      self.assertAllEqual(weight1, weight2)


if __name__ == "__main__":
  tf.config.experimental.enable_op_determinism()
  tf.test.main()
