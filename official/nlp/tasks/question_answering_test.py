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
"""Tests for official.nlp.tasks.question_answering."""
import itertools
import json
import os

from absl.testing import parameterized
import tensorflow as tf

from official.nlp.bert import configs
from official.nlp.bert import export_tfhub
from official.nlp.configs import bert
from official.nlp.configs import encoders
from official.nlp.data import question_answering_dataloader
from official.nlp.tasks import masked_lm
from official.nlp.tasks import question_answering


class QuestionAnsweringTaskTest(tf.test.TestCase, parameterized.TestCase):

  def setUp(self):
    super(QuestionAnsweringTaskTest, self).setUp()
    self._encoder_config = encoders.EncoderConfig(
        bert=encoders.BertEncoderConfig(vocab_size=30522, num_layers=1))
    self._train_data_config = question_answering_dataloader.QADataConfig(
        input_path="dummy", seq_length=128, global_batch_size=1)

    val_data = {
        "version":
            "1.1",
        "data": [{
            "paragraphs": [{
                "context":
                    "Sky is blue.",
                "qas": [{
                    "question":
                        "What is blue?",
                    "id":
                        "1234",
                    "answers": [{
                        "text": "Sky",
                        "answer_start": 0
                    }, {
                        "text": "Sky",
                        "answer_start": 0
                    }, {
                        "text": "Sky",
                        "answer_start": 0
                    }]
                }]
            }]
        }]
    }
    self._val_input_path = os.path.join(self.get_temp_dir(), "val_data.json")
    with tf.io.gfile.GFile(self._val_input_path, "w") as writer:
      writer.write(json.dumps(val_data, indent=4) + "\n")

    self._test_vocab = os.path.join(self.get_temp_dir(), "vocab.txt")
    with tf.io.gfile.GFile(self._test_vocab, "w") as writer:
      writer.write("[PAD]\n[UNK]\n[CLS]\n[SEP]\n[MASK]\nsky\nis\nblue\n")

  def _get_validation_data_config(self, version_2_with_negative=False):
    return question_answering_dataloader.QADataConfig(
        is_training=False,
        input_path=self._val_input_path,
        input_preprocessed_data_path=self.get_temp_dir(),
        seq_length=128,
        global_batch_size=1,
        version_2_with_negative=version_2_with_negative,
        vocab_file=self._test_vocab,
        tokenization="WordPiece",
        do_lower_case=True)

  def _run_task(self, config):
    task = question_answering.QuestionAnsweringTask(config)
    model = task.build_model()
    metrics = task.build_metrics()
    task.initialize(model)

    train_dataset = task.build_inputs(config.train_data)
    train_iterator = iter(train_dataset)
    optimizer = tf.keras.optimizers.SGD(lr=0.1)
    task.train_step(next(train_iterator), model, optimizer, metrics=metrics)

    val_dataset = task.build_inputs(config.validation_data)
    val_iterator = iter(val_dataset)
    logs = task.validation_step(next(val_iterator), model, metrics=metrics)
    # Mock that `logs` is from one replica.
    logs = {x: (logs[x],) for x in logs}
    logs = task.aggregate_logs(step_outputs=logs)
    metrics = task.reduce_aggregated_logs(logs)
    self.assertIn("final_f1", metrics)

  @parameterized.parameters(
      itertools.product(
          (False, True),
          ("WordPiece", "SentencePiece"),
      ))
  def test_task(self, version_2_with_negative, tokenization):
    # Saves a checkpoint.
    pretrain_cfg = bert.PretrainerConfig(
        encoder=self._encoder_config,
        cls_heads=[
            bert.ClsHeadConfig(
                inner_dim=10, num_classes=3, name="next_sentence")
        ])
    pretrain_model = masked_lm.MaskedLMTask(None).build_model(pretrain_cfg)
    ckpt = tf.train.Checkpoint(
        model=pretrain_model, **pretrain_model.checkpoint_items)
    saved_path = ckpt.save(self.get_temp_dir())

    config = question_answering.QuestionAnsweringConfig(
        init_checkpoint=saved_path,
        model=question_answering.ModelConfig(encoder=self._encoder_config),
        train_data=self._train_data_config,
        validation_data=self._get_validation_data_config(
            version_2_with_negative))
    self._run_task(config)

  def test_task_with_fit(self):
    config = question_answering.QuestionAnsweringConfig(
        model=question_answering.ModelConfig(encoder=self._encoder_config),
        train_data=self._train_data_config,
        validation_data=self._get_validation_data_config())
    task = question_answering.QuestionAnsweringTask(config)
    model = task.build_model()
    model = task.compile_model(
        model,
        optimizer=tf.keras.optimizers.SGD(lr=0.1),
        train_step=task.train_step,
        metrics=[tf.keras.metrics.SparseCategoricalAccuracy(name="accuracy")])
    dataset = task.build_inputs(config.train_data)
    logs = model.fit(dataset, epochs=1, steps_per_epoch=2)
    self.assertIn("loss", logs.history)
    self.assertIn("start_positions_accuracy", logs.history)
    self.assertIn("end_positions_accuracy", logs.history)

  def _export_bert_tfhub(self):
    bert_config = configs.BertConfig(
        vocab_size=30522,
        hidden_size=16,
        intermediate_size=32,
        max_position_embeddings=128,
        num_attention_heads=2,
        num_hidden_layers=1)
    _, encoder = export_tfhub.create_bert_model(bert_config)
    model_checkpoint_dir = os.path.join(self.get_temp_dir(), "checkpoint")
    checkpoint = tf.train.Checkpoint(model=encoder)
    checkpoint.save(os.path.join(model_checkpoint_dir, "test"))
    model_checkpoint_path = tf.train.latest_checkpoint(model_checkpoint_dir)

    vocab_file = os.path.join(self.get_temp_dir(), "uncased_vocab.txt")
    with tf.io.gfile.GFile(vocab_file, "w") as f:
      f.write("dummy content")

    hub_destination = os.path.join(self.get_temp_dir(), "hub")
    export_tfhub.export_bert_tfhub(bert_config, model_checkpoint_path,
                                   hub_destination, vocab_file)
    return hub_destination

  def test_task_with_hub(self):
    hub_module_url = self._export_bert_tfhub()
    config = question_answering.QuestionAnsweringConfig(
        hub_module_url=hub_module_url,
        model=question_answering.ModelConfig(encoder=self._encoder_config),
        train_data=self._train_data_config,
        validation_data=self._get_validation_data_config())
    self._run_task(config)

  @parameterized.named_parameters(("squad1", False), ("squad2", True))
  def test_predict(self, version_2_with_negative):
    validation_data = self._get_validation_data_config(
        version_2_with_negative=version_2_with_negative)

    config = question_answering.QuestionAnsweringConfig(
        model=question_answering.ModelConfig(encoder=self._encoder_config),
        train_data=self._train_data_config,
        validation_data=validation_data)
    task = question_answering.QuestionAnsweringTask(config)
    model = task.build_model()

    all_predictions, all_nbest, scores_diff = question_answering.predict(
        task, validation_data, model)
    self.assertLen(all_predictions, 1)
    self.assertLen(all_nbest, 1)
    if version_2_with_negative:
      self.assertLen(scores_diff, 1)
    else:
      self.assertEmpty(scores_diff)


if __name__ == "__main__":
  tf.test.main()
