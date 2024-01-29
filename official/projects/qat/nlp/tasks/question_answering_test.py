# Copyright 2023 The TensorFlow Authors. All Rights Reserved.
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

"""Tests for official.nlp.tasks.question_answering."""
import json
import os

from absl.testing import parameterized
import tensorflow as tf

from official.nlp.configs import encoders
from official.nlp.data import question_answering_dataloader
from official.nlp.tasks import question_answering as qa_cfg
from official.projects.qat.nlp.tasks import question_answering


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

  @parameterized.named_parameters(("squad1", False), ("squad2", True))
  def test_predict(self, version_2_with_negative):
    validation_data = self._get_validation_data_config(
        version_2_with_negative=version_2_with_negative)

    config = question_answering.QuantizedModelQAConfig(
        model=qa_cfg.ModelConfig(encoder=self._encoder_config),
        train_data=self._train_data_config,
        validation_data=validation_data)
    task = question_answering.QuantizedModelQATask(config)
    model = task.build_model()

    all_predictions, all_nbest, scores_diff = qa_cfg.predict(
        task, validation_data, model)
    self.assertLen(all_predictions, 1)
    self.assertLen(all_nbest, 1)
    if version_2_with_negative:
      self.assertLen(scores_diff, 1)
    else:
      self.assertEmpty(scores_diff)


if __name__ == "__main__":
  tf.test.main()
