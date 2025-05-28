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

"""Tests for official.nlp.data.tagging_data_lib."""
import os
import random

from absl.testing import parameterized
import tensorflow as tf, tf_keras

from official.nlp.data import tagging_data_lib
from official.nlp.tools import tokenization


def _create_fake_file(filename, labels, is_test):

  def write_one_sentence(writer, length):
    for _ in range(length):
      line = "hiworld"
      if not is_test:
        line += "\t%s" % (labels[random.randint(0, len(labels) - 1)])
      writer.write(line + "\n")

  # Writes two sentences with length of 3 and 12 respectively.
  with tf.io.gfile.GFile(filename, "w") as writer:
    write_one_sentence(writer, 3)
    writer.write("\n")
    write_one_sentence(writer, 12)


class TaggingDataLibTest(tf.test.TestCase, parameterized.TestCase):

  def setUp(self):
    super(TaggingDataLibTest, self).setUp()

    self.processors = {
        "panx": tagging_data_lib.PanxProcessor,
        "udpos": tagging_data_lib.UdposProcessor,
    }
    self.vocab_file = os.path.join(self.get_temp_dir(), "vocab.txt")
    with tf.io.gfile.GFile(self.vocab_file, "w") as writer:
      writer.write("\n".join(["[CLS]", "[SEP]", "hi", "##world", "[UNK]"]))

  @parameterized.parameters(
      {"task_type": "panx"},
      {"task_type": "udpos"},
  )
  def test_generate_tf_record(self, task_type):
    processor = self.processors[task_type]()
    input_data_dir = os.path.join(self.get_temp_dir(), task_type)
    tf.io.gfile.mkdir(input_data_dir)
    # Write fake train file.
    _create_fake_file(
        os.path.join(input_data_dir, "train-en.tsv"),
        processor.get_labels(),
        is_test=False)

    # Write fake dev file.
    _create_fake_file(
        os.path.join(input_data_dir, "dev-en.tsv"),
        processor.get_labels(),
        is_test=False)

    # Write fake test files.
    for lang in processor.supported_languages:
      _create_fake_file(
          os.path.join(input_data_dir, "test-%s.tsv" % lang),
          processor.get_labels(),
          is_test=True)

    output_path = os.path.join(self.get_temp_dir(), task_type, "output")
    tokenizer = tokenization.FullTokenizer(
        vocab_file=self.vocab_file, do_lower_case=True)
    metadata = tagging_data_lib.generate_tf_record_from_data_file(
        processor,
        input_data_dir,
        tokenizer,
        max_seq_length=8,
        train_data_output_path=os.path.join(output_path, "train.tfrecord"),
        eval_data_output_path=os.path.join(output_path, "eval.tfrecord"),
        test_data_output_path=os.path.join(output_path, "test_{}.tfrecord"),
        text_preprocessing=tokenization.convert_to_unicode)

    self.assertEqual(metadata["train_data_size"], 5)
    files = tf.io.gfile.glob(output_path + "/*")
    expected_files = []
    expected_files.append(os.path.join(output_path, "train.tfrecord"))
    expected_files.append(os.path.join(output_path, "eval.tfrecord"))
    for lang in processor.supported_languages:
      expected_files.append(
          os.path.join(output_path, "test_%s.tfrecord" % lang))

    self.assertCountEqual(files, expected_files)


if __name__ == "__main__":
  tf.test.main()
