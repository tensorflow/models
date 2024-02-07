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

"""Tests for third_party.tensorflow_models.official.nlp.data.classifier_data_lib."""

import os
import tempfile

from absl.testing import parameterized
import tensorflow as tf, tf_keras
import tensorflow_datasets as tfds

from official.nlp.data import classifier_data_lib
from official.nlp.tools import tokenization


def decode_record(record, name_to_features):
  """Decodes a record to a TensorFlow example."""
  return tf.io.parse_single_example(record, name_to_features)


class BertClassifierLibTest(tf.test.TestCase, parameterized.TestCase):

  def setUp(self):
    super(BertClassifierLibTest, self).setUp()
    self.model_dir = self.get_temp_dir()
    self.processors = {
        "CB": classifier_data_lib.CBProcessor,
        "SUPERGLUE-RTE": classifier_data_lib.SuperGLUERTEProcessor,
        "BOOLQ": classifier_data_lib.BoolQProcessor,
        "WIC": classifier_data_lib.WiCProcessor,
    }

    vocab_tokens = [
        "[UNK]", "[CLS]", "[SEP]", "want", "##want", "##ed", "wa", "un", "runn",
        "##ing", ","
    ]
    with tempfile.NamedTemporaryFile(delete=False) as vocab_writer:
      vocab_writer.write("".join([x + "\n" for x in vocab_tokens
                                 ]).encode("utf-8"))
    vocab_file = vocab_writer.name
    self.tokenizer = tokenization.FullTokenizer(vocab_file)

  @parameterized.parameters(
      {"task_type": "CB"},
      {"task_type": "BOOLQ"},
      {"task_type": "SUPERGLUE-RTE"},
      {"task_type": "WIC"},
  )
  def test_generate_dataset_from_tfds_processor(self, task_type):
    with tfds.testing.mock_data(num_examples=5):
      output_path = os.path.join(self.model_dir, task_type)

      processor = self.processors[task_type]()

      classifier_data_lib.generate_tf_record_from_data_file(
          processor,
          None,
          self.tokenizer,
          train_data_output_path=output_path,
          eval_data_output_path=output_path,
          test_data_output_path=output_path)
      files = tf.io.gfile.glob(output_path)
      self.assertNotEmpty(files)

      train_dataset = tf.data.TFRecordDataset(output_path)
      seq_length = 128
      label_type = tf.int64
      name_to_features = {
          "input_ids": tf.io.FixedLenFeature([seq_length], tf.int64),
          "input_mask": tf.io.FixedLenFeature([seq_length], tf.int64),
          "segment_ids": tf.io.FixedLenFeature([seq_length], tf.int64),
          "label_ids": tf.io.FixedLenFeature([], label_type),
      }
      train_dataset = train_dataset.map(
          lambda record: decode_record(record, name_to_features))

      # If data is retrieved without error, then all requirements
      # including data type/shapes are met.
      _ = next(iter(train_dataset))


if __name__ == "__main__":
  tf.test.main()
