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

"""A script to train sentencepiece model from tensorflow datasets.

Reserved tokens:
pad: 0,
eos: 1,
unk: 2
(bos is not reserved)
"""

import os
import tempfile
from typing import List, Tuple

from absl import app
from absl import flags
from absl import logging
import tensorflow as tf
import tensorflow_datasets as tfds

from sentencepiece import SentencePieceTrainer


FLAGS = flags.FLAGS
flags.DEFINE_string("output_model_path", None,
                    "Path to save the the sentencepiece model.")
flags.mark_flag_as_required("output_model_path")

flags.DEFINE_string("tfds_dir", None, "Directory of the tfds.")
flags.DEFINE_string("tfds_name", "wmt14_translate/de-en",
                    "Name of the dataset we generate vacabulay from.")
flags.DEFINE_string("tfds_split", "train", "Split of the dataset.")
flags.DEFINE_integer("vocab_size", 32000, "Size of vocabulary.")
flags.DEFINE_integer(
    "max_char", -1,
    "Maximum number of characters to use. "
    "If a non-positive number is provided, all sentences are used.")
flags.DEFINE_string("model_type", "bpe",
                    "Model algorithm: unigram, bpe, word or char.")
flags.DEFINE_float("character_coverage", 0.9995,
                   "Character coverage to determine the minimum symbols")
flags.DEFINE_list(
    "data_keys", ["en", "de"],
    "Comma-separated list of keys to use for training the vocabulary.")


def dump_chars_to_textfile(dataset: tf.data.Dataset,
                           data_keys: Tuple[str],
                           max_char: int = -1):
  """Write part of a TFDS sentence dataset to lines in a text file.

  Args:
    dataset: tf.dataset containing string-data.
    data_keys: what keys in dataset to dump from.
    max_char: max character to dump to text file.

  Returns:
    name of temp file with dataset bytes, exact number of characters dumped.
  """
  ds_iter = dataset.as_numpy_iterator()
  with tempfile.NamedTemporaryFile(delete=False) as outfp:
    char_count = 0
    while True:
      example = next(ds_iter, None)
      if example is None or (
          max_char > 0 and char_count > max_char):
        break
      for k in data_keys:
        line = example[k] + b"\n"
        char_count += len(line)
        outfp.write(line)
  return outfp.name


def train_sentencepiece(
    file_path: str,
    model_path: str,
    vocab_size: int,
    character_coverage: float,
    model_type: str):
  """Train SentencePiece tokenizer from subset of tf dataset.

  Args:
    file_path: path of data to train sentencepiece.
    model_path: path of model file to save vocab model to.
    vocab_size: size of vocab tokens to train.
    character_coverage: amount of characters covered by the model, good defaults
      are 0.9995 for languages with rich character set like Japanese or Chinese
      and 1.0 for other languages with small character set.
    model_type: type of sentencepiece vocab to train.

  Returns:
    path to the trained sentencepiece vocabulary model.
  """
  argstr = " ".join([
      f"--input={file_path}", f"--vocab_size={vocab_size}",
      f"--character_coverage={character_coverage}",
      f"--model_prefix={model_path}", f"--model_type={model_type}",
      "--bos_id=-1", "--pad_id=0", "--eos_id=1", "--unk_id=2"
  ])
  SentencePieceTrainer.Train(argstr)


def main(argv: List[str]):
  del argv
  builder = tfds.builder(FLAGS.tfds_name, data_dir=FLAGS.tfds_dir)
  ds = builder.as_dataset(split=FLAGS.tfds_split)
  tmp_filename = dump_chars_to_textfile(ds, FLAGS.data_keys, FLAGS.max_char)
  logging.info("Sentencepiece model will be placed here: %s",
               FLAGS.output_model_path)
  train_sentencepiece(tmp_filename,
                      FLAGS.output_model_path,
                      FLAGS.vocab_size,
                      FLAGS.character_coverage,
                      FLAGS.model_type)
  os.remove(tmp_filename)


if __name__ == "__main__":
  app.run(main)
