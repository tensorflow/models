# Copyright 2018 The TensorFlow Authors. All Rights Reserved.
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
"""Translate text or files using trained transformer model."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import tensorflow as tf

from official.transformer.utils import tokenizer

_DECODE_BATCH_SIZE = 32
_EXTRA_DECODE_LENGTH = 100
_BEAM_SIZE = 4
_ALPHA = 0.6


def _get_sorted_inputs(filename):
  """Read and sort lines from the file sorted by decreasing length.

  Args:
    filename: String name of file to read inputs from.
  Returns:
    Sorted list of inputs, and dictionary mapping original index->sorted index
    of each element.
  """
  with tf.io.gfile.GFile(filename) as f:
    records = f.read().split("\n")
    inputs = [record.strip() for record in records]
    if not inputs[-1]:
      inputs.pop()

  input_lens = [(i, len(line.split())) for i, line in enumerate(inputs)]
  sorted_input_lens = sorted(input_lens, key=lambda x: x[1], reverse=True)

  sorted_inputs = [None] * len(sorted_input_lens)
  sorted_keys = [0] * len(sorted_input_lens)
  for i, (index, _) in enumerate(sorted_input_lens):
    sorted_inputs[i] = inputs[index]
    sorted_keys[index] = i
  return sorted_inputs, sorted_keys


def _encode_and_add_eos(line, subtokenizer):
  """Encode line with subtokenizer, and add EOS id to the end."""
  return subtokenizer.encode(line) + [tokenizer.EOS_ID]


def _trim_and_decode(ids, subtokenizer):
  """Trim EOS and PAD tokens from ids, and decode to return a string."""
  try:
    index = list(ids).index(tokenizer.EOS_ID)
    return subtokenizer.decode(ids[:index])
  except ValueError:  # No EOS found in sequence
    return subtokenizer.decode(ids)


def translate_file(
    model, subtokenizer, input_file, output_file=None,
    print_all_translations=True):
  """Translate lines in file, and save to output file if specified.

  Args:
    model: Keras model used to generate the translations.
    subtokenizer: Subtokenizer object for encoding and decoding source and
       translated lines.
    input_file: file containing lines to translate
    output_file: file that stores the generated translations.
    print_all_translations: If true, all translations are printed to stdout.

  Raises:
    ValueError: if output file is invalid.
  """
  batch_size = _DECODE_BATCH_SIZE

  # Read and sort inputs by length. Keep dictionary (original index-->new index
  # in sorted list) to write translations in the original order.
  sorted_inputs, sorted_keys = _get_sorted_inputs(input_file)
  total_samples = len(sorted_inputs)
  num_decode_batches = (total_samples - 1) // batch_size + 1

  def input_generator():
    """Yield encoded strings from sorted_inputs."""
    for i in range(num_decode_batches):
      lines = [
          sorted_inputs[j + i * batch_size]
          for j in range(batch_size)
          if j + i * batch_size < total_samples
      ]
      lines = [_encode_and_add_eos(l, subtokenizer) for l in lines]
      batch = tf.keras.preprocessing.sequence.pad_sequences(
          lines, dtype="int64", padding="post")
      tf.compat.v1.logging.info("Decoding batch %d out of %d.", i,
                                num_decode_batches)
      yield batch

  translations = []
  for i, text in enumerate(input_generator()):
    val_outputs, _ = model.predict(text)

    length = len(val_outputs)
    for j in range(length):
      translation = _trim_and_decode(val_outputs[j], subtokenizer)
      translations.append(translation)
      if print_all_translations:
        tf.compat.v1.logging.info(
            "Translating:\n\tInput: %s\n\tOutput: %s" %
            (sorted_inputs[j + i * batch_size], translation))

  # Write translations in the order they appeared in the original file.
  if output_file is not None:
    if tf.io.gfile.isdir(output_file):
      raise ValueError("File output is a directory, will not save outputs to "
                       "file.")
    tf.compat.v1.logging.info("Writing to file %s" % output_file)
    with tf.compat.v1.gfile.Open(output_file, "w") as f:
      for i in sorted_keys:
        f.write("%s\n" % translations[i])


def translate_from_text(model, subtokenizer, txt):
  encoded_txt = _encode_and_add_eos(txt, subtokenizer)
  result = model.predict(encoded_txt)
  outputs = result["outputs"]
  tf.compat.v1.logging.info("Original: \"%s\"" % txt)
  translate_from_input(outputs, subtokenizer)


def translate_from_input(outputs, subtokenizer):
  translation = _trim_and_decode(outputs, subtokenizer)
  tf.compat.v1.logging.info("Translation: \"%s\"" % translation)
