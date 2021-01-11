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

# Import libraries
from absl import logging
import numpy as np
import tensorflow as tf

from official.nlp.transformer.utils import tokenizer

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


def translate_file(model,
                   params,
                   subtokenizer,
                   input_file,
                   output_file=None,
                   print_all_translations=True,
                   distribution_strategy=None):
  """Translate lines in file, and save to output file if specified.

  Args:
    model: A Keras model, used to generate the translations.
    params: A dictionary, containing the translation related parameters.
    subtokenizer: A subtokenizer object, used for encoding and decoding source
      and translated lines.
    input_file: A file containing lines to translate.
    output_file: A file that stores the generated translations.
    print_all_translations: A bool. If true, all translations are printed to
      stdout.
    distribution_strategy: A distribution strategy, used to perform inference
      directly with tf.function instead of Keras model.predict().

  Raises:
    ValueError: if output file is invalid.
  """
  batch_size = params["decode_batch_size"]

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
      if distribution_strategy:
        for j in range(batch_size - len(lines)):
          lines.append([tokenizer.EOS_ID])
      batch = tf.keras.preprocessing.sequence.pad_sequences(
          lines,
          maxlen=params["decode_max_length"],
          dtype="int32",
          padding="post")
      logging.info("Decoding batch %d out of %d.", i, num_decode_batches)
      yield batch

  @tf.function
  def predict_step(inputs):
    """Decoding step function for TPU runs."""

    def _step_fn(inputs):
      """Per replica step function."""
      tag = inputs[0]
      val_inputs = inputs[1]
      val_outputs, _ = model([val_inputs], training=False)
      return tag, val_outputs

    return distribution_strategy.run(_step_fn, args=(inputs,))

  translations = []
  if distribution_strategy:
    num_replicas = distribution_strategy.num_replicas_in_sync
    local_batch_size = params["decode_batch_size"] // num_replicas
  for i, text in enumerate(input_generator()):
    if distribution_strategy:
      text = np.reshape(text, [num_replicas, local_batch_size, -1])
      # Add tag to the input of each replica with the reordering logic after
      # outputs, to ensure the output order matches the input order.
      text = tf.constant(text)

      @tf.function
      def text_as_per_replica():
        replica_context = tf.distribute.get_replica_context()
        replica_id = replica_context.replica_id_in_sync_group
        return replica_id, text[replica_id]

      text = distribution_strategy.run(text_as_per_replica)
      outputs = distribution_strategy.experimental_local_results(
          predict_step(text))
      tags, unordered_val_outputs = outputs[0]
      tags = [tag.numpy() for tag in tags._values]
      unordered_val_outputs = [
          val_output.numpy() for val_output in unordered_val_outputs._values]
      # pylint: enable=protected-access
      val_outputs = [None] * len(tags)
      for k in range(len(tags)):
        val_outputs[tags[k]] = unordered_val_outputs[k]
      val_outputs = np.reshape(val_outputs, [params["decode_batch_size"], -1])
    else:
      val_outputs, _ = model.predict(text)

    length = len(val_outputs)
    for j in range(length):
      if j + i * batch_size < total_samples:
        translation = _trim_and_decode(val_outputs[j], subtokenizer)
        translations.append(translation)
        if print_all_translations:
          logging.info("Translating:\n\tInput: %s\n\tOutput: %s",
                       sorted_inputs[j + i * batch_size], translation)

  # Write translations in the order they appeared in the original file.
  if output_file is not None:
    if tf.io.gfile.isdir(output_file):
      raise ValueError("File output is a directory, will not save outputs to "
                       "file.")
    logging.info("Writing to file %s", output_file)
    with tf.io.gfile.GFile(output_file, "w") as f:
      for i in sorted_keys:
        f.write("%s\n" % translations[i])


def translate_from_text(model, subtokenizer, txt):
  encoded_txt = _encode_and_add_eos(txt, subtokenizer)
  result = model.predict(encoded_txt)
  outputs = result["outputs"]
  logging.info("Original: \"%s\"", txt)
  translate_from_input(outputs, subtokenizer)


def translate_from_input(outputs, subtokenizer):
  translation = _trim_and_decode(outputs, subtokenizer)
  logging.info("Translation: \"%s\"", translation)
