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

import os

# pylint: disable=g-bad-import-order
from absl import app as absl_app
from absl import flags
import tensorflow as tf
# pylint: enable=g-bad-import-order

from official.transformer.utils import tokenizer
from official.utils.flags import core as flags_core

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
  with tf.gfile.Open(filename) as f:
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
    estimator, subtokenizer, input_file, output_file=None,
    print_all_translations=True):
  """Translate lines in file, and save to output file if specified.

  Args:
    estimator: tf.Estimator used to generate the translations.
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
  num_decode_batches = (len(sorted_inputs) - 1) // batch_size + 1

  def input_generator():
    """Yield encoded strings from sorted_inputs."""
    for i, line in enumerate(sorted_inputs):
      if i % batch_size == 0:
        batch_num = (i // batch_size) + 1

        tf.logging.info("Decoding batch %d out of %d." %
                        (batch_num, num_decode_batches))
      yield _encode_and_add_eos(line, subtokenizer)

  def input_fn():
    """Created batched dataset of encoded inputs."""
    ds = tf.data.Dataset.from_generator(
        input_generator, tf.int64, tf.TensorShape([None]))
    ds = ds.padded_batch(batch_size, [None])
    return ds

  translations = []
  for i, prediction in enumerate(estimator.predict(input_fn)):
    translation = _trim_and_decode(prediction["outputs"], subtokenizer)
    translations.append(translation)

    if print_all_translations:
      tf.logging.info("Translating:\n\tInput: %s\n\tOutput: %s" %
                      (sorted_inputs[i], translation))

  # Write translations in the order they appeared in the original file.
  if output_file is not None:
    if tf.gfile.IsDirectory(output_file):
      raise ValueError("File output is a directory, will not save outputs to "
                       "file.")
    tf.logging.info("Writing to file %s" % output_file)
    with tf.gfile.Open(output_file, "w") as f:
      for i in sorted_keys:
        f.write("%s\n" % translations[i])


def translate_text(estimator, subtokenizer, txt):
  """Translate a single string."""
  encoded_txt = _encode_and_add_eos(txt, subtokenizer)

  def input_fn():
    ds = tf.data.Dataset.from_tensors(encoded_txt)
    ds = ds.batch(_DECODE_BATCH_SIZE)
    return ds

  predictions = estimator.predict(input_fn)
  translation = next(predictions)["outputs"]
  translation = _trim_and_decode(translation, subtokenizer)
  tf.logging.info("Translation of \"%s\": \"%s\"" % (txt, translation))


def main(unused_argv):
  from official.transformer import transformer_main

  tf.logging.set_verbosity(tf.logging.INFO)

  if FLAGS.text is None and FLAGS.file is None:
    tf.logging.warn("Nothing to translate. Make sure to call this script using "
                    "flags --text or --file.")
    return

  subtokenizer = tokenizer.Subtokenizer(FLAGS.vocab_file)

  # Set up estimator and params
  params = transformer_main.PARAMS_MAP[FLAGS.param_set]
  params["beam_size"] = _BEAM_SIZE
  params["alpha"] = _ALPHA
  params["extra_decode_length"] = _EXTRA_DECODE_LENGTH
  params["batch_size"] = _DECODE_BATCH_SIZE
  estimator = tf.estimator.Estimator(
      model_fn=transformer_main.model_fn, model_dir=FLAGS.model_dir,
      params=params)

  if FLAGS.text is not None:
    tf.logging.info("Translating text: %s" % FLAGS.text)
    translate_text(estimator, subtokenizer, FLAGS.text)

  if FLAGS.file is not None:
    input_file = os.path.abspath(FLAGS.file)
    tf.logging.info("Translating file: %s" % input_file)
    if not tf.gfile.Exists(FLAGS.file):
      raise ValueError("File does not exist: %s" % input_file)

    output_file = None
    if FLAGS.file_out is not None:
      output_file = os.path.abspath(FLAGS.file_out)
      tf.logging.info("File output specified: %s" % output_file)

    translate_file(estimator, subtokenizer, input_file, output_file)


def define_translate_flags():
  """Define flags used for translation script."""
  # Model flags
  flags.DEFINE_string(
      name="model_dir", short_name="md", default="/tmp/transformer_model",
      help=flags_core.help_wrap(
          "Directory containing Transformer model checkpoints."))
  flags.DEFINE_enum(
      name="param_set", short_name="mp", default="big",
      enum_values=["base", "big"],
      help=flags_core.help_wrap(
          "Parameter set to use when creating and training the model. The "
          "parameters define the input shape (batch size and max length), "
          "model configuration (size of embedding, # of hidden layers, etc.), "
          "and various other settings. The big parameter set increases the "
          "default batch size, embedding/hidden size, and filter size. For a "
          "complete list of parameters, please see model/model_params.py."))
  flags.DEFINE_string(
      name="vocab_file", short_name="vf", default=None,
      help=flags_core.help_wrap(
          "Path to subtoken vocabulary file. If data_download.py was used to "
          "download and encode the training data, look in the data_dir to find "
          "the vocab file."))
  flags.mark_flag_as_required("vocab_file")

  flags.DEFINE_string(
      name="text", default=None,
      help=flags_core.help_wrap(
          "Text to translate. Output will be printed to console."))
  flags.DEFINE_string(
      name="file", default=None,
      help=flags_core.help_wrap(
          "File containing text to translate. Translation will be printed to "
          "console and, if --file_out is provided, saved to an output file."))
  flags.DEFINE_string(
      name="file_out", default=None,
      help=flags_core.help_wrap(
          "If --file flag is specified, save translation to this file."))


if __name__ == "__main__":
  define_translate_flags()
  FLAGS = flags.FLAGS
  absl_app.run(main)
