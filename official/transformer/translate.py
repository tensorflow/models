# Copyright 2018 MLBenchmark Group. All Rights Reserved.
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

import argparse
import os
import sys

from six.moves import xrange  # pylint: disable=redefined-builtin
import tensorflow as tf

from data_download import VOCAB_FILE
from model import model_params
import transformer_main
from utils import tokenizer

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

  sorted_inputs = []
  sorted_keys = {}
  for i, (index, _) in enumerate(sorted_input_lens):
    sorted_inputs.append(inputs[index])
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

        print("Decoding batch %d out of %d." % (batch_num, num_decode_batches))
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
      print("Translating:")
      print("\tInput: %s" % sorted_inputs[i])
      print("\tOutput: %s\n" % translation)
      print("=" * 100)

  # Write translations in the order they appeared in the original file.
  if output_file is not None:
    if tf.gfile.IsDirectory(output_file):
      raise ValueError("File output is a directory, will not save outputs to "
                       "file.")
    tf.logging.info("Writing to file %s" % output_file)
    with tf.gfile.Open(output_file, "w") as f:
      for index in xrange(len(sorted_keys)):
        f.write("%s\n" % translations[sorted_keys[index]])


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
  print("Translation of \"%s\": \"%s\"" % (txt, translation))


def main(unused_argv):
  tf.logging.set_verbosity(tf.logging.INFO)

  if FLAGS.text is None and FLAGS.file is None:
    tf.logging.warn("Nothing to translate. Make sure to call this script using "
                    "flags --text or --file.")
    return

  subtokenizer = tokenizer.Subtokenizer(
      os.path.join(FLAGS.data_dir, FLAGS.vocab_file))

  if FLAGS.params == "base":
    params = model_params.TransformerBaseParams
  elif FLAGS.params == "big":
    params = model_params.TransformerBigParams
  else:
    raise ValueError("Invalid parameter set defined: %s."
                     "Expected 'base' or 'big.'" % FLAGS.params)

  # Set up estimator and params
  params.beam_size = _BEAM_SIZE
  params.alpha = _ALPHA
  params.extra_decode_length = _EXTRA_DECODE_LENGTH
  params.batch_size = _DECODE_BATCH_SIZE
  estimator = tf.estimator.Estimator(
      model_fn=transformer_main.model_fn, model_dir=FLAGS.model_dir, params=params)

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


if __name__ == "__main__":
  parser = argparse.ArgumentParser()

  # Model arguments
  parser.add_argument(
      "--data_dir", "-dd", type=str, default="/tmp/data/translate_ende",
      help="[default: %(default)s] Directory where vocab file is stored.",
      metavar="<DD>")
  parser.add_argument(
      "--vocab_file", "-vf", type=str, default=VOCAB_FILE,
      help="[default: %(default)s] Name of vocabulary file.",
      metavar="<vf>")
  parser.add_argument(
      "--model_dir", "-md", type=str, default="/tmp/transformer_model",
      help="[default: %(default)s] Directory containing Transformer model "
           "checkpoints.",
      metavar="<MD>")
  parser.add_argument(
      "--params", "-p", type=str, default="big", choices=["base", "big"],
      help="[default: %(default)s] Parameter used for trained model.",
      metavar="<P>")

  # Flags for specifying text/file to be translated.
  parser.add_argument(
      "--text", "-t", type=str, default=None,
      help="[default: %(default)s] Text to translate. Output will be printed "
           "to console.",
      metavar="<T>")
  parser.add_argument(
      "--file", "-f", type=str, default=None,
      help="[default: %(default)s] File containing text to translate. "
           "Translation will be printed to console and, if --file_out is "
           "provided, saved to an output file.",
      metavar="<F>")
  parser.add_argument(
      "--file_out", "-fo", type=str, default=None,
      help="[default: %(default)s] If --file flag is specified, save "
           "translation to this file.",
      metavar="<FO>")

  FLAGS, unparsed = parser.parse_known_args()
  main(sys.argv)
