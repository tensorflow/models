
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
"""Deep speech decoder."""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from nltk.metrics import distance
from six.moves import xrange
import tensorflow as tf


class DeepSpeechDecoder(object):
  """Basic decoder class from which all other decoders inherit.

  Implements several helper functions. Subclasses should implement the decode()
  method.
  """

  def __init__(self, labels, blank_index=28, space_index=27):
    """Decoder initialization.

    Arguments:
      labels (string): mapping from integers to characters.
      blank_index (int, optional): index for the blank '_' character.
        Defaults to 0.
      space_index (int, optional): index for the space ' ' character.
        Defaults to 28.
    """
    # e.g. labels = "[a-z]' _"
    self.labels = labels
    self.int_to_char = dict([(i, c) for (i, c) in enumerate(labels)])
    self.blank_index = blank_index
    self.space_index = space_index

  def convert_to_strings(self, sequences, sizes=None):
    """Given a list of numeric sequences, returns the corresponding strings."""
    strings = []
    for x in xrange(len(sequences)):
      seq_len = sizes[x] if sizes is not None else len(sequences[x])
      string = self._convert_to_string(sequences[x], seq_len)
      strings.append(string)
    return strings

  def _convert_to_string(self, sequence, sizes):
    return ''.join([self.int_to_char[sequence[i]] for i in range(sizes)])

  def process_strings(self, sequences, remove_repetitions=False):
    """Process strings.

    Given a list of strings, removes blanks and replace space character with
    space. Option to remove repetitions (e.g. 'abbca' -> 'abca').

    Arguments:
      sequences: list of 1-d array of integers
      remove_repetitions (boolean, optional): If true, repeating characters
        are removed. Defaults to False.

    Returns:
      The processed string.
    """
    processed_strings = []
    for sequence in sequences:
      string = self.process_string(remove_repetitions, sequence).strip()
      processed_strings.append(string)
    return processed_strings

  def process_string(self, remove_repetitions, sequence):
    """Process each given sequence."""
    seq_string = ''
    for i, char in enumerate(sequence):
      if char != self.int_to_char[self.blank_index]:
        # if this char is a repetition and remove_repetitions=true,
        # skip.
        if remove_repetitions and i != 0 and char == sequence[i - 1]:
          pass
        elif char == self.labels[self.space_index]:
          seq_string += ' '
        else:
          seq_string += char
    return seq_string

  def wer(self, output, target):
    """Computes the Word Error Rate (WER).

    WER is defined as the edit distance between the two provided sentences after
    tokenizing to words.

    Args:
      output: string of the decoded output.
      target: a string for the true transcript.

    Returns:
      A float number for the WER of the current sentence pair.
    """
    # Map each word to a new char.
    words = set(output.split() + target.split())
    word2char = dict(zip(words, range(len(words))))

    new_output = [chr(word2char[w]) for w in output.split()]
    new_target = [chr(word2char[w]) for w in target.split()]

    return distance.edit_distance(''.join(new_output), ''.join(new_target))

  def cer(self, output, target):
    """Computes the Character Error Rate (CER).

    CER is  defined as the edit distance between the given strings.

    Args:
      output: a string of the decoded output.
      target: a string for the ground truth transcript.

    Returns:
      A float number denoting the CER for the current sentence pair.
    """
    return distance.edit_distance(output, target)

  def batch_wer(self, decoded_output, targets):
    """Compute the aggregate WER for each batch.

    Args:
      decoded_output: 2d array of integers for the decoded output of a batch.
      targets: 2d array of integers for the labels of a batch.

    Returns:
      A float number for the aggregated WER for the current batch output.
    """
    # Convert numeric representation to string.
    decoded_strings = self.convert_to_strings(decoded_output)
    decoded_strings = self.process_strings(
        decoded_strings, remove_repetitions=True)
    target_strings = self.convert_to_strings(targets)
    target_strings = self.process_strings(
        target_strings, remove_repetitions=True)
    wer = 0
    for i in xrange(len(decoded_strings)):
      wer += self.wer(decoded_strings[i], target_strings[i]) / float(
          len(target_strings[i].split()))
    return wer

  def batch_cer(self, decoded_output, targets):
    """Compute the aggregate CER for each batch.

    Args:
      decoded_output: 2d array of integers for the decoded output of a batch.
      targets: 2d array of integers for the labels of a batch.

    Returns:
      A float number for the aggregated CER for the current batch output.
    """
    # Convert numeric representation to string.
    decoded_strings = self.convert_to_strings(decoded_output)
    decoded_strings = self.process_strings(
        decoded_strings, remove_repetitions=True)
    target_strings = self.convert_to_strings(targets)
    target_strings = self.process_strings(
        target_strings, remove_repetitions=True)
    cer = 0
    for i in xrange(len(decoded_strings)):
      cer += self.cer(decoded_strings[i], target_strings[i]) / float(
          len(target_strings[i]))
    return cer

  def decode(self, sequences, sizes=None):
    """Perform sequence decoding.

    Given a matrix of character probabilities, returns the decoder's best guess
    of the transcription.

    Arguments:
      sequences: 2D array of character probabilities, where sequences[c, t] is
        the probability of character c at time t.
      sizes(optional): Size of each sequence in the mini-batch.

    Returns:
      string: sequence of the model's best guess for the transcription.
    """
    strings = self.convert_to_strings(sequences, sizes)
    return self.process_strings(strings, remove_repetitions=True)


class GreedyDecoder(DeepSpeechDecoder):
  """Greedy decoder."""

  def decode(self, logits, seq_len):
    # Reshape to [max_time, batch_size, num_classes]
    logits = tf.transpose(logits, (1, 0, 2))
    decoded, _ = tf.nn.ctc_greedy_decoder(logits, seq_len)
    decoded_dense = tf.Session().run(tf.sparse_to_dense(
        decoded[0].indices, decoded[0].dense_shape, decoded[0].values))
    result = self.convert_to_strings(decoded_dense)
    return self.process_strings(result, remove_repetitions=True), decoded_dense
