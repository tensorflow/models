# Copyright 2016 The TensorFlow Authors. All Rights Reserved.
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
"""Unit tests for CaptionGenerator."""

import math



import numpy as np
import tensorflow as tf

from im2txt.inference_utils import caption_generator


class FakeVocab(object):
  """Fake Vocabulary for testing purposes."""

  def __init__(self):
    self.start_id = 0  # Word id denoting sentence start.
    self.end_id = 1  # Word id denoting sentence end.


class FakeModel(object):
  """Fake model for testing purposes."""

  def __init__(self):
    # Number of words in the vocab.
    self._vocab_size = 12

    # Dimensionality of the nominal model state.
    self._state_size = 1

    # Map of previous word to the probability distribution of the next word.
    self._probabilities = {
        0: {1: 0.1,
            2: 0.2,
            3: 0.3,
            4: 0.4},
        2: {5: 0.1,
            6: 0.9},
        3: {1: 0.1,
            7: 0.4,
            8: 0.5},
        4: {1: 0.3,
            9: 0.3,
            10: 0.4},
        5: {1: 1.0},
        6: {1: 1.0},
        7: {1: 1.0},
        8: {1: 1.0},
        9: {1: 0.5,
            11: 0.5},
        10: {1: 1.0},
        11: {1: 1.0},
    }

  # pylint: disable=unused-argument

  def feed_image(self, sess, encoded_image):
    # Return a nominal model state.
    return np.zeros([1, self._state_size])

  def inference_step(self, sess, input_feed, state_feed):
    # Compute the matrix of softmax distributions for the next batch of words.
    batch_size = input_feed.shape[0]
    softmax_output = np.zeros([batch_size, self._vocab_size])
    for batch_index, word_id in enumerate(input_feed):
      for next_word, probability in self._probabilities[word_id].items():
        softmax_output[batch_index, next_word] = probability

    # Nominal state and metadata.
    new_state = np.zeros([batch_size, self._state_size])
    metadata = None

    return softmax_output, new_state, metadata

  # pylint: enable=unused-argument


class CaptionGeneratorTest(tf.test.TestCase):

  def _assertExpectedCaptions(self,
                              expected_captions,
                              beam_size=3,
                              max_caption_length=20,
                              length_normalization_factor=0):
    """Tests that beam search generates the expected captions.

    Args:
      expected_captions: A sequence of pairs (sentence, probability), where
        sentence is a list of integer ids and probability is a float in [0, 1].
      beam_size: Parameter passed to beam_search().
      max_caption_length: Parameter passed to beam_search().
      length_normalization_factor: Parameter passed to beam_search().
    """
    expected_sentences = [c[0] for c in expected_captions]
    expected_probabilities = [c[1] for c in expected_captions]

    # Generate captions.
    generator = caption_generator.CaptionGenerator(
        model=FakeModel(),
        vocab=FakeVocab(),
        beam_size=beam_size,
        max_caption_length=max_caption_length,
        length_normalization_factor=length_normalization_factor)
    actual_captions = generator.beam_search(sess=None, encoded_image=None)

    actual_sentences = [c.sentence for c in actual_captions]
    actual_probabilities = [math.exp(c.logprob) for c in actual_captions]

    self.assertEqual(expected_sentences, actual_sentences)
    self.assertAllClose(expected_probabilities, actual_probabilities)

  def testBeamSize(self):
    # Beam size = 1.
    expected = [([0, 4, 10, 1], 0.16)]
    self._assertExpectedCaptions(expected, beam_size=1)

    # Beam size = 2.
    expected = [([0, 4, 10, 1], 0.16), ([0, 3, 8, 1], 0.15)]
    self._assertExpectedCaptions(expected, beam_size=2)

    # Beam size = 3.
    expected = [
        ([0, 2, 6, 1], 0.18), ([0, 4, 10, 1], 0.16), ([0, 3, 8, 1], 0.15)
    ]
    self._assertExpectedCaptions(expected, beam_size=3)

  def testMaxLength(self):
    # Max length = 1.
    expected = [([0], 1.0)]
    self._assertExpectedCaptions(expected, max_caption_length=1)

    # Max length = 2.
    # There are no complete sentences, so partial sentences are returned.
    expected = [([0, 4], 0.4), ([0, 3], 0.3), ([0, 2], 0.2)]
    self._assertExpectedCaptions(expected, max_caption_length=2)

    # Max length = 3.
    # There is at least one complete sentence, so only complete sentences are
    # returned.
    expected = [([0, 4, 1], 0.12), ([0, 3, 1], 0.03)]
    self._assertExpectedCaptions(expected, max_caption_length=3)

    # Max length = 4.
    expected = [
        ([0, 2, 6, 1], 0.18), ([0, 4, 10, 1], 0.16), ([0, 3, 8, 1], 0.15)
    ]
    self._assertExpectedCaptions(expected, max_caption_length=4)

  def testLengthNormalization(self):
    # Length normalization factor = 3.
    # The longest caption is returned first, despite having low probability,
    # because it has the highest log(probability)/length**3.
    expected = [
        ([0, 4, 9, 11, 1], 0.06),
        ([0, 2, 6, 1], 0.18),
        ([0, 4, 10, 1], 0.16),
        ([0, 3, 8, 1], 0.15),
    ]
    self._assertExpectedCaptions(
        expected, beam_size=4, length_normalization_factor=3)


if __name__ == '__main__':
  tf.test.main()
