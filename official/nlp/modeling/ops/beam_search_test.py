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

"""Test beam search helper methods."""

from absl.testing import parameterized
import tensorflow as tf, tf_keras

from official.nlp.modeling.ops import beam_search


class BeamSearchTests(tf.test.TestCase, parameterized.TestCase):

  def test_expand_to_beam_size(self):
    x = tf.ones([7, 4, 2, 5])
    x = beam_search.expand_to_beam_size(x, 3)
    shape = tf.shape(x)
    self.assertAllEqual([7, 3, 4, 2, 5], shape)

  def test_get_shape_keep_last_dim(self):
    y = tf.constant(4.0)
    x = tf.ones([7, tf.cast(tf.sqrt(y), tf.int32), 2, 5])
    shape = beam_search._get_shape_keep_last_dim(x)
    self.assertAllEqual([None, None, None, 5], shape.as_list())

  def test_flatten_beam_dim(self):
    x = tf.ones([7, 4, 2, 5])
    x = beam_search.flatten_beam_dim(x)
    self.assertAllEqual([28, 2, 5], tf.shape(x))

  def test_unflatten_beam_dim(self):
    x = tf.ones([28, 2, 5])
    x = beam_search._unflatten_beam_dim(x, 7, 4)
    self.assertAllEqual([7, 4, 2, 5], tf.shape(x))

  def test_gather_beams(self):
    x = tf.reshape(tf.range(24), [2, 3, 4])
    # x looks like:  [[[ 0  1  2  3]
    #                  [ 4  5  6  7]
    #                  [ 8  9 10 11]]
    #
    #                 [[12 13 14 15]
    #                  [16 17 18 19]
    #                  [20 21 22 23]]]

    y = beam_search.SequenceBeamSearch._gather_beams(x, [[1, 2], [0, 2]], 2, 2)
    self.assertAllEqual(
        [[[4, 5, 6, 7], [8, 9, 10, 11]], [[12, 13, 14, 15], [20, 21, 22, 23]]],
        y)

  @parameterized.named_parameters([
      ('padded_decode_true_with_name', True, 0.0, 'decoding'),
      ('padded_decode_false_with_name', False, 0.0, 'decoding'),
      ('padded_decode_true_without_name', True, 0.0, None),
      ('padded_decode_false_without_name', False, 0.0, None),
      ('padded_decode_false_with_noise', False, 0.5, 'decoding'),
  ])
  def test_sequence_beam_search(self, padded_decode, noise_multiplier, name):
    # batch_size*beam_size, max_decode_length, vocab_size
    probabilities = tf.constant([[[0.2, 0.7, 0.1], [0.5, 0.3, 0.2],
                                  [0.1, 0.8, 0.1]],
                                 [[0.1, 0.8, 0.1], [0.3, 0.4, 0.3],
                                  [0.2, 0.1, 0.7]]])
    # batch_size, max_decode_length, num_heads, embed_size per head
    x = tf.zeros([1, 3, 2, 32], dtype=tf.float32)
    cache = {'layer_%d' % layer: {'k': x, 'v': x} for layer in range(2)}

    def _get_test_symbols_to_logits_fn():
      """Test function that returns logits for next token."""

      def symbols_to_logits_fn(_, i, cache):
        logits = tf.cast(probabilities[:, i, :], tf.float32)
        return logits, cache
      return symbols_to_logits_fn

    predictions, _ = beam_search.sequence_beam_search(
        symbols_to_logits_fn=_get_test_symbols_to_logits_fn(),
        initial_ids=tf.zeros([1], dtype=tf.int32),
        initial_cache=cache,
        vocab_size=3,
        beam_size=2,
        alpha=0.6,
        max_decode_length=3,
        eos_id=9,
        padded_decode=padded_decode,
        dtype=tf.float32,
        noise_multiplier=noise_multiplier,
        decoding_name=name,
    )
    if noise_multiplier > 0:
      self.assertAllEqual([[[0, 1, 0, 1], [0, 0, 2, 2]]], predictions)
    else:
      self.assertAllEqual([[[0, 1, 0, 1], [0, 1, 1, 2]]], predictions)

  @parameterized.named_parameters([
      ('padded_decode_true_with_name', True, 0.0, 'decoding'),
      ('padded_decode_false_with_name', False, 0.0, 'decoding'),
      ('padded_decode_true_without_name', True, 0.0, None),
      ('padded_decode_false_without_name', False, 0.0, None),
      ('padded_decode_false_with_noise', False, 0.5, 'decoding'),
  ])
  def test_sequence_beam_search_multi_eos(
      self, padded_decode, noise_multiplier, name
  ):
    # batch_size*beam_size, max_decode_length, vocab_size
    probabilities = tf.constant([
        [[0.2, 0.7, 0.1], [0.5, 0.3, 0.2], [0.1, 0.8, 0.1]],
        [[0.1, 0.8, 0.1], [0.3, 0.4, 0.3], [0.2, 0.1, 0.7]],
    ])
    # batch_size, max_decode_length, num_heads, embed_size per head
    x = tf.zeros([1, 3, 2, 32], dtype=tf.float32)
    cache = {'layer_%d' % layer: {'k': x, 'v': x} for layer in range(2)}

    def _get_test_symbols_to_logits_fn():
      """Test function that returns logits for next token."""

      def symbols_to_logits_fn(_, i, cache):
        logits = tf.cast(probabilities[:, i, :], tf.float32)
        return logits, cache

      return symbols_to_logits_fn

    predictions, _ = beam_search.sequence_beam_search(
        symbols_to_logits_fn=_get_test_symbols_to_logits_fn(),
        initial_ids=tf.zeros([1], dtype=tf.int32),
        initial_cache=cache,
        vocab_size=3,
        beam_size=2,
        alpha=0.6,
        max_decode_length=3,
        eos_id=[9, 10],
        padded_decode=padded_decode,
        dtype=tf.float32,
        noise_multiplier=noise_multiplier,
        decoding_name=name,
    )
    if noise_multiplier > 0:
      self.assertAllEqual([[[0, 1, 0, 1], [0, 0, 2, 2]]], predictions)
    else:
      self.assertAllEqual([[[0, 1, 0, 1], [0, 1, 1, 2]]], predictions)


if __name__ == '__main__':
  tf.test.main()
