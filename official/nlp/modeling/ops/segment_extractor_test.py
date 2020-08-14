# Copyright 2020 The TensorFlow Authors. All Rights Reserved.
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

# encoding=utf-8
# Lint as: python3
"""Tests for sentence prediction labels."""
import functools

from absl.testing import parameterized
import tensorflow as tf

from official.nlp.modeling.ops import segment_extractor


class NextSentencePredictionTest(parameterized.TestCase, tf.test.TestCase):

  @parameterized.parameters([
      dict(
          test_description="all random",
          sentences=[[b"Hello there.", b"La la la.", b"Such is life."],
                     [b"Who let the dogs out?", b"Who?."]],
          expected_segment=[[
              b"Who let the dogs out?", b"Who?.", b"Who let the dogs out?"
          ], [b"Hello there.", b"Hello there."]],
          expected_labels=[
              [False, False, False],
              [False, False],
          ],
          random_threshold=0.0,
      ),
      dict(
          test_description="all next",
          sentences=[[b"Hello there.", b"La la la.", b"Such is life."],
                     [b"Who let the dogs out?", b"Who?."]],
          expected_segment=[
              [b"La la la.", b"Such is life.", b"Who let the dogs out?"],
              [b"Who?.", b"Hello there."],
          ],
          expected_labels=[
              [True, True, False],
              [True, False],
          ],
          random_threshold=1.0,
      ),
  ])
  def testNextSentencePrediction(self,
                                 sentences,
                                 expected_segment,
                                 expected_labels,
                                 random_threshold=0.5,
                                 test_description=""):
    sentences = tf.ragged.constant(sentences)
    # Set seed and rig the shuffle function to a deterministic reverse function
    # instead. This is so that we have consistent and deterministic results.
    extracted_segment, actual_labels = (
        segment_extractor.get_next_sentence_labels(
            sentences,
            random_threshold,
            random_fn=functools.partial(
                tf.random.stateless_uniform, seed=(2, 3))))
    self.assertAllEqual(expected_segment, extracted_segment)
    self.assertAllEqual(expected_labels, actual_labels)


class SentenceOrderLabelsTest(parameterized.TestCase, tf.test.TestCase):

  @parameterized.parameters([
      dict(
          test_description="all random",
          sentences=[[b"Hello there.", b"La la la.", b"Such is life."],
                     [b"Who let the dogs out?", b"Who?."]],
          expected_segment=[[
              b"Who let the dogs out?", b"Who?.", b"Who let the dogs out?"
          ], [b"Hello there.", b"Hello there."]],
          expected_labels=[[True, True, True], [True, True]],
          random_threshold=0.0,
          random_next_threshold=0.0,
      ),
      dict(
          test_description="all next",
          sentences=[[b"Hello there.", b"La la la.", b"Such is life."],
                     [b"Who let the dogs out?", b"Who?."]],
          expected_segment=[[
              b"La la la.", b"Such is life.", b"Who let the dogs out?"
          ], [b"Who?.", b"Hello there."]],
          expected_labels=[[True, True, True], [True, True]],
          random_threshold=1.0,
          random_next_threshold=0.0,
      ),
      dict(
          test_description="all preceeding",
          sentences=[[b"Hello there.", b"La la la.", b"Such is life."],
                     [b"Who let the dogs out?", b"Who?."]],
          expected_segment=[
              [b"La la la.", b"Hello there.", b"Hello there."],
              [b"Who?.", b"Who let the dogs out?"],
          ],
          expected_labels=[
              [True, False, False],
              [True, False],
          ],
          random_threshold=1.0,
          random_next_threshold=1.0,
      ),
  ])
  def testSentenceOrderPrediction(self,
                                  sentences,
                                  expected_segment,
                                  expected_labels,
                                  random_threshold=0.5,
                                  random_next_threshold=0.5,
                                  test_description=""):
    sentences = tf.ragged.constant(sentences)
    # Set seed and rig the shuffle function to a deterministic reverse function
    # instead. This is so that we have consistent and deterministic results.
    extracted_segment, actual_labels = (
        segment_extractor.get_sentence_order_labels(
            sentences,
            random_threshold=random_threshold,
            random_next_threshold=random_next_threshold,
            random_fn=functools.partial(
                tf.random.stateless_uniform, seed=(2, 3))))
    self.assertAllEqual(expected_segment, extracted_segment)
    self.assertAllEqual(expected_labels, actual_labels)


if __name__ == "__main__":
  tf.test.main()
