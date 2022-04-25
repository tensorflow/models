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

"""Tests for anchor_generator.py."""

from absl.testing import parameterized
import tensorflow as tf
from official.vision.ops import anchor_generator


class AnchorGeneratorTest(parameterized.TestCase, tf.test.TestCase):

  @parameterized.parameters(
      # Single scale anchor.
      (5, [1.0], [[[-16., -16., 48., 48.], [-16., 16., 48., 80.]],
                  [[16., -16., 80., 48.], [16., 16., 80., 80.]]]),
      # # Multi aspect ratio anchor.
      (6, [1.0, 4.0, 0.25],
       [[[-32., -32., 96., 96., 0., -96., 64., 160., -96., 0., 160., 64.]]]),
  )
  def testAnchorGeneration(self, level, aspect_ratios, expected_boxes):
    image_size = [64, 64]
    anchor_size = 2**(level + 1)
    stride = 2**level
    anchor_gen = anchor_generator._SingleAnchorGenerator(
        anchor_size=anchor_size,
        scales=[1.],
        aspect_ratios=aspect_ratios,
        stride=stride,
        clip_boxes=False)
    anchors = anchor_gen(image_size).numpy()
    self.assertAllClose(expected_boxes, anchors)

  @parameterized.parameters(
      # Single scale anchor.
      (5, [1.0], [[[0., 0., 48., 48.], [0., 16., 48., 64.]],
                  [[16., 0., 64., 48.], [16., 16., 64., 64.]]]),
      # # Multi aspect ratio anchor.
      (6, [1.0, 4.0, 0.25
          ], [[[0., 0., 64., 64., 0., 0., 64., 64., 0., 0., 64., 64.]]]),
  )
  def testAnchorGenerationClipped(self, level, aspect_ratios, expected_boxes):
    image_size = [64, 64]
    anchor_size = 2**(level + 1)
    stride = 2**level
    anchor_gen = anchor_generator._SingleAnchorGenerator(
        anchor_size=anchor_size,
        scales=[1.],
        aspect_ratios=aspect_ratios,
        stride=stride,
        clip_boxes=True)
    anchors = anchor_gen(image_size).numpy()
    self.assertAllClose(expected_boxes, anchors)


class MultiScaleAnchorGeneratorTest(parameterized.TestCase, tf.test.TestCase):

  @parameterized.parameters(
      # Multi scale anchor.
      (5, 6, [[1.0], [1.0]], [[-16, -16, 48, 48], [-16, 16, 48, 80],
                              [16, -16, 80, 48], [16, 16, 80, 80],
                              [-32, -32, 96, 96]]),)
  def testAnchorGeneration(self, min_level, max_level, aspect_ratios,
                           expected_boxes):
    image_size = [64, 64]
    levels = range(min_level, max_level + 1)
    anchor_sizes = [2**(level + 1) for level in levels]
    strides = [2**level for level in levels]
    anchor_gen = anchor_generator.AnchorGenerator(
        anchor_sizes=anchor_sizes,
        scales=[1.],
        aspect_ratios=aspect_ratios,
        strides=strides)
    anchors = anchor_gen(image_size)
    anchors = [tf.reshape(anchor, [-1, 4]) for anchor in anchors]
    anchors = tf.concat(anchors, axis=0).numpy()
    self.assertAllClose(expected_boxes, anchors)

  @parameterized.parameters(
      # Multi scale anchor.
      (5, 6, [[1.0], [1.0]], [[-16, -16, 48, 48], [-16, 16, 48, 80],
                              [16, -16, 80, 48], [16, 16, 80, 80],
                              [-32, -32, 96, 96]]),)
  def testAnchorGenerationClipped(self, min_level, max_level, aspect_ratios,
                                  expected_boxes):
    image_size = [64, 64]
    levels = range(min_level, max_level + 1)
    anchor_sizes = [2**(level + 1) for level in levels]
    strides = [2**level for level in levels]
    anchor_gen = anchor_generator.AnchorGenerator(
        anchor_sizes=anchor_sizes,
        scales=[1.],
        aspect_ratios=aspect_ratios,
        strides=strides,
        clip_boxes=False)
    anchors = anchor_gen(image_size)
    anchors = [tf.reshape(anchor, [-1, 4]) for anchor in anchors]
    anchors = tf.concat(anchors, axis=0).numpy()
    self.assertAllClose(expected_boxes, anchors)

  @parameterized.parameters(
      # Multi scale anchor.
      (5, 6, [1.0], {
          '5': [[[-16., -16., 48., 48.], [-16., 16., 48., 80.]],
                [[16., -16., 80., 48.], [16., 16., 80., 80.]]],
          '6': [[[-32, -32, 96, 96]]]
      }),)
  def testAnchorGenerationDict(self, min_level, max_level, aspect_ratios,
                               expected_boxes):
    image_size = [64, 64]
    levels = range(min_level, max_level + 1)
    anchor_sizes = dict((str(level), 2**(level + 1)) for level in levels)
    strides = dict((str(level), 2**level) for level in levels)
    anchor_gen = anchor_generator.AnchorGenerator(
        anchor_sizes=anchor_sizes,
        scales=[1.],
        aspect_ratios=aspect_ratios,
        strides=strides,
        clip_boxes=False)
    anchors = anchor_gen(image_size)
    for k in expected_boxes.keys():
      self.assertAllClose(expected_boxes[k], anchors[k].numpy())


if __name__ == '__main__':
  tf.test.main()
