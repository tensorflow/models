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

"""Tests EdgeTPU oriented layers and tools."""

from typing import Optional

from absl.testing import parameterized
import numpy as np
import tensorflow as tf, tf_keras
from official.vision.modeling.layers import edgetpu


def random_boxes(shape):
  a = tf.random.uniform(shape=shape+[2])
  b = tf.random.uniform(shape=shape+[2])
  l = tf.minimum(a, b)
  u = tf.maximum(a, b)
  return tf.concat([l, u], axis=-1)


def _maximum_activation_size(model):
  max_size = 0
  for layer in model.layers:
    outputs = layer.output
    if not isinstance(outputs, list):
      outputs = [outputs]
    for output in outputs:
      if hasattr(output, 'shape'):
        size = np.prod(output.shape)
        max_size = max(max_size, size)
  return max_size


def _deviation_and_margin(reference, valid, optimized):
  """Returns deviation and margin between two batched sets of indices."""
  deviation_rate = 0
  min_union = reference.shape[1] + optimized.shape[1]
  runs = reference.shape[0]
  for run in range(runs):
    reference_slice = {*reference[run, :valid[run]].numpy().tolist()}
    optimized_slice = {*optimized[run].numpy().astype(int).tolist()} - {-1}
    union_size = len(optimized_slice | reference_slice)
    symdiff_size = len(optimized_slice ^ reference_slice)
    deviation_rate += symdiff_size / union_size
    min_union = min(min_union, union_size)
  deviation_rate = deviation_rate / runs
  # six sigma estimate via LLN theorem
  margin = 6 * (deviation_rate / np.sqrt(runs) + 1 / (runs * min_union))
  return deviation_rate, margin


class NonMaxSuppressionTest(parameterized.TestCase, tf.test.TestCase):

  def setUp(self):
    super().setUp()
    tf.random.set_seed(42)

  def test_refinement_sample(self):
    """Tests difference in NMS behaviours.

    Runs on four boxes with following IOU table (only neighbours will qualify
    as similar boxes)

    box | 0    | 1    | 2    | 3
    --- | ---- | ---- | ---- | ----
    0   | 1    | 7/13 | 1/4  | 1/19
    1   | 7/13 | 1    | 7/13 | 1/4
    2   | 1/4  | 7/13 | 1    | 7/13
    3   | 1/19 | 1/4  | 7/13 | 1

    So 0 is best box, it eliminates 1, next is box 2 which is eleminated by 1
    if it is allowed (depending on number of refinements).
    """
    boxes: tf.Tensor = tf.constant(
        [
            # y1,  x1,  y2,  x2
            [0.0, 0.0, 1.0, 1.0],
            [0.0, 0.3, 1.0, 1.3],
            [0.0, 0.6, 1.0, 1.6],
            [0.0, 0.9, 1.0, 1.9],
        ],
        dtype=tf.float32)
    scores: tf.Tensor = tf.constant([
        1.0,
        0.9,
        0.8,
        0.7,
    ], dtype=tf.float32)
    self.assertAllEqual(
        edgetpu.non_max_suppression_padded(boxes, scores, 4, refinements=0),
        tf.constant([0.0, -1.0, -1.0, -1.0], dtype=tf.float32))
    self.assertAllEqual(
        edgetpu.non_max_suppression_padded(boxes, scores, 4, refinements=1),
        tf.constant([0.0, 2.0, -1.0, -1.0], dtype=tf.float32))

  @parameterized.parameters((16, 8, 200, [0.009, 0.004, 0.004]),
                            (31, 17, 100, [0.013, 0.004, 0.004]),
                            (71, 41, 100, [0.045, 0.003, 0.002]),
                            (150, 100, 100, [0.129, 0.010, 0.001]),
                            (300, 300, 100, [0.116, 0.016, 0.002]),
                            (600, 600, 50, [0.176, 0.032, 0.003]))
  def test_reference_match(self, n, top, runs, max_devs):
    """Compares that new optimized method is close to reference method.

    Runs two algorithms with same sets of input boxes and scores, and measures
    deviation between returned sets of prunned boxes.
    Read more about test results at ./g3doc/non_max_suppression.md
    (*) Avoid flakiness with safe boundary (go/python-tips/048): deviation
    between two sets is a positive number, which may vary from test to test.
    Doing multiple runs expected to reduce average deviation variation following
    LLN theorem. Therefore by having first test run we know upper deviation
    bound which algorithm would not exceed until broken (in any feasible amount
    of time in the future). Use of this safe boundary makes test non-flaky.

    Args:
      n: number of boxes and scores on input of the algorithm.
      top: limit of output boxes count.
      runs: for the statistical testing number of runs to performs to avoid
        tests flakiness.
      max_devs: series of mean limits on deviation between optimized and
        reference algorithms with different number of refinements. (Indexes of
        elements correspond to number of refinements) Please use margin based
        values proposed by failed test to avoid flaky testing.
    """
    boxes = random_boxes([runs, n])
    scores = tf.random.uniform(shape=[runs, n])
    reference, valid = tf.image.non_max_suppression_padded(
        boxes, scores, top, pad_to_max_output_size=True)
    for refinements, max_deviation in enumerate(max_devs):
      optimized = edgetpu.non_max_suppression_padded(
          boxes, scores, top, refinements=refinements)
      deviation, margin = _deviation_and_margin(reference, valid, optimized)
      self.assertLess(
          deviation,
          max_deviation,
          msg='Deviation rate between optimized and reference implementations is '
          'higher than expected. If you are tuning the test, recommended safe '
          'deviation rate is '
          f'{deviation} + {margin} = {deviation + margin}')

  @parameterized.parameters(([16], 8), ([91, 150], 100), ([20, 20, 200], 10))
  def test_sharded_match(self, shape: list[int], top: int):
    boxes = random_boxes(shape)
    scores = tf.random.uniform(shape=shape)
    optimized = edgetpu.non_max_suppression_padded(boxes, scores, top)
    reference = edgetpu._non_max_suppression_as_is(boxes, scores, top)
    self.assertAllEqual(optimized, reference)

  _sharded_nms = edgetpu.non_max_suppression_padded
  _stright_nms = edgetpu._non_max_suppression_as_is

  @parameterized.parameters(([16], 8, _sharded_nms, True),
                            ([16], 8, _stright_nms, True),
                            ([91, 150], 100, _sharded_nms, True),
                            ([91, 150], 100, _stright_nms, False),
                            ([20, 20, 200], 10, _sharded_nms, True),
                            ([20, 20, 200], 10, _stright_nms, False))
  def test_sharded_size(self, shape: list[int], top: int, algorithm,
                        fits_as_is: bool):
    scores = tf_keras.Input(shape=shape, batch_size=1)
    boxes = tf_keras.Input(shape=shape + [4], batch_size=1)
    optimized = algorithm(boxes, scores, top)
    model = tf_keras.Model(inputs=[boxes, scores], outputs=optimized)
    max_size = _maximum_activation_size(model)
    if fits_as_is:
      # Sharding done or not needed.
      self.assertLessEqual(max_size, edgetpu._RECOMMENDED_NMS_MEMORY)
    else:
      # Sharding needed.
      self.assertGreater(max_size, edgetpu._RECOMMENDED_NMS_MEMORY)

  def test_shard_tensors(self):
    a: tf.Tensor = tf.constant([[0, 1, 2, 3, 4]])
    b: tf.Tensor = tf.constant([[
        [0, 1, 2, 3, 4],
        [5, 6, 7, 8, 9],
        [10, 11, 12, 13, 14],
        [15, 16, 17, 18, 19],
        [20, 21, 22, 23, 24],
    ]])
    for i, (a_i, b_i) in enumerate(edgetpu.shard_tensors(1, 3, (a, b))):
      self.assertAllEqual(a_i, a[:, i * 3:i * 3 + 3])
      self.assertAllEqual(b_i, b[:, i * 3:i * 3 + 3, :])

  def test_top_k_sharded_fusion_arguments_validation(self):
    # Input scores is not pair of aggregation and shard.
    self.assertRaises(ValueError, edgetpu.concat_and_top_k, 100,
                      tf.zeros(shape=[1000]))
    # Input other values is not pairs of aggregation and shard.
    self.assertRaises(TypeError, edgetpu.concat_and_top_k, 100,
                      (None, tf.zeros(shape=[1000])), None,
                      tf.zeros(shape=[1000]))
    # Insufficient rank to do top_k
    self.assertRaises(IndexError, edgetpu.concat_and_top_k, 100,
                      (None, tf.constant(1.)))

  @parameterized.parameters(0, 1, 2)
  def test_top_k_sharded_fusion_vs_top_k_unsharded(self, axis: int):
    r"""Tests `horizontal` sharding using shard_tensors and concat_and_top_k.

    Will generate and test graph (on diagram 4 shards, in test 6 shards):
    Input
    -----
       |
    +-------+--------------------------------------------
    | Split |-----------------------                     \
    +-------+---                    \                     |
        |       \                    |                    |
    +-------+ +--------+ +-------+ +--------+ +-------+ +--------+ +-------+
    | top k |-| concat |-| top k |-| concat |-| top k |-| concat |-| top k |
    +-------+ +--------+ +-------+ +--------+ +-------+ +--------+ +-------+
                                                                      |
                                                                    Output
                                                                    ------

    Args:
      axis: test top_k axis (tensor rank will be axis + 1)
    """
    sample: tf.Tensor = tf.random.uniform(
        shape=axis * [1] + [10000], dtype=tf.float32)
    top_1000_direct: tf.Tensor = tf.math.top_k(sample, 1000).values
    top_1000_sharded: Optional[tf.Tensor] = None
    for (piece,) in edgetpu.shard_tensors(axis, 1500, (sample,)):
      (top_1000_sharded,) = edgetpu.concat_and_top_k(
          1000, (top_1000_sharded, piece))
    self.assertAllEqual(top_1000_direct, top_1000_sharded)

if __name__ == '__main__':
  tf.test.main()
