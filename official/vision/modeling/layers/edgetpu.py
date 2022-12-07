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

"""EdgeTPU oriented layers and tools."""
from typing import List, Optional, Sequence, Union, Iterable

import numpy as np
import tensorflow as tf

_or = tf.maximum
_and = tf.minimum
_reduce_or = tf.reduce_max


def _tensor_sum_vectors(a, b):
  a = tf.tile(tf.reshape(a, [1, -1, 1, a.shape[-1]]), [1, 1, a.shape[-1], 1])
  b = tf.tile(tf.reshape(b, [1, -1, a.shape[-1], 1]), [1, 1, 1, a.shape[-1]])
  return a + b


def _tensor_product_iou(boxes):
  """Computes pairwise IOU.

  Reason to use 4-D tensors is to follow TPU compiler preference.

  Args:
    boxes: A 2-D float `Tensor` of shape `[num_boxes, 4]`.

  Returns:
    A 4-D float `Tensor` of shape `[1, 1, num_boxes, num_boxes]` containing
    pairwise IOU.
  """
  boxes_size = boxes.shape[-2]
  # Code below will do frequent operands broadcasting.
  # TPU compiler has (empirically) less issues broadcasting if
  # - batch (first) dimension is 1. (Special consideration sharding)
  # - there are 4 dimensions. (Standard traversal mapping)
  # - last dimension is not 1. (Structure alignment)
  tpu_friendly_shape = [1, -1, 1, boxes_size]
  bottom, left, top, right = (
      tf.reshape(side, tpu_friendly_shape) for side in tf.split(boxes, 4, -1))
  height, width = top - bottom, right - left
  area = height * width
  area_sum = _tensor_sum_vectors(area, area)
  bottom_pad, left_pad, top_pad, right_pad = (
      tf.nn.relu(_tensor_sum_vectors(x, -x))
      for x in (-bottom, -left, top, right))
  height_pad, width_pad = bottom_pad + top_pad, left_pad + right_pad
  intersection = tf.nn.relu(height - height_pad) * tf.nn.relu(width - width_pad)
  union = area_sum - intersection
  iou = tf.math.divide(intersection, union + _same(union))
  return iou


def _greater(x):
  """Avoid non lowerable layers in boolean comparison.

  Logical operation results in tensor of boolean type. However in serving such
  a tensors cannot be cast to values because of NNAPI specs.
  `tf.where` operation result in `select` instruction lowering, which not runs
  well on all generations of edge-tpus.

  Args:
    x: any numeric tensor.

  Returns:
    tf.where(x > tf.zero_like(x), tf.one_like(x), tf.zero_like(x))
  """
  x_clip = tf.minimum(tf.nn.relu(x), tf.constant(1, dtype=x.dtype))
  return -tf.math.floor(-x_clip)


def _same(x):
  """Avoid non lowerable layers in boolean equality.

  Logical operation results in tensor of boolean type. However in serving such
  a tensors cannot be cast to values because of NNAPI specs.
  `tf.where` operation result in `select` instruction lowering, which not runs
  well on all generations of edge-tpus.

  Args:
    x: any numeric tensor.

  Returns:
    tf.where(x == tf.zero_like(x), tf.one_like(x), tf.zero_like(x))
  """
  x_clip = tf.minimum(tf.abs(x), tf.constant(1, dtype=x.dtype))
  return tf.constant(1, dtype=x.dtype) + tf.math.floor(-x_clip)


def shard_tensors(
    axis: int, block_size: int, tensors: Sequence[tf.Tensor]
) -> Union[List[Sequence[tf.Tensor]], 'Iterable[Sequence[tf.Tensor]]']:
  """Consistently splits multiple tensors sharding-style.

  Args:
    axis: axis to be used to split tensors
    block_size: block size to split tensors.
    tensors: list of tensors.

  Returns:
    List of shards, each shard has exactly one peace of each input tesnor.

  Raises:
    ValueError: if input tensors has different size of sharded dimension.
  """
  for validate_axis in range(axis + 1):
    consistent_length: int = tensors[0].shape[validate_axis]
    for tensor in tensors:
      if tensor.shape[validate_axis] != consistent_length:
        raise ValueError('Inconsistent shapes in shard_tensors: first is '
                         f'{tensors[0].shape} and other is {tensor.shape}')
  batch_size: int = tensors[0].shape[axis]
  if block_size >= batch_size:
    return [tensors]
  else:
    blocks = batch_size // block_size
    remainder = batch_size % block_size
    if remainder:
      tensor_parts = []
      for tensor in tensors:
        shape: tf.TensorShape = tensor.shape
        body: tf.Tensor = tf.slice(tensor, [0] * len(shape), [
            size if i != axis else blocks * block_size
            for i, size in enumerate(shape)
        ])
        tail: tf.Tensor = tf.slice(tensor, [
            0 if i != axis else (blocks * block_size)
            for i, _ in enumerate(shape)
        ], [
            size if i != axis else (size - blocks * block_size)
            for i, size in enumerate(shape)
        ])
        tensor_parts.append(tf.split(body, blocks, axis) + [tail])
      return zip(*tensor_parts)
    else:
      return zip(*[tf.split(tensor, blocks, axis) for tensor in tensors])


# TODO(b/258007436): Number is based on existing compiler limitations while
# running bf16 NMS on edgetpu. Remove manual sharing when compiler issue will be
# fixed.
_RECOMMENDED_NMS_MEMORY = 360000


def non_max_suppression_padded(boxes: tf.Tensor,
                               scores: tf.Tensor,
                               output_size: int,
                               iou_threshold: float = 0.5,
                               refinements: int = 0) -> tf.Tensor:
  """Selects a subset of boxes which have highest score among IOU-similar boxes.

  Prunes away boxes that have high intersection-over-union (IOU) overlap
  with boxes having higher score. Boxes are supplied as `[y1, x1, y2, x2]`,
  where `(y1, x1)` and `(y2, x2)` are the coordinates of any diagonal pair of
  box corners. Note that this algorithm is agnostic to the coordinate system.
  Thus translating or reflections of the coordinate system result in the same
  boxes being selected by the algorithm. The output of this operation is a
  set of integers indexing into the input collection of bounding boxes
  representing the selected boxes.

  Set will be returned padded on the right with `-1` values. The bounding
  box coordinates corresponding to the selected indices can then be obtained
  using the `tf.gather` operation.  For example:
    ```python
    selected_indices = vision.modeling.layers.non_max_suppression_padded(
        boxes, scores, max_output_size, iou_threshold)
    selected_boxes = tf.gather(boxes, selected_indices)
    ```

  See following documetation for implementation details.
  third_party/tensorflow_models/official/projects/edgetpu/vision/modeling/g3doc/non_max_suppression.md

  Args:
    boxes: A 2-D+ float `Tensor` of shape `[...batch_dims, num_boxes, 4]`.
    scores: A 1-D+ float `Tensor` of shape `[...batch_dims, num_boxes]`
      representing a single score corresponding to each box (each row of boxes).
    output_size: A scalar integer `Tensor` representing the maximum number of
      boxes to be selected by non-max suppression.
    iou_threshold: A float representing the threshold for deciding whether boxes
      overlap too much with respect to IOU.
    refinements: A number of extra refinement steps to make result closer to
      original sequential NMS.

  Returns:
    A 1-D+ integer `Tensor` of shape `[...batch_dims, output_size]` representing
    the selected indices from the boxes tensor and `-1` values for the padding.
  """
  # Does partitioning job to help compiler converge with memory.
  batch_shape = boxes.shape[:-2]
  batch_size = np.prod(batch_shape, dtype=np.int32)
  boxes_size, struct_size = boxes.shape[-2:]
  boxes = tf.reshape(boxes, [batch_size, boxes_size, struct_size])
  scores = tf.reshape(scores, [batch_size, boxes_size])
  block = max(1, _RECOMMENDED_NMS_MEMORY // (boxes_size * boxes_size))
  indices = []
  for boxes_i, scores_i in shard_tensors(0, block, (boxes, scores)):
    indices.append(
        _non_max_suppression_as_is(boxes_i, scores_i, output_size,
                                   iou_threshold, refinements))
  indices = tf.concat(indices, axis=0)
  return tf.reshape(indices, batch_shape + [output_size])


def _refine_nms_graph_to_original_algorithm(better: tf.Tensor) -> tf.Tensor:
  """Refines the relationship graph, bringing it closer to the iterative NMS.

  See `test_refinement_sample` unit tests for example, also comments in body of
  the algorithm, for the intuition.

  Args:
    better: is a tensor with zeros and ones so that
      [batch dims ..., box_1, box_2] represents the
      [adjacency matrix](https://en.wikipedia.org/wiki/Adjacency_matrix)
      for the [relation](https://en.wikipedia.org/wiki/Relation_(mathematics))
      `better` between boxes box_1 and box_2.

  Returns:
    Modification of tensor encoding adjacency matrix of `better` relation.
  """
  # good_box: is a tensor with zeros and ones so that
  # [batch dims ..., box_i] represents belonging of a box_i to the `good`
  # subset. `good` subset is defined as exactly those boxes that do not have any
  # `better` boxes.
  # INTUITION: In terms of oriented graph , this is subset of nodes nobody
  # points to as "I'm better than you". These nodes will never be suppressed in
  # the original NMS algorithm.
  good_box = tf.constant(1.) - _reduce_or(better, axis=-1)
  # good_better: is a tensor with zeros and ones so that
  # [batch dims ..., box_1, box_2] represents the adjacency matrix for the
  # `good_better` relation on all boxes set. `good_better` relation is defined
  # as relation between good box and boxes it is better than.
  # INTUITION: In terms of oriented graph, this is subset of edges, which
  # doesn't have any other inbound edges. These edges will represent
  # suppression actions in the original NMS algorithm.
  good_better = _and(tf.expand_dims(good_box, axis=-2), better)
  # not_bad_box: is a tensor with zeros and ones so that
  # [batch dims ..., box_i] represents belonging of a box_i to the `not_bad`
  # subset. `not_bad` subset is defined as boxes all that and only those that
  # does not have any `good_better` boxes.
  # INTUITION: These nodes are nodes which are not suppressed by `good` boxes
  # in the original NMS algorithm.
  not_bad_box = tf.constant(1.) - _reduce_or(good_better, axis=-1)
  # return: is a tensor with zeros and ones so that
  # [batch dims ..., box_1, box_2] represents the adjacency matrix for the
  # `better` relation on all boxes set which is closer to represent suppression
  # procedure in original NMS algorithm.
  return _and(tf.expand_dims(not_bad_box, axis=-2), better)


def _non_max_suppression_as_is(boxes: tf.Tensor,
                               scores: tf.Tensor,
                               output_size: int,
                               iou_threshold: float = 0.5,
                               refinements: int = 0) -> tf.Tensor:
  """Selects a subset of boxes which have highest score among IOU-similar boxes.

  Args:
    boxes: A 2-D+ float `Tensor` of shape `[...batch_dims, num_boxes, 4]`.
    scores: A 1-D+ float `Tensor` of shape `[...batch_dims, num_boxes]`
      representing a single score corresponding to each box (each row of boxes).
    output_size: A scalar integer `Tensor` representing the maximum number of
      boxes to be selected by non-max suppression.
    iou_threshold: A 0-D float tensor representing the threshold for deciding
      whether boxes overlap too much with respect to IOU.
    refinements: A number of extra refinement steps to make result closer to
      original sequencial NMS.

  Returns:
    A 1-D+ integer `Tensor` of shape `[...batch_dims, output_size]` representing
    the selected indices from the boxes tensor and `-1` values for the padding.
  """
  batch_shape = boxes.shape[:-2]
  batch_size = np.prod(batch_shape, dtype=np.int32)
  boxes_size = boxes.shape[-2]
  if boxes.shape[-1] != 4:
    raise ValueError(f'Boxes shape ({boxes.shape}) last dimension must be 4 '
                     'to represent [y1, x1, y2, x2] boxes coordinates')
  if scores.shape != boxes.shape[:-1]:
    raise ValueError(f'Boxes shape ({boxes.shape}) and scores shape '
                     f'({scores.shape}) do not match.')
  order = tf.range(boxes_size, dtype=tf.float32)
  relative_order = _tensor_sum_vectors(order, -order)
  relative_scores = _tensor_sum_vectors(scores, -scores)
  similar = _greater(_tensor_product_iou(boxes) - iou_threshold)
  worse = _greater(relative_scores)
  same_later = _and(_same(relative_scores), _greater(relative_order))
  similar_worse_or_same_later = _and(similar, _or(worse, same_later))
  for _ in range(refinements):
    similar_worse_or_same_later = _refine_nms_graph_to_original_algorithm(
        similar_worse_or_same_later)
  prunable = _reduce_or(similar_worse_or_same_later, axis=-1)
  remaining = tf.constant(1.) - prunable
  scores = tf.reshape(tf.exp(scores), [1, 1, batch_size, boxes_size])
  remaining = tf.reshape(remaining, [1, 1, batch_size, boxes_size])
  # top_k runs on TPU cores, let it happen, TPU tiles implementation is slower.
  top_k = tf.math.top_k(scores * remaining, output_size)
  indices = (
      tf.cast(top_k.indices, top_k.values.dtype) * _greater(top_k.values) -
      _same(top_k.values))
  return tf.reshape(indices, batch_shape + [output_size])


def concat_and_top_k(
    top_k: int, scores_pair: 'tuple[Optional[tf.Tensor], tf.Tensor]',
    *other_pairs: 'tuple[Optional[tf.Tensor], tf.Tensor]'
) -> 'tuple[tf.Tensor, ...]':
  """Combines shards of top_k operation, when sharded along filtered dimension.

  General idea is that sometimes top_k dimension is very large, while top_k is
  moderately low. (Keep in mind sample of 15K pre-top_k dimension and 150 top_k)
  In that case it is possible to break top_k input into groups significantly
  larger than top_k and significatly lower than pre-top_l (Keep in mind 1500).
  We do top_k over first 1500 elements, than join 150 remaining with new 1500
  elements (1750 in total), repeat top_k. This function provides repeatedly used
  method which will concat and top_k in that case.

  For example with top_k = 2 and scores_pair = ([10, 6], [9, 8, 7]), output
  scores will be [10, 9].

  Other pairs are filtered using indexes generated from scores. This is a preaty
  common case of filtering structure by its score.

  For example with one extra pair of box per score:
  top_k = 2
  scores_pair =  ([10,             6],
                  [9,              8,            7])
  other_pairs = [([[0, 0, 10, 10], [0, 0, 6, 6]],
                  [[1, 1, 9, 9],   [1, 1, 8, 8], [1, 1, 7, 7]])]
  Output is:
  ([10, 9], [[0, 0, 10, 10], [1, 1, 9, 9]])

  See also 'test_top_k_sharded_fusion' unit test with end to end example.

  Args:
    top_k: is top_k argument of sharded tf.math.top_k.
    scores_pair: Tuple (<previous shards combination>, <additional shard>)
      scores to be aggregated using top_k.
    *other_pairs: Tuples (<previous shards combination>, <additional shard>)
      other values to be aggregated using indexes of top_k scores.

  Returns:
    Tuple of scores based top_k aggregations with additional shards.
  """
  scores, scores_shard = scores_pair
  if other_pairs:
    others, others_shard = zip(*other_pairs)
  else:
    others = others_shard = []
  # Same as tf.rank, but avoiding tensor form for graph mode execution.
  top_k_dim: int = len(scores_shard.shape) - 1
  if scores is None:
    # First shard becomes aggregation
    scores = scores_shard
    others = others_shard
  else:
    # Merge shard into aggregation
    scores = tf.concat([scores, scores_shard], top_k_dim)
    others = [
        tf.concat([other, other_shard], top_k_dim)
        for other, other_shard in zip(others, others_shard)
    ]
  # When shards are uneven some will be smaller than requested top_k
  if scores.shape[top_k_dim] > top_k:
    scores, indices = tf.nn.top_k(scores, top_k)
    others = [
        tf.gather(other, indices, axis=top_k_dim, batch_dims=top_k_dim)
        for other in others
    ]
  return scores, *others
