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

"""Pix2Seq required utility library."""
import copy

import tensorflow as tf, tf_keras
from official.projects.pix2seq.configs import pix2seq as pix2seq_cfg


def decode_object_seq_to_bbox(
    logits, pred_seq, quantization_bins, coord_vocab_shift
):
  """Decode objects (label & bbox) for seq from `build_response_seq_from_bbox`.

  Assume yxyxc format with truncation at the end for any uneven extra tokens.

      Replace class tokens with argmax instead of sampling.
  Args:
      logits: `float` output logits in shape of (bsz, max_seq_len, vocab_size).
      pred_seq: `int` pred sequence in shape of (bsz, max_seq_len).
      quantization_bins: `int` for bins.
      coord_vocab_shift: `int`, shifting coordinates by a specified integer.

  Returns:
      pred_class: `int` of shape (bsz, max_instances_per_image).
      pred_bbox: `float` of shape (bsz, max_instances_per_image, 4).
      pred_score: `float` of shape (bsz, max_instances_per_image).
  """
  _, seqlen, vocab_size = logits.shape

  if seqlen % 5 != 0:  # truncate out the last few tokens.
    pred_seq = pred_seq[..., : -(seqlen % 5)]
    logits = logits[..., : -(seqlen % 5), :]
  pred_class_p = tf.nn.softmax(logits)[:, 4::5]  # (bsz, instances, vocab_size)
  mask_s1 = [0.0] * (pix2seq_cfg.BASE_VOCAB_SHIFT)  # reserved.
  mask_s2 = [1.0] * (
      coord_vocab_shift - pix2seq_cfg.BASE_VOCAB_SHIFT
  )  # labels.
  mask_s3 = [0.0] * (vocab_size - coord_vocab_shift)  # coordinates and others.
  mask = tf.constant(mask_s1 + mask_s2 + mask_s3)
  pred_class = tf.argmax(pred_class_p * mask[tf.newaxis, tf.newaxis, :], -1)

  pred_num = logits[:, 4::5] * mask[tf.newaxis, tf.newaxis, :]
  pred_num = tf.reduce_sum(
      tf.cast(
          tf.math.greater(tf.math.reduce_max(pred_num, axis=-1), 0), tf.int32
      ),
      axis=-1,
  )

  pred_score = tf.reduce_sum(
      pred_class_p * tf.one_hot(pred_class, vocab_size), -1
  )
  pred_class = tf.maximum(pred_class - pix2seq_cfg.BASE_VOCAB_SHIFT, 0)
  pred_bbox = seq_to_bbox(pred_seq - coord_vocab_shift, quantization_bins)
  return pred_class, pred_bbox, pred_score, pred_num


def seq_to_bbox(seq, quantization_bins, seq_format='yxyx_name'):
  """Returns [0, 1] normalized yxyx bbox from token sequence."""
  # [batch, 5*num_instances]
  assert seq.shape.rank == 2, seq.shape.as_list()
  # [batch, num_instances, 1]
  if seq_format.startswith('name'):
    ymin = tf.expand_dims(seq[:, 1::5], -1)
    xmin = tf.expand_dims(seq[:, 2::5], -1)
    ymax = tf.expand_dims(seq[:, 3::5], -1)
    xmax = tf.expand_dims(seq[:, 4::5], -1)
  else:
    ymin = tf.expand_dims(seq[:, 0::5], -1)
    xmin = tf.expand_dims(seq[:, 1::5], -1)
    ymax = tf.expand_dims(seq[:, 2::5], -1)
    xmax = tf.expand_dims(seq[:, 3::5], -1)
  if seq_format in ['name_cycxhw', 'cycxhw_name']:
    ycnt, xcnt, ysize, xsize = ymin, xmin, ymax, xmax
    ymin = ycnt - ysize // 2
    xmin = xcnt - xsize // 2
    ymax = ycnt + ysize // 2
    xmax = xcnt + xsize // 2
  quantized_box = tf.concat([ymin, xmin, ymax, xmax], axis=-1)
  quantized_box = dequantize(quantized_box, quantization_bins)
  return tf.minimum(tf.maximum(quantized_box, 0), 1)


def quantize(coordinates, bins):
  """Quantization of (normalized) coordinates in [0, 1]."""
  coordinates = tf.cast(tf.round(coordinates * (bins - 1)), tf.int64)
  coordinates = tf.clip_by_value(coordinates, 0, bins - 1)
  return coordinates


def dequantize(boxes, bins):
  """Dequantization of discrete tokens of coordinates in [0, bins-1]."""
  boxes = tf.cast(boxes, tf.float32)
  boxes = boxes / (bins - 1)
  return boxes


def truncation_bbox(bbox):
  return tf.minimum(tf.maximum(bbox, 0.0), 1.0)


def jitter_bbox(bbox, min_range=0.0, max_range=0.05, truncation=True):
  """Jitter the bbox.

  Args:
    bbox: `float` tensor of shape (n, 4), ranged between 0 and 1.
    min_range: min jitter range in ratio to bbox size.
    max_range: max jitter range in ratio to bbox size.
    truncation: whether to truncate resulting bbox to remain [0, 1].
  Note: To create noisy positives, set min_range=0, which enables truncated
    normal distribution. max_range <=0.05: noisy duplicates, <=0.02: near
    duplicate. To create negatives: set min_range >= 0.1 to avoid false
    negatives; suggested max_range <=0.4 to avoid too much randomness.

  Returns:
    jittered bbox.
  """
  n = tf.shape(bbox)[0]
  h = bbox[:, 2] - bbox[:, 0]
  w = bbox[:, 3] - bbox[:, 1]
  noise = tf.stack([h, w, h, w], -1)
  if min_range == 0:
    noise_rate = tf.random.truncated_normal(
        [n, 4], mean=0, stddev=max_range / 2.0, dtype=bbox.dtype
    )
  else:
    noise_rate1 = tf.random.uniform([n, 4], min_range, max_range)
    noise_rate2 = tf.random.uniform([n, 4], -max_range, -min_range)
    selector = tf.cast(tf.random.uniform([n, 4], 0, 1) < 0.5, tf.float32)
    noise_rate = noise_rate1 * selector + noise_rate2 * (1.0 - selector)
  bbox = bbox + noise * noise_rate
  return truncation_bbox(bbox) if truncation else bbox


def shift_bbox(bbox, truncation=True):
  """Shifting bbox without changing the bbox height and width."""
  n = tf.shape(bbox)[0]
  # randomly sample new bbox centers.
  cy = tf.random.uniform([n, 1], 0, 1)
  cx = tf.random.uniform([n, 1], 0, 1)
  h = bbox[:, 2:3] - bbox[:, 0:1]
  w = bbox[:, 3:4] - bbox[:, 1:2]
  bbox = tf.concat(
      [
          cy - tf.abs(h) / 2,
          cx - tf.abs(w) / 2,
          cy + tf.abs(h) / 2,
          cx + tf.abs(w) / 2,
      ],
      -1,
  )
  return truncation_bbox(bbox) if truncation else bbox


def random_bbox(n, max_size=1.0, truncation=True):
  """Generating random n bbox with max size specified within [0, 1]."""
  cy = tf.random.uniform([n, 1], 0, 1)
  cx = tf.random.uniform([n, 1], 0, 1)
  h = tf.random.truncated_normal([n, 1], 0, max_size / 2.0)
  w = tf.random.truncated_normal([n, 1], 0, max_size / 2.0)
  bbox = tf.concat(
      [
          cy - tf.abs(h) / 2,
          cx - tf.abs(w) / 2,
          cy + tf.abs(h) / 2,
          cx + tf.abs(w) / 2,
      ],
      -1,
  )
  return truncation_bbox(bbox) if truncation else bbox


def augment_bbox(bbox, bbox_label, max_jitter, n_noise_bbox, mix_rate=0.0):
  """Augment bbox.

  There are two types of noises to add:

    1. Bad bbox: jittered bbox, shifted bbox, or random bbox.
    2. Duplicated bbox.
  Args:
    bbox: `float` tensor of shape (n, 4), ranged between 0 and 1.
    bbox_label: `int` tensor of shape (n,).
    max_jitter: `float` scalar specifying max jitter range for positive bbox.
    n_noise_bbox: `int` scalar tensor specifying size of the extra noise to add.
    mix_rate: `float`. Probability of injecting the bad bbox in the middle of
      original bbox, followed by dup bbox at the end; otherwise simply append
      all noises at the end of original bbox.

  Returns:
    bbox_new: augmented bbox that's `n_noise_bbox` larger than original.
    label_new: new label for bbox_new.
    is_real: a `float` 0/1 indicator for whether a bbox is real.
    is_noise: a `float` 0/1 indicator for whether a bbox is extra.
  """
  n = tf.shape(bbox)[0]
  dup_bbox_size = tf.random.uniform([], 0, n_noise_bbox + 1, dtype=tf.int32)
  dup_bbox_size = 0 if n == 0 else dup_bbox_size
  bad_bbox_size = n_noise_bbox - dup_bbox_size
  multiplier = 1 if n == 0 else tf.math.floordiv(n_noise_bbox, n) + 1
  bbox_tiled = tf.tile(bbox, [multiplier, 1])

  # Create bad bbox.
  bbox_tiled = tf.random.shuffle(bbox_tiled)
  bad_bbox_shift = shift_bbox(bbox_tiled[:bad_bbox_size], truncation=True)
  bad_bbox_random = random_bbox(bad_bbox_size, max_size=1.0, truncation=True)
  bad_bbox = tf.concat([bad_bbox_shift, bad_bbox_random], 0)
  bad_bbox = tf.random.shuffle(bad_bbox)[:bad_bbox_size]
  bad_bbox_label = tf.zeros([bad_bbox_size], dtype=bbox_label.dtype) + (
      pix2seq_cfg.FAKE_CLASS_TOKEN - pix2seq_cfg.BASE_VOCAB_SHIFT
  )

  # Create dup bbox.
  bbox_tiled = tf.random.shuffle(bbox_tiled)
  dup_bbox = jitter_bbox(
      bbox_tiled[:dup_bbox_size], min_range=0, max_range=0.1, truncation=True
  )
  dup_bbox_label = tf.zeros([dup_bbox_size], dtype=bbox_label.dtype) + (
      pix2seq_cfg.FAKE_CLASS_TOKEN - pix2seq_cfg.BASE_VOCAB_SHIFT
  )

  # Jitter positive bbox.
  if max_jitter > 0:
    bbox = jitter_bbox(bbox, min_range=0, max_range=max_jitter, truncation=True)

  if tf.random.uniform([]) < mix_rate:
    # Mix the bbox with bad bbox, appneded by dup bbox.
    bbox_new = tf.concat([bbox, bad_bbox], 0)
    bbox_new_label = tf.concat([bbox_label, bad_bbox_label], 0)
    idx = tf.random.shuffle(tf.range(tf.shape(bbox_new)[0]))
    bbox_new = tf.gather(bbox_new, idx)
    bbox_new_label = tf.gather(bbox_new_label, idx)
    bbox_new = tf.concat([bbox_new, dup_bbox], 0)
    bbox_new_label = tf.concat([bbox_new_label, dup_bbox_label], 0)
  else:
    # Merge bad bbox and dup bbox into noise bbox.
    noise_bbox = tf.concat([bad_bbox, dup_bbox], 0)
    noise_bbox_label = tf.concat([bad_bbox_label, dup_bbox_label], 0)

    if n_noise_bbox > 0:
      idx = tf.random.shuffle(tf.range(n_noise_bbox))
      noise_bbox = tf.gather(noise_bbox, idx)
      noise_bbox_label = tf.gather(noise_bbox_label, idx)

    # Append noise bbox to bbox and create mask.
    bbox_new = tf.concat([bbox, noise_bbox], 0)
    bbox_new_label = tf.concat([bbox_label, noise_bbox_label], 0)

  return bbox_new, bbox_new_label


def inject_noise_bbox(boxes, classes, max_instances_per_image):
  boxes = copy.copy(boxes)
  classes = copy.copy(classes)
  num_instances = tf.shape(boxes)[0]
  if num_instances < max_instances_per_image:
    n_noise_bbox = max_instances_per_image - num_instances
    boxes, classes = augment_bbox(boxes, classes, 0.0, n_noise_bbox)
  return boxes, classes


def build_prompt_seq_from_task_id(
    task_vocab_id: int, response_seq=None, prompt_shape=None
):
  """Build prompt seq just using task id.

  Args:
    task_vocab_id: Vocab id for the task.
    response_seq: an (optional) discerte target sequen with shape (bsz, ..., k).
    prompt_shape: an (optional) tuple for prompt shape. One and only one of
      `response_seq` and `prompt_shape` should be specified.

  Returns:
    discrete input sequence of task id with shape (bsz, ..., 1).
  """
  task_id = tf.constant(task_vocab_id)
  if response_seq is not None:
    prompt_seq = tf.zeros_like(response_seq[..., :1]) + tf.cast(
        task_id, response_seq.dtype
    )
  if prompt_shape is not None:
    assert response_seq is None, 'double specification'
    prompt_seq = tf.zeros(prompt_shape, dtype=tf.int64) + tf.cast(
        task_id, dtype=tf.int64
    )
  return prompt_seq


def clip_or_pad_to_max_len(data, max_len, dim):
  """Pad the data tensor to max length on dim."""
  shape = shape_as_list(data)
  padding_shape, clipped_shape = copy.copy(shape), copy.copy(shape)
  padding_shape[dim] = tf.maximum(0, max_len - padding_shape[dim])
  clipped_shape[dim] = tf.minimum(clipped_shape[dim], max_len)

  paddings = tf.zeros(padding_shape, dtype=data.dtype)
  clipped_data = tf.slice(data, tf.zeros_like(shape), clipped_shape)
  return tf.concat([clipped_data, paddings], axis=dim)


def shape_as_list(t):
  # Assumes rank of `t` is statically known.
  shape = t.shape.as_list()
  dynamic_shape = tf.shape(t)
  return [
      shape[i] if shape[i] is not None else dynamic_shape[i]
      for i in range(len(shape))
  ]


def reorder_object_instances(boxes, classes, order):
  """Must be called _before_ padding to max instances."""
  if order == 'none':
    return classes, boxes

  assert boxes.shape.rank == 2, 'Must be unbatched'
  boxes = tf.reshape(boxes, [-1, 2, 2])

  if order == 'random':
    idx = tf.random.shuffle(tf.range(tf.shape(boxes)[0]))
  elif order == 'area':
    areas = tf.cast(
        tf.reduce_prod(boxes[:, 1, :] - boxes[:, 0, :], axis=1), tf.int64
    )  # approximated size.
    idx = tf.argsort(areas, direction='DESCENDING')
  elif order == 'dist2ori':
    y, x = boxes[:, 0], boxes[:, 1]  # using top-left corner.
    dist2ori = tf.square(y) + tf.square(x)
    idx = tf.argsort(dist2ori, direction='ASCENDING')
  else:
    raise ValueError('Unknown order {}'.format(order))

  boxes = tf.reshape(boxes, [-1, 4])
  boxes = tf.gather(boxes, idx)
  classes = tf.gather(classes, idx)

  return boxes, classes


def scale_points(points, scale):
  """Scales points.

  Args:
    points: Tensor with shape [num_points * 2], [batch, num_points * 2] or
      [batch, instances, num_points * 2] where points are organized in (y, x)
      format.
    scale: Tensor with shape [2] or [batch, 2].

  Returns:
    Tensor with same shape as points.
  """
  points_orig = points
  orig_shape = tf.shape(points)
  coords_len = points.shape[-1]
  if points.shape.rank == 1:
    points = tf.reshape(points, [coords_len // 2, 2])
  elif points.shape.rank == 2:
    points = tf.reshape(points, [-1, coords_len // 2, 2])
  else:
    points = tf.reshape(points, [-1, orig_shape[1], coords_len // 2, 2])
    scale = tf.expand_dims(scale, -2)
  points = points * scale
  points = tf.reshape(points, orig_shape)
  points = preserve_reserved_tokens(points, points_orig)
  return points


def preserve_reserved_tokens(points, points_orig):
  """Preserve reserved tokens in points according to points_orig."""
  return replace_reserved_tokens(
      points, points_orig, dict(zip(pix2seq_cfg.FLOATS, pix2seq_cfg.FLOATS))
  )


def replace_reserved_tokens(seq, ref_seq, replacements):
  for key, replacement in replacements.items():
    seq = tf.where(
        tf.equal(ref_seq, key), tf.constant(replacement, seq.dtype), seq
    )
  return seq


def tf_float32(t):
  return tf.cast(t, tf.float32)
