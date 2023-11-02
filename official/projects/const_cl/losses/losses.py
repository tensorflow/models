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

"""The losses for ConST-CL."""

from typing import Mapping

import tensorflow as tf, tf_keras

from tensorflow.compiler.tf2xla.python import xla  # pylint: disable=g-direct-tensorflow-import
from official.projects.video_ssl.losses import losses as video_ssl_losses

tpu_cross_replica_concat = video_ssl_losses.tpu_cross_replica_concat


_LARGE_NUM = 1e9


class ContrastiveLoss(object):
  """InfoNCE loss.

  Reference: Oord et al. "Representation learning with contrastive
    predictive coding" NeurIPS 2019.
  """

  def __init__(self,
               normalize_inputs: bool,
               temperature: float):
    """Computes contrastive loss.

    Args:
      normalize_inputs: whether or not to l2 normalize the inputs vector.
      temperature: temperature in the InfoNCE contrastive loss.
    """
    self._normalize_inputs = normalize_inputs
    self._temperature = temperature

  def __call__(self,
               inputs: tf.Tensor,
               num_replicas: int = 1) -> Mapping[str, tf.Tensor]:
    """Calculates the loss.

    Args:
      inputs: the embeddings (in shape [2*B, C]) from video clips after the
        projection head.
      num_replicas: the number of TPU replicas.

    Returns:
      a dictionary contains calculated loss and statistics.
    """
    inputs1, inputs2 = tf.split(inputs, num_or_size_splits=2, axis=0)
    if self._normalize_inputs:
      inputs1 = tf.math.l2_normalize(inputs1, -1)
      inputs2 = tf.math.l2_normalize(inputs2, -1)
    batch_size = tf.shape(inputs1)[0]

    if num_replicas == 1:
      # This is the local version.
      inputs1_large = inputs1
      inputs2_large = inputs2
      labels = tf.one_hot(tf.range(batch_size), batch_size * 2)
      masks = tf.one_hot(tf.range(batch_size), batch_size)
    else:
      # This is the cross-tpu version.
      inputs1_large = tpu_cross_replica_concat(inputs1, num_replicas)
      inputs2_large = tpu_cross_replica_concat(inputs2, num_replicas)
      enlarged_batch_size = tf.shape(inputs1_large)[0]
      replica_id = tf.cast(tf.cast(xla.replica_id(), tf.uint32), tf.int32)
      labels_idx = tf.range(batch_size) + replica_id * batch_size
      labels = tf.one_hot(labels_idx, enlarged_batch_size * 2)
      masks = tf.one_hot(labels_idx, enlarged_batch_size)

    logits_aa = tf.matmul(
        inputs1, inputs1_large, transpose_b=True) / self._temperature
    logits_aa = logits_aa - tf.cast(masks, logits_aa.dtype) * _LARGE_NUM
    logits_bb = tf.matmul(
        inputs2, inputs2_large, transpose_b=True) / self._temperature
    logits_bb = logits_bb - tf.cast(masks, logits_bb.dtype) * _LARGE_NUM
    logits_ab = tf.matmul(
        inputs1, inputs2_large, transpose_b=True) / self._temperature
    logits_ba = tf.matmul(
        inputs2, inputs1_large, transpose_b=True) / self._temperature

    loss_a = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(
        labels, tf.concat([logits_ab, logits_aa], 1)))
    loss_b = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(
        labels, tf.concat([logits_ba, logits_bb], 1)))
    loss = loss_a + loss_b

    contrast_prob = tf.nn.softmax(logits_ab)
    contrast_entropy = - tf.reduce_mean(
        tf.reduce_sum(contrast_prob * tf.math.log(contrast_prob + 1e-8), -1))

    contrast_acc = tf.equal(tf.argmax(labels, 1), tf.argmax(logits_ab, axis=1))
    contrast_acc = tf.reduce_mean(tf.cast(contrast_acc, tf.float32))

    return {
        'loss': loss,
        'contrastive_accuracy': contrast_acc,
        'contrastive_entropy': contrast_entropy,
    }


class InstanceContrastiveLoss(object):
  """Instance Contrastive Loss.

  Reference: Yuan et al. "Contextualized Spatio-Temporal Contrastive Learning
    with Self-Supervision" CVPR 2022.
  """

  def __init__(self,
               normalize_inputs: bool,
               temperature: float):
    self._normalize_inputs = normalize_inputs
    self._temperature = temperature

  def __call__(self,
               predictions: Mapping[str, tf.Tensor],
               num_replicas: int = 1) -> Mapping[str, tf.Tensor]:
    """Computes contrastive loss for spatio-temporal instance embeddings.

    Args:
      predictions: a dictionary of the model outputs, contains
        'instances_a2b': the reconstructed instance features from view a -> b.
          In shape [B, N, C].
        'instances_b2a': the reconstructed instance features from view b -> a.
          In shape [B, N, C].
        'instances_a': the target instance features in view a. In shape
          [B, N, C].
        'instances_b': the target instance features in view b. In shape
          [B, N, C].
        'masks_a': the vaidity boolean mask for instances in view a. In shape
          [B, N].
        'masks_b': the vaidity boolean mask for instances in view b. In shape
          [B, N].
      num_replicas: the number of TPU replicas.

    Returns:
      A loss scalar.
      The staticstics for positive examples.
      The staticstics for negative examples.
    """

    inst_a2b = predictions['instances_a2b']
    inst_b2a = predictions['instances_b2a']
    inst_a = predictions['instances_a']
    inst_b = predictions['instances_b']
    masks_a = tf.cast(predictions['masks_a'][..., None], dtype=inst_a.dtype)
    masks_b = tf.cast(predictions['masks_b'][..., None], dtype=inst_b.dtype)

    if self._normalize_inputs:
      inst_a2b = tf.math.l2_normalize(inst_a2b, axis=-1)
      inst_b2a = tf.math.l2_normalize(inst_b2a, axis=-1)
      inst_a = tf.math.l2_normalize(inst_a, axis=-1)
      inst_b = tf.math.l2_normalize(inst_b, axis=-1)

    b, n = inst_a.shape.as_list()[:2]
    batch_index = tf.range(b)

    # Computes similarity based on raw features in view a and b.
    similarity_ab = tf.einsum('ijc,ikc->ijk', inst_a, inst_b)

    # Loss on translated_a2b.
    similarity_ab_index = tf.argmax(similarity_ab, axis=2, output_type=tf.int32)
    lookup_a2b_index = tf.stack(
        [tf.tile(batch_index[:, None], [1, n]), similarity_ab_index], axis=-1)
    loss_and_stats_a = self._compute_constrastive_loss(
        positive_lookup_index=lookup_a2b_index,
        inst_translated=inst_a2b,
        inst_target=inst_b,
        inst_mask=masks_a,
        num_replicas=num_replicas)

    # Loss on translated_b2a.
    similarity_ba_index = tf.argmax(similarity_ab, axis=1, output_type=tf.int32)
    lookup_b2a_index = tf.stack(
        [tf.tile(batch_index[:, None], [1, n]), similarity_ba_index], axis=-1)
    loss_and_stats_b = self._compute_constrastive_loss(
        positive_lookup_index=lookup_b2a_index,
        inst_translated=inst_b2a,
        inst_target=inst_a,
        inst_mask=masks_b,
        num_replicas=num_replicas)

    loss_and_stats = {}
    for key in loss_and_stats_a:
      loss_and_stats[key] = 0.5 * (
          loss_and_stats_a[key] + loss_and_stats_b[key])
    return loss_and_stats

  def _get_negative_similarity_statistics(
      self,
      logits: tf.Tensor,
      batch_masks: tf.Tensor,
      inst_mask: tf.Tensor) -> Mapping[str, tf.Tensor]:
    """Gets negative examples similarity statistics.

    Args:
      logits: the logits matrix.
      batch_masks: the batch validity mask.
      inst_mask: the instance validity mask.

    Returns:
      logs: a dictionary of logs.
    """
    # logits = [b, n, bl, n]
    # batch_masks = [b, n, bl, n]
    # inst_mask = [b, n, 1]
    inst_mask = tf.cast(inst_mask, logits.dtype)
    batch_masks = tf.cast(batch_masks, logits.dtype)
    batch_masks = tf.ones_like(batch_masks) - batch_masks
    masks = batch_masks * inst_mask[..., None]
    # Recover the raw similarity and mask self-similarity, which will be
    # removed from negative samples.
    similarity = logits * masks * self._temperature
    similarity_mean = tf.reduce_sum(similarity) / tf.reduce_sum(masks)

    similarity_masks = tf.squeeze(inst_mask, axis=-1)
    similarity_max = similarity - (1.0 - masks) * _LARGE_NUM
    similarity_max = tf.reduce_max(similarity_max, axis=[-1, -2])
    similarity_max = tf.reduce_sum(
        similarity_max * similarity_masks) / tf.reduce_sum(similarity_masks)

    similarity_min = similarity + (1.0 - masks) * _LARGE_NUM
    similarity_min = tf.reduce_min(similarity_min, axis=[-1, -2])
    similarity_min = tf.reduce_sum(
        similarity_min * similarity_masks) / tf.reduce_sum(similarity_masks)
    logs = {
        'negative_similarity_mean': similarity_mean,
        'negative_similarity_min': similarity_min,
        'negative_similarity_max': similarity_max,
    }
    return logs

  def _get_positive_similarity_statistics(
      self,
      logits: tf.Tensor,
      inst_mask: tf.Tensor) -> Mapping[str, tf.Tensor]:
    """Gets positive examples similarity statistics.

    Args:
      logits: the logits matrix.
      inst_mask: the instance validity mask.

    Returns:
      logs: a dictionary of logs.
    """
    # logits in shape [b, n]
    # inst_mask in shape [b, n, 1]
    inst_mask = tf.squeeze(inst_mask, axis=-1)
    inst_mask = tf.cast(inst_mask, dtype=logits.dtype)
    similarity = logits * inst_mask * self._temperature

    num_instances = tf.reduce_sum(inst_mask)
    similarity_mean = tf.reduce_sum(similarity) / num_instances

    similarity_max = similarity - (1.0 - inst_mask) * _LARGE_NUM
    similarity_max = tf.reduce_max(similarity_max)

    similarity_min = similarity + (1.0 - inst_mask) * _LARGE_NUM
    similarity_min = tf.reduce_min(similarity_min)

    logs = {
        'positive_similarity_mean': similarity_mean,
        'positive_similarity_min': similarity_min,
        'positive_similarity_max': similarity_max,
    }
    return logs

  def _compute_constrastive_loss(
      self,
      positive_lookup_index: tf.Tensor,
      inst_translated: tf.Tensor,
      inst_target: tf.Tensor,
      inst_mask: tf.Tensor,
      num_replicas: int = 1) -> Mapping[str, tf.Tensor]:
    """Computes constrastive loss.

    Args:
      positive_lookup_index: the index tensor to look-up the corresponding
        features in inst_target. In shape [B, N].
      inst_translated: a float tensor of shape [B, N, C] of translated instance
        features by the transformer head.
      inst_target: a float tensor of shape [B, N, C] of instance features on the
        target domain. Note that the order of inst_target is not necessarily
        matched to inst_translated.
      inst_mask: a boolean tensor of shape [B, N, 1] suggesting valid instances
        in inst_translated.
      num_replicas: the number of TPU replicas.

    Returns:
      loss_and_stats: a dictionary of loss and intermediate statistics.
    """
    b, n = inst_translated.shape.as_list()[:2]

    if num_replicas == 1:
      inst_target_large = inst_target
      b_large = tf.shape(inst_target_large)[0]
      labels_idx = tf.range(b)
    else:
      inst_target_large = tpu_cross_replica_concat(
          inst_target,
          num_replicas)
      b_large = tf.shape(inst_target_large)[0]
      # NOTE: make sure to use xla.replica_id() here and in
      # tpu_cross_replica_concat to consistently align the replica_id.
      # replicator.replica_id != xla.replica_id()
      replica_id = tf.cast(tf.cast(xla.replica_id(), tf.uint32), tf.int32)
      labels_idx = tf.range(b) + replica_id * b

    # [B, BL], 1 indicates positive batches.
    batch_masks = tf.one_hot(labels_idx, b_large)
    # [B, N, BL, N]
    batch_masks = tf.tile(batch_masks[:, None, :, None], [1, n, 1, n])

    # Construct negative examples.
    logits_negative = tf.einsum(
        'ijc,pqc->ijpq',
        inst_translated, inst_target_large) / self._temperature
    # Get negative statistics.
    negative_stats = self._get_negative_similarity_statistics(
        logits_negative, batch_masks, inst_mask)
    logits_negative = logits_negative - tf.cast(
        batch_masks, logits_negative.dtype) * _LARGE_NUM
    logits_negative = tf.reshape(logits_negative, [b * n, b_large * n])

    # Construct positive examples.
    inst_matched = tf.gather_nd(
        inst_target, positive_lookup_index, name='matched_inst')
    logits_positive = tf.einsum(
        'ijc,ijc->ij',
        inst_translated, inst_matched) / self._temperature
    # Get positive statistics.
    positive_stats = self._get_positive_similarity_statistics(
        logits_positive, inst_mask)
    logits_positive = tf.reshape(logits_positive, [b * n, 1])

    logits_all = tf.concat([logits_positive, logits_negative], axis=1)
    loss_pos = tf.reduce_logsumexp(logits_positive, 1)
    loss_all = tf.reduce_logsumexp(logits_all, 1)
    loss = (loss_all - loss_pos) * tf.reshape(inst_mask, [b * n])

    # Average across instances.
    loss = tf.math.divide_no_nan(
        tf.reduce_sum(loss), tf.reduce_sum(inst_mask))

    loss_and_stats = {'loss': loss}
    loss_and_stats.update(negative_stats)
    loss_and_stats.update(positive_stats)
    return loss_and_stats
