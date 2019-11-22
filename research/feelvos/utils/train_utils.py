# Copyright 2018 The TensorFlow Authors All Rights Reserved.
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

"""Utility functions for training."""
import collections
import six
import tensorflow as tf

from deeplab.core import preprocess_utils
from deeplab.utils import train_utils
from feelvos.utils import embedding_utils
from feelvos.utils import eval_utils

slim = tf.contrib.slim
add_softmax_cross_entropy_loss_for_each_scale = (
    train_utils.add_softmax_cross_entropy_loss_for_each_scale)
get_model_gradient_multipliers = train_utils.get_model_gradient_multipliers
get_model_learning_rate = train_utils.get_model_learning_rate
resolve_shape = preprocess_utils.resolve_shape


def add_triplet_loss_for_each_scale(batch_size, num_frames_per_video,
                                    embedding_dim, scales_to_embeddings,
                                    labels, scope):
  """Adds triplet loss for logits of each scale.

  Args:
    batch_size: Int, the number of video chunks sampled per batch
    num_frames_per_video: Int, the number of frames per video.
    embedding_dim: Int, the dimension of the learned embedding
    scales_to_embeddings: A map from embedding names for different scales to
      embeddings. The embeddings have shape [batch, embeddings_height,
      embeddings_width, embedding_dim].
    labels: Groundtruth labels with shape [batch, image_height, image_width, 1].
    scope: String, the scope for the loss.

  Raises:
    ValueError: labels is None.
  """
  if labels is None:
    raise ValueError('No label for triplet loss.')
  for scale, embeddings in scales_to_embeddings.iteritems():
    loss_scope = None
    if scope:
      loss_scope = '%s_%s' % (scope, scale)
    # Label is downsampled to the same size as logits.
    scaled_labels = tf.image.resize_nearest_neighbor(
        labels,
        resolve_shape(embeddings, 4)[1:3],
        align_corners=True)
    # Reshape from [batch * num_frames, ...] to [batch, num_frames, ...].
    h = tf.shape(embeddings)[1]
    w = tf.shape(embeddings)[2]
    new_labels_shape = tf.stack([batch_size, num_frames_per_video, h, w, 1])
    reshaped_labels = tf.reshape(scaled_labels, new_labels_shape)
    new_embeddings_shape = tf.stack([batch_size, num_frames_per_video, h, w,
                                     -1])
    reshaped_embeddings = tf.reshape(embeddings, new_embeddings_shape)

    with tf.name_scope(loss_scope):
      total_loss = tf.constant(0, dtype=tf.float32)
      for n in range(batch_size):
        embedding = reshaped_embeddings[n]
        label = reshaped_labels[n]
        n_pixels = h * w
        n_anchors_used = 256
        sampled_anchor_indices = tf.random_shuffle(tf.range(n_pixels))[
            :n_anchors_used]
        anchors_pool = tf.reshape(embedding[0], [-1, embedding_dim])
        anchors_pool_classes = tf.reshape(label[0], [-1])
        anchors = tf.gather(anchors_pool, sampled_anchor_indices)
        anchor_classes = tf.gather(anchors_pool_classes, sampled_anchor_indices)

        pos_neg_pool = tf.reshape(embedding[1:], [-1, embedding_dim])
        pos_neg_pool_classes = tf.reshape(label[1:], [-1])
        dists = embedding_utils.pairwise_distances(anchors, pos_neg_pool)
        pos_mask = tf.equal(anchor_classes[:, tf.newaxis],
                            pos_neg_pool_classes[tf.newaxis, :])
        neg_mask = tf.logical_not(pos_mask)
        pos_mask_f = tf.cast(pos_mask, tf.float32)
        neg_mask_f = tf.cast(neg_mask, tf.float32)
        pos_dists = pos_mask_f * dists + 1e20 * neg_mask_f
        neg_dists = neg_mask_f * dists + 1e20 * pos_mask_f
        pos_dists_min = tf.reduce_min(pos_dists, axis=1)
        neg_dists_min = tf.reduce_min(neg_dists, axis=1)
        margin = 1.0
        loss = tf.nn.relu(pos_dists_min - neg_dists_min + margin)
        # Handle case that no positive is present (per anchor).
        any_pos = tf.reduce_any(pos_mask, axis=1)
        loss *= tf.cast(any_pos, tf.float32)
        # Average over anchors
        loss = tf.reduce_mean(loss, axis=0)
        total_loss += loss
      total_loss /= batch_size
      # Scale the loss up a bit.
      total_loss *= 3.0
      tf.add_to_collection(tf.GraphKeys.LOSSES, total_loss)


def add_dynamic_softmax_cross_entropy_loss_for_each_scale(
    scales_to_logits, labels, ignore_label, loss_weight=1.0,
    upsample_logits=True, scope=None, top_k_percent_pixels=1.0,
    hard_example_mining_step=100000):
  """Adds softmax cross entropy loss per scale for logits with varying classes.

  Also adds summaries for mIoU.

  Args:
    scales_to_logits: A map from logits names for different scales to logits.
      The logits are a list of length batch_size of tensors of shape
      [time, logits_height, logits_width, num_classes].
    labels: Groundtruth labels with shape [batch_size * time, image_height,
      image_width, 1].
    ignore_label: Integer, label to ignore.
    loss_weight: Float, loss weight.
    upsample_logits: Boolean, upsample logits or not.
    scope: String, the scope for the loss.
    top_k_percent_pixels: A float, the value lies in [0.0, 1.0]. When its
      value < 1.0, only compute the loss for the top k percent pixels (e.g.,
      the top 20% pixels). This is useful for hard pixel mining.
    hard_example_mining_step: An integer, the training step in which the
      hard exampling mining kicks off. Note that we gradually reduce the
      mining percent to the top_k_percent_pixels. For example, if
      hard_example_mining_step=100K and top_k_percent_pixels=0.25, then
      mining percent will gradually reduce from 100% to 25% until 100K steps
      after which we only mine top 25% pixels.

  Raises:
    ValueError: Label or logits is None.
  """
  if labels is None:
    raise ValueError('No label for softmax cross entropy loss.')

  if top_k_percent_pixels < 0 or top_k_percent_pixels > 1:
    raise ValueError('Unexpected value of top_k_percent_pixels.')

  for scale, logits in six.iteritems(scales_to_logits):
    loss_scope = None
    if scope:
      loss_scope = '%s_%s' % (scope, scale)

    if upsample_logits:
      # Label is not downsampled, and instead we upsample logits.
      assert isinstance(logits, collections.Sequence)
      logits = [tf.image.resize_bilinear(
          x,
          preprocess_utils.resolve_shape(labels, 4)[1:3],
          align_corners=True) for x in logits]
      scaled_labels = labels
    else:
      # Label is downsampled to the same size as logits.
      assert isinstance(logits, collections.Sequence)
      scaled_labels = tf.image.resize_nearest_neighbor(
          labels,
          preprocess_utils.resolve_shape(logits[0], 4)[1:3],
          align_corners=True)

    batch_size = len(logits)
    num_time = preprocess_utils.resolve_shape(logits[0])[0]
    reshaped_labels = tf.reshape(
        scaled_labels, ([batch_size, num_time] +
                        preprocess_utils.resolve_shape(scaled_labels)[1:]))
    for n, logits_n in enumerate(logits):
      labels_n = reshaped_labels[n]
      labels_n = tf.reshape(labels_n, shape=[-1])
      not_ignore_mask = tf.to_float(tf.not_equal(labels_n,
                                                 ignore_label)) * loss_weight
      num_classes_n = tf.shape(logits_n)[-1]
      one_hot_labels = slim.one_hot_encoding(
          labels_n, num_classes_n, on_value=1.0, off_value=0.0)
      logits_n_flat = tf.reshape(logits_n, shape=[-1, num_classes_n])
      if top_k_percent_pixels == 1.0:
        tf.losses.softmax_cross_entropy(
            one_hot_labels,
            logits_n_flat,
            weights=not_ignore_mask,
            scope=loss_scope)
      else:
        # Only compute the loss for top k percent pixels.
        # First, compute the loss for all pixels. Note we do not put the loss
        # to loss_collection and set reduction = None to keep the shape.
        num_pixels = tf.to_float(tf.shape(logits_n_flat)[0])
        pixel_losses = tf.losses.softmax_cross_entropy(
            one_hot_labels,
            logits_n_flat,
            weights=not_ignore_mask,
            scope='pixel_losses',
            loss_collection=None,
            reduction=tf.losses.Reduction.NONE)
        # Compute the top_k_percent pixels based on current training step.
        if hard_example_mining_step == 0:
          # Directly focus on the top_k pixels.
          top_k_pixels = tf.to_int32(top_k_percent_pixels * num_pixels)
        else:
          # Gradually reduce the mining percent to top_k_percent_pixels.
          global_step = tf.to_float(tf.train.get_or_create_global_step())
          ratio = tf.minimum(1.0, global_step / hard_example_mining_step)
          top_k_pixels = tf.to_int32(
              (ratio * top_k_percent_pixels + (1.0 - ratio)) * num_pixels)
        _, top_k_indices = tf.nn.top_k(pixel_losses,
                                       k=top_k_pixels,
                                       sorted=True,
                                       name='top_k_percent_pixels')
        # Compute the loss for the top k percent pixels.
        tf.losses.softmax_cross_entropy(
            tf.gather(one_hot_labels, top_k_indices),
            tf.gather(logits_n_flat, top_k_indices),
            weights=tf.gather(not_ignore_mask, top_k_indices),
            scope=loss_scope)

      pred_n = tf.argmax(logits_n, axis=-1, output_type=tf.int32)[
          ..., tf.newaxis]
      labels_n = labels[n * num_time: (n + 1) * num_time]
      miou = eval_utils.calculate_multi_object_miou_tf(pred_n, labels_n)
      tf.summary.scalar('miou', miou)


def get_model_init_fn(train_logdir,
                      tf_initial_checkpoint,
                      initialize_last_layer,
                      last_layers,
                      ignore_missing_vars=False):
  """Gets the function initializing model variables from a checkpoint.

  Args:
    train_logdir: Log directory for training.
    tf_initial_checkpoint: TensorFlow checkpoint for initialization.
    initialize_last_layer: Initialize last layer or not.
    last_layers: Last layers of the model.
    ignore_missing_vars: Ignore missing variables in the checkpoint.

  Returns:
    Initialization function.
  """
  if tf_initial_checkpoint is None:
    tf.logging.info('Not initializing the model from a checkpoint.')
    return None

  if tf.train.latest_checkpoint(train_logdir):
    tf.logging.info('Ignoring initialization; other checkpoint exists')
    return None

  tf.logging.info('Initializing model from path: %s', tf_initial_checkpoint)

  # Variables that will not be restored.
  exclude_list = ['global_step']
  if not initialize_last_layer:
    exclude_list.extend(last_layers)

  variables_to_restore = slim.get_variables_to_restore(exclude=exclude_list)

  if variables_to_restore:
    return slim.assign_from_checkpoint_fn(
        tf_initial_checkpoint,
        variables_to_restore,
        ignore_missing_vars=ignore_missing_vars)
  return None
