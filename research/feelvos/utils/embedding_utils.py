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

"""Utilities for the instance embedding for segmentation."""

import numpy as np
import tensorflow as tf
from deeplab import model
from deeplab.core import preprocess_utils
from feelvos.utils import mask_damaging

slim = tf.contrib.slim
resolve_shape = preprocess_utils.resolve_shape
WRONG_LABEL_PADDING_DISTANCE = 1e20

# With correlation_cost local matching will be much faster. But we provide a
# slow fallback for convenience.
USE_CORRELATION_COST = False
if USE_CORRELATION_COST:
  # pylint: disable=g-import-not-at-top
  from correlation_cost.python.ops import correlation_cost_op


def pairwise_distances(x, y):
  """Computes pairwise squared l2 distances between tensors x and y.

  Args:
    x: Tensor of shape [n, feature_dim].
    y: Tensor of shape [m, feature_dim].

  Returns:
    Float32 distances tensor of shape [n, m].
  """
  # d[i,j] = (x[i] - y[j]) * (x[i] - y[j])'
  # = sum(x[i]^2, 1) + sum(y[j]^2, 1) - 2 * x[i] * y[j]'
  xs = tf.reduce_sum(x * x, axis=1)[:, tf.newaxis]
  ys = tf.reduce_sum(y * y, axis=1)[tf.newaxis, :]
  d = xs + ys - 2 * tf.matmul(x, y, transpose_b=True)
  return d


def pairwise_distances2(x, y):
  """Computes pairwise squared l2 distances between tensors x and y.

  Naive implementation, high memory use. Could be useful to test the more
  efficient implementation.

  Args:
    x: Tensor of shape [n, feature_dim].
    y: Tensor of shape [m, feature_dim].

  Returns:
    distances of shape [n, m].
  """
  return tf.reduce_sum(tf.squared_difference(
      x[:, tf.newaxis], y[tf.newaxis, :]), axis=-1)


def cross_correlate(x, y, max_distance=9):
  """Efficiently computes the cross correlation of x and y.

  Optimized implementation using correlation_cost.
  Note that we do not normalize by the feature dimension.

  Args:
    x: Float32 tensor of shape [height, width, feature_dim].
    y: Float32 tensor of shape [height, width, feature_dim].
    max_distance: Integer, the maximum distance in pixel coordinates
      per dimension which is considered to be in the search window.

  Returns:
    Float32 tensor of shape [height, width, (2 * max_distance + 1) ** 2].
  """
  with tf.name_scope('cross_correlation'):
    corr = correlation_cost_op.correlation_cost(
        x[tf.newaxis], y[tf.newaxis], kernel_size=1,
        max_displacement=max_distance, stride_1=1, stride_2=1,
        pad=max_distance)
    corr = tf.squeeze(corr, axis=0)
    # This correlation implementation takes the mean over the feature_dim,
    # but we want sum here, so multiply by feature_dim.
    feature_dim = resolve_shape(x)[-1]
    corr *= feature_dim
    return corr


def local_pairwise_distances(x, y, max_distance=9):
  """Computes pairwise squared l2 distances using a local search window.

  Optimized implementation using correlation_cost.

  Args:
    x: Float32 tensor of shape [height, width, feature_dim].
    y: Float32 tensor of shape [height, width, feature_dim].
    max_distance: Integer, the maximum distance in pixel coordinates
      per dimension which is considered to be in the search window.

  Returns:
    Float32 distances tensor of shape
      [height, width, (2 * max_distance + 1) ** 2].
  """
  with tf.name_scope('local_pairwise_distances'):
    # d[i,j] = (x[i] - y[j]) * (x[i] - y[j])'
    # = sum(x[i]^2, -1) + sum(y[j]^2, -1) - 2 * x[i] * y[j]'
    corr = cross_correlate(x, y, max_distance=max_distance)
    xs = tf.reduce_sum(x * x, axis=2)[..., tf.newaxis]
    ys = tf.reduce_sum(y * y, axis=2)[..., tf.newaxis]
    ones_ys = tf.ones_like(ys)
    ys = cross_correlate(ones_ys, ys, max_distance=max_distance)
    d = xs + ys - 2 * corr
    # Boundary should be set to Inf.
    boundary = tf.equal(
        cross_correlate(ones_ys, ones_ys, max_distance=max_distance), 0)
    d = tf.where(boundary, tf.fill(tf.shape(d), tf.constant(np.float('inf'))),
                 d)
    return d


def local_pairwise_distances2(x, y, max_distance=9):
  """Computes pairwise squared l2 distances using a local search window.

  Naive implementation using map_fn.
  Used as a slow fallback for when correlation_cost is not available.

  Args:
    x: Float32 tensor of shape [height, width, feature_dim].
    y: Float32 tensor of shape [height, width, feature_dim].
    max_distance: Integer, the maximum distance in pixel coordinates
      per dimension which is considered to be in the search window.

  Returns:
    Float32 distances tensor of shape
      [height, width, (2 * max_distance + 1) ** 2].
  """
  with tf.name_scope('local_pairwise_distances2'):
    padding_val = 1e20
    padded_y = tf.pad(y, [[max_distance, max_distance],
                          [max_distance, max_distance], [0, 0]],
                      constant_values=padding_val)
    height, width, _ = resolve_shape(x)
    dists = []
    for y_start in range(2 * max_distance + 1):
      y_end = y_start + height
      y_slice = padded_y[y_start:y_end]
      for x_start in range(2 * max_distance + 1):
        x_end = x_start + width
        offset_y = y_slice[:, x_start:x_end]
        dist = tf.reduce_sum(tf.squared_difference(x, offset_y), axis=2)
        dists.append(dist)
    dists = tf.stack(dists, axis=2)
    return dists


def majority_vote(labels):
  """Performs a label majority vote along axis 1.

  Second try, hopefully this time more efficient.
  We assume that the labels are contiguous starting from 0.
  It will also work for non-contiguous labels, but be inefficient.

  Args:
    labels: Int tensor of shape [n, k]

  Returns:
    The majority of labels along axis 1
  """
  max_label = tf.reduce_max(labels)
  one_hot = tf.one_hot(labels, depth=max_label + 1)
  summed = tf.reduce_sum(one_hot, axis=1)
  majority = tf.argmax(summed, axis=1)
  return majority


def assign_labels_by_nearest_neighbors(reference_embeddings, query_embeddings,
                                       reference_labels, k=1):
  """Segments by nearest neighbor query wrt the reference frame.

  Args:
    reference_embeddings: Tensor of shape [height, width, embedding_dim],
      the embedding vectors for the reference frame
    query_embeddings: Tensor of shape [n_query_images, height, width,
      embedding_dim], the embedding vectors for the query frames
    reference_labels: Tensor of shape [height, width, 1], the class labels of
      the reference frame
    k: Integer, the number of nearest neighbors to use

  Returns:
    The labels of the nearest neighbors as [n_query_frames, height, width, 1]
    tensor

  Raises:
    ValueError: If k < 1.
  """
  if k < 1:
    raise ValueError('k must be at least 1')
  dists = flattened_pairwise_distances(reference_embeddings, query_embeddings)
  if k == 1:
    nn_indices = tf.argmin(dists, axis=1)[..., tf.newaxis]
  else:
    _, nn_indices = tf.nn.top_k(-dists, k, sorted=False)
  reference_labels = tf.reshape(reference_labels, [-1])
  nn_labels = tf.gather(reference_labels, nn_indices)
  if k == 1:
    nn_labels = tf.squeeze(nn_labels, axis=1)
  else:
    nn_labels = majority_vote(nn_labels)
  height = tf.shape(reference_embeddings)[0]
  width = tf.shape(reference_embeddings)[1]
  n_query_frames = query_embeddings.shape[0]
  nn_labels = tf.reshape(nn_labels, [n_query_frames, height, width, 1])
  return nn_labels


def flattened_pairwise_distances(reference_embeddings, query_embeddings):
  """Calculates flattened tensor of pairwise distances between ref and query.

  Args:
    reference_embeddings: Tensor of shape [..., embedding_dim],
      the embedding vectors for the reference frame
    query_embeddings: Tensor of shape [n_query_images, height, width,
      embedding_dim], the embedding vectors for the query frames.

  Returns:
    A distance tensor of shape [reference_embeddings.size / embedding_dim,
    query_embeddings.size / embedding_dim]
  """
  embedding_dim = resolve_shape(query_embeddings)[-1]
  reference_embeddings = tf.reshape(reference_embeddings, [-1, embedding_dim])
  first_dim = -1
  query_embeddings = tf.reshape(query_embeddings, [first_dim, embedding_dim])
  dists = pairwise_distances(query_embeddings, reference_embeddings)
  return dists


def nearest_neighbor_features_per_object(
    reference_embeddings, query_embeddings, reference_labels,
    max_neighbors_per_object, k_nearest_neighbors, gt_ids=None, n_chunks=100):
  """Calculates the distance to the nearest neighbor per object.

  For every pixel of query_embeddings calculate the distance to the
  nearest neighbor in the (possibly subsampled) reference_embeddings per object.

  Args:
    reference_embeddings: Tensor of shape [height, width, embedding_dim],
      the embedding vectors for the reference frame.
    query_embeddings: Tensor of shape [n_query_images, height, width,
      embedding_dim], the embedding vectors for the query frames.
    reference_labels: Tensor of shape [height, width, 1], the class labels of
      the reference frame.
    max_neighbors_per_object: Integer, the maximum number of candidates
      for the nearest neighbor query per object after subsampling,
      or 0 for no subsampling.
    k_nearest_neighbors: Integer, the number of nearest neighbors to use.
    gt_ids: Int tensor of shape [n_objs] of the sorted unique ground truth
      ids in the first frame. If None, it will be derived from
      reference_labels.
    n_chunks: Integer, the number of chunks to use to save memory
      (set to 1 for no chunking).

  Returns:
    nn_features: A float32 tensor of nearest neighbor features of shape
      [n_query_images, height, width, n_objects, feature_dim].
    gt_ids: An int32 tensor of the unique sorted object ids present
      in the reference labels.
  """
  with tf.name_scope('nn_features_per_object'):
    reference_labels_flat = tf.reshape(reference_labels, [-1])
    if gt_ids is None:
      ref_obj_ids, _ = tf.unique(reference_labels_flat)
      ref_obj_ids = tf.contrib.framework.sort(ref_obj_ids)
      gt_ids = ref_obj_ids
    embedding_dim = resolve_shape(reference_embeddings)[-1]
    reference_embeddings_flat = tf.reshape(reference_embeddings,
                                           [-1, embedding_dim])

    reference_embeddings_flat, reference_labels_flat = (
        subsample_reference_embeddings_and_labels(reference_embeddings_flat,
                                                  reference_labels_flat,
                                                  gt_ids,
                                                  max_neighbors_per_object))
    shape = resolve_shape(query_embeddings)
    query_embeddings_flat = tf.reshape(query_embeddings, [-1, embedding_dim])
    nn_features = _nearest_neighbor_features_per_object_in_chunks(
        reference_embeddings_flat, query_embeddings_flat, reference_labels_flat,
        gt_ids, k_nearest_neighbors, n_chunks)
    nn_features_dim = resolve_shape(nn_features)[-1]
    nn_features_reshaped = tf.reshape(nn_features,
                                      tf.stack(shape[:3] + [tf.size(gt_ids),
                                                            nn_features_dim]))
    return nn_features_reshaped, gt_ids


def _nearest_neighbor_features_per_object_in_chunks(
    reference_embeddings_flat, query_embeddings_flat, reference_labels_flat,
    ref_obj_ids, k_nearest_neighbors, n_chunks):
  """Calculates the nearest neighbor features per object in chunks to save mem.

  Uses chunking to bound the memory use.

  Args:
    reference_embeddings_flat: Tensor of shape [n, embedding_dim],
      the embedding vectors for the reference frame.
    query_embeddings_flat: Tensor of shape [m, embedding_dim], the embedding
      vectors for the query frames.
    reference_labels_flat: Tensor of shape [n], the class labels of the
      reference frame.
    ref_obj_ids: int tensor of unique object ids in the reference labels.
    k_nearest_neighbors: Integer, the number of nearest neighbors to use.
    n_chunks: Integer, the number of chunks to use to save memory
      (set to 1 for no chunking).

  Returns:
    nn_features: A float32 tensor of nearest neighbor features of shape
      [m, n_objects, feature_dim].
  """
  chunk_size = tf.cast(tf.ceil(tf.cast(tf.shape(query_embeddings_flat)[0],
                                       tf.float32) / n_chunks), tf.int32)
  wrong_label_mask = tf.not_equal(reference_labels_flat,
                                  ref_obj_ids[:, tf.newaxis])
  all_features = []
  for n in range(n_chunks):
    if n_chunks == 1:
      query_embeddings_flat_chunk = query_embeddings_flat
    else:
      chunk_start = n * chunk_size
      chunk_end = (n + 1) * chunk_size
      query_embeddings_flat_chunk = query_embeddings_flat[chunk_start:chunk_end]
    # Use control dependencies to make sure that the chunks are not processed
    # in parallel which would prevent any peak memory savings.
    with tf.control_dependencies(all_features):
      features = _nn_features_per_object_for_chunk(
          reference_embeddings_flat, query_embeddings_flat_chunk,
          wrong_label_mask, k_nearest_neighbors
      )
    all_features.append(features)
  if n_chunks == 1:
    nn_features = all_features[0]
  else:
    nn_features = tf.concat(all_features, axis=0)
  return nn_features


def _nn_features_per_object_for_chunk(
    reference_embeddings, query_embeddings, wrong_label_mask,
    k_nearest_neighbors):
  """Extracts features for each object using nearest neighbor attention.

  Args:
    reference_embeddings: Tensor of shape [n_chunk, embedding_dim],
      the embedding vectors for the reference frame.
    query_embeddings: Tensor of shape [m_chunk, embedding_dim], the embedding
      vectors for the query frames.
    wrong_label_mask:
    k_nearest_neighbors: Integer, the number of nearest neighbors to use.

  Returns:
    nn_features: A float32 tensor of nearest neighbor features of shape
      [m_chunk, n_objects, feature_dim].
  """
  reference_embeddings_key = reference_embeddings
  query_embeddings_key = query_embeddings
  dists = flattened_pairwise_distances(reference_embeddings_key,
                                       query_embeddings_key)
  dists = (dists[:, tf.newaxis, :] +
           tf.cast(wrong_label_mask[tf.newaxis, :, :], tf.float32) *
           WRONG_LABEL_PADDING_DISTANCE)
  if k_nearest_neighbors == 1:
    features = tf.reduce_min(dists, axis=2, keepdims=True)
  else:
    # Find the closest k and combine them according to attention_feature_type
    dists, _ = tf.nn.top_k(-dists, k=k_nearest_neighbors)
    dists = -dists
    # If not enough real neighbors were found, pad with the farthest real
    # neighbor.
    valid_mask = tf.less(dists, WRONG_LABEL_PADDING_DISTANCE)
    masked_dists = dists * tf.cast(valid_mask, tf.float32)
    pad_dist = tf.tile(tf.reduce_max(masked_dists, axis=2)[..., tf.newaxis],
                       multiples=[1, 1, k_nearest_neighbors])
    dists = tf.where(valid_mask, dists, pad_dist)
    # take mean of distances
    features = tf.reduce_mean(dists, axis=2, keepdims=True)
  return features


def create_embedding_segmentation_features(features, feature_dimension,
                                           n_layers, kernel_size, reuse,
                                           atrous_rates=None):
  """Extracts features which can be used to estimate the final segmentation.

  Args:
    features: input features of shape [batch, height, width, features]
    feature_dimension: Integer, the dimensionality used in the segmentation
      head layers.
    n_layers: Integer, the number of layers in the segmentation head.
    kernel_size: Integer, the kernel size used in the segmentation head.
    reuse: reuse mode for the variable_scope.
    atrous_rates: List of integers of length n_layers, the atrous rates to use.

  Returns:
    Features to be used to estimate the segmentation labels of shape
      [batch, height, width, embedding_seg_feat_dim].
  """
  if atrous_rates is None or not atrous_rates:
    atrous_rates = [1 for _ in range(n_layers)]
  assert len(atrous_rates) == n_layers
  with tf.variable_scope('embedding_seg', reuse=reuse):
    for n in range(n_layers):
      features = model.split_separable_conv2d(
          features, feature_dimension, kernel_size=kernel_size,
          rate=atrous_rates[n], scope='split_separable_conv2d_{}'.format(n))
    return features


def add_image_summaries(images, nn_features, logits, batch_size,
                        prev_frame_nn_features=None):
  """Adds image summaries of input images, attention features and logits.

  Args:
    images: Image tensor of shape [batch, height, width, channels].
    nn_features: Nearest neighbor attention features of shape
      [batch_size, height, width, n_objects, 1].
    logits: Float32 tensor of logits.
    batch_size: Integer, the number of videos per clone per mini-batch.
    prev_frame_nn_features: Nearest neighbor attention features wrt. the
      last frame of shape [batch_size, height, width, n_objects, 1].
      Can be None.
  """
  # Separate reference and query images.
  reshaped_images = tf.reshape(images, tf.stack(
      [batch_size, -1] + resolve_shape(images)[1:]))
  reference_images = reshaped_images[:, 0]
  query_images = reshaped_images[:, 1:]
  query_images_reshaped = tf.reshape(query_images, tf.stack(
      [-1] + resolve_shape(images)[1:]))
  tf.summary.image('ref_images', reference_images, max_outputs=batch_size)
  tf.summary.image('query_images', query_images_reshaped, max_outputs=10)
  predictions = tf.cast(
      tf.argmax(logits, axis=-1), tf.uint8)[..., tf.newaxis]
  # Scale up so that we can actually see something.
  tf.summary.image('predictions', predictions * 32, max_outputs=10)
  # We currently only show the first dimension of the features for background
  # and the first foreground object.
  tf.summary.image('nn_fg_features', nn_features[..., 0:1, 0],
                   max_outputs=batch_size)
  if prev_frame_nn_features is not None:
    tf.summary.image('nn_fg_features_prev', prev_frame_nn_features[..., 0:1, 0],
                     max_outputs=batch_size)
  tf.summary.image('nn_bg_features', nn_features[..., 1:2, 0],
                   max_outputs=batch_size)
  if prev_frame_nn_features is not None:
    tf.summary.image('nn_bg_features_prev',
                     prev_frame_nn_features[..., 1:2, 0],
                     max_outputs=batch_size)


def get_embeddings(images, model_options, embedding_dimension):
  """Extracts embedding vectors for images. Should only be used for inference.

  Args:
    images: A tensor of shape [batch, height, width, channels].
    model_options: A ModelOptions instance to configure models.
    embedding_dimension: Integer, the dimension of the embedding.

  Returns:
    embeddings: A tensor of shape [batch, height, width, embedding_dimension].
  """
  features, end_points = model.extract_features(
      images,
      model_options,
      is_training=False)

  if model_options.decoder_output_stride is not None:
    decoder_output_stride = min(model_options.decoder_output_stride)
    if model_options.crop_size is None:
      height = tf.shape(images)[1]
      width = tf.shape(images)[2]
    else:
      height, width = model_options.crop_size
    features = model.refine_by_decoder(
        features,
        end_points,
        crop_size=[height, width],
        decoder_output_stride=[decoder_output_stride],
        decoder_use_separable_conv=model_options.decoder_use_separable_conv,
        model_variant=model_options.model_variant,
        is_training=False)

  with tf.variable_scope('embedding'):
    embeddings = split_separable_conv2d_with_identity_initializer(
        features, embedding_dimension, scope='split_separable_conv2d')
  return embeddings


def get_logits_with_matching(images,
                             model_options,
                             weight_decay=0.0001,
                             reuse=None,
                             is_training=False,
                             fine_tune_batch_norm=False,
                             reference_labels=None,
                             batch_size=None,
                             num_frames_per_video=None,
                             embedding_dimension=None,
                             max_neighbors_per_object=0,
                             k_nearest_neighbors=1,
                             use_softmax_feedback=True,
                             initial_softmax_feedback=None,
                             embedding_seg_feature_dimension=256,
                             embedding_seg_n_layers=4,
                             embedding_seg_kernel_size=7,
                             embedding_seg_atrous_rates=None,
                             normalize_nearest_neighbor_distances=True,
                             also_attend_to_previous_frame=True,
                             damage_initial_previous_frame_mask=False,
                             use_local_previous_frame_attention=True,
                             previous_frame_attention_window_size=15,
                             use_first_frame_matching=True,
                             also_return_embeddings=False,
                             ref_embeddings=None):
  """Gets the logits by atrous/image spatial pyramid pooling using attention.

  Args:
    images: A tensor of size [batch, height, width, channels].
    model_options: A ModelOptions instance to configure models.
    weight_decay: The weight decay for model variables.
    reuse: Reuse the model variables or not.
    is_training: Is training or not.
    fine_tune_batch_norm: Fine-tune the batch norm parameters or not.
    reference_labels: The segmentation labels of the reference frame on which
      attention is applied.
    batch_size: Integer, the number of videos on a batch
    num_frames_per_video: Integer, the number of frames per video
    embedding_dimension: Integer, the dimension of the embedding
    max_neighbors_per_object: Integer, the maximum number of candidates
      for the nearest neighbor query per object after subsampling.
      Can be 0 for no subsampling.
    k_nearest_neighbors: Integer, the number of nearest neighbors to use.
    use_softmax_feedback: Boolean, whether to give the softmax predictions of
      the last frame as additional input to the segmentation head.
    initial_softmax_feedback: List of Float32 tensors, or None. Can be used to
      initialize the softmax predictions used for the feedback loop.
      Only has an effect if use_softmax_feedback is True.
    embedding_seg_feature_dimension: Integer, the dimensionality used in the
      segmentation head layers.
    embedding_seg_n_layers: Integer, the number of layers in the segmentation
      head.
    embedding_seg_kernel_size: Integer, the kernel size used in the
      segmentation head.
    embedding_seg_atrous_rates: List of integers of length
      embedding_seg_n_layers, the atrous rates to use for the segmentation head.
    normalize_nearest_neighbor_distances: Boolean, whether to normalize the
      nearest neighbor distances to [0,1] using sigmoid, scale and shift.
    also_attend_to_previous_frame: Boolean, whether to also use nearest
      neighbor attention with respect to the previous frame.
    damage_initial_previous_frame_mask: Boolean, whether to artificially damage
      the initial previous frame mask. Only has an effect if
      also_attend_to_previous_frame is True.
    use_local_previous_frame_attention: Boolean, whether to restrict the
      previous frame attention to a local search window.
      Only has an effect, if also_attend_to_previous_frame is True.
    previous_frame_attention_window_size: Integer, the window size used for
      local previous frame attention, if use_local_previous_frame_attention
      is True.
    use_first_frame_matching: Boolean, whether to extract features by matching
      to the reference frame. This should always be true except for ablation
      experiments.
    also_return_embeddings: Boolean, whether to return the embeddings as well.
    ref_embeddings: Tuple of
      (first_frame_embeddings, previous_frame_embeddings),
      each of shape [batch, height, width, embedding_dimension], or None.
  Returns:
    outputs_to_logits: A map from output_type to logits.
    If also_return_embeddings is True, it will also return an embeddings
      tensor of shape [batch, height, width, embedding_dimension].
  """
  features, end_points = model.extract_features(
      images,
      model_options,
      weight_decay=weight_decay,
      reuse=reuse,
      is_training=is_training,
      fine_tune_batch_norm=fine_tune_batch_norm)

  if model_options.decoder_output_stride:
    decoder_output_stride = min(model_options.decoder_output_stride)
    if model_options.crop_size is None:
      height = tf.shape(images)[1]
      width = tf.shape(images)[2]
    else:
      height, width = model_options.crop_size
    decoder_height = model.scale_dimension(height, 1.0 / decoder_output_stride)
    decoder_width = model.scale_dimension(width, 1.0 / decoder_output_stride)
    features = model.refine_by_decoder(
        features,
        end_points,
        crop_size=[height, width],
        decoder_output_stride=[decoder_output_stride],
        decoder_use_separable_conv=model_options.decoder_use_separable_conv,
        model_variant=model_options.model_variant,
        weight_decay=weight_decay,
        reuse=reuse,
        is_training=is_training,
        fine_tune_batch_norm=fine_tune_batch_norm)

  with tf.variable_scope('embedding', reuse=reuse):
    embeddings = split_separable_conv2d_with_identity_initializer(
        features, embedding_dimension, scope='split_separable_conv2d')
    embeddings = tf.identity(embeddings, name='embeddings')
  scaled_reference_labels = tf.image.resize_nearest_neighbor(
      reference_labels,
      resolve_shape(embeddings, 4)[1:3],
      align_corners=True)
  h, w = decoder_height, decoder_width
  if num_frames_per_video is None:
    num_frames_per_video = tf.size(embeddings) // (
        batch_size * h * w * embedding_dimension)
  new_labels_shape = tf.stack([batch_size, -1, h, w, 1])
  reshaped_reference_labels = tf.reshape(scaled_reference_labels,
                                         new_labels_shape)
  new_embeddings_shape = tf.stack([batch_size,
                                   num_frames_per_video, h, w,
                                   embedding_dimension])
  reshaped_embeddings = tf.reshape(embeddings, new_embeddings_shape)
  all_nn_features = []
  all_ref_obj_ids = []
  # To keep things simple, we do all this separate for each sequence for now.
  for n in range(batch_size):
    embedding = reshaped_embeddings[n]
    if ref_embeddings is None:
      n_chunks = 100
      reference_embedding = embedding[0]
      if also_attend_to_previous_frame or use_softmax_feedback:
        queries_embedding = embedding[2:]
      else:
        queries_embedding = embedding[1:]
    else:
      if USE_CORRELATION_COST:
        n_chunks = 20
      else:
        n_chunks = 500
      reference_embedding = ref_embeddings[0][n]
      queries_embedding = embedding
    reference_labels = reshaped_reference_labels[n][0]
    nn_features_n, ref_obj_ids = nearest_neighbor_features_per_object(
        reference_embedding, queries_embedding, reference_labels,
        max_neighbors_per_object, k_nearest_neighbors, n_chunks=n_chunks)
    if normalize_nearest_neighbor_distances:
      nn_features_n = (tf.nn.sigmoid(nn_features_n) - 0.5) * 2
    all_nn_features.append(nn_features_n)
    all_ref_obj_ids.append(ref_obj_ids)

  feat_dim = resolve_shape(features)[-1]
  features = tf.reshape(features, tf.stack(
      [batch_size, num_frames_per_video, h, w, feat_dim]))
  if ref_embeddings is None:
    # Strip the features for the reference frame.
    if also_attend_to_previous_frame or use_softmax_feedback:
      features = features[:, 2:]
    else:
      features = features[:, 1:]

  # To keep things simple, we do all this separate for each sequence for now.
  outputs_to_logits = {output: [] for
                       output in model_options.outputs_to_num_classes}
  for n in range(batch_size):
    features_n = features[n]
    nn_features_n = all_nn_features[n]
    nn_features_n_tr = tf.transpose(nn_features_n, [3, 0, 1, 2, 4])
    n_objs = tf.shape(nn_features_n_tr)[0]
    # Repeat features for every object.
    features_n_tiled = tf.tile(features_n[tf.newaxis],
                               multiples=[n_objs, 1, 1, 1, 1])
    prev_frame_labels = None
    if also_attend_to_previous_frame:
      prev_frame_labels = reshaped_reference_labels[n, 1]
      if is_training and damage_initial_previous_frame_mask:
        # Damage the previous frame masks.
        prev_frame_labels = mask_damaging.damage_masks(prev_frame_labels,
                                                       dilate=False)
      tf.summary.image('prev_frame_labels',
                       tf.cast(prev_frame_labels[tf.newaxis],
                               tf.uint8) * 32)
      initial_softmax_feedback_n = create_initial_softmax_from_labels(
          prev_frame_labels, reshaped_reference_labels[n][0],
          decoder_output_stride=None, reduce_labels=True)
    elif initial_softmax_feedback is not None:
      initial_softmax_feedback_n = initial_softmax_feedback[n]
    else:
      initial_softmax_feedback_n = None
    if initial_softmax_feedback_n is None:
      last_softmax = tf.zeros((n_objs, h, w, 1), dtype=tf.float32)
    else:
      last_softmax = tf.transpose(initial_softmax_feedback_n, [2, 0, 1])[
          ..., tf.newaxis]
    assert len(model_options.outputs_to_num_classes) == 1
    output = model_options.outputs_to_num_classes.keys()[0]
    logits = []
    n_ref_frames = 1
    prev_frame_nn_features_n = None
    if also_attend_to_previous_frame or use_softmax_feedback:
      n_ref_frames += 1
    if ref_embeddings is not None:
      n_ref_frames = 0
    for t in range(num_frames_per_video - n_ref_frames):
      to_concat = [features_n_tiled[:, t]]
      if use_first_frame_matching:
        to_concat.append(nn_features_n_tr[:, t])
      if use_softmax_feedback:
        to_concat.append(last_softmax)
      if also_attend_to_previous_frame:
        assert normalize_nearest_neighbor_distances, (
            'previous frame attention currently only works when normalized '
            'distances are used')
        embedding = reshaped_embeddings[n]
        if ref_embeddings is None:
          last_frame_embedding = embedding[t + 1]
          query_embeddings = embedding[t + 2, tf.newaxis]
        else:
          last_frame_embedding = ref_embeddings[1][0]
          query_embeddings = embedding
        if use_local_previous_frame_attention:
          assert query_embeddings.shape[0] == 1
          prev_frame_nn_features_n = (
              local_previous_frame_nearest_neighbor_features_per_object(
                  last_frame_embedding,
                  query_embeddings[0],
                  prev_frame_labels,
                  all_ref_obj_ids[n],
                  max_distance=previous_frame_attention_window_size)
          )
        else:
          prev_frame_nn_features_n, _ = (
              nearest_neighbor_features_per_object(
                  last_frame_embedding, query_embeddings, prev_frame_labels,
                  max_neighbors_per_object, k_nearest_neighbors,
                  gt_ids=all_ref_obj_ids[n]))
          prev_frame_nn_features_n = (tf.nn.sigmoid(
              prev_frame_nn_features_n) - 0.5) * 2
        prev_frame_nn_features_n_sq = tf.squeeze(prev_frame_nn_features_n,
                                                 axis=0)
        prev_frame_nn_features_n_tr = tf.transpose(
            prev_frame_nn_features_n_sq, [2, 0, 1, 3])
        to_concat.append(prev_frame_nn_features_n_tr)
      features_n_concat_t = tf.concat(to_concat, axis=-1)
      embedding_seg_features_n_t = (
          create_embedding_segmentation_features(
              features_n_concat_t, embedding_seg_feature_dimension,
              embedding_seg_n_layers, embedding_seg_kernel_size,
              reuse or n > 0, atrous_rates=embedding_seg_atrous_rates))
      logits_t = model.get_branch_logits(
          embedding_seg_features_n_t,
          1,
          model_options.atrous_rates,
          aspp_with_batch_norm=model_options.aspp_with_batch_norm,
          kernel_size=model_options.logits_kernel_size,
          weight_decay=weight_decay,
          reuse=reuse or n > 0 or t > 0,
          scope_suffix=output
      )
      logits.append(logits_t)
      prev_frame_labels = tf.transpose(tf.argmax(logits_t, axis=0),
                                       [2, 0, 1])
      last_softmax = tf.nn.softmax(logits_t, axis=0)
    logits = tf.stack(logits, axis=1)
    logits_shape = tf.stack(
        [n_objs, num_frames_per_video - n_ref_frames] +
        resolve_shape(logits)[2:-1])
    logits_reshaped = tf.reshape(logits, logits_shape)
    logits_transposed = tf.transpose(logits_reshaped, [1, 2, 3, 0])
    outputs_to_logits[output].append(logits_transposed)

    add_image_summaries(
        images[n * num_frames_per_video: (n+1) * num_frames_per_video],
        nn_features_n,
        logits_transposed,
        batch_size=1,
        prev_frame_nn_features=prev_frame_nn_features_n)
  if also_return_embeddings:
    return outputs_to_logits, embeddings
  else:
    return outputs_to_logits


def subsample_reference_embeddings_and_labels(
    reference_embeddings_flat, reference_labels_flat, ref_obj_ids,
    max_neighbors_per_object):
  """Subsamples the reference embedding vectors and labels.

  After subsampling, at most max_neighbors_per_object items will remain per
    class.

  Args:
    reference_embeddings_flat: Tensor of shape [n, embedding_dim],
      the embedding vectors for the reference frame.
    reference_labels_flat: Tensor of shape [n, 1],
      the class labels of the reference frame.
    ref_obj_ids: An int32 tensor of the unique object ids present
      in the reference labels.
    max_neighbors_per_object: Integer, the maximum number of candidates
      for the nearest neighbor query per object after subsampling,
      or 0 for no subsampling.

  Returns:
    reference_embeddings_flat: Tensor of shape [n_sub, embedding_dim],
      the subsampled embedding vectors for the reference frame.
    reference_labels_flat: Tensor of shape [n_sub, 1],
      the class labels of the reference frame.
  """
  if max_neighbors_per_object == 0:
    return reference_embeddings_flat, reference_labels_flat
  same_label_mask = tf.equal(reference_labels_flat[tf.newaxis, :],
                             ref_obj_ids[:, tf.newaxis])
  max_neighbors_per_object_repeated = tf.tile(
      tf.constant(max_neighbors_per_object)[tf.newaxis],
      multiples=[tf.size(ref_obj_ids)])
  # Somehow map_fn on GPU caused trouble sometimes, so let's put it on CPU
  # for now.
  with tf.device('cpu:0'):
    subsampled_indices = tf.map_fn(_create_subsampling_mask,
                                   (same_label_mask,
                                    max_neighbors_per_object_repeated),
                                   dtype=tf.int64,
                                   name='subsample_labels_map_fn',
                                   parallel_iterations=1)
  mask = tf.not_equal(subsampled_indices, tf.constant(-1, dtype=tf.int64))
  masked_indices = tf.boolean_mask(subsampled_indices, mask)
  reference_embeddings_flat = tf.gather(reference_embeddings_flat,
                                        masked_indices)
  reference_labels_flat = tf.gather(reference_labels_flat, masked_indices)
  return reference_embeddings_flat, reference_labels_flat


def _create_subsampling_mask(args):
  """Creates boolean mask which can be used to subsample the labels.

  Args:
    args: tuple of (label_mask, max_neighbors_per_object), where label_mask
      is the mask to be subsampled and max_neighbors_per_object is a int scalar,
      the maximum number of neighbors to be retained after subsampling.

  Returns:
    The boolean mask for subsampling the labels.
  """
  label_mask, max_neighbors_per_object = args
  indices = tf.squeeze(tf.where(label_mask), axis=1)
  shuffled_indices = tf.random_shuffle(indices)
  subsampled_indices = shuffled_indices[:max_neighbors_per_object]
  n_pad = max_neighbors_per_object - tf.size(subsampled_indices)
  padded_label = -1
  padding = tf.fill((n_pad,), tf.constant(padded_label, dtype=tf.int64))
  padded = tf.concat([subsampled_indices, padding], axis=0)
  return padded


def conv2d_identity_initializer(scale=1.0, mean=0, stddev=3e-2):
  """Creates an identity initializer for TensorFlow conv2d.

  We add a small amount of normal noise to the initialization matrix.
  Code copied from lcchen@.

  Args:
    scale: The scale coefficient for the identity weight matrix.
    mean: A 0-D Tensor or Python value of type `dtype`. The mean of the
      truncated normal distribution.
    stddev: A 0-D Tensor or Python value of type `dtype`. The standard deviation
      of the truncated normal distribution.

  Returns:
    An identity initializer function for TensorFlow conv2d.
  """
  def _initializer(shape, dtype=tf.float32, partition_info=None):
    """Returns the identity matrix scaled by `scale`.

    Args:
      shape: A tuple of int32 numbers indicating the shape of the initializing
         matrix.
      dtype: The data type of the initializing matrix.
      partition_info: (Optional) variable_scope._PartitionInfo object holding
      additional information about how the variable is partitioned. This input
        is not used in our case, but is required by TensorFlow.

    Returns:
      A identity matrix.

    Raises:
      ValueError: If len(shape) != 4, or shape[0] != shape[1], or shape[0] is
        not odd, or shape[1] is not odd..
    """
    del partition_info
    if len(shape) != 4:
      raise ValueError('Expect shape length to be 4.')
    if shape[0] != shape[1]:
      raise ValueError('Expect shape[0] = shape[1].')
    if shape[0] % 2 != 1:
      raise ValueError('Expect shape[0] to be odd value.')
    if shape[1] % 2 != 1:
      raise ValueError('Expect shape[1] to be odd value.')
    weights = np.zeros(shape, dtype=np.float32)
    center_y = shape[0] / 2
    center_x = shape[1] / 2
    min_channel = min(shape[2], shape[3])
    for i in range(min_channel):
      weights[center_y, center_x, i, i] = scale
    return tf.constant(weights, dtype=dtype) + tf.truncated_normal(
        shape, mean=mean, stddev=stddev, dtype=dtype)

  return _initializer


def split_separable_conv2d_with_identity_initializer(
    inputs,
    filters,
    kernel_size=3,
    rate=1,
    weight_decay=0.00004,
    scope=None):
  """Splits a separable conv2d into depthwise and pointwise conv2d.

  This operation differs from `tf.layers.separable_conv2d` as this operation
  applies activation function between depthwise and pointwise conv2d.

  Args:
    inputs: Input tensor with shape [batch, height, width, channels].
    filters: Number of filters in the 1x1 pointwise convolution.
    kernel_size: A list of length 2: [kernel_height, kernel_width] of
      of the filters. Can be an int if both values are the same.
    rate: Atrous convolution rate for the depthwise convolution.
    weight_decay: The weight decay to use for regularizing the model.
    scope: Optional scope for the operation.

  Returns:
    Computed features after split separable conv2d.
  """
  initializer = conv2d_identity_initializer()
  outputs = slim.separable_conv2d(
      inputs,
      None,
      kernel_size=kernel_size,
      depth_multiplier=1,
      rate=rate,
      weights_initializer=initializer,
      weights_regularizer=None,
      scope=scope + '_depthwise')
  return slim.conv2d(
      outputs,
      filters,
      1,
      weights_initializer=initializer,
      weights_regularizer=slim.l2_regularizer(weight_decay),
      scope=scope + '_pointwise')


def create_initial_softmax_from_labels(last_frame_labels, reference_labels,
                                       decoder_output_stride, reduce_labels):
  """Creates initial softmax predictions from last frame labels.

  Args:
    last_frame_labels: last frame labels of shape [1, height, width, 1].
    reference_labels: reference frame labels of shape [1, height, width, 1].
    decoder_output_stride: Integer, the stride of the decoder. Can be None, in
      this case it's assumed that the last_frame_labels and reference_labels
      are already scaled to the decoder output resolution.
    reduce_labels: Boolean, whether to reduce the depth of the softmax one_hot
      encoding to the actual number of labels present in the reference frame
      (otherwise the depth will be the highest label index + 1).

  Returns:
    init_softmax: the initial softmax predictions.
  """
  if decoder_output_stride is None:
    labels_output_size = last_frame_labels
    reference_labels_output_size = reference_labels
  else:
    h = tf.shape(last_frame_labels)[1]
    w = tf.shape(last_frame_labels)[2]
    h_sub = model.scale_dimension(h, 1.0 / decoder_output_stride)
    w_sub = model.scale_dimension(w, 1.0 / decoder_output_stride)
    labels_output_size = tf.image.resize_nearest_neighbor(
        last_frame_labels, [h_sub, w_sub], align_corners=True)
    reference_labels_output_size = tf.image.resize_nearest_neighbor(
        reference_labels, [h_sub, w_sub], align_corners=True)
  if reduce_labels:
    unique_labels, _ = tf.unique(tf.reshape(reference_labels_output_size, [-1]))
    depth = tf.size(unique_labels)
  else:
    depth = tf.reduce_max(reference_labels_output_size) + 1
  one_hot_assertion = tf.assert_less(tf.reduce_max(labels_output_size), depth)
  with tf.control_dependencies([one_hot_assertion]):
    init_softmax = tf.one_hot(tf.squeeze(labels_output_size,
                                         axis=-1),
                              depth=depth,
                              dtype=tf.float32)
  return init_softmax


def local_previous_frame_nearest_neighbor_features_per_object(
    prev_frame_embedding, query_embedding, prev_frame_labels,
    gt_ids, max_distance=9):
  """Computes nearest neighbor features while only allowing local matches.

  Args:
    prev_frame_embedding: Tensor of shape [height, width, embedding_dim],
      the embedding vectors for the last frame.
    query_embedding: Tensor of shape [height, width, embedding_dim],
      the embedding vectors for the query frames.
    prev_frame_labels: Tensor of shape [height, width, 1], the class labels of
      the previous frame.
    gt_ids: Int Tensor of shape [n_objs] of the sorted unique ground truth
      ids in the first frame.
    max_distance: Integer, the maximum distance allowed for local matching.

  Returns:
    nn_features: A float32 np.array of nearest neighbor features of shape
      [1, height, width, n_objects, 1].
  """
  with tf.name_scope(
      'local_previous_frame_nearest_neighbor_features_per_object'):
    if USE_CORRELATION_COST:
      tf.logging.info('Using correlation_cost.')
      d = local_pairwise_distances(query_embedding, prev_frame_embedding,
                                   max_distance=max_distance)
    else:
      # Slow fallback in case correlation_cost is not available.
      tf.logging.warn('correlation cost is not available, using slow fallback '
                      'implementation.')
      d = local_pairwise_distances2(query_embedding, prev_frame_embedding,
                                    max_distance=max_distance)
    d = (tf.nn.sigmoid(d) - 0.5) * 2
    height = tf.shape(prev_frame_embedding)[0]
    width = tf.shape(prev_frame_embedding)[1]

    # Create offset versions of the mask.
    if USE_CORRELATION_COST:
      # New, faster code with cross-correlation via correlation_cost.
      # Due to padding we have to add 1 to the labels.
      offset_labels = correlation_cost_op.correlation_cost(
          tf.ones((1, height, width, 1)),
          tf.cast(prev_frame_labels + 1, tf.float32)[tf.newaxis],
          kernel_size=1,
          max_displacement=max_distance, stride_1=1, stride_2=1,
          pad=max_distance)
      offset_labels = tf.squeeze(offset_labels, axis=0)[..., tf.newaxis]
      # Subtract the 1 again and round.
      offset_labels = tf.round(offset_labels - 1)
      offset_masks = tf.equal(
          offset_labels,
          tf.cast(gt_ids, tf.float32)[tf.newaxis, tf.newaxis, tf.newaxis, :])
    else:
      # Slower code, without dependency to correlation_cost
      masks = tf.equal(prev_frame_labels, gt_ids[tf.newaxis, tf.newaxis])
      padded_masks = tf.pad(masks,
                            [[max_distance, max_distance],
                             [max_distance, max_distance],
                             [0, 0]])
      offset_masks = []
      for y_start in range(2 * max_distance + 1):
        y_end = y_start + height
        masks_slice = padded_masks[y_start:y_end]
        for x_start in range(2 * max_distance + 1):
          x_end = x_start + width
          offset_mask = masks_slice[:, x_start:x_end]
          offset_masks.append(offset_mask)
      offset_masks = tf.stack(offset_masks, axis=2)

    pad = tf.ones((height, width, (2 * max_distance + 1) ** 2, tf.size(gt_ids)))
    d_tiled = tf.tile(d[..., tf.newaxis], multiples=(1, 1, 1, tf.size(gt_ids)))
    d_masked = tf.where(offset_masks, d_tiled, pad)
    dists = tf.reduce_min(d_masked, axis=2)
    dists = tf.reshape(dists, (1, height, width, tf.size(gt_ids), 1))
    return dists
