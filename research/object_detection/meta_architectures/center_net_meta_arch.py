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
"""The CenterNet meta architecture as described in the "Objects as Points" paper [1].

[1]: https://arxiv.org/abs/1904.07850

"""

import abc
import collections
import functools
import tensorflow.compat.v1 as tf
import tensorflow.compat.v2 as tf2

from object_detection.core import box_list
from object_detection.core import box_list_ops
from object_detection.core import keypoint_ops
from object_detection.core import model
from object_detection.core import standard_fields as fields
from object_detection.core import target_assigner as cn_assigner
from object_detection.utils import shape_utils
from object_detection.utils import target_assigner_utils as ta_utils
from object_detection.utils import tf_version


# Number of channels needed to predict size and offsets.
NUM_OFFSET_CHANNELS = 2
NUM_SIZE_CHANNELS = 2

# Error range for detecting peaks.
PEAK_EPSILON = 1e-6


class CenterNetFeatureExtractor(tf.keras.Model):
  """Base class for feature extractors for the CenterNet meta architecture.

  Child classes are expected to override the _output_model property which will
  return 1 or more tensors predicted by the feature extractor.

  """
  __metaclass__ = abc.ABCMeta

  def __init__(self, name=None, channel_means=(0., 0., 0.),
               channel_stds=(1., 1., 1.), bgr_ordering=False):
    """Initializes a CenterNet feature extractor.

    Args:
      name: str, the name used for the underlying keras model.
      channel_means: A tuple of floats, denoting the mean of each channel
        which will be subtracted from it. If None or empty, we use 0s.
      channel_stds: A tuple of floats, denoting the standard deviation of each
        channel. Each channel will be divided by its standard deviation value.
        If None or empty, we use 1s.
      bgr_ordering: bool, if set will change the channel ordering to be in the
        [blue, red, green] order.
    """
    super(CenterNetFeatureExtractor, self).__init__(name=name)

    if channel_means is None or len(channel_means) == 0:  # pylint:disable=g-explicit-length-test
      channel_means = [0., 0., 0.]

    if channel_stds is None or len(channel_stds) == 0:  # pylint:disable=g-explicit-length-test
      channel_stds = [1., 1., 1.]

    self._channel_means = channel_means
    self._channel_stds = channel_stds
    self._bgr_ordering = bgr_ordering

  def preprocess(self, inputs):
    """Converts a batch of unscaled images to a scale suitable for the model.

    This method normalizes the image using the given `channel_means` and
    `channels_stds` values at initialization time while optionally flipping
    the channel order if `bgr_ordering` is set.

    Args:
      inputs: a [batch, height, width, channels] float32 tensor

    Returns:
      outputs: a [batch, height, width, channels] float32 tensor

    """

    if self._bgr_ordering:
      red, green, blue = tf.unstack(inputs, axis=3)
      inputs = tf.stack([blue, green, red], axis=3)

    channel_means = tf.reshape(tf.constant(self._channel_means),
                               [1, 1, 1, -1])
    channel_stds = tf.reshape(tf.constant(self._channel_stds),
                              [1, 1, 1, -1])

    return (inputs - channel_means)/channel_stds

  def preprocess_reverse(self, preprocessed_inputs):
    """Undo the preprocessing and return the raw image.

    This is a convenience function for some algorithms that require access
    to the raw inputs.

    Args:
      preprocessed_inputs: A [batch_size, height, width, channels] float
        tensor preprocessed_inputs from the preprocess function.

    Returns:
      images: A [batch_size, height, width, channels] float tensor with
        the preprocessing removed.
    """
    channel_means = tf.reshape(tf.constant(self._channel_means),
                               [1, 1, 1, -1])
    channel_stds = tf.reshape(tf.constant(self._channel_stds),
                              [1, 1, 1, -1])
    inputs = (preprocessed_inputs * channel_stds) + channel_means

    if self._bgr_ordering:
      blue, green, red = tf.unstack(inputs, axis=3)
      inputs = tf.stack([red, green, blue], axis=3)

    return inputs

  @property
  @abc.abstractmethod
  def out_stride(self):
    """The stride in the output image of the network."""
    pass

  @property
  @abc.abstractmethod
  def num_feature_outputs(self):
    """Ther number of feature outputs returned by the feature extractor."""
    pass

  @property
  def classification_backbone(self):
    raise NotImplementedError(
        'Classification backbone not supported for {}'.format(type(self)))


def make_prediction_net(num_out_channels, kernel_sizes=(3), num_filters=(256),
                        bias_fill=None, use_depthwise=False, name=None,
                        unit_height_conv=True):
  """Creates a network to predict the given number of output channels.

  This function is intended to make the prediction heads for the CenterNet
  meta architecture.

  Args:
    num_out_channels: Number of output channels.
    kernel_sizes: A list representing the sizes of the conv kernel in the
      intermediate layer. Note that the length of the list indicates the number
      of intermediate conv layers and it must be the same as the length of the
      num_filters.
    num_filters: A list representing the number of filters in the intermediate
      conv layer. Note that the length of the list indicates the number of
      intermediate conv layers.
    bias_fill: If not None, is used to initialize the bias in the final conv
      layer.
    use_depthwise: If true, use SeparableConv2D to construct the Sequential
      layers instead of Conv2D.
    name: Optional name for the prediction net.
    unit_height_conv: If True, Conv2Ds have asymmetric kernels with height=1.

  Returns:
    net: A keras module which when called on an input tensor of size
      [batch_size, height, width, num_in_channels] returns an output
      of size [batch_size, height, width, num_out_channels]
  """
  if isinstance(kernel_sizes, int) and isinstance(num_filters, int):
    kernel_sizes = [kernel_sizes]
    num_filters = [num_filters]
  assert len(kernel_sizes) == len(num_filters)
  if use_depthwise:
    conv_fn = tf.keras.layers.SeparableConv2D
  else:
    conv_fn = tf.keras.layers.Conv2D

  # We name the convolution operations explicitly because Keras, by default,
  # uses different names during training and evaluation. By setting the names
  # here, we avoid unexpected pipeline breakage in TF1.
  out_conv = tf.keras.layers.Conv2D(
      num_out_channels,
      kernel_size=1,
      name='conv1' if tf_version.is_tf1() else None)

  if bias_fill is not None:
    out_conv.bias_initializer = tf.keras.initializers.constant(bias_fill)

  layers = []
  for idx, (kernel_size,
            num_filter) in enumerate(zip(kernel_sizes, num_filters)):
    layers.append(
        conv_fn(
            num_filter,
            kernel_size=[1, kernel_size] if unit_height_conv else kernel_size,
            padding='same',
            name='conv2_%d' % idx if tf_version.is_tf1() else None))
    layers.append(tf.keras.layers.ReLU())
  layers.append(out_conv)
  net = tf.keras.Sequential(layers, name=name)
  return net


def _to_float32(x):
  return tf.cast(x, tf.float32)


def _get_shape(tensor, num_dims):
  assert len(tensor.shape.as_list()) == num_dims
  return shape_utils.combined_static_and_dynamic_shape(tensor)


def _flatten_spatial_dimensions(batch_images):
  batch_size, height, width, channels = _get_shape(batch_images, 4)
  return tf.reshape(batch_images, [batch_size, height * width,
                                   channels])


def _multi_range(limit,
                 value_repetitions=1,
                 range_repetitions=1,
                 dtype=tf.int32):
  """Creates a sequence with optional value duplication and range repetition.

  As an example (see the Args section for more details),
  _multi_range(limit=2, value_repetitions=3, range_repetitions=4) returns:

  [0, 0, 0, 1, 1, 1, 0, 0, 0, 1, 1, 1, 0, 0, 0, 1, 1, 1, 0, 0, 0, 1, 1, 1]

  Args:
    limit: A 0-D Tensor (scalar). Upper limit of sequence, exclusive.
    value_repetitions: Integer. The number of times a value in the sequence is
      repeated. With value_repetitions=3, the result is [0, 0, 0, 1, 1, 1, ..].
    range_repetitions: Integer. The number of times the range is repeated. With
      range_repetitions=3, the result is [0, 1, 2, .., 0, 1, 2, ..].
    dtype: The type of the elements of the resulting tensor.

  Returns:
    A 1-D tensor of type `dtype` and size
      [`limit` * `value_repetitions` * `range_repetitions`] that contains the
      specified range with given repetitions.
  """
  return tf.reshape(
      tf.tile(
          tf.expand_dims(tf.range(limit, dtype=dtype), axis=-1),
          multiples=[range_repetitions, value_repetitions]), [-1])


def top_k_feature_map_locations(feature_map, max_pool_kernel_size=3, k=100,
                                per_channel=False):
  """Returns the top k scores and their locations in a feature map.

  Given a feature map, the top k values (based on activation) are returned. If
  `per_channel` is True, the top k values **per channel** are returned. Note
  that when k equals to 1, ths function uses reduce_max and argmax instead of
  top_k to make the logics more efficient.

  The `max_pool_kernel_size` argument allows for selecting local peaks in a
  region. This filtering is done per channel, so nothing prevents two values at
  the same location to be returned.

  Args:
    feature_map: [batch, height, width, channels] float32 feature map.
    max_pool_kernel_size: integer, the max pool kernel size to use to pull off
      peak score locations in a neighborhood (independently for each channel).
      For example, to make sure no two neighboring values (in the same channel)
      are returned, set max_pool_kernel_size=3. If None or 1, will not apply max
      pooling.
    k: The number of highest scoring locations to return.
    per_channel: If True, will return the top k scores and locations per
      feature map channel. If False, the top k across the entire feature map
      (height x width x channels) are returned.

  Returns:
    Tuple of
    scores: A [batch, N] float32 tensor with scores from the feature map in
      descending order. If per_channel is False, N = k. Otherwise,
      N = k * channels, and the first k elements correspond to channel 0, the
      second k correspond to channel 1, etc.
    y_indices: A [batch, N] int tensor with y indices of the top k feature map
      locations. If per_channel is False, N = k. Otherwise,
      N = k * channels.
    x_indices: A [batch, N] int tensor with x indices of the top k feature map
      locations. If per_channel is False, N = k. Otherwise,
      N = k * channels.
    channel_indices: A [batch, N] int tensor with channel indices of the top k
      feature map locations. If per_channel is False, N = k. Otherwise,
      N = k * channels.
  """
  if not max_pool_kernel_size or max_pool_kernel_size == 1:
    feature_map_peaks = feature_map
  else:
    feature_map_max_pool = tf.nn.max_pool(
        feature_map, ksize=max_pool_kernel_size, strides=1, padding='SAME')

    feature_map_peak_mask = tf.math.abs(
        feature_map - feature_map_max_pool) < PEAK_EPSILON

    # Zero out everything that is not a peak.
    feature_map_peaks = (
        feature_map * _to_float32(feature_map_peak_mask))

  batch_size, _, width, num_channels = _get_shape(feature_map, 4)

  if per_channel:
    if k == 1:
      feature_map_flattened = tf.reshape(
          feature_map_peaks, [batch_size, -1, num_channels])
      scores = tf.math.reduce_max(feature_map_flattened, axis=1)
      peak_flat_indices = tf.math.argmax(
          feature_map_flattened, axis=1, output_type=tf.dtypes.int32)
      peak_flat_indices = tf.expand_dims(peak_flat_indices, axis=-1)
    else:
      # Perform top k over batch and channels.
      feature_map_peaks_transposed = tf.transpose(feature_map_peaks,
                                                  perm=[0, 3, 1, 2])
      feature_map_peaks_transposed = tf.reshape(
          feature_map_peaks_transposed, [batch_size, num_channels, -1])
      # safe_k will be used whenever there are fewer positions in the heatmap
      # than the requested number of locations to score. In that case, all
      # positions are returned in sorted order. To ensure consistent shapes for
      # downstream ops the outputs are padded with zeros. Safe_k is also
      # fine for TPU because TPUs require a fixed input size so the number of
      # positions will also be fixed.
      safe_k = tf.minimum(k, tf.shape(feature_map_peaks_transposed)[-1])
      scores, peak_flat_indices = tf.math.top_k(
          feature_map_peaks_transposed, k=safe_k)
      scores = tf.pad(scores, [(0, 0), (0, 0), (0, k - safe_k)])
      peak_flat_indices = tf.pad(peak_flat_indices,
                                 [(0, 0), (0, 0), (0, k - safe_k)])
      scores = tf.ensure_shape(scores, (batch_size, num_channels, k))
      peak_flat_indices = tf.ensure_shape(peak_flat_indices,
                                          (batch_size, num_channels, k))
    # Convert the indices such that they represent the location in the full
    # (flattened) feature map of size [batch, height * width * channels].
    channel_idx = tf.range(num_channels)[tf.newaxis, :, tf.newaxis]
    peak_flat_indices = num_channels * peak_flat_indices + channel_idx
    scores = tf.reshape(scores, [batch_size, -1])
    peak_flat_indices = tf.reshape(peak_flat_indices, [batch_size, -1])
  else:
    if k == 1:
      feature_map_peaks_flat = tf.reshape(feature_map_peaks, [batch_size, -1])
      scores = tf.math.reduce_max(feature_map_peaks_flat, axis=1, keepdims=True)
      peak_flat_indices = tf.expand_dims(tf.math.argmax(
          feature_map_peaks_flat, axis=1, output_type=tf.dtypes.int32), axis=-1)
    else:
      feature_map_peaks_flat = tf.reshape(feature_map_peaks, [batch_size, -1])
      safe_k = tf.minimum(k, tf.shape(feature_map_peaks_flat)[1])
      scores, peak_flat_indices = tf.math.top_k(feature_map_peaks_flat,
                                                k=safe_k)

  # Get x, y and channel indices corresponding to the top indices in the flat
  # array.
  y_indices, x_indices, channel_indices = (
      row_col_channel_indices_from_flattened_indices(
          peak_flat_indices, width, num_channels))
  return scores, y_indices, x_indices, channel_indices


def prediction_tensors_to_boxes(y_indices, x_indices, height_width_predictions,
                                offset_predictions):
  """Converts CenterNet class-center, offset and size predictions to boxes.

  Args:
    y_indices: A [batch, num_boxes] int32 tensor with y indices corresponding to
      object center locations (expressed in output coordinate frame).
    x_indices: A [batch, num_boxes] int32 tensor with x indices corresponding to
      object center locations (expressed in output coordinate frame).
    height_width_predictions: A float tensor of shape [batch_size, height,
      width, 2] representing the height and width of a box centered at each
      pixel.
    offset_predictions: A float tensor of shape [batch_size, height, width, 2]
      representing the y and x offsets of a box centered at each pixel. This
      helps reduce the error from downsampling.

  Returns:
    detection_boxes: A tensor of shape [batch_size, num_boxes, 4] holding the
      the raw bounding box coordinates of boxes.
  """
  batch_size, num_boxes = _get_shape(y_indices, 2)
  _, height, width, _ = _get_shape(height_width_predictions, 4)
  height, width = tf.cast(height, tf.float32), tf.cast(width, tf.float32)

  # TF Lite does not support tf.gather with batch_dims > 0, so we need to use
  # tf_gather_nd instead and here we prepare the indices for that.
  combined_indices = tf.stack([
      _multi_range(batch_size, value_repetitions=num_boxes),
      tf.reshape(y_indices, [-1]),
      tf.reshape(x_indices, [-1])
  ], axis=1)
  new_height_width = tf.gather_nd(height_width_predictions, combined_indices)
  new_height_width = tf.reshape(new_height_width, [batch_size, num_boxes, 2])

  new_offsets = tf.gather_nd(offset_predictions, combined_indices)
  offsets = tf.reshape(new_offsets, [batch_size, num_boxes, 2])

  y_indices = _to_float32(y_indices)
  x_indices = _to_float32(x_indices)

  height_width = tf.maximum(new_height_width, 0)
  heights, widths = tf.unstack(height_width, axis=2)
  y_offsets, x_offsets = tf.unstack(offsets, axis=2)

  ymin = y_indices + y_offsets - heights / 2.0
  xmin = x_indices + x_offsets - widths / 2.0
  ymax = y_indices + y_offsets + heights / 2.0
  xmax = x_indices + x_offsets + widths / 2.0

  ymin = tf.clip_by_value(ymin, 0., height)
  xmin = tf.clip_by_value(xmin, 0., width)
  ymax = tf.clip_by_value(ymax, 0., height)
  xmax = tf.clip_by_value(xmax, 0., width)
  boxes = tf.stack([ymin, xmin, ymax, xmax], axis=2)

  return boxes


def prediction_tensors_to_temporal_offsets(
    y_indices, x_indices, offset_predictions):
  """Converts CenterNet temporal offset map predictions to batched format.

  This function is similar to the box offset conversion function, as both
  temporal offsets and box offsets are size-2 vectors.

  Args:
    y_indices: A [batch, num_boxes] int32 tensor with y indices corresponding to
      object center locations (expressed in output coordinate frame).
    x_indices: A [batch, num_boxes] int32 tensor with x indices corresponding to
      object center locations (expressed in output coordinate frame).
    offset_predictions: A float tensor of shape [batch_size, height, width, 2]
      representing the y and x offsets of a box's center across adjacent frames.

  Returns:
    offsets: A tensor of shape [batch_size, num_boxes, 2] holding the
      the object temporal offsets of (y, x) dimensions.

  """
  batch_size, num_boxes = _get_shape(y_indices, 2)

  # TF Lite does not support tf.gather with batch_dims > 0, so we need to use
  # tf_gather_nd instead and here we prepare the indices for that.
  combined_indices = tf.stack([
      _multi_range(batch_size, value_repetitions=num_boxes),
      tf.reshape(y_indices, [-1]),
      tf.reshape(x_indices, [-1])
  ], axis=1)

  new_offsets = tf.gather_nd(offset_predictions, combined_indices)
  offsets = tf.reshape(new_offsets, [batch_size, num_boxes, -1])

  return offsets


def prediction_tensors_to_keypoint_candidates(keypoint_heatmap_predictions,
                                              keypoint_heatmap_offsets,
                                              keypoint_score_threshold=0.1,
                                              max_pool_kernel_size=1,
                                              max_candidates=20,
                                              keypoint_depths=None):
  """Convert keypoint heatmap predictions and offsets to keypoint candidates.

  Args:
    keypoint_heatmap_predictions: A float tensor of shape [batch_size, height,
      width, num_keypoints] representing the per-keypoint heatmaps.
    keypoint_heatmap_offsets: A float tensor of shape [batch_size, height,
      width, 2] (or [batch_size, height, width, 2 * num_keypoints] if
      'per_keypoint_offset' is set True) representing the per-keypoint offsets.
    keypoint_score_threshold: float, the threshold for considering a keypoint a
      candidate.
    max_pool_kernel_size: integer, the max pool kernel size to use to pull off
      peak score locations in a neighborhood. For example, to make sure no two
      neighboring values for the same keypoint are returned, set
      max_pool_kernel_size=3. If None or 1, will not apply any local filtering.
    max_candidates: integer, maximum number of keypoint candidates per keypoint
      type.
    keypoint_depths: (optional) A float tensor of shape [batch_size, height,
      width, 1] (or [batch_size, height, width, num_keypoints] if
      'per_keypoint_depth' is set True) representing the per-keypoint depths.

  Returns:
    keypoint_candidates: A tensor of shape
      [batch_size, max_candidates, num_keypoints, 2] holding the
      location of keypoint candidates in [y, x] format (expressed in absolute
      coordinates in the output coordinate frame).
    keypoint_scores: A float tensor of shape
      [batch_size, max_candidates, num_keypoints] with the scores for each
      keypoint candidate. The scores come directly from the heatmap predictions.
    num_keypoint_candidates: An integer tensor of shape
      [batch_size, num_keypoints] with the number of candidates for each
      keypoint type, as it's possible to filter some candidates due to the score
      threshold.
    depth_candidates: A tensor of shape [batch_size, max_candidates,
      num_keypoints] representing the estimated depth of each keypoint
      candidate. Return None if the input keypoint_depths is None.
  """
  batch_size, _, _, num_keypoints = _get_shape(keypoint_heatmap_predictions, 4)
  # Get x, y and channel indices corresponding to the top indices in the
  # keypoint heatmap predictions.
  # Note that the top k candidates are produced for **each keypoint type**.
  # Might be worth eventually trying top k in the feature map, independent of
  # the keypoint type.
  keypoint_scores, y_indices, x_indices, channel_indices = (
      top_k_feature_map_locations(keypoint_heatmap_predictions,
                                  max_pool_kernel_size=max_pool_kernel_size,
                                  k=max_candidates,
                                  per_channel=True))

  # TF Lite does not support tf.gather with batch_dims > 0, so we need to use
  # tf_gather_nd instead and here we prepare the indices for that.
  _, num_indices = _get_shape(y_indices, 2)
  combined_indices = tf.stack([
      _multi_range(batch_size, value_repetitions=num_indices),
      tf.reshape(y_indices, [-1]),
      tf.reshape(x_indices, [-1])
  ], axis=1)

  selected_offsets_flat = tf.gather_nd(keypoint_heatmap_offsets,
                                       combined_indices)
  selected_offsets = tf.reshape(selected_offsets_flat,
                                [batch_size, num_indices, -1])

  y_indices = _to_float32(y_indices)
  x_indices = _to_float32(x_indices)

  _, _, num_channels = _get_shape(selected_offsets, 3)
  if num_channels > 2:
    # Offsets are per keypoint and the last dimension of selected_offsets
    # contains all those offsets, so reshape the offsets to make sure that the
    # last dimension contains (y_offset, x_offset) for a single keypoint.
    reshaped_offsets = tf.reshape(selected_offsets,
                                  [batch_size, num_indices, -1, 2])

    # TF Lite does not support tf.gather with batch_dims > 0, so we need to use
    # tf_gather_nd instead and here we prepare the indices for that. In this
    # case, channel_indices indicates which keypoint to use the offset from.
    channel_combined_indices = tf.stack([
        _multi_range(batch_size, value_repetitions=num_indices),
        _multi_range(num_indices, range_repetitions=batch_size),
        tf.reshape(channel_indices, [-1])
    ], axis=1)

    offsets = tf.gather_nd(reshaped_offsets, channel_combined_indices)
    offsets = tf.reshape(offsets, [batch_size, num_indices, -1])
  else:
    offsets = selected_offsets
  y_offsets, x_offsets = tf.unstack(offsets, axis=2)

  keypoint_candidates = tf.stack([y_indices + y_offsets,
                                  x_indices + x_offsets], axis=2)
  keypoint_candidates = tf.reshape(
      keypoint_candidates,
      [batch_size, num_keypoints, max_candidates, 2])
  keypoint_candidates = tf.transpose(keypoint_candidates, [0, 2, 1, 3])
  keypoint_scores = tf.reshape(
      keypoint_scores,
      [batch_size, num_keypoints, max_candidates])
  keypoint_scores = tf.transpose(keypoint_scores, [0, 2, 1])
  num_candidates = tf.reduce_sum(
      tf.to_int32(keypoint_scores >= keypoint_score_threshold), axis=1)

  depth_candidates = None
  if keypoint_depths is not None:
    selected_depth_flat = tf.gather_nd(keypoint_depths, combined_indices)
    selected_depth = tf.reshape(selected_depth_flat,
                                [batch_size, num_indices, -1])
    _, _, num_depth_channels = _get_shape(selected_depth, 3)
    if num_depth_channels > 1:
      combined_indices = tf.stack([
          _multi_range(batch_size, value_repetitions=num_indices),
          _multi_range(num_indices, range_repetitions=batch_size),
          tf.reshape(channel_indices, [-1])
      ], axis=1)
      depth = tf.gather_nd(selected_depth, combined_indices)
      depth = tf.reshape(depth, [batch_size, num_indices, -1])
    else:
      depth = selected_depth
    depth_candidates = tf.reshape(depth,
                                  [batch_size, num_keypoints, max_candidates])
    depth_candidates = tf.transpose(depth_candidates, [0, 2, 1])

  return keypoint_candidates, keypoint_scores, num_candidates, depth_candidates


def argmax_feature_map_locations(feature_map):
  """Returns the peak locations in the feature map."""
  batch_size, _, width, num_channels = _get_shape(feature_map, 4)

  feature_map_flattened = tf.reshape(
      feature_map, [batch_size, -1, num_channels])
  peak_flat_indices = tf.math.argmax(
      feature_map_flattened, axis=1, output_type=tf.dtypes.int32)
  # Get x and y indices corresponding to the top indices in the flat array.
  y_indices, x_indices = (
      row_col_indices_from_flattened_indices(peak_flat_indices, width))
  channel_indices = tf.tile(
      tf.range(num_channels)[tf.newaxis, :], [batch_size, 1])
  return y_indices, x_indices, channel_indices


def prediction_tensors_to_single_instance_kpts(
    keypoint_heatmap_predictions,
    keypoint_heatmap_offsets,
    keypoint_score_heatmap=None):
  """Convert keypoint heatmap predictions and offsets to keypoint candidates.

  Args:
    keypoint_heatmap_predictions: A float tensor of shape [batch_size, height,
      width, num_keypoints] representing the per-keypoint heatmaps which is
      used for finding the best keypoint candidate locations.
    keypoint_heatmap_offsets: A float tensor of shape [batch_size, height,
      width, 2] (or [batch_size, height, width, 2 * num_keypoints] if
      'per_keypoint_offset' is set True) representing the per-keypoint offsets.
    keypoint_score_heatmap: (optional) A float tensor of shape [batch_size,
      height, width, num_keypoints] representing the heatmap which is used for
      reporting the confidence scores. If not provided, then the values in the
      keypoint_heatmap_predictions will be used.

  Returns:
    keypoint_candidates: A tensor of shape
      [batch_size, max_candidates, num_keypoints, 2] holding the
      location of keypoint candidates in [y, x] format (expressed in absolute
      coordinates in the output coordinate frame).
    keypoint_scores: A float tensor of shape
      [batch_size, max_candidates, num_keypoints] with the scores for each
      keypoint candidate. The scores come directly from the heatmap predictions.
    num_keypoint_candidates: An integer tensor of shape
      [batch_size, num_keypoints] with the number of candidates for each
      keypoint type, as it's possible to filter some candidates due to the score
      threshold.
  """
  batch_size, _, _, num_keypoints = _get_shape(
      keypoint_heatmap_predictions, 4)
  # Get x, y and channel indices corresponding to the top indices in the
  # keypoint heatmap predictions.
  y_indices, x_indices, channel_indices = argmax_feature_map_locations(
      keypoint_heatmap_predictions)

  # TF Lite does not support tf.gather with batch_dims > 0, so we need to use
  # tf_gather_nd instead and here we prepare the indices for that.
  _, num_keypoints = _get_shape(y_indices, 2)
  combined_indices = tf.stack([
      _multi_range(batch_size, value_repetitions=num_keypoints),
      tf.reshape(y_indices, [-1]),
      tf.reshape(x_indices, [-1]),
  ], axis=1)

  # shape: [num_keypoints, num_keypoints * 2]
  selected_offsets_flat = tf.gather_nd(keypoint_heatmap_offsets,
                                       combined_indices)
  # shape: [num_keypoints, num_keypoints, 2].
  selected_offsets_flat = tf.reshape(
      selected_offsets_flat, [num_keypoints, num_keypoints, -1])
  # shape: [num_keypoints].
  channel_indices = tf.keras.backend.flatten(channel_indices)
  # shape: [num_keypoints, 2].
  retrieve_indices = tf.stack([channel_indices, channel_indices], axis=1)
  # shape: [num_keypoints, 2]
  selected_offsets = tf.gather_nd(selected_offsets_flat, retrieve_indices)
  y_offsets, x_offsets = tf.unstack(selected_offsets, axis=1)

  keypoint_candidates = tf.stack([
      tf.cast(y_indices, dtype=tf.float32) + tf.expand_dims(y_offsets, axis=0),
      tf.cast(x_indices, dtype=tf.float32) + tf.expand_dims(x_offsets, axis=0)
  ], axis=2)
  keypoint_candidates = tf.expand_dims(keypoint_candidates, axis=0)

  # Append the channel indices back to retrieve the keypoint scores from the
  # heatmap.
  combined_indices = tf.concat(
      [combined_indices, tf.expand_dims(channel_indices, axis=-1)], axis=1)
  if keypoint_score_heatmap is None:
    keypoint_scores = tf.gather_nd(
        keypoint_heatmap_predictions, combined_indices)
  else:
    keypoint_scores = tf.gather_nd(keypoint_score_heatmap, combined_indices)
  keypoint_scores = tf.expand_dims(
      tf.expand_dims(keypoint_scores, axis=0), axis=0)
  return keypoint_candidates, keypoint_scores


def _score_to_distance_map(y_grid, x_grid, heatmap, points_y, points_x,
                           score_distance_offset):
  """Rescores heatmap using the distance information.

  Rescore the heatmap scores using the formula:
  score / (d + score_distance_offset), where the d is the distance from each
  pixel location to the target point location.

  Args:
    y_grid: A float tensor with shape [height, width] representing the
      y-coordinate of each pixel grid.
    x_grid: A float tensor with shape [height, width] representing the
      x-coordinate of each pixel grid.
    heatmap: A float tensor with shape [1, height, width, channel]
      representing the heatmap to be rescored.
    points_y: A float tensor with shape [channel] representing the y
      coordinates of the target points for each channel.
    points_x: A float tensor with shape [channel] representing the x
      coordinates of the target points for each channel.
    score_distance_offset: A constant used in the above formula.

  Returns:
    A float tensor with shape [1, height, width, channel] representing the
    rescored heatmap.
  """
  y_diff = y_grid[:, :, tf.newaxis] - points_y
  x_diff = x_grid[:, :, tf.newaxis] - points_x
  distance = tf.math.sqrt(y_diff**2 + x_diff**2)
  return tf.math.divide(heatmap, distance + score_distance_offset)


def prediction_to_single_instance_keypoints(
    object_heatmap,
    keypoint_heatmap,
    keypoint_offset,
    keypoint_regression,
    kp_params,
    keypoint_depths=None):
  """Postprocess function to predict single instance keypoints.

  This is a simplified postprocessing function based on the assumption that
  there is only one instance in the image. If there are multiple instances in
  the image, the model prefers to predict the one that is closest to the image
  center. Here is a high-level description of what this function does:
    1) Object heatmap re-weighted by the distance between each pixel to the
       image center is used to determine the instance center.
    2) Regressed keypoint locations are retrieved from the instance center. The
       Gaussian kernel is applied to the regressed keypoint locations to
       re-weight the keypoint heatmap. This is to select the keypoints that are
       associated with the center instance without using top_k op.
    3) The keypoint locations are computed by the re-weighted keypoint heatmap
       and the keypoint offset.

  Args:
    object_heatmap: A float tensor of shape [1, height, width, 1] representing
      the heapmap of the class.
    keypoint_heatmap: A float tensor of shape [1, height, width, num_keypoints]
      representing the per-keypoint heatmaps.
    keypoint_offset: A float tensor of shape [1, height, width, 2] (or [1,
      height, width, 2 * num_keypoints] if 'per_keypoint_offset' is set True)
      representing the per-keypoint offsets.
    keypoint_regression: A float  tensor of shape [1, height, width, 2 *
      num_keypoints] representing the joint regression prediction.
    kp_params: A `KeypointEstimationParams` object with parameters for a single
      keypoint class.
    keypoint_depths: (optional) A float tensor of shape [batch_size, height,
      width, 1] (or [batch_size, height, width, num_keypoints] if
      'per_keypoint_depth' is set True) representing the per-keypoint depths.

  Returns:
    A tuple of two tensors:
      keypoint_candidates: A float tensor with shape [1, 1, num_keypoints, 2]
        representing the yx-coordinates of the keypoints in the output feature
        map space.
      keypoint_scores: A float tensor with shape [1, 1, num_keypoints]
        representing the keypoint prediction scores.

  Raises:
    ValueError: if the input keypoint_std_dev doesn't have valid number of
      elements (1 or num_keypoints).
  """
  # TODO(yuhuic): add the keypoint depth prediction logics in the browser
  # postprocessing back.
  del keypoint_depths

  num_keypoints = len(kp_params.keypoint_std_dev)
  batch_size, height, width, _ = _get_shape(keypoint_heatmap, 4)

  # Create the image center location.
  image_center_y = tf.convert_to_tensor([0.5 * height], dtype=tf.float32)
  image_center_x = tf.convert_to_tensor([0.5 * width], dtype=tf.float32)
  (y_grid, x_grid) = ta_utils.image_shape_to_grids(height, width)
  # Rescore the object heatmap by the distnace to the image center.
  object_heatmap = _score_to_distance_map(
      y_grid, x_grid, object_heatmap, image_center_y,
      image_center_x, kp_params.score_distance_offset)

  # Pick the highest score and location of the weighted object heatmap.
  y_indices, x_indices, _ = argmax_feature_map_locations(object_heatmap)
  _, num_indices = _get_shape(y_indices, 2)
  combined_indices = tf.stack([
      _multi_range(batch_size, value_repetitions=num_indices),
      tf.reshape(y_indices, [-1]),
      tf.reshape(x_indices, [-1])
  ], axis=1)

  # Select the regression vectors from the object center.
  selected_regression_flat = tf.gather_nd(keypoint_regression, combined_indices)
  # shape: [num_keypoints, 2]
  regression_offsets = tf.reshape(selected_regression_flat, [num_keypoints, -1])
  (y_reg, x_reg) = tf.unstack(regression_offsets, axis=1)
  y_regressed = tf.cast(y_indices, dtype=tf.float32) + y_reg
  x_regressed = tf.cast(x_indices, dtype=tf.float32) + x_reg

  if kp_params.candidate_ranking_mode == 'score_distance_ratio':
    reweighted_keypoint_heatmap = _score_to_distance_map(
        y_grid, x_grid, keypoint_heatmap, y_regressed, x_regressed,
        kp_params.score_distance_offset)
  else:
    raise ValueError('Unsupported candidate_ranking_mode: %s' %
                     kp_params.candidate_ranking_mode)

  # Get the keypoint locations/scores:
  #   keypoint_candidates: [1, 1, num_keypoints, 2]
  #   keypoint_scores: [1, 1, num_keypoints]
  #   depth_candidates: [1, 1, num_keypoints]
  (keypoint_candidates, keypoint_scores
   ) = prediction_tensors_to_single_instance_kpts(
       reweighted_keypoint_heatmap,
       keypoint_offset,
       keypoint_score_heatmap=keypoint_heatmap)
  return keypoint_candidates, keypoint_scores, None


def _gaussian_weighted_map_const_multi(
    y_grid, x_grid, heatmap, points_y, points_x, boxes,
    gaussian_denom_ratio):
  """Rescores heatmap using the distance information.

  The function is called when the candidate_ranking_mode in the
  KeypointEstimationParams is set to be 'gaussian_weighted_const'. The
  keypoint candidates are ranked using the formula:
    heatmap_score * exp((-distances^2) / (gaussian_denom))

  where 'gaussian_denom' is determined by:
    min(output_feature_height, output_feature_width) * gaussian_denom_ratio

  the 'distances' are the distances between the grid coordinates and the target
  points.

  Note that the postfix 'const' refers to the fact that the denominator is a
  constant given the input image size, not scaled by the size of each of the
  instances.

  Args:
    y_grid: A float tensor with shape [height, width] representing the
      y-coordinate of each pixel grid.
    x_grid: A float tensor with shape [height, width] representing the
      x-coordinate of each pixel grid.
    heatmap: A float tensor with shape [height, width, num_keypoints]
      representing the heatmap to be rescored.
    points_y: A float tensor with shape [num_instances, num_keypoints]
      representing the y coordinates of the target points for each channel.
    points_x: A float tensor with shape [num_instances, num_keypoints]
      representing the x coordinates of the target points for each channel.
    boxes: A tensor of shape [num_instances, 4] with predicted bounding boxes
      for each instance, expressed in the output coordinate frame.
    gaussian_denom_ratio: A constant used in the above formula that determines
      the denominator of the Gaussian kernel.

  Returns:
    A float tensor with shape [height, width, channel] representing
    the rescored heatmap.
  """
  num_instances, _ = _get_shape(boxes, 2)
  height, width, num_keypoints = _get_shape(heatmap, 3)

  # [height, width, num_instances, num_keypoints].
  # Note that we intentionally avoid using tf.newaxis as TfLite converter
  # doesn't like it.
  y_diff = (
      tf.reshape(y_grid, [height, width, 1, 1]) -
      tf.reshape(points_y, [1, 1, num_instances, num_keypoints]))
  x_diff = (
      tf.reshape(x_grid, [height, width, 1, 1]) -
      tf.reshape(points_x, [1, 1, num_instances, num_keypoints]))
  distance_square = y_diff * y_diff + x_diff * x_diff

  y_min, x_min, y_max, x_max = tf.split(boxes, 4, axis=1)

  # Make the mask with all 1.0 in the box regions.
  # Shape: [height, width, num_instances]
  in_boxes = tf.math.logical_and(
      tf.math.logical_and(
          tf.reshape(y_grid, [height, width, 1]) >= tf.reshape(
              y_min, [1, 1, num_instances]),
          tf.reshape(y_grid, [height, width, 1]) < tf.reshape(
              y_max, [1, 1, num_instances])),
      tf.math.logical_and(
          tf.reshape(x_grid, [height, width, 1]) >= tf.reshape(
              x_min, [1, 1, num_instances]),
          tf.reshape(x_grid, [height, width, 1]) < tf.reshape(
              x_max, [1, 1, num_instances])))
  in_boxes = tf.cast(in_boxes, dtype=tf.float32)

  gaussian_denom = tf.cast(
      tf.minimum(height, width), dtype=tf.float32) * gaussian_denom_ratio
  # shape: [height, width, num_instances, num_keypoints]
  gaussian_map = tf.exp((-1 * distance_square) / gaussian_denom)
  return tf.expand_dims(heatmap, axis=2) * gaussian_map * tf.reshape(
      in_boxes, [height, width, num_instances, 1])


def prediction_tensors_to_multi_instance_kpts(
    keypoint_heatmap_predictions,
    keypoint_heatmap_offsets,
    keypoint_score_heatmap=None):
  """Converts keypoint heatmap predictions and offsets to keypoint candidates.

  This function is similar to the 'prediction_tensors_to_single_instance_kpts'
  function except that the input keypoint_heatmap_predictions is prepared to
  have an additional 'num_instances' dimension for multi-instance prediction.

  Args:
    keypoint_heatmap_predictions: A float tensor of shape [height,
      width, num_instances, num_keypoints] representing the per-keypoint and
      per-instance heatmaps which is used for finding the best keypoint
      candidate locations.
    keypoint_heatmap_offsets: A float tensor of shape [height,
      width, 2 * num_keypoints] representing the per-keypoint offsets.
    keypoint_score_heatmap: (optional) A float tensor of shape [height, width,
      num_keypoints] representing the heatmap which is used for reporting the
      confidence scores. If not provided, then the values in the
      keypoint_heatmap_predictions will be used.

  Returns:
    keypoint_candidates: A tensor of shape
      [1, max_candidates, num_keypoints, 2] holding the
      location of keypoint candidates in [y, x] format (expressed in absolute
      coordinates in the output coordinate frame).
    keypoint_scores: A float tensor of shape
      [1, max_candidates, num_keypoints] with the scores for each
      keypoint candidate. The scores come directly from the heatmap predictions.
  """
  height, width, num_instances, num_keypoints = _get_shape(
      keypoint_heatmap_predictions, 4)

  # [height * width, num_instances * num_keypoints].
  feature_map_flattened = tf.reshape(
      keypoint_heatmap_predictions,
      [-1, num_instances * num_keypoints])

  # [num_instances * num_keypoints].
  peak_flat_indices = tf.math.argmax(
      feature_map_flattened, axis=0, output_type=tf.dtypes.int32)

  # Get x and y indices corresponding to the top indices in the flat array.
  y_indices, x_indices = (
      row_col_indices_from_flattened_indices(peak_flat_indices, width))
  # [num_instances * num_keypoints].
  y_indices = tf.reshape(y_indices, [-1])
  x_indices = tf.reshape(x_indices, [-1])

  # Prepare the indices to gather the offsets from the keypoint_heatmap_offsets.
  kpts_idx = _multi_range(
      limit=num_keypoints, value_repetitions=1,
      range_repetitions=num_instances)
  combined_indices = tf.stack([
      y_indices,
      x_indices,
      kpts_idx
  ], axis=1)

  keypoint_heatmap_offsets = tf.reshape(
      keypoint_heatmap_offsets, [height, width, num_keypoints, 2])
  # Retrieve the keypoint offsets: shape:
  # [num_instance * num_keypoints, 2].
  selected_offsets_flat = tf.gather_nd(keypoint_heatmap_offsets,
                                       combined_indices)
  y_offsets, x_offsets = tf.unstack(selected_offsets_flat, axis=1)

  keypoint_candidates = tf.stack([
      tf.cast(y_indices, dtype=tf.float32) + tf.expand_dims(y_offsets, axis=0),
      tf.cast(x_indices, dtype=tf.float32) + tf.expand_dims(x_offsets, axis=0)
  ], axis=2)
  keypoint_candidates = tf.reshape(
      keypoint_candidates, [num_instances, num_keypoints, 2])

  if keypoint_score_heatmap is None:
    keypoint_scores = tf.gather_nd(
        tf.reduce_max(keypoint_heatmap_predictions, axis=2), combined_indices)
  else:
    keypoint_scores = tf.gather_nd(keypoint_score_heatmap, combined_indices)
  return tf.expand_dims(keypoint_candidates, axis=0), tf.reshape(
      keypoint_scores, [1, num_instances, num_keypoints])


def prediction_to_keypoints_argmax(
    prediction_dict,
    object_y_indices,
    object_x_indices,
    boxes,
    task_name,
    kp_params):
  """Postprocess function to predict multi instance keypoints with argmax op.

  This is a different implementation of the original keypoint postprocessing
  function such that it avoids using topk op (replaced by argmax) as it runs
  much slower in the browser. Note that in this function, we assume the
  batch_size to be 1 to avoid using 5D tensors which cause issues when
  converting to the TfLite model.

  Args:
    prediction_dict: a dictionary holding predicted tensors, returned from the
      predict() method. This dictionary should contain keypoint prediction
      feature maps for each keypoint task.
    object_y_indices: A float tensor of shape [batch_size, max_instances]
      representing the location indices of the object centers.
    object_x_indices: A float tensor of shape [batch_size, max_instances]
      representing the location indices of the object centers.
    boxes: A tensor of shape [batch_size, num_instances, 4] with predicted
      bounding boxes for each instance, expressed in the output coordinate
      frame.
    task_name: string, the name of the task this namedtuple corresponds to.
      Note that it should be an unique identifier of the task.
    kp_params: A `KeypointEstimationParams` object with parameters for a single
      keypoint class.

  Returns:
    A tuple of two tensors:
      keypoint_candidates: A float tensor with shape [batch_size,
        num_instances, num_keypoints, 2] representing the yx-coordinates of
        the keypoints in the output feature map space.
      keypoint_scores: A float tensor with shape [batch_size, num_instances,
        num_keypoints] representing the keypoint prediction scores.

  Raises:
    ValueError: if the candidate_ranking_mode is not supported.
  """
  keypoint_heatmap = tf.squeeze(tf.nn.sigmoid(prediction_dict[
      get_keypoint_name(task_name, KEYPOINT_HEATMAP)][-1]), axis=0)
  keypoint_offset = tf.squeeze(prediction_dict[
      get_keypoint_name(task_name, KEYPOINT_OFFSET)][-1], axis=0)
  keypoint_regression = tf.squeeze(prediction_dict[
      get_keypoint_name(task_name, KEYPOINT_REGRESSION)][-1], axis=0)
  height, width, num_keypoints = _get_shape(keypoint_heatmap, 3)

  # Create the y,x grids: [height, width]
  (y_grid, x_grid) = ta_utils.image_shape_to_grids(height, width)

  # Prepare the indices to retrieve the information from object centers.
  num_instances = _get_shape(object_y_indices, 2)[1]
  combined_obj_indices = tf.stack([
      tf.reshape(object_y_indices, [-1]),
      tf.reshape(object_x_indices, [-1])
  ], axis=1)

  # Select the regression vectors from the object center.
  selected_regression_flat = tf.gather_nd(
      keypoint_regression, combined_obj_indices)
  selected_regression = tf.reshape(
      selected_regression_flat, [num_instances, num_keypoints, 2])
  (y_reg, x_reg) = tf.unstack(selected_regression, axis=2)

  # shape: [num_instances, num_keypoints].
  y_regressed = tf.cast(
      tf.reshape(object_y_indices, [num_instances, 1]),
      dtype=tf.float32) + y_reg
  x_regressed = tf.cast(
      tf.reshape(object_x_indices, [num_instances, 1]),
      dtype=tf.float32) + x_reg

  if kp_params.candidate_ranking_mode == 'gaussian_weighted_const':
    rescored_heatmap = _gaussian_weighted_map_const_multi(
        y_grid, x_grid, keypoint_heatmap, y_regressed, x_regressed,
        tf.squeeze(boxes, axis=0), kp_params.gaussian_denom_ratio)

    # shape: [height, width, num_keypoints].
    keypoint_score_heatmap = tf.math.reduce_max(rescored_heatmap, axis=2)
  else:
    raise ValueError(
        'Unsupported ranking mode in the multipose no topk method: %s' %
        kp_params.candidate_ranking_mode)
  (keypoint_candidates,
   keypoint_scores) = prediction_tensors_to_multi_instance_kpts(
       keypoint_heatmap_predictions=rescored_heatmap,
       keypoint_heatmap_offsets=keypoint_offset,
       keypoint_score_heatmap=keypoint_score_heatmap)
  return keypoint_candidates, keypoint_scores


def regressed_keypoints_at_object_centers(regressed_keypoint_predictions,
                                          y_indices, x_indices):
  """Returns the regressed keypoints at specified object centers.

  The original keypoint predictions are regressed relative to each feature map
  location. The returned keypoints are expressed in absolute coordinates in the
  output frame (i.e. the center offsets are added to each individual regressed
  set of keypoints).

  Args:
    regressed_keypoint_predictions: A float tensor of shape
      [batch_size, height, width, 2 * num_keypoints] holding regressed
      keypoints. The last dimension has keypoint coordinates ordered as follows:
      [y0, x0, y1, x1, ..., y{J-1}, x{J-1}] where J is the number of keypoints.
    y_indices: A [batch, num_instances] int tensor holding y indices for object
      centers. These indices correspond to locations in the output feature map.
    x_indices: A [batch, num_instances] int tensor holding x indices for object
      centers. These indices correspond to locations in the output feature map.

  Returns:
    A float tensor of shape [batch_size, num_objects, 2 * num_keypoints] where
    regressed keypoints are gathered at the provided locations, and converted
    to absolute coordinates in the output coordinate frame.
  """
  batch_size, num_instances = _get_shape(y_indices, 2)

  # TF Lite does not support tf.gather with batch_dims > 0, so we need to use
  # tf_gather_nd instead and here we prepare the indices for that.
  combined_indices = tf.stack([
      _multi_range(batch_size, value_repetitions=num_instances),
      tf.reshape(y_indices, [-1]),
      tf.reshape(x_indices, [-1])
  ], axis=1)

  relative_regressed_keypoints = tf.gather_nd(regressed_keypoint_predictions,
                                              combined_indices)
  relative_regressed_keypoints = tf.reshape(
      relative_regressed_keypoints,
      [batch_size, num_instances, -1, 2])
  relative_regressed_keypoints_y, relative_regressed_keypoints_x = tf.unstack(
      relative_regressed_keypoints, axis=3)
  y_indices = _to_float32(tf.expand_dims(y_indices, axis=-1))
  x_indices = _to_float32(tf.expand_dims(x_indices, axis=-1))
  absolute_regressed_keypoints = tf.stack(
      [y_indices + relative_regressed_keypoints_y,
       x_indices + relative_regressed_keypoints_x],
      axis=3)
  return tf.reshape(absolute_regressed_keypoints,
                    [batch_size, num_instances, -1])


def sdr_scaled_ranking_score(
    keypoint_scores, distances, bboxes, score_distance_multiplier):
  """Score-to-distance-ratio method to rank keypoint candidates.

  This corresponds to the ranking method: 'score_scaled_distance_ratio'. The
  keypoint candidates are ranked using the formula:
    ranking_score = score / (distance + offset)

  where 'score' is the keypoint heatmap scores, 'distance' is the distance
  between the heatmap peak location and the regressed joint location,
  'offset' is a function of the predicted bounding box:
    offset = max(bbox height, bbox width) * score_distance_multiplier

  The ranking score is used to find the best keypoint candidate for snapping
  regressed joints.

  Args:
    keypoint_scores: A float tensor of shape
      [batch_size, max_candidates, num_keypoints] indicating the scores for
      keypoint candidates.
    distances: A float tensor of shape
      [batch_size, num_instances, max_candidates, num_keypoints] indicating the
      distances between the keypoint candidates and the joint regression
      locations of each instances.
    bboxes: A tensor of shape [batch_size, num_instances, 4] with predicted
      bounding boxes for each instance, expressed in the output coordinate
      frame. If not provided, boxes will be computed from regressed keypoints.
    score_distance_multiplier: A scalar used to multiply the bounding box size
      to be the offset in the score-to-distance-ratio formula.

  Returns:
    A float tensor of shape [batch_size, num_instances, max_candidates,
    num_keypoints] representing the ranking scores of each keypoint candidates.
  """
  # Get ymin, xmin, ymax, xmax bounding box coordinates.
  # Shape: [batch_size, num_instances]
  ymin, xmin, ymax, xmax = tf.unstack(bboxes, axis=2)

  # Shape: [batch_size, num_instances].
  offsets = tf.math.maximum(
      ymax - ymin, xmax - xmin) * score_distance_multiplier

  # Shape: [batch_size, num_instances, max_candidates, num_keypoints]
  ranking_scores = keypoint_scores[:, tf.newaxis, :, :] / (
      distances + offsets[:, :, tf.newaxis, tf.newaxis])
  return ranking_scores


def gaussian_weighted_score(
    keypoint_scores, distances, keypoint_std_dev, bboxes):
  """Gaussian weighted method to rank keypoint candidates.

  This corresponds to the ranking method: 'gaussian_weighted'. The
  keypoint candidates are ranked using the formula:
    score * exp((-distances^2) / (2 * sigma^2))

  where 'score' is the keypoint heatmap score, 'distances' is the distance
  between the heatmap peak location and the regressed joint location and 'sigma'
  is a Gaussian standard deviation used in generating the Gausian heatmap target
  multiplied by the 'std_dev_multiplier'.

  The ranking score is used to find the best keypoint candidate for snapping
  regressed joints.

  Args:
    keypoint_scores: A float tensor of shape
      [batch_size, max_candidates, num_keypoints] indicating the scores for
      keypoint candidates.
    distances: A float tensor of shape
      [batch_size, num_instances, max_candidates, num_keypoints] indicating the
      distances between the keypoint candidates and the joint regression
      locations of each instances.
    keypoint_std_dev: A list of float represent the standard deviation of the
      Gaussian kernel used to generate the keypoint heatmap. It is to provide
      the flexibility of using different sizes of Gaussian kernel for each
      keypoint class.
    bboxes: A tensor of shape [batch_size, num_instances, 4] with predicted
      bounding boxes for each instance, expressed in the output coordinate
      frame. If not provided, boxes will be computed from regressed keypoints.

  Returns:
    A float tensor of shape [batch_size, num_instances, max_candidates,
    num_keypoints] representing the ranking scores of each keypoint candidates.
  """
  # Get ymin, xmin, ymax, xmax bounding box coordinates.
  # Shape: [batch_size, num_instances]
  ymin, xmin, ymax, xmax = tf.unstack(bboxes, axis=2)

  # shape: [num_keypoints]
  keypoint_std_dev = tf.constant(keypoint_std_dev)

  # shape: [batch_size, num_instances]
  sigma = cn_assigner._compute_std_dev_from_box_size(  # pylint: disable=protected-access
      ymax - ymin, xmax - xmin, min_overlap=0.7)
  # shape: [batch_size, num_instances, num_keypoints]
  sigma = keypoint_std_dev[tf.newaxis, tf.newaxis, :] * sigma[:, :, tf.newaxis]
  (_, _, max_candidates, _) = _get_shape(distances, 4)
  # shape: [batch_size, num_instances, max_candidates, num_keypoints]
  sigma = tf.tile(
      sigma[:, :, tf.newaxis, :], multiples=[1, 1, max_candidates, 1])

  gaussian_map = tf.exp((-1 * distances * distances) / (2 * sigma * sigma))
  return keypoint_scores[:, tf.newaxis, :, :] * gaussian_map


def refine_keypoints(regressed_keypoints,
                     keypoint_candidates,
                     keypoint_scores,
                     num_keypoint_candidates,
                     bboxes=None,
                     unmatched_keypoint_score=0.1,
                     box_scale=1.2,
                     candidate_search_scale=0.3,
                     candidate_ranking_mode='min_distance',
                     score_distance_offset=1e-6,
                     keypoint_depth_candidates=None,
                     keypoint_score_threshold=0.1,
                     score_distance_multiplier=0.1,
                     keypoint_std_dev=None):
  """Refines regressed keypoints by snapping to the nearest candidate keypoints.

  The initial regressed keypoints represent a full set of keypoints regressed
  from the centers of the objects. The keypoint candidates are estimated
  independently from heatmaps, and are not associated with any object instances.
  This function refines the regressed keypoints by "snapping" to the
  nearest/highest score/highest score-distance ratio (depending on the
  candidate_ranking_mode) candidate of the same keypoint type (e.g. "nose").
  If no candidates are nearby, the regressed keypoint remains unchanged.

  In order to snap a regressed keypoint to a candidate keypoint, the following
  must be satisfied:
  - the candidate keypoint must be of the same type as the regressed keypoint
  - the candidate keypoint must not lie outside the predicted boxes (or the
    boxes which encloses the regressed keypoints for the instance if `bboxes` is
    not provided). Note that the box is scaled by
    `regressed_box_scale` in height and width, to provide some margin around the
    keypoints
  - the distance to the closest candidate keypoint cannot exceed
    candidate_search_scale * max(height, width), where height and width refer to
    the bounding box for the instance.

  Note that the same candidate keypoint is allowed to snap to regressed
  keypoints in difference instances.

  Args:
    regressed_keypoints: A float tensor of shape
      [batch_size, num_instances, num_keypoints, 2] with the initial regressed
      keypoints.
    keypoint_candidates: A tensor of shape
      [batch_size, max_candidates, num_keypoints, 2] holding the location of
      keypoint candidates in [y, x] format (expressed in absolute coordinates in
      the output coordinate frame).
    keypoint_scores: A float tensor of shape
      [batch_size, max_candidates, num_keypoints] indicating the scores for
      keypoint candidates.
    num_keypoint_candidates: An integer tensor of shape
      [batch_size, num_keypoints] indicating the number of valid candidates for
      each keypoint type, as there may be padding (dim 1) of
      `keypoint_candidates` and `keypoint_scores`.
    bboxes: A tensor of shape [batch_size, num_instances, 4] with predicted
      bounding boxes for each instance, expressed in the output coordinate
      frame. If not provided, boxes will be computed from regressed keypoints.
    unmatched_keypoint_score: float, the default score to use for regressed
      keypoints that are not successfully snapped to a nearby candidate.
    box_scale: float, the multiplier to expand the bounding boxes (either the
      provided boxes or those which tightly cover the regressed keypoints) for
      an instance. This scale is typically larger than 1.0 when not providing
      `bboxes`.
    candidate_search_scale: float, the scale parameter that multiplies the
      largest dimension of a bounding box. The resulting distance becomes a
      search radius for candidates in the vicinity of each regressed keypoint.
    candidate_ranking_mode: A string as one of ['min_distance',
      'score_distance_ratio', 'score_scaled_distance_ratio',
      'gaussian_weighted'] indicating how to select the candidate. If invalid
      value is provided, an ValueError will be raised.
    score_distance_offset: The distance offset to apply in the denominator when
      candidate_ranking_mode is 'score_distance_ratio'. The metric to maximize
      in this scenario is score / (distance + score_distance_offset). Larger
      values of score_distance_offset make the keypoint score gain more relative
      importance.
    keypoint_depth_candidates: (optional) A float tensor of shape
      [batch_size, max_candidates, num_keypoints] indicating the depths for
      keypoint candidates.
    keypoint_score_threshold: float, The heatmap score threshold for
      a keypoint to become a valid candidate.
    score_distance_multiplier: A scalar used to multiply the bounding box size
      to be the offset in the score-to-distance-ratio formula.
    keypoint_std_dev: A list of float represent the standard deviation of the
      Gaussian kernel used to rank the keypoint candidates. It offers the
      flexibility of using different sizes of Gaussian kernel for each keypoint
      class. Only applicable when the candidate_ranking_mode equals to
      'gaussian_weighted'.

  Returns:
    A tuple with:
    refined_keypoints: A float tensor of shape
      [batch_size, num_instances, num_keypoints, 2] with the final, refined
      keypoints.
    refined_scores: A float tensor of shape
      [batch_size, num_instances, num_keypoints] with scores associated with all
      instances and keypoints in `refined_keypoints`.

  Raises:
    ValueError: if provided candidate_ranking_mode is not one of
      ['min_distance', 'score_distance_ratio']
  """
  batch_size, num_instances, num_keypoints, _ = (
      shape_utils.combined_static_and_dynamic_shape(regressed_keypoints))
  max_candidates = keypoint_candidates.shape[1]

  # Replace all invalid (i.e. padded) keypoint candidates with NaN.
  # This will prevent them from being considered.
  range_tiled = tf.tile(
      tf.reshape(tf.range(max_candidates), [1, max_candidates, 1]),
      [batch_size, 1, num_keypoints])
  num_candidates_tiled = tf.tile(tf.expand_dims(num_keypoint_candidates, 1),
                                 [1, max_candidates, 1])
  invalid_candidates = range_tiled >= num_candidates_tiled

  # Pairwise squared distances between regressed keypoints and candidate
  # keypoints (for a single keypoint type).
  # Shape [batch_size, num_instances, 1, num_keypoints, 2].
  regressed_keypoint_expanded = tf.expand_dims(regressed_keypoints,
                                               axis=2)
  # Shape [batch_size, 1, max_candidates, num_keypoints, 2].
  keypoint_candidates_expanded = tf.expand_dims(
      keypoint_candidates, axis=1)
  # Use explicit tensor shape broadcasting (since the tensor dimensions are
  # expanded to 5D) to make it tf.lite compatible.
  regressed_keypoint_expanded = tf.tile(
      regressed_keypoint_expanded, multiples=[1, 1, max_candidates, 1, 1])
  keypoint_candidates_expanded = tf.tile(
      keypoint_candidates_expanded, multiples=[1, num_instances, 1, 1, 1])
  # Replace tf.math.squared_difference by "-" operator and tf.multiply ops since
  # tf.lite convert doesn't support squared_difference with undetermined
  # dimension.
  diff = regressed_keypoint_expanded - keypoint_candidates_expanded
  sqrd_distances = tf.math.reduce_sum(tf.multiply(diff, diff), axis=-1)
  distances = tf.math.sqrt(sqrd_distances)

  # Replace the invalid candidated with large constant (10^5) to make sure the
  # following reduce_min/argmin behaves properly.
  max_dist = 1e5
  distances = tf.where(
      tf.tile(
          tf.expand_dims(invalid_candidates, axis=1),
          multiples=[1, num_instances, 1, 1]),
      tf.ones_like(distances) * max_dist,
      distances
  )

  # Determine the candidates that have the minimum distance to the regressed
  # keypoints. Shape [batch_size, num_instances, num_keypoints].
  min_distances = tf.math.reduce_min(distances, axis=2)
  if candidate_ranking_mode == 'min_distance':
    nearby_candidate_inds = tf.math.argmin(distances, axis=2)
  elif candidate_ranking_mode == 'score_distance_ratio':
    # tiled_keypoint_scores:
    # Shape [batch_size, num_instances, max_candidates, num_keypoints].
    tiled_keypoint_scores = tf.tile(
        tf.expand_dims(keypoint_scores, axis=1),
        multiples=[1, num_instances, 1, 1])
    ranking_scores = tiled_keypoint_scores / (distances + score_distance_offset)
    nearby_candidate_inds = tf.math.argmax(ranking_scores, axis=2)
  elif candidate_ranking_mode == 'score_scaled_distance_ratio':
    ranking_scores = sdr_scaled_ranking_score(
        keypoint_scores, distances, bboxes, score_distance_multiplier)
    nearby_candidate_inds = tf.math.argmax(ranking_scores, axis=2)
  elif candidate_ranking_mode == 'gaussian_weighted':
    ranking_scores = gaussian_weighted_score(
        keypoint_scores, distances, keypoint_std_dev, bboxes)
    nearby_candidate_inds = tf.math.argmax(ranking_scores, axis=2)
    weighted_scores = tf.math.reduce_max(ranking_scores, axis=2)
  else:
    raise ValueError('Not recognized candidate_ranking_mode: %s' %
                     candidate_ranking_mode)

  # Gather the coordinates and scores corresponding to the closest candidates.
  # Shape of tensors are [batch_size, num_instances, num_keypoints, 2] and
  # [batch_size, num_instances, num_keypoints], respectively.
  (nearby_candidate_coords, nearby_candidate_scores,
   nearby_candidate_depths) = (
       _gather_candidates_at_indices(keypoint_candidates, keypoint_scores,
                                     nearby_candidate_inds,
                                     keypoint_depth_candidates))

  # If the ranking mode is 'gaussian_weighted', we use the ranking scores as the
  # final keypoint confidence since their values are in between [0, 1].
  if candidate_ranking_mode == 'gaussian_weighted':
    nearby_candidate_scores = weighted_scores

  if bboxes is None:
    # Filter out the chosen candidate with score lower than unmatched
    # keypoint score.
    mask = tf.cast(nearby_candidate_scores <
                   keypoint_score_threshold, tf.int32)
  else:
    bboxes_flattened = tf.reshape(bboxes, [-1, 4])

    # Scale the bounding boxes.
    # Shape [batch_size, num_instances, 4].
    boxlist = box_list.BoxList(bboxes_flattened)
    boxlist_scaled = box_list_ops.scale_height_width(
        boxlist, box_scale, box_scale)
    bboxes_scaled = boxlist_scaled.get()
    bboxes = tf.reshape(bboxes_scaled, [batch_size, num_instances, 4])

    # Get ymin, xmin, ymax, xmax bounding box coordinates, tiled per keypoint.
    # Shape [batch_size, num_instances, num_keypoints].
    bboxes_tiled = tf.tile(tf.expand_dims(bboxes, 2), [1, 1, num_keypoints, 1])
    ymin, xmin, ymax, xmax = tf.unstack(bboxes_tiled, axis=3)

    # Produce a mask that indicates whether the original regressed keypoint
    # should be used instead of a candidate keypoint.
    # Shape [batch_size, num_instances, num_keypoints].
    search_radius = (
        tf.math.maximum(ymax - ymin, xmax - xmin) * candidate_search_scale)
    mask = (tf.cast(nearby_candidate_coords[:, :, :, 0] < ymin, tf.int32) +
            tf.cast(nearby_candidate_coords[:, :, :, 0] > ymax, tf.int32) +
            tf.cast(nearby_candidate_coords[:, :, :, 1] < xmin, tf.int32) +
            tf.cast(nearby_candidate_coords[:, :, :, 1] > xmax, tf.int32) +
            # Filter out the chosen candidate with score lower than unmatched
            # keypoint score.
            tf.cast(nearby_candidate_scores <
                    keypoint_score_threshold, tf.int32) +
            tf.cast(min_distances > search_radius, tf.int32))
  mask = mask > 0

  # Create refined keypoints where candidate keypoints replace original
  # regressed keypoints if they are in the vicinity of the regressed keypoints.
  # Shape [batch_size, num_instances, num_keypoints, 2].
  refined_keypoints = tf.where(
      tf.tile(tf.expand_dims(mask, -1), [1, 1, 1, 2]),
      regressed_keypoints,
      nearby_candidate_coords)

  # Update keypoints scores. In the case where we use the original regressed
  # keypoints, we use a default score of `unmatched_keypoint_score`.
  # Shape [batch_size, num_instances, num_keypoints].
  refined_scores = tf.where(
      mask,
      unmatched_keypoint_score * tf.ones_like(nearby_candidate_scores),
      nearby_candidate_scores)

  refined_depths = None
  if nearby_candidate_depths is not None:
    refined_depths = tf.where(mask, tf.zeros_like(nearby_candidate_depths),
                              nearby_candidate_depths)

  return refined_keypoints, refined_scores, refined_depths


def _pad_to_full_keypoint_dim(keypoint_coords, keypoint_scores, keypoint_inds,
                              num_total_keypoints):
  """Scatter keypoint elements into tensors with full keypoints dimension.

  Args:
    keypoint_coords: a [batch_size, num_instances, num_keypoints, 2] float32
      tensor.
    keypoint_scores: a [batch_size, num_instances, num_keypoints] float32
      tensor.
    keypoint_inds: a list of integers that indicate the keypoint indices for
      this specific keypoint class. These indices are used to scatter into
      tensors that have a `num_total_keypoints` dimension.
    num_total_keypoints: The total number of keypoints that this model predicts.

  Returns:
    A tuple with
    keypoint_coords_padded: a
      [batch_size, num_instances, num_total_keypoints,2] float32 tensor.
    keypoint_scores_padded: a [batch_size, num_instances, num_total_keypoints]
      float32 tensor.
  """
  batch_size, num_instances, _, _ = (
      shape_utils.combined_static_and_dynamic_shape(keypoint_coords))
  kpt_coords_transposed = tf.transpose(keypoint_coords, [2, 0, 1, 3])
  kpt_scores_transposed = tf.transpose(keypoint_scores, [2, 0, 1])
  kpt_inds_tensor = tf.expand_dims(keypoint_inds, axis=-1)
  kpt_coords_scattered = tf.scatter_nd(
      indices=kpt_inds_tensor,
      updates=kpt_coords_transposed,
      shape=[num_total_keypoints, batch_size, num_instances, 2])
  kpt_scores_scattered = tf.scatter_nd(
      indices=kpt_inds_tensor,
      updates=kpt_scores_transposed,
      shape=[num_total_keypoints, batch_size, num_instances])
  keypoint_coords_padded = tf.transpose(kpt_coords_scattered, [1, 2, 0, 3])
  keypoint_scores_padded = tf.transpose(kpt_scores_scattered, [1, 2, 0])
  return keypoint_coords_padded, keypoint_scores_padded


def _pad_to_full_instance_dim(keypoint_coords, keypoint_scores, instance_inds,
                              max_instances):
  """Scatter keypoint elements into tensors with full instance dimension.

  Args:
    keypoint_coords: a [batch_size, num_instances, num_keypoints, 2] float32
      tensor.
    keypoint_scores: a [batch_size, num_instances, num_keypoints] float32
      tensor.
    instance_inds: a list of integers that indicate the instance indices for
      these keypoints. These indices are used to scatter into tensors
      that have a `max_instances` dimension.
    max_instances: The maximum number of instances detected by the model.

  Returns:
    A tuple with
    keypoint_coords_padded: a [batch_size, max_instances, num_keypoints, 2]
      float32 tensor.
    keypoint_scores_padded: a [batch_size, max_instances, num_keypoints]
      float32 tensor.
  """
  batch_size, _, num_keypoints, _ = (
      shape_utils.combined_static_and_dynamic_shape(keypoint_coords))
  kpt_coords_transposed = tf.transpose(keypoint_coords, [1, 0, 2, 3])
  kpt_scores_transposed = tf.transpose(keypoint_scores, [1, 0, 2])
  instance_inds = tf.expand_dims(instance_inds, axis=-1)
  kpt_coords_scattered = tf.scatter_nd(
      indices=instance_inds,
      updates=kpt_coords_transposed,
      shape=[max_instances, batch_size, num_keypoints, 2])
  kpt_scores_scattered = tf.scatter_nd(
      indices=instance_inds,
      updates=kpt_scores_transposed,
      shape=[max_instances, batch_size, num_keypoints])
  keypoint_coords_padded = tf.transpose(kpt_coords_scattered, [1, 0, 2, 3])
  keypoint_scores_padded = tf.transpose(kpt_scores_scattered, [1, 0, 2])
  return keypoint_coords_padded, keypoint_scores_padded


def _gather_candidates_at_indices(keypoint_candidates,
                                  keypoint_scores,
                                  indices,
                                  keypoint_depth_candidates=None):
  """Gathers keypoint candidate coordinates and scores at indices.

  Args:
    keypoint_candidates: a float tensor of shape [batch_size, max_candidates,
      num_keypoints, 2] with candidate coordinates.
    keypoint_scores: a float tensor of shape [batch_size, max_candidates,
      num_keypoints] with keypoint scores.
    indices: an integer tensor of shape [batch_size, num_indices, num_keypoints]
      with indices.
    keypoint_depth_candidates: (optional) a float tensor of shape [batch_size,
      max_candidates, num_keypoints] with keypoint depths.

  Returns:
    A tuple with
    gathered_keypoint_candidates: a float tensor of shape [batch_size,
      num_indices, num_keypoints, 2] with gathered coordinates.
    gathered_keypoint_scores: a float tensor of shape [batch_size,
      num_indices, num_keypoints].
    gathered_keypoint_depths: a float tensor of shape [batch_size,
      num_indices, num_keypoints]. Return None if the input
      keypoint_depth_candidates is None.
  """
  batch_size, num_indices, num_keypoints = _get_shape(indices, 3)

  # Transpose tensors so that all batch dimensions are up front.
  keypoint_candidates_transposed = tf.transpose(keypoint_candidates,
                                                [0, 2, 1, 3])
  keypoint_scores_transposed = tf.transpose(keypoint_scores, [0, 2, 1])
  nearby_candidate_inds_transposed = tf.transpose(indices, [0, 2, 1])

  # TF Lite does not support tf.gather with batch_dims > 0, so we need to use
  # tf_gather_nd instead and here we prepare the indices for that.
  combined_indices = tf.stack([
      _multi_range(
          batch_size,
          value_repetitions=num_keypoints * num_indices,
          dtype=tf.int64),
      _multi_range(
          num_keypoints,
          value_repetitions=num_indices,
          range_repetitions=batch_size,
          dtype=tf.int64),
      tf.reshape(nearby_candidate_inds_transposed, [-1])
  ], axis=1)

  nearby_candidate_coords_transposed = tf.gather_nd(
      keypoint_candidates_transposed, combined_indices)
  nearby_candidate_coords_transposed = tf.reshape(
      nearby_candidate_coords_transposed,
      [batch_size, num_keypoints, num_indices, -1])

  nearby_candidate_scores_transposed = tf.gather_nd(keypoint_scores_transposed,
                                                    combined_indices)
  nearby_candidate_scores_transposed = tf.reshape(
      nearby_candidate_scores_transposed,
      [batch_size, num_keypoints, num_indices])

  gathered_keypoint_candidates = tf.transpose(
      nearby_candidate_coords_transposed, [0, 2, 1, 3])
  # The reshape operation above may result in a singleton last dimension, but
  # downstream code requires it to always be at least 2-valued.
  original_shape = tf.shape(gathered_keypoint_candidates)
  new_shape = tf.concat((original_shape[:3],
                         [tf.maximum(original_shape[3], 2)]), 0)
  gathered_keypoint_candidates = tf.reshape(gathered_keypoint_candidates,
                                            new_shape)
  gathered_keypoint_scores = tf.transpose(nearby_candidate_scores_transposed,
                                          [0, 2, 1])

  gathered_keypoint_depths = None
  if keypoint_depth_candidates is not None:
    keypoint_depths_transposed = tf.transpose(keypoint_depth_candidates,
                                              [0, 2, 1])
    nearby_candidate_depths_transposed = tf.gather_nd(
        keypoint_depths_transposed, combined_indices)
    nearby_candidate_depths_transposed = tf.reshape(
        nearby_candidate_depths_transposed,
        [batch_size, num_keypoints, num_indices])
    gathered_keypoint_depths = tf.transpose(nearby_candidate_depths_transposed,
                                            [0, 2, 1])
  return (gathered_keypoint_candidates, gathered_keypoint_scores,
          gathered_keypoint_depths)


def flattened_indices_from_row_col_indices(row_indices, col_indices, num_cols):
  """Get the index in a flattened array given row and column indices."""
  return (row_indices * num_cols) + col_indices


def row_col_channel_indices_from_flattened_indices(indices, num_cols,
                                                   num_channels):
  """Computes row, column and channel indices from flattened indices.

  Args:
    indices: An integer tensor of any shape holding the indices in the flattened
      space.
    num_cols: Number of columns in the image (width).
    num_channels: Number of channels in the image.

  Returns:
    row_indices: The row indices corresponding to each of the input indices.
      Same shape as indices.
    col_indices: The column indices corresponding to each of the input indices.
      Same shape as indices.
    channel_indices. The channel indices corresponding to each of the input
      indices.

  """
  # Be careful with this function when running a model in float16 precision
  # (e.g. TF.js with WebGL) because the array indices may not be represented
  # accurately if they are too large, resulting in incorrect channel indices.
  # See:
  # https://en.wikipedia.org/wiki/Half-precision_floating-point_format#Precision_limitations_on_integer_values
  #
  # Avoid using mod operator to make the ops more easy to be compatible with
  # different environments, e.g. WASM.
  row_indices = (indices // num_channels) // num_cols
  col_indices = (indices // num_channels) - row_indices * num_cols
  channel_indices_temp = indices // num_channels
  channel_indices = indices - channel_indices_temp * num_channels

  return row_indices, col_indices, channel_indices


def row_col_indices_from_flattened_indices(indices, num_cols):
  """Computes row and column indices from flattened indices.

  Args:
    indices: An integer tensor of any shape holding the indices in the flattened
      space.
    num_cols: Number of columns in the image (width).

  Returns:
    row_indices: The row indices corresponding to each of the input indices.
      Same shape as indices.
    col_indices: The column indices corresponding to each of the input indices.
      Same shape as indices.

  """
  # Avoid using mod operator to make the ops more easy to be compatible with
  # different environments, e.g. WASM.
  row_indices = indices // num_cols
  col_indices = indices - row_indices * num_cols

  return row_indices, col_indices


def get_valid_anchor_weights_in_flattened_image(true_image_shapes, height,
                                                width):
  """Computes valid anchor weights for an image assuming pixels will be flattened.

  This function is useful when we only want to penalize valid areas in the
  image in the case when padding is used. The function assumes that the loss
  function will be applied after flattening the spatial dimensions and returns
  anchor weights accordingly.

  Args:
    true_image_shapes: An integer tensor of shape [batch_size, 3] representing
      the true image shape (without padding) for each sample in the batch.
    height: height of the prediction from the network.
    width: width of the prediction from the network.

  Returns:
    valid_anchor_weights: a float tensor of shape [batch_size, height * width]
    with 1s in locations where the spatial coordinates fall within the height
    and width in true_image_shapes.
  """

  indices = tf.reshape(tf.range(height * width), [1, -1])
  batch_size = tf.shape(true_image_shapes)[0]
  batch_indices = tf.ones((batch_size, 1), dtype=tf.int32) * indices

  y_coords, x_coords, _ = row_col_channel_indices_from_flattened_indices(
      batch_indices, width, 1)

  max_y, max_x = true_image_shapes[:, 0], true_image_shapes[:, 1]
  max_x = _to_float32(tf.expand_dims(max_x, 1))
  max_y = _to_float32(tf.expand_dims(max_y, 1))

  x_coords = _to_float32(x_coords)
  y_coords = _to_float32(y_coords)

  valid_mask = tf.math.logical_and(x_coords < max_x, y_coords < max_y)

  return _to_float32(valid_mask)


def convert_strided_predictions_to_normalized_boxes(boxes, stride,
                                                    true_image_shapes):
  """Converts predictions in the output space to normalized boxes.

  Boxes falling outside the valid image boundary are clipped to be on the
  boundary.

  Args:
    boxes: A tensor of shape [batch_size, num_boxes, 4] holding the raw
     coordinates of boxes in the model's output space.
    stride: The stride in the output space.
    true_image_shapes: A tensor of shape [batch_size, 3] representing the true
      shape of the input not considering padding.

  Returns:
    boxes: A tensor of shape [batch_size, num_boxes, 4] representing the
      coordinates of the normalized boxes.
  """
  # Note: We use tf ops instead of functions in box_list_ops to make this
  # function compatible with dynamic batch size.
  boxes = boxes * stride
  true_image_shapes = tf.tile(true_image_shapes[:, tf.newaxis, :2], [1, 1, 2])
  boxes = boxes / tf.cast(true_image_shapes, tf.float32)
  boxes = tf.clip_by_value(boxes, 0.0, 1.0)
  return boxes


def convert_strided_predictions_to_normalized_keypoints(
    keypoint_coords, keypoint_scores, stride, true_image_shapes,
    clip_out_of_frame_keypoints=False):
  """Converts predictions in the output space to normalized keypoints.

  If clip_out_of_frame_keypoints=False, keypoint coordinates falling outside
  the valid image boundary are normalized but not clipped; If
  clip_out_of_frame_keypoints=True, keypoint coordinates falling outside the
  valid image boundary are clipped to the closest image boundary and the scores
  will be set to 0.0.

  Args:
    keypoint_coords: A tensor of shape
      [batch_size, num_instances, num_keypoints, 2] holding the raw coordinates
      of keypoints in the model's output space.
    keypoint_scores: A tensor of shape
      [batch_size, num_instances, num_keypoints] holding the keypoint scores.
    stride: The stride in the output space.
    true_image_shapes: A tensor of shape [batch_size, 3] representing the true
      shape of the input not considering padding.
    clip_out_of_frame_keypoints: A boolean indicating whether keypoints outside
      the image boundary should be clipped. If True, keypoint coords will be
      clipped to image boundary. If False, keypoints are normalized but not
      filtered based on their location.

  Returns:
    keypoint_coords_normalized: A tensor of shape
      [batch_size, num_instances, num_keypoints, 2] representing the coordinates
      of the normalized keypoints.
    keypoint_scores: A tensor of shape
      [batch_size, num_instances, num_keypoints] representing the updated
      keypoint scores.
  """
  # Flatten keypoints and scores.
  batch_size, _, _, _ = (
      shape_utils.combined_static_and_dynamic_shape(keypoint_coords))

  # Scale and normalize keypoints.
  true_heights, true_widths, _ = tf.unstack(true_image_shapes, axis=1)
  yscale = float(stride) / tf.cast(true_heights, tf.float32)
  xscale = float(stride) / tf.cast(true_widths, tf.float32)
  yx_scale = tf.stack([yscale, xscale], axis=1)
  keypoint_coords_normalized = keypoint_coords * tf.reshape(
      yx_scale, [batch_size, 1, 1, 2])

  if clip_out_of_frame_keypoints:
    # Determine the keypoints that are in the true image regions.
    valid_indices = tf.logical_and(
        tf.logical_and(keypoint_coords_normalized[:, :, :, 0] >= 0.0,
                       keypoint_coords_normalized[:, :, :, 0] <= 1.0),
        tf.logical_and(keypoint_coords_normalized[:, :, :, 1] >= 0.0,
                       keypoint_coords_normalized[:, :, :, 1] <= 1.0))
    batch_window = tf.tile(
        tf.constant([[0.0, 0.0, 1.0, 1.0]], dtype=tf.float32),
        multiples=[batch_size, 1])
    def clip_to_window(inputs):
      keypoints, window = inputs
      return keypoint_ops.clip_to_window(keypoints, window)

    keypoint_coords_normalized = shape_utils.static_or_dynamic_map_fn(
        clip_to_window, [keypoint_coords_normalized, batch_window],
        dtype=tf.float32, back_prop=False)
    keypoint_scores = tf.where(valid_indices, keypoint_scores,
                               tf.zeros_like(keypoint_scores))
  return keypoint_coords_normalized, keypoint_scores


def convert_strided_predictions_to_instance_masks(
    boxes, classes, masks, true_image_shapes,
    densepose_part_heatmap=None, densepose_surface_coords=None, stride=4,
    mask_height=256, mask_width=256, score_threshold=0.5,
    densepose_class_index=-1):
  """Converts predicted full-image masks into instance masks.

  For each predicted detection box:
    * Crop and resize the predicted mask (and optionally DensePose coordinates)
      based on the detected bounding box coordinates and class prediction. Uses
      bilinear resampling.
    * Binarize the mask using the provided score threshold.

  Args:
    boxes: A tensor of shape [batch, max_detections, 4] holding the predicted
      boxes, in normalized coordinates (relative to the true image dimensions).
    classes: An integer tensor of shape [batch, max_detections] containing the
      detected class for each box (0-indexed).
    masks: A [batch, output_height, output_width, num_classes] float32
      tensor with class probabilities.
    true_image_shapes: A tensor of shape [batch, 3] representing the true
      shape of the inputs not considering padding.
    densepose_part_heatmap: (Optional) A [batch, output_height, output_width,
      num_parts] float32 tensor with part scores (i.e. logits).
    densepose_surface_coords: (Optional) A [batch, output_height, output_width,
      2 * num_parts] float32 tensor with predicted part coordinates (in
      vu-format).
    stride: The stride in the output space.
    mask_height: The desired resized height for instance masks.
    mask_width: The desired resized width for instance masks.
    score_threshold: The threshold at which to convert predicted mask
       into foreground pixels.
    densepose_class_index: The class index (0-indexed) corresponding to the
      class which has DensePose labels (e.g. person class).

  Returns:
    A tuple of masks and surface_coords.
    instance_masks: A [batch_size, max_detections, mask_height, mask_width]
      uint8 tensor with predicted foreground mask for each
      instance. If DensePose tensors are provided, then each pixel value in the
      mask encodes the 1-indexed part.
    surface_coords: A [batch_size, max_detections, mask_height, mask_width, 2]
      float32 tensor with (v, u) coordinates. Note that v, u coordinates are
      only defined on instance masks, and the coordinates at each location of
      the foreground mask correspond to coordinates on a local part coordinate
      system (the specific part can be inferred from the `instance_masks`
      output. If DensePose feature maps are not passed to this function, this
      output will be None.

  Raises:
    ValueError: If one but not both of `densepose_part_heatmap` and
    `densepose_surface_coords` is provided.
  """
  batch_size, output_height, output_width, _ = (
      shape_utils.combined_static_and_dynamic_shape(masks))
  input_height = stride * output_height
  input_width = stride * output_width

  true_heights, true_widths, _ = tf.unstack(true_image_shapes, axis=1)
  # If necessary, create dummy DensePose tensors to simplify the map function.
  densepose_present = True
  if ((densepose_part_heatmap is not None) ^
      (densepose_surface_coords is not None)):
    raise ValueError('To use DensePose, both `densepose_part_heatmap` and '
                     '`densepose_surface_coords` must be provided')
  if densepose_part_heatmap is None and densepose_surface_coords is None:
    densepose_present = False
    densepose_part_heatmap = tf.zeros(
        (batch_size, output_height, output_width, 1), dtype=tf.float32)
    densepose_surface_coords = tf.zeros(
        (batch_size, output_height, output_width, 2), dtype=tf.float32)
  crop_and_threshold_fn = functools.partial(
      crop_and_threshold_masks, input_height=input_height,
      input_width=input_width, mask_height=mask_height, mask_width=mask_width,
      score_threshold=score_threshold,
      densepose_class_index=densepose_class_index)

  instance_masks, surface_coords = shape_utils.static_or_dynamic_map_fn(
      crop_and_threshold_fn,
      elems=[boxes, classes, masks, densepose_part_heatmap,
             densepose_surface_coords, true_heights, true_widths],
      dtype=[tf.uint8, tf.float32],
      back_prop=False)
  surface_coords = surface_coords if densepose_present else None
  return instance_masks, surface_coords


def crop_and_threshold_masks(elems, input_height, input_width, mask_height=256,
                             mask_width=256, score_threshold=0.5,
                             densepose_class_index=-1):
  """Crops and thresholds masks based on detection boxes.

  Args:
    elems: A tuple of
      boxes - float32 tensor of shape [max_detections, 4]
      classes - int32 tensor of shape [max_detections] (0-indexed)
      masks - float32 tensor of shape [output_height, output_width, num_classes]
      part_heatmap - float32 tensor of shape [output_height, output_width,
        num_parts]
      surf_coords - float32 tensor of shape [output_height, output_width,
        2 * num_parts]
      true_height - scalar int tensor
      true_width - scalar int tensor
    input_height: Input height to network.
    input_width: Input width to network.
    mask_height: Height for resizing mask crops.
    mask_width: Width for resizing mask crops.
    score_threshold: The threshold at which to convert predicted mask
      into foreground pixels.
    densepose_class_index: scalar int tensor with the class index (0-indexed)
      for DensePose.

  Returns:
    A tuple of
    all_instances: A [max_detections, mask_height, mask_width] uint8 tensor
      with a predicted foreground mask for each instance. Background is encoded
      as 0, and foreground is encoded as a positive integer. Specific part
      indices are encoded as 1-indexed parts (for classes that have part
      information).
    surface_coords: A [max_detections, mask_height, mask_width, 2]
      float32 tensor with (v, u) coordinates. for each part.
  """
  (boxes, classes, masks, part_heatmap, surf_coords, true_height,
   true_width) = elems
  # Boxes are in normalized coordinates relative to true image shapes. Convert
  # coordinates to be normalized relative to input image shapes (since masks
  # may still have padding).
  boxlist = box_list.BoxList(boxes)
  y_scale = true_height / input_height
  x_scale = true_width / input_width
  boxlist = box_list_ops.scale(boxlist, y_scale, x_scale)
  boxes = boxlist.get()
  # Convert masks from [output_height, output_width, num_classes] to
  # [num_classes, output_height, output_width, 1].
  num_classes = tf.shape(masks)[-1]
  masks_4d = tf.transpose(masks, perm=[2, 0, 1])[:, :, :, tf.newaxis]
  # Tile part and surface coordinate masks for all classes.
  part_heatmap_4d = tf.tile(part_heatmap[tf.newaxis, :, :, :],
                            multiples=[num_classes, 1, 1, 1])
  surf_coords_4d = tf.tile(surf_coords[tf.newaxis, :, :, :],
                           multiples=[num_classes, 1, 1, 1])
  feature_maps_concat = tf.concat([masks_4d, part_heatmap_4d, surf_coords_4d],
                                  axis=-1)
  # The following tensor has shape
  # [max_detections, mask_height, mask_width, 1 + 3 * num_parts].
  cropped_masks = tf2.image.crop_and_resize(
      feature_maps_concat,
      boxes=boxes,
      box_indices=classes,
      crop_size=[mask_height, mask_width],
      method='bilinear')

  # Split the cropped masks back into instance masks, part masks, and surface
  # coordinates.
  num_parts = tf.shape(part_heatmap)[-1]
  instance_masks, part_heatmap_cropped, surface_coords_cropped = tf.split(
      cropped_masks, [1, num_parts, 2 * num_parts], axis=-1)

  # Threshold the instance masks. Resulting tensor has shape
  # [max_detections, mask_height, mask_width, 1].
  instance_masks_int = tf.cast(
      tf.math.greater_equal(instance_masks, score_threshold), dtype=tf.int32)

  # Produce a binary mask that is 1.0 only:
  #  - in the foreground region for an instance
  #  - in detections corresponding to the DensePose class
  det_with_parts = tf.equal(classes, densepose_class_index)
  det_with_parts = tf.cast(
      tf.reshape(det_with_parts, [-1, 1, 1, 1]), dtype=tf.int32)
  instance_masks_with_parts = tf.math.multiply(instance_masks_int,
                                               det_with_parts)

  # Similarly, produce a binary mask that holds the foreground masks only for
  # instances without parts (i.e. non-DensePose classes).
  det_without_parts = 1 - det_with_parts
  instance_masks_without_parts = tf.math.multiply(instance_masks_int,
                                                  det_without_parts)

  # Assemble a tensor that has standard instance segmentation masks for
  # non-DensePose classes (with values in [0, 1]), and part segmentation masks
  # for DensePose classes (with vaues in [0, 1, ..., num_parts]).
  part_mask_int_zero_indexed = tf.math.argmax(
      part_heatmap_cropped, axis=-1, output_type=tf.int32)[:, :, :, tf.newaxis]
  part_mask_int_one_indexed = part_mask_int_zero_indexed + 1
  all_instances = (instance_masks_without_parts +
                   instance_masks_with_parts * part_mask_int_one_indexed)

  # Gather the surface coordinates for the parts.
  surface_coords_cropped = tf.reshape(
      surface_coords_cropped, [-1, mask_height, mask_width, num_parts, 2])
  surface_coords = gather_surface_coords_for_parts(surface_coords_cropped,
                                                   part_mask_int_zero_indexed)
  surface_coords = (
      surface_coords * tf.cast(instance_masks_with_parts, tf.float32))

  return [tf.squeeze(all_instances, axis=3), surface_coords]


def gather_surface_coords_for_parts(surface_coords_cropped,
                                    highest_scoring_part):
  """Gathers the (v, u) coordinates for the highest scoring DensePose parts.

  Args:
    surface_coords_cropped: A [max_detections, height, width, num_parts, 2]
      float32 tensor with (v, u) surface coordinates.
    highest_scoring_part: A [max_detections, height, width] integer tensor with
      the highest scoring part (0-indexed) indices for each location.

  Returns:
    A [max_detections, height, width, 2] float32 tensor with the (v, u)
    coordinates selected from the highest scoring parts.
  """
  max_detections, height, width, num_parts, _ = (
      shape_utils.combined_static_and_dynamic_shape(surface_coords_cropped))
  flattened_surface_coords = tf.reshape(surface_coords_cropped, [-1, 2])
  flattened_part_ids = tf.reshape(highest_scoring_part, [-1])

  # Produce lookup indices that represent the locations of the highest scoring
  # parts in the `flattened_surface_coords` tensor.
  flattened_lookup_indices = (
      num_parts * tf.range(max_detections * height * width) +
      flattened_part_ids)

  vu_coords_flattened = tf.gather(flattened_surface_coords,
                                  flattened_lookup_indices, axis=0)
  return tf.reshape(vu_coords_flattened, [max_detections, height, width, 2])


def predicted_embeddings_at_object_centers(embedding_predictions,
                                           y_indices, x_indices):
  """Returns the predicted embeddings at specified object centers.

  Args:
    embedding_predictions: A float tensor of shape [batch_size, height, width,
      reid_embed_size] holding predicted embeddings.
    y_indices: A [batch, num_instances] int tensor holding y indices for object
      centers. These indices correspond to locations in the output feature map.
    x_indices: A [batch, num_instances] int tensor holding x indices for object
      centers. These indices correspond to locations in the output feature map.

  Returns:
    A float tensor of shape [batch_size, num_objects, reid_embed_size] where
    predicted embeddings are gathered at the provided locations.
  """
  batch_size, _, width, _ = _get_shape(embedding_predictions, 4)
  flattened_indices = flattened_indices_from_row_col_indices(
      y_indices, x_indices, width)
  _, num_instances = _get_shape(flattened_indices, 2)
  embeddings_flat = _flatten_spatial_dimensions(embedding_predictions)
  embeddings = tf.gather(embeddings_flat, flattened_indices, batch_dims=1)
  embeddings = tf.reshape(embeddings, [batch_size, num_instances, -1])

  return embeddings


def mask_from_true_image_shape(data_shape, true_image_shapes):
  """Get a binary mask based on the true_image_shape.

  Args:
    data_shape: a possibly static (4,) tensor for the shape of the feature
      map.
    true_image_shapes: int32 tensor of shape [batch, 3] where each row is of
      the form [height, width, channels] indicating the shapes of true
      images in the resized images, as resized images can be padded with
      zeros.
  Returns:
    a [batch, data_height, data_width, 1] tensor of 1.0 wherever data_height
    is less than height, etc.
  """
  mask_h = tf.cast(
      tf.range(data_shape[1]) < true_image_shapes[:, tf.newaxis, 0],
      tf.float32)
  mask_w = tf.cast(
      tf.range(data_shape[2]) < true_image_shapes[:, tf.newaxis, 1],
      tf.float32)
  mask = tf.expand_dims(
      mask_h[:, :, tf.newaxis] * mask_w[:, tf.newaxis, :], 3)
  return mask


class ObjectDetectionParams(
    collections.namedtuple('ObjectDetectionParams', [
        'localization_loss', 'scale_loss_weight', 'offset_loss_weight',
        'task_loss_weight', 'scale_head_num_filters',
        'scale_head_kernel_sizes', 'offset_head_num_filters',
        'offset_head_kernel_sizes'
    ])):
  """Namedtuple to host object detection related parameters.

  This is a wrapper class over the fields that are either the hyper-parameters
  or the loss functions needed for the object detection task. The class is
  immutable after constructed. Please see the __new__ function for detailed
  information for each fields.
  """

  __slots__ = ()

  def __new__(cls,
              localization_loss,
              scale_loss_weight,
              offset_loss_weight,
              task_loss_weight=1.0,
              scale_head_num_filters=(256),
              scale_head_kernel_sizes=(3),
              offset_head_num_filters=(256),
              offset_head_kernel_sizes=(3)):
    """Constructor with default values for ObjectDetectionParams.

    Args:
      localization_loss: a object_detection.core.losses.Loss object to compute
        the loss for the center offset and height/width predictions in
        CenterNet.
      scale_loss_weight: float, The weight for localizing box size. Note that
        the scale loss is dependent on the input image size, since we penalize
        the raw height and width. This constant may need to be adjusted
        depending on the input size.
      offset_loss_weight: float, The weight for localizing center offsets.
      task_loss_weight: float, the weight of the object detection loss.
      scale_head_num_filters: filter numbers of the convolutional layers used
        by the object detection box scale prediction head.
      scale_head_kernel_sizes: kernel size of the convolutional layers used
        by the object detection box scale prediction head.
      offset_head_num_filters: filter numbers of the convolutional layers used
        by the object detection box offset prediction head.
      offset_head_kernel_sizes: kernel size of the convolutional layers used
        by the object detection box offset prediction head.

    Returns:
      An initialized ObjectDetectionParams namedtuple.
    """
    return super(ObjectDetectionParams,
                 cls).__new__(cls, localization_loss, scale_loss_weight,
                              offset_loss_weight, task_loss_weight,
                              scale_head_num_filters, scale_head_kernel_sizes,
                              offset_head_num_filters, offset_head_kernel_sizes)


class KeypointEstimationParams(
    collections.namedtuple('KeypointEstimationParams', [
        'task_name', 'class_id', 'keypoint_indices', 'classification_loss',
        'localization_loss', 'keypoint_labels', 'keypoint_std_dev',
        'keypoint_heatmap_loss_weight', 'keypoint_offset_loss_weight',
        'keypoint_regression_loss_weight', 'keypoint_candidate_score_threshold',
        'heatmap_bias_init', 'num_candidates_per_keypoint', 'task_loss_weight',
        'peak_max_pool_kernel_size', 'unmatched_keypoint_score', 'box_scale',
        'candidate_search_scale', 'candidate_ranking_mode',
        'offset_peak_radius', 'per_keypoint_offset', 'predict_depth',
        'per_keypoint_depth', 'keypoint_depth_loss_weight',
        'score_distance_offset', 'clip_out_of_frame_keypoints',
        'rescore_instances', 'heatmap_head_num_filters',
        'heatmap_head_kernel_sizes', 'offset_head_num_filters',
        'offset_head_kernel_sizes', 'regress_head_num_filters',
        'regress_head_kernel_sizes', 'score_distance_multiplier',
        'std_dev_multiplier', 'rescoring_threshold', 'gaussian_denom_ratio',
        'argmax_postprocessing'
    ])):
  """Namedtuple to host object detection related parameters.

  This is a wrapper class over the fields that are either the hyper-parameters
  or the loss functions needed for the keypoint estimation task. The class is
  immutable after constructed. Please see the __new__ function for detailed
  information for each fields.
  """

  __slots__ = ()

  def __new__(cls,
              task_name,
              class_id,
              keypoint_indices,
              classification_loss,
              localization_loss,
              keypoint_labels=None,
              keypoint_std_dev=None,
              keypoint_heatmap_loss_weight=1.0,
              keypoint_offset_loss_weight=1.0,
              keypoint_regression_loss_weight=1.0,
              keypoint_candidate_score_threshold=0.1,
              heatmap_bias_init=-2.19,
              num_candidates_per_keypoint=100,
              task_loss_weight=1.0,
              peak_max_pool_kernel_size=3,
              unmatched_keypoint_score=0.1,
              box_scale=1.2,
              candidate_search_scale=0.3,
              candidate_ranking_mode='min_distance',
              offset_peak_radius=0,
              per_keypoint_offset=False,
              predict_depth=False,
              per_keypoint_depth=False,
              keypoint_depth_loss_weight=1.0,
              score_distance_offset=1e-6,
              clip_out_of_frame_keypoints=False,
              rescore_instances=False,
              heatmap_head_num_filters=(256),
              heatmap_head_kernel_sizes=(3),
              offset_head_num_filters=(256),
              offset_head_kernel_sizes=(3),
              regress_head_num_filters=(256),
              regress_head_kernel_sizes=(3),
              score_distance_multiplier=0.1,
              std_dev_multiplier=1.0,
              rescoring_threshold=0.0,
              argmax_postprocessing=False,
              gaussian_denom_ratio=0.1):
    """Constructor with default values for KeypointEstimationParams.

    Args:
      task_name: string, the name of the task this namedtuple corresponds to.
        Note that it should be an unique identifier of the task.
      class_id: int, the ID of the class that contains the target keypoints to
        considered in this task. For example, if the task is human pose
        estimation, the class id should correspond to the "human" class. Note
        that the ID is 0-based, meaning that class 0 corresponds to the first
        non-background object class.
      keypoint_indices: A list of integers representing the indicies of the
        keypoints to be considered in this task. This is used to retrieve the
        subset of the keypoints from gt_keypoints that should be considered in
        this task.
      classification_loss: an object_detection.core.losses.Loss object to
        compute the loss for the class predictions in CenterNet.
      localization_loss: an object_detection.core.losses.Loss object to compute
        the loss for the center offset and height/width predictions in
        CenterNet.
      keypoint_labels: A list of strings representing the label text of each
        keypoint, e.g. "nose", 'left_shoulder". Note that the length of this
        list should be equal to keypoint_indices.
      keypoint_std_dev: A list of float represent the standard deviation of the
        Gaussian kernel used to generate the keypoint heatmap. It is to provide
        the flexibility of using different sizes of Gaussian kernel for each
        keypoint class.
      keypoint_heatmap_loss_weight: float, The weight for the keypoint heatmap.
      keypoint_offset_loss_weight: float, The weight for the keypoint offsets
        loss.
      keypoint_regression_loss_weight: float, The weight for keypoint regression
        loss. Note that the loss is dependent on the input image size, since we
        penalize the raw height and width. This constant may need to be adjusted
        depending on the input size.
      keypoint_candidate_score_threshold: float, The heatmap score threshold for
        a keypoint to become a valid candidate.
      heatmap_bias_init: float, the initial value of bias in the convolutional
        kernel of the class prediction head. If set to None, the bias is
        initialized with zeros.
      num_candidates_per_keypoint: The maximum number of candidates to retrieve
        for each keypoint.
      task_loss_weight: float, the weight of the keypoint estimation loss.
      peak_max_pool_kernel_size: Max pool kernel size to use to pull off peak
        score locations in a neighborhood (independently for each keypoint
        types).
      unmatched_keypoint_score: The default score to use for regressed keypoints
        that are not successfully snapped to a nearby candidate.
      box_scale: The multiplier to expand the bounding boxes (either the
        provided boxes or those which tightly cover the regressed keypoints).
      candidate_search_scale: The scale parameter that multiplies the largest
        dimension of a bounding box. The resulting distance becomes a search
        radius for candidates in the vicinity of each regressed keypoint.
      candidate_ranking_mode: One of ['min_distance', 'score_distance_ratio',
        'score_scaled_distance_ratio', 'gaussian_weighted'] indicating how to
        select the keypoint candidate.
      offset_peak_radius: The radius (in the unit of output pixel) around
        groundtruth heatmap peak to assign the offset targets. If set 0, then
        the offset target will only be assigned to the heatmap peak (same
        behavior as the original paper).
      per_keypoint_offset: A bool indicates whether to assign offsets for each
        keypoint channel separately. If set False, the output offset target has
        the shape [batch_size, out_height, out_width, 2] (same behavior as the
        original paper). If set True, the output offset target has the shape
        [batch_size, out_height, out_width, 2 * num_keypoints] (recommended when
        the offset_peak_radius is not zero).
      predict_depth: A bool indicates whether to predict the depth of each
        keypoints.
      per_keypoint_depth: A bool indicates whether the model predicts the depth
        of each keypoints in independent channels. Similar to
        per_keypoint_offset but for the keypoint depth.
      keypoint_depth_loss_weight: The weight of the keypoint depth loss.
      score_distance_offset: The distance offset to apply in the denominator
        when candidate_ranking_mode is 'score_distance_ratio'. The metric to
        maximize in this scenario is score / (distance + score_distance_offset).
        Larger values of score_distance_offset make the keypoint score gain more
        relative importance.
      clip_out_of_frame_keypoints: Whether keypoints outside the image frame
        should be clipped back to the image boundary. If True, the keypoints
        that are clipped have scores set to 0.0.
      rescore_instances: Whether to rescore instances based on a combination of
        detection score and keypoint scores.
      heatmap_head_num_filters: filter numbers of the convolutional layers used
        by the keypoint heatmap prediction head.
      heatmap_head_kernel_sizes: kernel size of the convolutional layers used
        by the keypoint heatmap prediction head.
      offset_head_num_filters: filter numbers of the convolutional layers used
        by the keypoint offset prediction head.
      offset_head_kernel_sizes: kernel size of the convolutional layers used
        by the keypoint offset prediction head.
      regress_head_num_filters: filter numbers of the convolutional layers used
        by the keypoint regression prediction head.
      regress_head_kernel_sizes: kernel size of the convolutional layers used
        by the keypoint regression prediction head.
      score_distance_multiplier: A scalar used to multiply the bounding box size
        to be used as the offset in the score-to-distance-ratio formula.
      std_dev_multiplier: A scalar used to multiply the standard deviation to
        control the Gaussian kernel which used to weight the candidates.
      rescoring_threshold: A scalar used when "rescore_instances" is set to
        True. The detection score of an instance is set to be the average over
        the scores of the keypoints which their scores higher than the
        threshold.
      argmax_postprocessing: Whether to use the keypoint postprocessing logic
        that replaces the topk op with argmax. Usually used when exporting the
        model for predicting keypoints of multiple instances in the browser.
      gaussian_denom_ratio: The ratio used to multiply the image size to
        determine the denominator of the Gaussian formula. Only applicable when
        the candidate_ranking_mode is set to be 'gaussian_weighted_const'.

    Returns:
      An initialized KeypointEstimationParams namedtuple.
    """
    return super(KeypointEstimationParams, cls).__new__(
        cls, task_name, class_id, keypoint_indices, classification_loss,
        localization_loss, keypoint_labels, keypoint_std_dev,
        keypoint_heatmap_loss_weight, keypoint_offset_loss_weight,
        keypoint_regression_loss_weight, keypoint_candidate_score_threshold,
        heatmap_bias_init, num_candidates_per_keypoint, task_loss_weight,
        peak_max_pool_kernel_size, unmatched_keypoint_score, box_scale,
        candidate_search_scale, candidate_ranking_mode, offset_peak_radius,
        per_keypoint_offset, predict_depth, per_keypoint_depth,
        keypoint_depth_loss_weight, score_distance_offset,
        clip_out_of_frame_keypoints, rescore_instances,
        heatmap_head_num_filters, heatmap_head_kernel_sizes,
        offset_head_num_filters, offset_head_kernel_sizes,
        regress_head_num_filters, regress_head_kernel_sizes,
        score_distance_multiplier, std_dev_multiplier, rescoring_threshold,
        argmax_postprocessing, gaussian_denom_ratio)


class ObjectCenterParams(
    collections.namedtuple('ObjectCenterParams', [
        'classification_loss', 'object_center_loss_weight', 'heatmap_bias_init',
        'min_box_overlap_iou', 'max_box_predictions', 'use_labeled_classes',
        'keypoint_weights_for_center', 'center_head_num_filters',
        'center_head_kernel_sizes', 'peak_max_pool_kernel_size'
    ])):
  """Namedtuple to store object center prediction related parameters."""

  __slots__ = ()

  def __new__(cls,
              classification_loss,
              object_center_loss_weight,
              heatmap_bias_init=-2.19,
              min_box_overlap_iou=0.7,
              max_box_predictions=100,
              use_labeled_classes=False,
              keypoint_weights_for_center=None,
              center_head_num_filters=(256),
              center_head_kernel_sizes=(3),
              peak_max_pool_kernel_size=3):
    """Constructor with default values for ObjectCenterParams.

    Args:
      classification_loss: an object_detection.core.losses.Loss object to
        compute the loss for the class predictions in CenterNet.
      object_center_loss_weight: float, The weight for the object center loss.
      heatmap_bias_init: float, the initial value of bias in the convolutional
        kernel of the object center prediction head. If set to None, the bias is
        initialized with zeros.
      min_box_overlap_iou: float, the minimum IOU overlap that predicted boxes
        need have with groundtruth boxes to not be penalized. This is used for
        computing the class specific center heatmaps.
      max_box_predictions: int, the maximum number of boxes to predict.
      use_labeled_classes: boolean, compute the loss only labeled classes.
      keypoint_weights_for_center: (optional) The keypoint weights used for
        calculating the location of object center. If provided, the number of
        weights need to be the same as the number of keypoints. The object
        center is calculated by the weighted mean of the keypoint locations. If
        not provided, the object center is determined by the center of the
        bounding box (default behavior).
      center_head_num_filters: filter numbers of the convolutional layers used
        by the object center prediction head.
      center_head_kernel_sizes: kernel size of the convolutional layers used
        by the object center prediction head.
      peak_max_pool_kernel_size: Max pool kernel size to use to pull off peak
        score locations in a neighborhood for the object detection heatmap.
    Returns:
      An initialized ObjectCenterParams namedtuple.
    """
    return super(ObjectCenterParams,
                 cls).__new__(cls, classification_loss,
                              object_center_loss_weight, heatmap_bias_init,
                              min_box_overlap_iou, max_box_predictions,
                              use_labeled_classes, keypoint_weights_for_center,
                              center_head_num_filters, center_head_kernel_sizes,
                              peak_max_pool_kernel_size)


class MaskParams(
    collections.namedtuple('MaskParams', [
        'classification_loss', 'task_loss_weight', 'mask_height', 'mask_width',
        'score_threshold', 'heatmap_bias_init', 'mask_head_num_filters',
        'mask_head_kernel_sizes'
    ])):
  """Namedtuple to store mask prediction related parameters."""

  __slots__ = ()

  def __new__(cls,
              classification_loss,
              task_loss_weight=1.0,
              mask_height=256,
              mask_width=256,
              score_threshold=0.5,
              heatmap_bias_init=-2.19,
              mask_head_num_filters=(256),
              mask_head_kernel_sizes=(3)):
    """Constructor with default values for MaskParams.

    Args:
      classification_loss: an object_detection.core.losses.Loss object to
        compute the loss for the semantic segmentation predictions in CenterNet.
      task_loss_weight: float, The loss weight for the segmentation task.
      mask_height: The height of the resized instance segmentation mask.
      mask_width: The width of the resized instance segmentation mask.
      score_threshold: The threshold at which to convert predicted mask
        probabilities (after passing through sigmoid) into foreground pixels.
      heatmap_bias_init: float, the initial value of bias in the convolutional
        kernel of the semantic segmentation prediction head. If set to None, the
        bias is initialized with zeros.
      mask_head_num_filters: filter numbers of the convolutional layers used
        by the mask prediction head.
      mask_head_kernel_sizes: kernel size of the convolutional layers used
        by the mask prediction head.

    Returns:
      An initialized MaskParams namedtuple.
    """
    return super(MaskParams,
                 cls).__new__(cls, classification_loss,
                              task_loss_weight, mask_height, mask_width,
                              score_threshold, heatmap_bias_init,
                              mask_head_num_filters, mask_head_kernel_sizes)


class DensePoseParams(
    collections.namedtuple('DensePoseParams', [
        'class_id', 'classification_loss', 'localization_loss',
        'part_loss_weight', 'coordinate_loss_weight', 'num_parts',
        'task_loss_weight', 'upsample_to_input_res', 'upsample_method',
        'heatmap_bias_init'
    ])):
  """Namedtuple to store DensePose prediction related parameters."""

  __slots__ = ()

  def __new__(cls,
              class_id,
              classification_loss,
              localization_loss,
              part_loss_weight=1.0,
              coordinate_loss_weight=1.0,
              num_parts=24,
              task_loss_weight=1.0,
              upsample_to_input_res=True,
              upsample_method='bilinear',
              heatmap_bias_init=-2.19):
    """Constructor with default values for DensePoseParams.

    Args:
      class_id: the ID of the class that contains the DensePose groundtruth.
        This should typically correspond to the "person" class. Note that the ID
        is 0-based, meaning that class 0 corresponds to the first non-background
        object class.
      classification_loss: an object_detection.core.losses.Loss object to
        compute the loss for the body part predictions in CenterNet.
      localization_loss: an object_detection.core.losses.Loss object to compute
        the loss for the surface coordinate regression in CenterNet.
      part_loss_weight: The loss weight to apply to part prediction.
      coordinate_loss_weight: The loss weight to apply to surface coordinate
        prediction.
      num_parts: The number of DensePose parts to predict.
      task_loss_weight: float, the loss weight for the DensePose task.
      upsample_to_input_res: Whether to upsample the DensePose feature maps to
        the input resolution before applying loss. Note that the prediction
        outputs are still at the standard CenterNet output stride.
      upsample_method: Method for upsampling DensePose feature maps. Options are
        either 'bilinear' or 'nearest'). This takes no effect when
        `upsample_to_input_res` is False.
      heatmap_bias_init: float, the initial value of bias in the convolutional
        kernel of the part prediction head. If set to None, the
        bias is initialized with zeros.

    Returns:
      An initialized DensePoseParams namedtuple.
    """
    return super(DensePoseParams,
                 cls).__new__(cls, class_id, classification_loss,
                              localization_loss, part_loss_weight,
                              coordinate_loss_weight, num_parts,
                              task_loss_weight, upsample_to_input_res,
                              upsample_method, heatmap_bias_init)


class TrackParams(
    collections.namedtuple('TrackParams', [
        'num_track_ids', 'reid_embed_size', 'num_fc_layers',
        'classification_loss', 'task_loss_weight'
    ])):
  """Namedtuple to store tracking prediction related parameters."""

  __slots__ = ()

  def __new__(cls,
              num_track_ids,
              reid_embed_size,
              num_fc_layers,
              classification_loss,
              task_loss_weight=1.0):
    """Constructor with default values for TrackParams.

    Args:
      num_track_ids: int. The maximum track ID in the dataset. Used for ReID
        embedding classification task.
      reid_embed_size: int. The embedding size for ReID task.
      num_fc_layers: int. The number of (fully-connected, batch-norm, relu)
        layers for track ID classification head.
      classification_loss: an object_detection.core.losses.Loss object to
        compute the loss for the ReID embedding in CenterNet.
      task_loss_weight: float, the loss weight for the tracking task.

    Returns:
      An initialized TrackParams namedtuple.
    """
    return super(TrackParams,
                 cls).__new__(cls, num_track_ids, reid_embed_size,
                              num_fc_layers, classification_loss,
                              task_loss_weight)


class TemporalOffsetParams(
    collections.namedtuple('TemporalOffsetParams', [
        'localization_loss', 'task_loss_weight'
    ])):
  """Namedtuple to store temporal offset related parameters."""

  __slots__ = ()

  def __new__(cls,
              localization_loss,
              task_loss_weight=1.0):
    """Constructor with default values for TrackParams.

    Args:
      localization_loss: an object_detection.core.losses.Loss object to
        compute the loss for the temporal offset in CenterNet.
      task_loss_weight: float, the loss weight for the temporal offset
        task.

    Returns:
      An initialized TemporalOffsetParams namedtuple.
    """
    return super(TemporalOffsetParams,
                 cls).__new__(cls, localization_loss, task_loss_weight)

# The following constants are used to generate the keys of the
# (prediction, loss, target assigner,...) dictionaries used in CenterNetMetaArch
# class.
DETECTION_TASK = 'detection_task'
OBJECT_CENTER = 'object_center'
BOX_SCALE = 'box/scale'
BOX_OFFSET = 'box/offset'
KEYPOINT_REGRESSION = 'keypoint/regression'
KEYPOINT_HEATMAP = 'keypoint/heatmap'
KEYPOINT_OFFSET = 'keypoint/offset'
KEYPOINT_DEPTH = 'keypoint/depth'
SEGMENTATION_TASK = 'segmentation_task'
SEGMENTATION_HEATMAP = 'segmentation/heatmap'
DENSEPOSE_TASK = 'densepose_task'
DENSEPOSE_HEATMAP = 'densepose/heatmap'
DENSEPOSE_REGRESSION = 'densepose/regression'
LOSS_KEY_PREFIX = 'Loss'
TRACK_TASK = 'track_task'
TRACK_REID = 'track/reid'
TEMPORALOFFSET_TASK = 'temporal_offset_task'
TEMPORAL_OFFSET = 'track/offset'


def get_keypoint_name(task_name, head_name):
  return '%s/%s' % (task_name, head_name)


def get_num_instances_from_weights(groundtruth_weights_list):
  """Computes the number of instances/boxes from the weights in a batch.

  Args:
    groundtruth_weights_list: A list of float tensors with shape
      [max_num_instances] representing whether there is an actual instance in
      the image (with non-zero value) or is padded to match the
      max_num_instances (with value 0.0). The list represents the batch
      dimension.

  Returns:
    A scalar integer tensor incidating how many instances/boxes are in the
    images in the batch. Note that this function is usually used to normalize
    the loss so the minimum return value is 1 to avoid weird behavior.
  """
  num_instances = tf.reduce_sum(
      [tf.math.count_nonzero(w) for w in groundtruth_weights_list])
  num_instances = tf.maximum(num_instances, 1)
  return num_instances


class CenterNetMetaArch(model.DetectionModel):
  """The CenterNet meta architecture [1].

  [1]: https://arxiv.org/abs/1904.07850
  """

  def __init__(self,
               is_training,
               add_summaries,
               num_classes,
               feature_extractor,
               image_resizer_fn,
               object_center_params,
               object_detection_params=None,
               keypoint_params_dict=None,
               mask_params=None,
               densepose_params=None,
               track_params=None,
               temporal_offset_params=None,
               use_depthwise=False,
               compute_heatmap_sparse=False,
               non_max_suppression_fn=None,
               unit_height_conv=False):
    """Initializes a CenterNet model.

    Args:
      is_training: Set to True if this model is being built for training.
      add_summaries: Whether to add tf summaries in the model.
      num_classes: int, The number of classes that the model should predict.
      feature_extractor: A CenterNetFeatureExtractor to use to extract features
        from an image.
      image_resizer_fn: a callable for image resizing.  This callable always
        takes a rank-3 image tensor (corresponding to a single image) and
        returns a rank-3 image tensor, possibly with new spatial dimensions and
        a 1-D tensor of shape [3] indicating shape of true image within the
        resized image tensor as the resized image tensor could be padded. See
        builders/image_resizer_builder.py.
      object_center_params: An ObjectCenterParams namedtuple. This object holds
        the hyper-parameters for object center prediction. This is required by
        either object detection or keypoint estimation tasks.
      object_detection_params: An ObjectDetectionParams namedtuple. This object
        holds the hyper-parameters necessary for object detection. Please see
        the class definition for more details.
      keypoint_params_dict: A dictionary that maps from task name to the
        corresponding KeypointEstimationParams namedtuple. This object holds the
        hyper-parameters necessary for multiple keypoint estimations. Please
        see the class definition for more details.
      mask_params: A MaskParams namedtuple. This object
        holds the hyper-parameters for segmentation. Please see the class
        definition for more details.
      densepose_params: A DensePoseParams namedtuple. This object holds the
        hyper-parameters for DensePose prediction. Please see the class
        definition for more details. Note that if this is provided, it is
        expected that `mask_params` is also provided.
      track_params: A TrackParams namedtuple. This object
        holds the hyper-parameters for tracking. Please see the class
        definition for more details.
      temporal_offset_params: A TemporalOffsetParams namedtuple. This object
        holds the hyper-parameters for offset prediction based tracking.
      use_depthwise: If true, all task heads will be constructed using
        separable_conv. Otherwise, standard convoltuions will be used.
      compute_heatmap_sparse: bool, whether or not to use the sparse version of
        the Op that computes the center heatmaps. The sparse version scales
        better with number of channels in the heatmap, but in some cases is
        known to cause an OOM error. See b/170989061.
      non_max_suppression_fn: Optional Non Max Suppression function to apply.
      unit_height_conv: If True, Conv2Ds in prediction heads have asymmetric
        kernels with height=1.
    """
    assert object_detection_params or keypoint_params_dict
    # Shorten the name for convenience and better formatting.
    self._is_training = is_training
    # The Objects as Points paper attaches loss functions to multiple
    # (`num_feature_outputs`) feature maps in the the backbone. E.g.
    # for the hourglass  backbone, `num_feature_outputs` is 2.
    self._num_classes = num_classes
    self._feature_extractor = feature_extractor
    self._num_feature_outputs = feature_extractor.num_feature_outputs
    self._stride = self._feature_extractor.out_stride
    self._image_resizer_fn = image_resizer_fn
    self._center_params = object_center_params
    self._od_params = object_detection_params
    self._kp_params_dict = keypoint_params_dict
    self._mask_params = mask_params
    if densepose_params is not None and mask_params is None:
      raise ValueError('To run DensePose prediction, `mask_params` must also '
                       'be supplied.')
    self._densepose_params = densepose_params
    self._track_params = track_params
    self._temporal_offset_params = temporal_offset_params

    self._use_depthwise = use_depthwise
    self._compute_heatmap_sparse = compute_heatmap_sparse

    # subclasses may not implement the unit_height_conv arg, so only provide it
    # as a kwarg if it is True.
    kwargs = {'unit_height_conv': unit_height_conv} if unit_height_conv else {}
    # Construct the prediction head nets.
    self._prediction_head_dict = self._construct_prediction_heads(
        num_classes,
        self._num_feature_outputs,
        class_prediction_bias_init=self._center_params.heatmap_bias_init,
        **kwargs)
    # Initialize the target assigners.
    self._target_assigner_dict = self._initialize_target_assigners(
        stride=self._stride,
        min_box_overlap_iou=self._center_params.min_box_overlap_iou)

    # Will be used in VOD single_frame_meta_arch for tensor reshape.
    self._batched_prediction_tensor_names = []
    self._non_max_suppression_fn = non_max_suppression_fn

    super(CenterNetMetaArch, self).__init__(num_classes)

  def set_trainability_by_layer_traversal(self, trainable):
    """Sets trainability layer by layer.

    The commonly-seen `model.trainable = False` method does not traverse
    the children layer. For example, if the parent is not trainable, we won't
    be able to set individual layers as trainable/non-trainable differentially.

    Args:
      trainable: (bool) Setting this for the model layer by layer except for
        the parent itself.
    """
    for layer in self._flatten_layers(include_self=False):
      layer.trainable = trainable

  @property
  def prediction_head_dict(self):
    return self._prediction_head_dict

  @property
  def batched_prediction_tensor_names(self):
    if not self._batched_prediction_tensor_names:
      raise RuntimeError('Must call predict() method to get batched prediction '
                         'tensor names.')
    return self._batched_prediction_tensor_names

  def _make_prediction_net_list(self, num_feature_outputs, num_out_channels,
                                kernel_sizes=(3), num_filters=(256),
                                bias_fill=None, name=None,
                                unit_height_conv=False):
    prediction_net_list = []
    for i in range(num_feature_outputs):
      prediction_net_list.append(
          make_prediction_net(
              num_out_channels,
              kernel_sizes=kernel_sizes,
              num_filters=num_filters,
              bias_fill=bias_fill,
              use_depthwise=self._use_depthwise,
              name='{}_{}'.format(name, i) if name else name,
              unit_height_conv=unit_height_conv))
    return prediction_net_list

  def _construct_prediction_heads(self, num_classes, num_feature_outputs,
                                  class_prediction_bias_init,
                                  unit_height_conv=False):
    """Constructs the prediction heads based on the specific parameters.

    Args:
      num_classes: An integer indicating how many classes in total to predict.
      num_feature_outputs: An integer indicating how many feature outputs to use
        for calculating the loss. The Objects as Points paper attaches loss
        functions to multiple (`num_feature_outputs`) feature maps in the the
        backbone. E.g. for the hourglass backbone, `num_feature_outputs` is 2.
      class_prediction_bias_init: float, the initial value of bias in the
        convolutional kernel of the class prediction head. If set to None, the
        bias is initialized with zeros.
      unit_height_conv: If True, Conv2Ds have asymmetric kernels with height=1.

    Returns:
      A dictionary of keras modules generated by calling make_prediction_net
      function. It will also create and set a private member of the class when
      learning the tracking task.
    """
    prediction_heads = {}
    prediction_heads[OBJECT_CENTER] = self._make_prediction_net_list(
        num_feature_outputs,
        num_classes,
        kernel_sizes=self._center_params.center_head_kernel_sizes,
        num_filters=self._center_params.center_head_num_filters,
        bias_fill=class_prediction_bias_init,
        name='center',
        unit_height_conv=unit_height_conv)

    if self._od_params is not None:
      prediction_heads[BOX_SCALE] = self._make_prediction_net_list(
          num_feature_outputs,
          NUM_SIZE_CHANNELS,
          kernel_sizes=self._od_params.scale_head_kernel_sizes,
          num_filters=self._od_params.scale_head_num_filters,
          name='box_scale',
          unit_height_conv=unit_height_conv)
      prediction_heads[BOX_OFFSET] = self._make_prediction_net_list(
          num_feature_outputs,
          NUM_OFFSET_CHANNELS,
          kernel_sizes=self._od_params.offset_head_kernel_sizes,
          num_filters=self._od_params.offset_head_num_filters,
          name='box_offset',
          unit_height_conv=unit_height_conv)

    if self._kp_params_dict is not None:
      for task_name, kp_params in self._kp_params_dict.items():
        num_keypoints = len(kp_params.keypoint_indices)
        prediction_heads[get_keypoint_name(
            task_name, KEYPOINT_HEATMAP)] = self._make_prediction_net_list(
                num_feature_outputs,
                num_keypoints,
                kernel_sizes=kp_params.heatmap_head_kernel_sizes,
                num_filters=kp_params.heatmap_head_num_filters,
                bias_fill=kp_params.heatmap_bias_init,
                name='kpt_heatmap',
                unit_height_conv=unit_height_conv)
        prediction_heads[get_keypoint_name(
            task_name, KEYPOINT_REGRESSION)] = self._make_prediction_net_list(
                num_feature_outputs,
                NUM_OFFSET_CHANNELS * num_keypoints,
                kernel_sizes=kp_params.regress_head_kernel_sizes,
                num_filters=kp_params.regress_head_num_filters,
                name='kpt_regress',
                unit_height_conv=unit_height_conv)

        if kp_params.per_keypoint_offset:
          prediction_heads[get_keypoint_name(
              task_name, KEYPOINT_OFFSET)] = self._make_prediction_net_list(
                  num_feature_outputs,
                  NUM_OFFSET_CHANNELS * num_keypoints,
                  kernel_sizes=kp_params.offset_head_kernel_sizes,
                  num_filters=kp_params.offset_head_num_filters,
                  name='kpt_offset',
                  unit_height_conv=unit_height_conv)
        else:
          prediction_heads[get_keypoint_name(
              task_name, KEYPOINT_OFFSET)] = self._make_prediction_net_list(
                  num_feature_outputs,
                  NUM_OFFSET_CHANNELS,
                  kernel_sizes=kp_params.offset_head_kernel_sizes,
                  num_filters=kp_params.offset_head_num_filters,
                  name='kpt_offset',
                  unit_height_conv=unit_height_conv)

        if kp_params.predict_depth:
          num_depth_channel = (
              num_keypoints if kp_params.per_keypoint_depth else 1)
          prediction_heads[get_keypoint_name(
              task_name, KEYPOINT_DEPTH)] = self._make_prediction_net_list(
                  num_feature_outputs, num_depth_channel, name='kpt_depth',
                  unit_height_conv=unit_height_conv)

    if self._mask_params is not None:
      prediction_heads[SEGMENTATION_HEATMAP] = self._make_prediction_net_list(
          num_feature_outputs,
          num_classes,
          kernel_sizes=self._mask_params.mask_head_kernel_sizes,
          num_filters=self._mask_params.mask_head_num_filters,
          bias_fill=self._mask_params.heatmap_bias_init,
          name='seg_heatmap',
          unit_height_conv=unit_height_conv)

    if self._densepose_params is not None:
      prediction_heads[DENSEPOSE_HEATMAP] = self._make_prediction_net_list(
          num_feature_outputs,
          self._densepose_params.num_parts,
          bias_fill=self._densepose_params.heatmap_bias_init,
          name='dense_pose_heatmap',
          unit_height_conv=unit_height_conv)
      prediction_heads[DENSEPOSE_REGRESSION] = self._make_prediction_net_list(
          num_feature_outputs,
          2 * self._densepose_params.num_parts,
          name='dense_pose_regress',
          unit_height_conv=unit_height_conv)

    if self._track_params is not None:
      prediction_heads[TRACK_REID] = self._make_prediction_net_list(
          num_feature_outputs,
          self._track_params.reid_embed_size,
          name='track_reid',
          unit_height_conv=unit_height_conv)

      # Creates a classification network to train object embeddings by learning
      # a projection from embedding space to object track ID space.
      self.track_reid_classification_net = tf.keras.Sequential()
      for _ in range(self._track_params.num_fc_layers - 1):
        self.track_reid_classification_net.add(
            tf.keras.layers.Dense(self._track_params.reid_embed_size))
        self.track_reid_classification_net.add(
            tf.keras.layers.BatchNormalization())
        self.track_reid_classification_net.add(tf.keras.layers.ReLU())
      self.track_reid_classification_net.add(
          tf.keras.layers.Dense(self._track_params.num_track_ids))
    if self._temporal_offset_params is not None:
      prediction_heads[TEMPORAL_OFFSET] = self._make_prediction_net_list(
          num_feature_outputs, NUM_OFFSET_CHANNELS, name='temporal_offset',
          unit_height_conv=unit_height_conv)
    return prediction_heads

  def _initialize_target_assigners(self, stride, min_box_overlap_iou):
    """Initializes the target assigners and puts them in a dictionary.

    Args:
      stride: An integer indicating the stride of the image.
      min_box_overlap_iou: float, the minimum IOU overlap that predicted boxes
        need have with groundtruth boxes to not be penalized. This is used for
        computing the class specific center heatmaps.

    Returns:
      A dictionary of initialized target assigners for each task.
    """
    target_assigners = {}
    keypoint_weights_for_center = (
        self._center_params.keypoint_weights_for_center)
    if not keypoint_weights_for_center:
      target_assigners[OBJECT_CENTER] = (
          cn_assigner.CenterNetCenterHeatmapTargetAssigner(
              stride, min_box_overlap_iou, self._compute_heatmap_sparse))
      self._center_from_keypoints = False
    else:
      # Determining the object center location by keypoint location is only
      # supported when there is exactly one keypoint prediction task and no
      # object detection task is specified.
      assert len(self._kp_params_dict) == 1 and self._od_params is None
      kp_params = next(iter(self._kp_params_dict.values()))
      # The number of keypoint_weights_for_center needs to be the same as the
      # number of keypoints.
      assert len(keypoint_weights_for_center) == len(kp_params.keypoint_indices)
      target_assigners[OBJECT_CENTER] = (
          cn_assigner.CenterNetCenterHeatmapTargetAssigner(
              stride,
              min_box_overlap_iou,
              self._compute_heatmap_sparse,
              keypoint_class_id=kp_params.class_id,
              keypoint_indices=kp_params.keypoint_indices,
              keypoint_weights_for_center=keypoint_weights_for_center))
      self._center_from_keypoints = True
    if self._od_params is not None:
      target_assigners[DETECTION_TASK] = (
          cn_assigner.CenterNetBoxTargetAssigner(stride))
    if self._kp_params_dict is not None:
      for task_name, kp_params in self._kp_params_dict.items():
        target_assigners[task_name] = (
            cn_assigner.CenterNetKeypointTargetAssigner(
                stride=stride,
                class_id=kp_params.class_id,
                keypoint_indices=kp_params.keypoint_indices,
                keypoint_std_dev=kp_params.keypoint_std_dev,
                peak_radius=kp_params.offset_peak_radius,
                per_keypoint_offset=kp_params.per_keypoint_offset,
                compute_heatmap_sparse=self._compute_heatmap_sparse,
                per_keypoint_depth=kp_params.per_keypoint_depth))
    if self._mask_params is not None:
      target_assigners[SEGMENTATION_TASK] = (
          cn_assigner.CenterNetMaskTargetAssigner(stride, boxes_scale=1.05))
    if self._densepose_params is not None:
      dp_stride = 1 if self._densepose_params.upsample_to_input_res else stride
      target_assigners[DENSEPOSE_TASK] = (
          cn_assigner.CenterNetDensePoseTargetAssigner(dp_stride))
    if self._track_params is not None:
      target_assigners[TRACK_TASK] = (
          cn_assigner.CenterNetTrackTargetAssigner(
              stride, self._track_params.num_track_ids))
    if self._temporal_offset_params is not None:
      target_assigners[TEMPORALOFFSET_TASK] = (
          cn_assigner.CenterNetTemporalOffsetTargetAssigner(stride))

    return target_assigners

  def _compute_object_center_loss(self, input_height, input_width,
                                  object_center_predictions, per_pixel_weights,
                                  maximum_normalized_coordinate=1.1):
    """Computes the object center loss.

    Args:
      input_height: An integer scalar tensor representing input image height.
      input_width: An integer scalar tensor representing input image width.
      object_center_predictions: A list of float tensors of shape [batch_size,
        out_height, out_width, num_classes] representing the object center
        feature maps.
      per_pixel_weights: A float tensor of shape [batch_size,
        out_height * out_width, 1] with 1s in locations where the spatial
        coordinates fall within the height and width in true_image_shapes.
      maximum_normalized_coordinate: Maximum coordinate value to be considered
        as normalized, default to 1.1. This is used to check bounds during
        converting normalized coordinates to absolute coordinates.

    Returns:
      A float scalar tensor representing the object center loss per instance.
    """
    gt_classes_list = self.groundtruth_lists(fields.BoxListFields.classes)
    gt_weights_list = self.groundtruth_lists(fields.BoxListFields.weights)

    if self._center_params.use_labeled_classes:
      gt_labeled_classes_list = self.groundtruth_lists(
          fields.InputDataFields.groundtruth_labeled_classes)
      batch_labeled_classes = tf.stack(gt_labeled_classes_list, axis=0)
      batch_labeled_classes_shape = tf.shape(batch_labeled_classes)
      batch_labeled_classes = tf.reshape(
          batch_labeled_classes,
          [batch_labeled_classes_shape[0], 1, batch_labeled_classes_shape[-1]])
      per_pixel_weights = per_pixel_weights * batch_labeled_classes

    # Convert the groundtruth to targets.
    assigner = self._target_assigner_dict[OBJECT_CENTER]
    if self._center_from_keypoints:
      gt_keypoints_list = self.groundtruth_lists(fields.BoxListFields.keypoints)
      heatmap_targets = assigner.assign_center_targets_from_keypoints(
          height=input_height,
          width=input_width,
          gt_classes_list=gt_classes_list,
          gt_keypoints_list=gt_keypoints_list,
          gt_weights_list=gt_weights_list,
          maximum_normalized_coordinate=maximum_normalized_coordinate)
    else:
      gt_boxes_list = self.groundtruth_lists(fields.BoxListFields.boxes)
      heatmap_targets = assigner.assign_center_targets_from_boxes(
          height=input_height,
          width=input_width,
          gt_boxes_list=gt_boxes_list,
          gt_classes_list=gt_classes_list,
          gt_weights_list=gt_weights_list,
          maximum_normalized_coordinate=maximum_normalized_coordinate)

    flattened_heatmap_targets = _flatten_spatial_dimensions(heatmap_targets)
    num_boxes = _to_float32(get_num_instances_from_weights(gt_weights_list))

    loss = 0.0
    object_center_loss = self._center_params.classification_loss
    # Loop through each feature output head.
    for pred in object_center_predictions:
      pred = _flatten_spatial_dimensions(pred)
      loss += object_center_loss(
          pred, flattened_heatmap_targets, weights=per_pixel_weights)
    loss_per_instance = tf.reduce_sum(loss) / (
        float(len(object_center_predictions)) * num_boxes)
    return loss_per_instance

  def _compute_object_detection_losses(self, input_height, input_width,
                                       prediction_dict, per_pixel_weights,
                                       maximum_normalized_coordinate=1.1):
    """Computes the weighted object detection losses.

    This wrapper function calls the function which computes the losses for
    object detection task and applies corresponding weights to the losses.

    Args:
      input_height: An integer scalar tensor representing input image height.
      input_width: An integer scalar tensor representing input image width.
      prediction_dict: A dictionary holding predicted tensors output by
        "predict" function. See "predict" function for more detailed
        description.
      per_pixel_weights: A float tensor of shape [batch_size,
        out_height * out_width, 1] with 1s in locations where the spatial
        coordinates fall within the height and width in true_image_shapes.
      maximum_normalized_coordinate: Maximum coordinate value to be considered
        as normalized, default to 1.1. This is used to check bounds during
        converting normalized coordinates to absolute coordinates.

    Returns:
      A dictionary of scalar float tensors representing the weighted losses for
      object detection task:
         BOX_SCALE: the weighted scale (height/width) loss.
         BOX_OFFSET: the weighted object offset loss.
    """
    od_scale_loss, od_offset_loss = self._compute_box_scale_and_offset_loss(
        scale_predictions=prediction_dict[BOX_SCALE],
        offset_predictions=prediction_dict[BOX_OFFSET],
        input_height=input_height,
        input_width=input_width,
        maximum_normalized_coordinate=maximum_normalized_coordinate)
    loss_dict = {}
    loss_dict[BOX_SCALE] = (
        self._od_params.scale_loss_weight * od_scale_loss)
    loss_dict[BOX_OFFSET] = (
        self._od_params.offset_loss_weight * od_offset_loss)
    return loss_dict

  def _compute_box_scale_and_offset_loss(self, input_height, input_width,
                                         scale_predictions, offset_predictions,
                                         maximum_normalized_coordinate=1.1):
    """Computes the scale loss of the object detection task.

    Args:
      input_height: An integer scalar tensor representing input image height.
      input_width: An integer scalar tensor representing input image width.
      scale_predictions: A list of float tensors of shape [batch_size,
        out_height, out_width, 2] representing the prediction heads of the model
        for object scale (i.e height and width).
      offset_predictions: A list of float tensors of shape [batch_size,
        out_height, out_width, 2] representing the prediction heads of the model
        for object offset.
      maximum_normalized_coordinate: Maximum coordinate value to be considered
        as normalized, default to 1.1. This is used to check bounds during
        converting normalized coordinates to absolute coordinates.

    Returns:
      A tuple of two losses:
        scale_loss: A float scalar tensor representing the object height/width
          loss normalized by total number of boxes.
        offset_loss: A float scalar tensor representing the object offset loss
          normalized by total number of boxes
    """
    # TODO(vighneshb) Explore a size invariant version of scale loss.
    gt_boxes_list = self.groundtruth_lists(fields.BoxListFields.boxes)
    gt_weights_list = self.groundtruth_lists(fields.BoxListFields.weights)
    num_boxes = _to_float32(get_num_instances_from_weights(gt_weights_list))
    num_predictions = float(len(scale_predictions))

    assigner = self._target_assigner_dict[DETECTION_TASK]
    (batch_indices, batch_height_width_targets, batch_offset_targets,
     batch_weights) = assigner.assign_size_and_offset_targets(
         height=input_height,
         width=input_width,
         gt_boxes_list=gt_boxes_list,
         gt_weights_list=gt_weights_list,
         maximum_normalized_coordinate=maximum_normalized_coordinate)
    batch_weights = tf.expand_dims(batch_weights, -1)

    scale_loss = 0
    offset_loss = 0
    localization_loss_fn = self._od_params.localization_loss
    for scale_pred, offset_pred in zip(scale_predictions, offset_predictions):
      # Compute the scale loss.
      scale_pred = cn_assigner.get_batch_predictions_from_indices(
          scale_pred, batch_indices)
      scale_loss += localization_loss_fn(
          scale_pred, batch_height_width_targets, weights=batch_weights)
      # Compute the offset loss.
      offset_pred = cn_assigner.get_batch_predictions_from_indices(
          offset_pred, batch_indices)
      offset_loss += localization_loss_fn(
          offset_pred, batch_offset_targets, weights=batch_weights)
    scale_loss = tf.reduce_sum(scale_loss) / (
        num_predictions * num_boxes)
    offset_loss = tf.reduce_sum(offset_loss) / (
        num_predictions * num_boxes)
    return scale_loss, offset_loss

  def _compute_keypoint_estimation_losses(self, task_name, input_height,
                                          input_width, prediction_dict,
                                          per_pixel_weights):
    """Computes the weighted keypoint losses."""
    kp_params = self._kp_params_dict[task_name]
    heatmap_key = get_keypoint_name(task_name, KEYPOINT_HEATMAP)
    offset_key = get_keypoint_name(task_name, KEYPOINT_OFFSET)
    regression_key = get_keypoint_name(task_name, KEYPOINT_REGRESSION)
    depth_key = get_keypoint_name(task_name, KEYPOINT_DEPTH)
    heatmap_loss = self._compute_kp_heatmap_loss(
        input_height=input_height,
        input_width=input_width,
        task_name=task_name,
        heatmap_predictions=prediction_dict[heatmap_key],
        classification_loss_fn=kp_params.classification_loss,
        per_pixel_weights=per_pixel_weights)
    offset_loss = self._compute_kp_offset_loss(
        input_height=input_height,
        input_width=input_width,
        task_name=task_name,
        offset_predictions=prediction_dict[offset_key],
        localization_loss_fn=kp_params.localization_loss)
    reg_loss = self._compute_kp_regression_loss(
        input_height=input_height,
        input_width=input_width,
        task_name=task_name,
        regression_predictions=prediction_dict[regression_key],
        localization_loss_fn=kp_params.localization_loss)

    loss_dict = {}
    loss_dict[heatmap_key] = (
        kp_params.keypoint_heatmap_loss_weight * heatmap_loss)
    loss_dict[offset_key] = (
        kp_params.keypoint_offset_loss_weight * offset_loss)
    loss_dict[regression_key] = (
        kp_params.keypoint_regression_loss_weight * reg_loss)
    if kp_params.predict_depth:
      depth_loss = self._compute_kp_depth_loss(
          input_height=input_height,
          input_width=input_width,
          task_name=task_name,
          depth_predictions=prediction_dict[depth_key],
          localization_loss_fn=kp_params.localization_loss)
      loss_dict[depth_key] = kp_params.keypoint_depth_loss_weight * depth_loss
    return loss_dict

  def _compute_kp_heatmap_loss(self, input_height, input_width, task_name,
                               heatmap_predictions, classification_loss_fn,
                               per_pixel_weights):
    """Computes the heatmap loss of the keypoint estimation task.

    Args:
      input_height: An integer scalar tensor representing input image height.
      input_width: An integer scalar tensor representing input image width.
      task_name: A string representing the name of the keypoint task.
      heatmap_predictions: A list of float tensors of shape [batch_size,
        out_height, out_width, num_keypoints] representing the prediction heads
        of the model for keypoint heatmap.
      classification_loss_fn: An object_detection.core.losses.Loss object to
        compute the loss for the class predictions in CenterNet.
      per_pixel_weights: A float tensor of shape [batch_size,
        out_height * out_width, 1] with 1s in locations where the spatial
        coordinates fall within the height and width in true_image_shapes.

    Returns:
      loss: A float scalar tensor representing the object keypoint heatmap loss
        normalized by number of instances.
    """
    gt_keypoints_list = self.groundtruth_lists(fields.BoxListFields.keypoints)
    gt_classes_list = self.groundtruth_lists(fields.BoxListFields.classes)
    gt_weights_list = self.groundtruth_lists(fields.BoxListFields.weights)
    gt_boxes_list = self.groundtruth_lists(fields.BoxListFields.boxes)

    assigner = self._target_assigner_dict[task_name]
    (keypoint_heatmap, num_instances_per_kp_type,
     valid_mask_batch) = assigner.assign_keypoint_heatmap_targets(
         height=input_height,
         width=input_width,
         gt_keypoints_list=gt_keypoints_list,
         gt_weights_list=gt_weights_list,
         gt_classes_list=gt_classes_list,
         gt_boxes_list=gt_boxes_list)
    flattened_valid_mask = _flatten_spatial_dimensions(valid_mask_batch)
    flattened_heapmap_targets = _flatten_spatial_dimensions(keypoint_heatmap)
    # Sum over the number of instances per keypoint types to get the total
    # number of keypoints. Note that this is used to normalized the loss and we
    # keep the minimum value to be 1 to avoid generating weird loss value when
    # no keypoint is in the image batch.
    num_instances = tf.maximum(
        tf.cast(tf.reduce_sum(num_instances_per_kp_type), dtype=tf.float32),
        1.0)
    loss = 0.0
    # Loop through each feature output head.
    for pred in heatmap_predictions:
      pred = _flatten_spatial_dimensions(pred)
      unweighted_loss = classification_loss_fn(
          pred,
          flattened_heapmap_targets,
          weights=tf.ones_like(per_pixel_weights))
      # Apply the weights after the loss function to have full control over it.
      loss += unweighted_loss * per_pixel_weights * flattened_valid_mask
    loss = tf.reduce_sum(loss) / (
        float(len(heatmap_predictions)) * num_instances)
    return loss

  def _compute_kp_offset_loss(self, input_height, input_width, task_name,
                              offset_predictions, localization_loss_fn):
    """Computes the offset loss of the keypoint estimation task.

    Args:
      input_height: An integer scalar tensor representing input image height.
      input_width: An integer scalar tensor representing input image width.
      task_name: A string representing the name of the keypoint task.
      offset_predictions: A list of float tensors of shape [batch_size,
        out_height, out_width, 2] representing the prediction heads of the model
        for keypoint offset.
      localization_loss_fn: An object_detection.core.losses.Loss object to
        compute the loss for the keypoint offset predictions in CenterNet.

    Returns:
      loss: A float scalar tensor representing the keypoint offset loss
        normalized by number of total keypoints.
    """
    gt_keypoints_list = self.groundtruth_lists(fields.BoxListFields.keypoints)
    gt_classes_list = self.groundtruth_lists(fields.BoxListFields.classes)
    gt_weights_list = self.groundtruth_lists(fields.BoxListFields.weights)

    assigner = self._target_assigner_dict[task_name]
    (batch_indices, batch_offsets,
     batch_weights) = assigner.assign_keypoints_offset_targets(
         height=input_height,
         width=input_width,
         gt_keypoints_list=gt_keypoints_list,
         gt_weights_list=gt_weights_list,
         gt_classes_list=gt_classes_list)

    # Keypoint offset loss.
    loss = 0.0
    for prediction in offset_predictions:
      batch_size, out_height, out_width, channels = _get_shape(prediction, 4)
      if channels > 2:
        prediction = tf.reshape(
            prediction, shape=[batch_size, out_height, out_width, -1, 2])
      prediction = cn_assigner.get_batch_predictions_from_indices(
          prediction, batch_indices)
      # The dimensions passed are not as per the doc string but the loss
      # still computes the correct value.
      unweighted_loss = localization_loss_fn(
          prediction,
          batch_offsets,
          weights=tf.expand_dims(tf.ones_like(batch_weights), -1))
      # Apply the weights after the loss function to have full control over it.
      loss += batch_weights * tf.reduce_sum(unweighted_loss, axis=1)

    loss = tf.reduce_sum(loss) / (
        float(len(offset_predictions)) *
        tf.maximum(tf.reduce_sum(batch_weights), 1.0))
    return loss

  def _compute_kp_regression_loss(self, input_height, input_width, task_name,
                                  regression_predictions, localization_loss_fn):
    """Computes the keypoint regression loss of the keypoint estimation task.

    Args:
      input_height: An integer scalar tensor representing input image height.
      input_width: An integer scalar tensor representing input image width.
      task_name: A string representing the name of the keypoint task.
      regression_predictions: A list of float tensors of shape [batch_size,
        out_height, out_width, 2 * num_keypoints] representing the prediction
        heads of the model for keypoint regression offset.
      localization_loss_fn: An object_detection.core.losses.Loss object to
        compute the loss for the keypoint regression offset predictions in
        CenterNet.

    Returns:
      loss: A float scalar tensor representing the keypoint regression offset
        loss normalized by number of total keypoints.
    """
    gt_boxes_list = self.groundtruth_lists(fields.BoxListFields.boxes)
    gt_keypoints_list = self.groundtruth_lists(fields.BoxListFields.keypoints)
    gt_classes_list = self.groundtruth_lists(fields.BoxListFields.classes)
    gt_weights_list = self.groundtruth_lists(fields.BoxListFields.weights)
    # keypoint regression offset loss.
    assigner = self._target_assigner_dict[task_name]
    (batch_indices, batch_regression_offsets,
     batch_weights) = assigner.assign_joint_regression_targets(
         height=input_height,
         width=input_width,
         gt_keypoints_list=gt_keypoints_list,
         gt_classes_list=gt_classes_list,
         gt_weights_list=gt_weights_list,
         gt_boxes_list=gt_boxes_list)

    loss = 0.0
    for prediction in regression_predictions:
      batch_size, out_height, out_width, _ = _get_shape(prediction, 4)
      reshaped_prediction = tf.reshape(
          prediction, shape=[batch_size, out_height, out_width, -1, 2])
      reg_prediction = cn_assigner.get_batch_predictions_from_indices(
          reshaped_prediction, batch_indices)
      unweighted_loss = localization_loss_fn(
          reg_prediction,
          batch_regression_offsets,
          weights=tf.expand_dims(tf.ones_like(batch_weights), -1))
      # Apply the weights after the loss function to have full control over it.
      loss += batch_weights * tf.reduce_sum(unweighted_loss, axis=1)

    loss = tf.reduce_sum(loss) / (
        float(len(regression_predictions)) *
        tf.maximum(tf.reduce_sum(batch_weights), 1.0))
    return loss

  def _compute_kp_depth_loss(self, input_height, input_width, task_name,
                             depth_predictions, localization_loss_fn):
    """Computes the loss of the keypoint depth estimation.

    Args:
      input_height: An integer scalar tensor representing input image height.
      input_width: An integer scalar tensor representing input image width.
      task_name: A string representing the name of the keypoint task.
      depth_predictions: A list of float tensors of shape [batch_size,
        out_height, out_width, 1 (or num_keypoints)] representing the prediction
        heads of the model for keypoint depth.
      localization_loss_fn: An object_detection.core.losses.Loss object to
        compute the loss for the keypoint offset predictions in CenterNet.

    Returns:
      loss: A float scalar tensor representing the keypoint depth loss
        normalized by number of total keypoints.
    """
    kp_params = self._kp_params_dict[task_name]
    gt_keypoints_list = self.groundtruth_lists(fields.BoxListFields.keypoints)
    gt_classes_list = self.groundtruth_lists(fields.BoxListFields.classes)
    gt_weights_list = self.groundtruth_lists(fields.BoxListFields.weights)
    gt_keypoint_depths_list = self.groundtruth_lists(
        fields.BoxListFields.keypoint_depths)
    gt_keypoint_depth_weights_list = self.groundtruth_lists(
        fields.BoxListFields.keypoint_depth_weights)

    assigner = self._target_assigner_dict[task_name]
    (batch_indices, batch_depths,
     batch_weights) = assigner.assign_keypoints_depth_targets(
         height=input_height,
         width=input_width,
         gt_keypoints_list=gt_keypoints_list,
         gt_weights_list=gt_weights_list,
         gt_classes_list=gt_classes_list,
         gt_keypoint_depths_list=gt_keypoint_depths_list,
         gt_keypoint_depth_weights_list=gt_keypoint_depth_weights_list)

    # Keypoint offset loss.
    loss = 0.0
    for prediction in depth_predictions:
      if kp_params.per_keypoint_depth:
        prediction = tf.expand_dims(prediction, axis=-1)
      selected_depths = cn_assigner.get_batch_predictions_from_indices(
          prediction, batch_indices)
      # The dimensions passed are not as per the doc string but the loss
      # still computes the correct value.
      unweighted_loss = localization_loss_fn(
          selected_depths,
          batch_depths,
          weights=tf.expand_dims(tf.ones_like(batch_weights), -1))
      # Apply the weights after the loss function to have full control over it.
      loss += batch_weights * tf.squeeze(unweighted_loss, axis=1)

    loss = tf.reduce_sum(loss) / (
        float(len(depth_predictions)) *
        tf.maximum(tf.reduce_sum(batch_weights), 1.0))
    return loss

  def _compute_segmentation_losses(self, prediction_dict, per_pixel_weights):
    """Computes all the losses associated with segmentation.

    Args:
      prediction_dict: The dictionary returned from the predict() method.
      per_pixel_weights: A float tensor of shape [batch_size,
        out_height * out_width, 1] with 1s in locations where the spatial
        coordinates fall within the height and width in true_image_shapes.

    Returns:
      A dictionary with segmentation losses.
    """
    segmentation_heatmap = prediction_dict[SEGMENTATION_HEATMAP]
    mask_loss = self._compute_mask_loss(
        segmentation_heatmap, per_pixel_weights)
    losses = {
        SEGMENTATION_HEATMAP: mask_loss
    }
    return losses

  def _compute_mask_loss(self, segmentation_predictions,
                         per_pixel_weights):
    """Computes the mask loss.

    Args:
      segmentation_predictions: A list of float32 tensors of shape [batch_size,
        out_height, out_width, num_classes].
      per_pixel_weights: A float tensor of shape [batch_size,
        out_height * out_width, 1] with 1s in locations where the spatial
        coordinates fall within the height and width in true_image_shapes.

    Returns:
      A float scalar tensor representing the mask loss.
    """
    gt_boxes_list = self.groundtruth_lists(fields.BoxListFields.boxes)
    gt_masks_list = self.groundtruth_lists(fields.BoxListFields.masks)
    gt_mask_weights_list = None
    if self.groundtruth_has_field(fields.BoxListFields.mask_weights):
      gt_mask_weights_list = self.groundtruth_lists(
          fields.BoxListFields.mask_weights)
    gt_classes_list = self.groundtruth_lists(fields.BoxListFields.classes)

    # Convert the groundtruth to targets.
    assigner = self._target_assigner_dict[SEGMENTATION_TASK]
    heatmap_targets, heatmap_weight = assigner.assign_segmentation_targets(
        gt_masks_list=gt_masks_list,
        gt_classes_list=gt_classes_list,
        gt_boxes_list=gt_boxes_list,
        gt_mask_weights_list=gt_mask_weights_list)

    flattened_heatmap_targets = _flatten_spatial_dimensions(heatmap_targets)
    flattened_heatmap_mask = _flatten_spatial_dimensions(
        heatmap_weight[:, :, :, tf.newaxis])
    per_pixel_weights *= flattened_heatmap_mask

    loss = 0.0
    mask_loss_fn = self._mask_params.classification_loss

    total_pixels_in_loss = tf.math.maximum(
        tf.reduce_sum(per_pixel_weights), 1)

    # Loop through each feature output head.
    for pred in segmentation_predictions:
      pred = _flatten_spatial_dimensions(pred)
      loss += mask_loss_fn(
          pred, flattened_heatmap_targets, weights=per_pixel_weights)
    # TODO(ronnyvotel): Consider other ways to normalize loss.
    total_loss = tf.reduce_sum(loss) / (
        float(len(segmentation_predictions)) * total_pixels_in_loss)
    return total_loss

  def _compute_densepose_losses(self, input_height, input_width,
                                prediction_dict):
    """Computes the weighted DensePose losses.

    Args:
      input_height: An integer scalar tensor representing input image height.
      input_width: An integer scalar tensor representing input image width.
      prediction_dict: A dictionary holding predicted tensors output by the
        "predict" function. See the "predict" function for more detailed
        description.

    Returns:
      A dictionary of scalar float tensors representing the weighted losses for
      the DensePose task:
         DENSEPOSE_HEATMAP: the weighted part segmentation loss.
         DENSEPOSE_REGRESSION: the weighted part surface coordinate loss.
    """
    dp_heatmap_loss, dp_regression_loss = (
        self._compute_densepose_part_and_coordinate_losses(
            input_height=input_height,
            input_width=input_width,
            part_predictions=prediction_dict[DENSEPOSE_HEATMAP],
            surface_coord_predictions=prediction_dict[DENSEPOSE_REGRESSION]))
    loss_dict = {}
    loss_dict[DENSEPOSE_HEATMAP] = (
        self._densepose_params.part_loss_weight * dp_heatmap_loss)
    loss_dict[DENSEPOSE_REGRESSION] = (
        self._densepose_params.coordinate_loss_weight * dp_regression_loss)
    return loss_dict

  def _compute_densepose_part_and_coordinate_losses(
      self, input_height, input_width, part_predictions,
      surface_coord_predictions):
    """Computes the individual losses for the DensePose task.

    Args:
      input_height: An integer scalar tensor representing input image height.
      input_width: An integer scalar tensor representing input image width.
      part_predictions: A list of float tensors of shape [batch_size,
        out_height, out_width, num_parts].
      surface_coord_predictions: A list of float tensors of shape [batch_size,
        out_height, out_width, 2 * num_parts].

    Returns:
      A tuple with two scalar loss tensors: part_prediction_loss and
      surface_coord_loss.
    """
    gt_dp_num_points_list = self.groundtruth_lists(
        fields.BoxListFields.densepose_num_points)
    gt_dp_part_ids_list = self.groundtruth_lists(
        fields.BoxListFields.densepose_part_ids)
    gt_dp_surface_coords_list = self.groundtruth_lists(
        fields.BoxListFields.densepose_surface_coords)
    gt_weights_list = self.groundtruth_lists(fields.BoxListFields.weights)

    assigner = self._target_assigner_dict[DENSEPOSE_TASK]
    batch_indices, batch_part_ids, batch_surface_coords, batch_weights = (
        assigner.assign_part_and_coordinate_targets(
            height=input_height,
            width=input_width,
            gt_dp_num_points_list=gt_dp_num_points_list,
            gt_dp_part_ids_list=gt_dp_part_ids_list,
            gt_dp_surface_coords_list=gt_dp_surface_coords_list,
            gt_weights_list=gt_weights_list))

    part_prediction_loss = 0
    surface_coord_loss = 0
    classification_loss_fn = self._densepose_params.classification_loss
    localization_loss_fn = self._densepose_params.localization_loss
    num_predictions = float(len(part_predictions))
    num_valid_points = tf.math.count_nonzero(batch_weights)
    num_valid_points = tf.cast(tf.math.maximum(num_valid_points, 1), tf.float32)
    for part_pred, surface_coord_pred in zip(part_predictions,
                                             surface_coord_predictions):
      # Potentially upsample the feature maps, so that better quality (i.e.
      # higher res) groundtruth can be applied.
      if self._densepose_params.upsample_to_input_res:
        part_pred = tf.keras.layers.UpSampling2D(
            self._stride, interpolation=self._densepose_params.upsample_method)(
                part_pred)
        surface_coord_pred = tf.keras.layers.UpSampling2D(
            self._stride, interpolation=self._densepose_params.upsample_method)(
                surface_coord_pred)
      # Compute the part prediction loss.
      part_pred = cn_assigner.get_batch_predictions_from_indices(
          part_pred, batch_indices[:, 0:3])
      part_prediction_loss += classification_loss_fn(
          part_pred[:, tf.newaxis, :],
          batch_part_ids[:, tf.newaxis, :],
          weights=batch_weights[:, tf.newaxis, tf.newaxis])
      # Compute the surface coordinate loss.
      batch_size, out_height, out_width, _ = _get_shape(
          surface_coord_pred, 4)
      surface_coord_pred = tf.reshape(
          surface_coord_pred, [batch_size, out_height, out_width, -1, 2])
      surface_coord_pred = cn_assigner.get_batch_predictions_from_indices(
          surface_coord_pred, batch_indices)
      surface_coord_loss += localization_loss_fn(
          surface_coord_pred,
          batch_surface_coords,
          weights=batch_weights[:, tf.newaxis])
    part_prediction_loss = tf.reduce_sum(part_prediction_loss) / (
        num_predictions * num_valid_points)
    surface_coord_loss = tf.reduce_sum(surface_coord_loss) / (
        num_predictions * num_valid_points)
    return part_prediction_loss, surface_coord_loss

  def _compute_track_losses(self, input_height, input_width, prediction_dict):
    """Computes all the losses associated with tracking.

    Args:
      input_height: An integer scalar tensor representing input image height.
      input_width: An integer scalar tensor representing input image width.
      prediction_dict: The dictionary returned from the predict() method.

    Returns:
      A dictionary with tracking losses.
    """
    object_reid_predictions = prediction_dict[TRACK_REID]
    embedding_loss = self._compute_track_embedding_loss(
        input_height=input_height,
        input_width=input_width,
        object_reid_predictions=object_reid_predictions)
    losses = {
        TRACK_REID: embedding_loss
    }
    return losses

  def _compute_track_embedding_loss(self, input_height, input_width,
                                    object_reid_predictions):
    """Computes the object ReID loss.

    The embedding is trained as a classification task where the target is the
    ID of each track among all tracks in the whole dataset.

    Args:
      input_height: An integer scalar tensor representing input image height.
      input_width: An integer scalar tensor representing input image width.
      object_reid_predictions: A list of float tensors of shape [batch_size,
        out_height, out_width, reid_embed_size] representing the object
        embedding feature maps.

    Returns:
      A float scalar tensor representing the object ReID loss per instance.
    """
    gt_track_ids_list = self.groundtruth_lists(fields.BoxListFields.track_ids)
    gt_boxes_list = self.groundtruth_lists(fields.BoxListFields.boxes)
    gt_weights_list = self.groundtruth_lists(fields.BoxListFields.weights)
    num_boxes = _to_float32(get_num_instances_from_weights(gt_weights_list))

    # Convert the groundtruth to targets.
    assigner = self._target_assigner_dict[TRACK_TASK]
    batch_indices, batch_weights, track_targets = assigner.assign_track_targets(
        height=input_height,
        width=input_width,
        gt_track_ids_list=gt_track_ids_list,
        gt_boxes_list=gt_boxes_list,
        gt_weights_list=gt_weights_list)
    batch_weights = tf.expand_dims(batch_weights, -1)

    loss = 0.0
    object_reid_loss = self._track_params.classification_loss
    # Loop through each feature output head.
    for pred in object_reid_predictions:
      embedding_pred = cn_assigner.get_batch_predictions_from_indices(
          pred, batch_indices)

      reid_classification = self.track_reid_classification_net(embedding_pred)

      loss += object_reid_loss(
          reid_classification, track_targets, weights=batch_weights)

    loss_per_instance = tf.reduce_sum(loss) / (
        float(len(object_reid_predictions)) * num_boxes)

    return loss_per_instance

  def _compute_temporal_offset_loss(self, input_height,
                                    input_width, prediction_dict):
    """Computes the temporal offset loss for tracking.

    Args:
      input_height: An integer scalar tensor representing input image height.
      input_width: An integer scalar tensor representing input image width.
      prediction_dict: The dictionary returned from the predict() method.

    Returns:
      A dictionary with track/temporal_offset losses.
    """
    gt_boxes_list = self.groundtruth_lists(fields.BoxListFields.boxes)
    gt_offsets_list = self.groundtruth_lists(
        fields.BoxListFields.temporal_offsets)
    gt_match_list = self.groundtruth_lists(
        fields.BoxListFields.track_match_flags)
    gt_weights_list = self.groundtruth_lists(fields.BoxListFields.weights)
    num_boxes = tf.cast(
        get_num_instances_from_weights(gt_weights_list), tf.float32)

    offset_predictions = prediction_dict[TEMPORAL_OFFSET]
    num_predictions = float(len(offset_predictions))

    assigner = self._target_assigner_dict[TEMPORALOFFSET_TASK]
    (batch_indices, batch_offset_targets,
     batch_weights) = assigner.assign_temporal_offset_targets(
         height=input_height,
         width=input_width,
         gt_boxes_list=gt_boxes_list,
         gt_offsets_list=gt_offsets_list,
         gt_match_list=gt_match_list,
         gt_weights_list=gt_weights_list)
    batch_weights = tf.expand_dims(batch_weights, -1)

    offset_loss_fn = self._temporal_offset_params.localization_loss
    loss_dict = {}
    offset_loss = 0
    for offset_pred in offset_predictions:
      offset_pred = cn_assigner.get_batch_predictions_from_indices(
          offset_pred, batch_indices)
      offset_loss += offset_loss_fn(offset_pred[:, None],
                                    batch_offset_targets[:, None],
                                    weights=batch_weights)
    offset_loss = tf.reduce_sum(offset_loss) / (num_predictions * num_boxes)
    loss_dict[TEMPORAL_OFFSET] = offset_loss
    return loss_dict

  def _should_clip_keypoints(self):
    """Returns a boolean indicating whether keypoint clipping should occur.

    If there is only one keypoint task, clipping is controlled by the field
    `clip_out_of_frame_keypoints`. If there are multiple keypoint tasks,
    clipping logic is defined based on unanimous agreement of keypoint
    parameters. If there is any ambiguity, clip_out_of_frame_keypoints is set
    to False (default).
    """
    kp_params_iterator = iter(self._kp_params_dict.values())
    if len(self._kp_params_dict) == 1:
      kp_params = next(kp_params_iterator)
      return kp_params.clip_out_of_frame_keypoints

    # Multi-task setting.
    kp_params = next(kp_params_iterator)
    should_clip = kp_params.clip_out_of_frame_keypoints
    for kp_params in kp_params_iterator:
      if kp_params.clip_out_of_frame_keypoints != should_clip:
        return False
    return should_clip

  def _rescore_instances(self, classes, scores, keypoint_scores):
    """Rescores instances based on detection and keypoint scores.

    Args:
      classes: A [batch, max_detections] int32 tensor with detection classes.
      scores: A [batch, max_detections] float32 tensor with detection scores.
      keypoint_scores: A [batch, max_detections, total_num_keypoints] float32
        tensor with keypoint scores.

    Returns:
      A [batch, max_detections] float32 tensor with possibly altered detection
      scores.
    """
    batch, max_detections, total_num_keypoints = (
        shape_utils.combined_static_and_dynamic_shape(keypoint_scores))
    classes_tiled = tf.tile(classes[:, :, tf.newaxis],
                            multiples=[1, 1, total_num_keypoints])
    # TODO(yuhuic): Investigate whether this function will create subgraphs in
    # tflite that will cause the model to run slower at inference.
    for kp_params in self._kp_params_dict.values():
      if not kp_params.rescore_instances:
        continue
      class_id = kp_params.class_id
      keypoint_indices = kp_params.keypoint_indices
      kpt_mask = tf.reduce_sum(
          tf.one_hot(keypoint_indices, depth=total_num_keypoints), axis=0)
      kpt_mask_tiled = tf.tile(kpt_mask[tf.newaxis, tf.newaxis, :],
                               multiples=[batch, max_detections, 1])
      class_and_keypoint_mask = tf.math.logical_and(
          classes_tiled == class_id,
          kpt_mask_tiled == 1.0)
      class_and_keypoint_mask_float = tf.cast(class_and_keypoint_mask,
                                              dtype=tf.float32)
      visible_keypoints = tf.math.greater(
          keypoint_scores, kp_params.rescoring_threshold)
      keypoint_scores = tf.where(
          visible_keypoints, keypoint_scores, tf.zeros_like(keypoint_scores))
      num_visible_keypoints = tf.reduce_sum(
          class_and_keypoint_mask_float *
          tf.cast(visible_keypoints, tf.float32), axis=-1)
      num_visible_keypoints = tf.math.maximum(num_visible_keypoints, 1.0)
      scores_for_class = (1./num_visible_keypoints) * (
          tf.reduce_sum(class_and_keypoint_mask_float *
                        scores[:, :, tf.newaxis] *
                        keypoint_scores, axis=-1))
      scores = tf.where(classes == class_id,
                        scores_for_class,
                        scores)
    return scores

  def preprocess(self, inputs):
    outputs = shape_utils.resize_images_and_return_shapes(
        inputs, self._image_resizer_fn)
    resized_inputs, true_image_shapes = outputs

    return (self._feature_extractor.preprocess(resized_inputs),
            true_image_shapes)

  def predict(self, preprocessed_inputs, _):
    """Predicts CenterNet prediction tensors given an input batch.

    Feature extractors are free to produce predictions from multiple feature
    maps and therefore we return a dictionary mapping strings to lists.
    E.g. the hourglass backbone produces two feature maps.

    Args:
      preprocessed_inputs: a [batch, height, width, channels] float32 tensor
        representing a batch of images.

    Returns:
      prediction_dict: a dictionary holding predicted tensors with
        'preprocessed_inputs' - The input image after being resized and
          preprocessed by the feature extractor.
        'extracted_features' - The output of the feature extractor.
        'object_center' - A list of size num_feature_outputs containing
          float tensors of size [batch_size, output_height, output_width,
          num_classes] representing the predicted object center heatmap logits.
        'box/scale' - [optional] A list of size num_feature_outputs holding
          float tensors of size [batch_size, output_height, output_width, 2]
          representing the predicted box height and width at each output
          location. This field exists only when object detection task is
          specified.
        'box/offset' - [optional] A list of size num_feature_outputs holding
          float tensors of size [batch_size, output_height, output_width, 2]
          representing the predicted y and x offsets at each output location.
        '$TASK_NAME/keypoint_heatmap' - [optional]  A list of size
          num_feature_outputs holding float tensors of size [batch_size,
          output_height, output_width, num_keypoints] representing the predicted
          keypoint heatmap logits.
        '$TASK_NAME/keypoint_offset' - [optional] A list of size
          num_feature_outputs holding float tensors of size [batch_size,
          output_height, output_width, 2] representing the predicted keypoint
          offsets at each output location.
        '$TASK_NAME/keypoint_regression' - [optional] A list of size
          num_feature_outputs holding float tensors of size [batch_size,
          output_height, output_width, 2 * num_keypoints] representing the
          predicted keypoint regression at each output location.
        'segmentation/heatmap' - [optional] A list of size num_feature_outputs
          holding float tensors of size [batch_size, output_height,
          output_width, num_classes] representing the mask logits.
        'densepose/heatmap' - [optional] A list of size num_feature_outputs
          holding float tensors of size [batch_size, output_height,
          output_width, num_parts] representing the mask logits for each part.
        'densepose/regression' - [optional] A list of size num_feature_outputs
          holding float tensors of size [batch_size, output_height,
          output_width, 2 * num_parts] representing the DensePose surface
          coordinate predictions.
        Note the $TASK_NAME is provided by the KeypointEstimation namedtuple
        used to differentiate between different keypoint tasks.
    """
    features_list = self._feature_extractor(preprocessed_inputs)

    predictions = {}
    for head_name, heads in self._prediction_head_dict.items():
      predictions[head_name] = [
          head(feature) for (feature, head) in zip(features_list, heads)
      ]
    predictions['extracted_features'] = features_list
    predictions['preprocessed_inputs'] = preprocessed_inputs

    self._batched_prediction_tensor_names = predictions.keys()
    return predictions

  def loss(
      self, prediction_dict, true_image_shapes, scope=None,
      maximum_normalized_coordinate=1.1):
    """Computes scalar loss tensors with respect to provided groundtruth.

    This function implements the various CenterNet losses.

    Args:
      prediction_dict: a dictionary holding predicted tensors returned by
        "predict" function.
      true_image_shapes: int32 tensor of shape [batch, 3] where each row is of
        the form [height, width, channels] indicating the shapes of true images
        in the resized images, as resized images can be padded with zeros.
      scope: Optional scope name.
      maximum_normalized_coordinate: Maximum coordinate value to be considered
        as normalized, default to 1.1. This is used to check bounds during
        converting normalized coordinates to absolute coordinates.

    Returns:
      A dictionary mapping the keys [
        'Loss/object_center',
        'Loss/box/scale',  (optional)
        'Loss/box/offset', (optional)
        'Loss/$TASK_NAME/keypoint/heatmap', (optional)
        'Loss/$TASK_NAME/keypoint/offset', (optional)
        'Loss/$TASK_NAME/keypoint/regression', (optional)
        'Loss/segmentation/heatmap', (optional)
        'Loss/densepose/heatmap', (optional)
        'Loss/densepose/regression', (optional)
        'Loss/track/reid'] (optional)
        'Loss/track/offset'] (optional)
        scalar tensors corresponding to the losses for different tasks. Note the
        $TASK_NAME is provided by the KeypointEstimation namedtuple used to
        differentiate between different keypoint tasks.
    """

    _, input_height, input_width, _ = _get_shape(
        prediction_dict['preprocessed_inputs'], 4)

    output_height, output_width = (tf.maximum(input_height // self._stride, 1),
                                   tf.maximum(input_width // self._stride, 1))

    # TODO(vighneshb) Explore whether using floor here is safe.
    output_true_image_shapes = tf.ceil(
        tf.cast(true_image_shapes, tf.float32) / self._stride)
    valid_anchor_weights = get_valid_anchor_weights_in_flattened_image(
        output_true_image_shapes, output_height, output_width)
    valid_anchor_weights = tf.expand_dims(valid_anchor_weights, 2)

    object_center_loss = self._compute_object_center_loss(
        object_center_predictions=prediction_dict[OBJECT_CENTER],
        input_height=input_height,
        input_width=input_width,
        per_pixel_weights=valid_anchor_weights,
        maximum_normalized_coordinate=maximum_normalized_coordinate)
    losses = {
        OBJECT_CENTER:
            self._center_params.object_center_loss_weight * object_center_loss
    }
    if self._od_params is not None:
      od_losses = self._compute_object_detection_losses(
          input_height=input_height,
          input_width=input_width,
          prediction_dict=prediction_dict,
          per_pixel_weights=valid_anchor_weights,
          maximum_normalized_coordinate=maximum_normalized_coordinate)
      for key in od_losses:
        od_losses[key] = od_losses[key] * self._od_params.task_loss_weight
      losses.update(od_losses)

    if self._kp_params_dict is not None:
      for task_name, params in self._kp_params_dict.items():
        kp_losses = self._compute_keypoint_estimation_losses(
            task_name=task_name,
            input_height=input_height,
            input_width=input_width,
            prediction_dict=prediction_dict,
            per_pixel_weights=valid_anchor_weights)
        for key in kp_losses:
          kp_losses[key] = kp_losses[key] * params.task_loss_weight
        losses.update(kp_losses)

    if self._mask_params is not None:
      seg_losses = self._compute_segmentation_losses(
          prediction_dict=prediction_dict,
          per_pixel_weights=valid_anchor_weights)
      for key in seg_losses:
        seg_losses[key] = seg_losses[key] * self._mask_params.task_loss_weight
      losses.update(seg_losses)

    if self._densepose_params is not None:
      densepose_losses = self._compute_densepose_losses(
          input_height=input_height,
          input_width=input_width,
          prediction_dict=prediction_dict)
      for key in densepose_losses:
        densepose_losses[key] = (
            densepose_losses[key] * self._densepose_params.task_loss_weight)
      losses.update(densepose_losses)

    if self._track_params is not None:
      track_losses = self._compute_track_losses(
          input_height=input_height,
          input_width=input_width,
          prediction_dict=prediction_dict)
      for key in track_losses:
        track_losses[key] = (
            track_losses[key] * self._track_params.task_loss_weight)
      losses.update(track_losses)

    if self._temporal_offset_params is not None:
      offset_losses = self._compute_temporal_offset_loss(
          input_height=input_height,
          input_width=input_width,
          prediction_dict=prediction_dict)
      for key in offset_losses:
        offset_losses[key] = (
            offset_losses[key] * self._temporal_offset_params.task_loss_weight)
      losses.update(offset_losses)

    # Prepend the LOSS_KEY_PREFIX to the keys in the dictionary such that the
    # losses will be grouped together in Tensorboard.
    return dict([('%s/%s' % (LOSS_KEY_PREFIX, key), val)
                 for key, val in losses.items()])

  def postprocess(self, prediction_dict, true_image_shapes, **params):
    """Produces boxes given a prediction dict returned by predict().

    Although predict returns a list of tensors, only the last tensor in
    each list is used for making box predictions.

    Args:
      prediction_dict: a dictionary holding predicted tensors from "predict"
        function.
      true_image_shapes: int32 tensor of shape [batch, 3] where each row is of
        the form [height, width, channels] indicating the shapes of true images
        in the resized images, as resized images can be padded with zeros.
      **params: Currently ignored.

    Returns:
      detections: a dictionary containing the following fields
        detection_boxes - A tensor of shape [batch, max_detections, 4]
          holding the predicted boxes.
        detection_boxes_strided: A tensor of shape [batch_size, num_detections,
          4] holding the predicted boxes in absolute coordinates of the
          feature extractor's final layer output.
        detection_scores: A tensor of shape [batch, max_detections] holding
          the predicted score for each box.
        detection_multiclass_scores: A tensor of shape [batch, max_detection,
          num_classes] holding multiclass score for each box.
        detection_classes: An integer tensor of shape [batch, max_detections]
          containing the detected class for each box.
        num_detections: An integer tensor of shape [batch] containing the
          number of detected boxes for each sample in the batch.
        detection_keypoints: (Optional) A float tensor of shape [batch,
          max_detections, num_keypoints, 2] with normalized keypoints. Any
          invalid keypoints have their coordinates and scores set to 0.0.
        detection_keypoint_scores: (Optional) A float tensor of shape [batch,
          max_detection, num_keypoints] with scores for each keypoint.
        detection_masks: (Optional) A uint8 tensor of shape [batch,
          max_detections, mask_height, mask_width] with masks for each
          detection. Background is specified with 0, and foreground is specified
          with positive integers (1 for standard instance segmentation mask, and
          1-indexed parts for DensePose task).
        detection_surface_coords: (Optional) A float32 tensor of shape [batch,
          max_detection, mask_height, mask_width, 2] with DensePose surface
          coordinates, in (v, u) format.
        detection_embeddings: (Optional) A float tensor of shape [batch,
          max_detections, reid_embed_size] containing object embeddings.
    """
    object_center_prob = tf.nn.sigmoid(prediction_dict[OBJECT_CENTER][-1])

    if true_image_shapes is None:
      # If true_image_shapes is not provided, we assume the whole image is valid
      # and infer the true_image_shapes from the object_center_prob shape.
      batch_size, strided_height, strided_width, _ = _get_shape(
          object_center_prob, 4)
      true_image_shapes = tf.stack(
          [strided_height * self._stride, strided_width * self._stride,
           tf.constant(len(self._feature_extractor._channel_means))])   # pylint: disable=protected-access
      true_image_shapes = tf.stack([true_image_shapes] * batch_size, axis=0)
    else:
      # Mask object centers by true_image_shape. [batch, h, w, 1]
      object_center_mask = mask_from_true_image_shape(
          _get_shape(object_center_prob, 4), true_image_shapes)
      object_center_prob *= object_center_mask

    # Get x, y and channel indices corresponding to the top indices in the class
    # center predictions.
    detection_scores, y_indices, x_indices, channel_indices = (
        top_k_feature_map_locations(
            object_center_prob,
            max_pool_kernel_size=self._center_params.peak_max_pool_kernel_size,
            k=self._center_params.max_box_predictions))
    multiclass_scores = tf.gather_nd(
        object_center_prob, tf.stack([y_indices, x_indices], -1), batch_dims=1)
    num_detections = tf.reduce_sum(
        tf.cast(detection_scores > 0, tf.int32), axis=1)
    postprocess_dict = {
        fields.DetectionResultFields.detection_scores: detection_scores,
        fields.DetectionResultFields.detection_multiclass_scores:
            multiclass_scores,
        fields.DetectionResultFields.detection_classes: channel_indices,
        fields.DetectionResultFields.num_detections: num_detections,
    }

    boxes_strided = None
    if self._od_params:
      boxes_strided = (
          prediction_tensors_to_boxes(y_indices, x_indices,
                                      prediction_dict[BOX_SCALE][-1],
                                      prediction_dict[BOX_OFFSET][-1]))

      boxes = convert_strided_predictions_to_normalized_boxes(
          boxes_strided, self._stride, true_image_shapes)

      postprocess_dict.update({
          fields.DetectionResultFields.detection_boxes: boxes,
          'detection_boxes_strided': boxes_strided
      })

    if self._kp_params_dict:
      # If the model is trained to predict only one class of object and its
      # keypoint, we fall back to a simpler postprocessing function which uses
      # the ops that are supported by tf.lite on GPU.
      clip_keypoints = self._should_clip_keypoints()
      if len(self._kp_params_dict) == 1 and self._num_classes == 1:
        task_name, kp_params = next(iter(self._kp_params_dict.items()))
        keypoint_depths = None
        if kp_params.argmax_postprocessing:
          keypoints, keypoint_scores = (
              prediction_to_keypoints_argmax(
                  prediction_dict,
                  object_y_indices=y_indices,
                  object_x_indices=x_indices,
                  boxes=boxes_strided,
                  task_name=task_name,
                  kp_params=kp_params))
        else:
          (keypoints, keypoint_scores,
           keypoint_depths) = self._postprocess_keypoints_single_class(
               prediction_dict, channel_indices, y_indices, x_indices,
               boxes_strided, num_detections)
        keypoints, keypoint_scores = (
            convert_strided_predictions_to_normalized_keypoints(
                keypoints, keypoint_scores, self._stride, true_image_shapes,
                clip_out_of_frame_keypoints=clip_keypoints))
        if keypoint_depths is not None:
          postprocess_dict.update({
              fields.DetectionResultFields.detection_keypoint_depths:
                  keypoint_depths
          })
      else:
        # Multi-class keypoint estimation task does not support depth
        # estimation.
        assert all([
            not kp_dict.predict_depth
            for kp_dict in self._kp_params_dict.values()
        ])
        keypoints, keypoint_scores = self._postprocess_keypoints_multi_class(
            prediction_dict, channel_indices, y_indices, x_indices,
            boxes_strided, num_detections)
        keypoints, keypoint_scores = (
            convert_strided_predictions_to_normalized_keypoints(
                keypoints, keypoint_scores, self._stride, true_image_shapes,
                clip_out_of_frame_keypoints=clip_keypoints))

      # Update instance scores based on keypoints.
      scores = self._rescore_instances(
          channel_indices, detection_scores, keypoint_scores)
      postprocess_dict.update({
          fields.DetectionResultFields.detection_scores: scores,
          fields.DetectionResultFields.detection_keypoints: keypoints,
          fields.DetectionResultFields.detection_keypoint_scores:
              keypoint_scores
      })
      if self._od_params is None:
        # Still output the box prediction by enclosing the keypoints for
        # evaluation purpose.
        boxes = keypoint_ops.keypoints_to_enclosing_bounding_boxes(
            keypoints, keypoints_axis=2)
        postprocess_dict.update({
            fields.DetectionResultFields.detection_boxes: boxes,
        })

    if self._mask_params:
      masks = tf.nn.sigmoid(prediction_dict[SEGMENTATION_HEATMAP][-1])
      densepose_part_heatmap, densepose_surface_coords = None, None
      densepose_class_index = 0
      if self._densepose_params:
        densepose_part_heatmap = prediction_dict[DENSEPOSE_HEATMAP][-1]
        densepose_surface_coords = prediction_dict[DENSEPOSE_REGRESSION][-1]
        densepose_class_index = self._densepose_params.class_id
      instance_masks, surface_coords = (
          convert_strided_predictions_to_instance_masks(
              boxes, channel_indices, masks, true_image_shapes,
              densepose_part_heatmap, densepose_surface_coords,
              stride=self._stride, mask_height=self._mask_params.mask_height,
              mask_width=self._mask_params.mask_width,
              score_threshold=self._mask_params.score_threshold,
              densepose_class_index=densepose_class_index))
      postprocess_dict[
          fields.DetectionResultFields.detection_masks] = instance_masks
      if self._densepose_params:
        postprocess_dict[
            fields.DetectionResultFields.detection_surface_coords] = (
                surface_coords)

    if self._track_params:
      embeddings = self._postprocess_embeddings(prediction_dict,
                                                y_indices, x_indices)
      postprocess_dict.update({
          fields.DetectionResultFields.detection_embeddings: embeddings
      })

    if self._temporal_offset_params:
      offsets = prediction_tensors_to_temporal_offsets(
          y_indices, x_indices,
          prediction_dict[TEMPORAL_OFFSET][-1])
      postprocess_dict[fields.DetectionResultFields.detection_offsets] = offsets

    if self._non_max_suppression_fn:
      boxes = tf.expand_dims(
          postprocess_dict.pop(fields.DetectionResultFields.detection_boxes),
          axis=-2)
      multiclass_scores = postprocess_dict[
          fields.DetectionResultFields.detection_multiclass_scores]
      num_valid_boxes = postprocess_dict.pop(
          fields.DetectionResultFields.num_detections)
      # Remove scores and classes as NMS will compute these form multiclass
      # scores.
      postprocess_dict.pop(fields.DetectionResultFields.detection_scores)
      postprocess_dict.pop(fields.DetectionResultFields.detection_classes)
      (nmsed_boxes, nmsed_scores, nmsed_classes, _, nmsed_additional_fields,
       num_detections) = self._non_max_suppression_fn(
           boxes,
           multiclass_scores,
           additional_fields=postprocess_dict,
           num_valid_boxes=num_valid_boxes)
      postprocess_dict = nmsed_additional_fields
      postprocess_dict[
          fields.DetectionResultFields.detection_boxes] = nmsed_boxes
      postprocess_dict[
          fields.DetectionResultFields.detection_scores] = nmsed_scores
      postprocess_dict[
          fields.DetectionResultFields.detection_classes] = nmsed_classes
      postprocess_dict[
          fields.DetectionResultFields.num_detections] = num_detections
      postprocess_dict.update(nmsed_additional_fields)
    return postprocess_dict

  def postprocess_single_instance_keypoints(
      self,
      prediction_dict,
      true_image_shapes):
    """Postprocess for predicting single instance keypoints.

    This postprocess function is a special case of predicting the keypoint of
    a single instance in the image (original CenterNet postprocess supports
    multi-instance prediction). Due to the simplification assumption, this
    postprocessing function achieves much faster inference time.
    Here is a short list of the modifications made in this function:

      1) Assume the model predicts only single class keypoint.
      2) Assume there is only one instance in the image. If multiple instances
         appear in the image, the model tends to predict the one that is closer
         to the image center (the other ones are considered as background and
         are rejected by the model).
      3) Avoid using top_k ops in the postprocessing logics since it is slower
         than using argmax.
      4) The predictions other than the keypoints are ignored, e.g. boxes.
      5) The input batch size is assumed to be 1.

    Args:
      prediction_dict: a dictionary holding predicted tensors from "predict"
        function.
      true_image_shapes: int32 tensor of shape [batch, 3] where each row is of
        the form [height, width, channels] indicating the shapes of true images
        in the resized images, as resized images can be padded with zeros.

    Returns:
      detections: a dictionary containing the following fields
        detection_keypoints: A float tensor of shape
          [1, 1, num_keypoints, 2] with normalized keypoints. Any invalid
          keypoints have their coordinates and scores set to 0.0.
        detection_keypoint_scores: A float tensor of shape
          [1, 1, num_keypoints] with scores for each keypoint.
    """
    # The number of keypoint task is expected to be 1.
    assert len(self._kp_params_dict) == 1
    task_name, kp_params = next(iter(self._kp_params_dict.items()))
    keypoint_heatmap = tf.nn.sigmoid(prediction_dict[get_keypoint_name(
        task_name, KEYPOINT_HEATMAP)][-1])
    keypoint_offset = prediction_dict[get_keypoint_name(task_name,
                                                        KEYPOINT_OFFSET)][-1]
    keypoint_regression = prediction_dict[get_keypoint_name(
        task_name, KEYPOINT_REGRESSION)][-1]
    object_heatmap = tf.nn.sigmoid(prediction_dict[OBJECT_CENTER][-1])

    keypoint_depths = None
    if kp_params.predict_depth:
      keypoint_depths = prediction_dict[get_keypoint_name(
          task_name, KEYPOINT_DEPTH)][-1]
    keypoints, keypoint_scores, keypoint_depths = (
        prediction_to_single_instance_keypoints(
            object_heatmap=object_heatmap,
            keypoint_heatmap=keypoint_heatmap,
            keypoint_offset=keypoint_offset,
            keypoint_regression=keypoint_regression,
            kp_params=kp_params,
            keypoint_depths=keypoint_depths))

    keypoints, keypoint_scores = (
        convert_strided_predictions_to_normalized_keypoints(
            keypoints,
            keypoint_scores,
            self._stride,
            true_image_shapes,
            clip_out_of_frame_keypoints=False))
    postprocess_dict = {
        fields.DetectionResultFields.detection_keypoints: keypoints,
        fields.DetectionResultFields.detection_keypoint_scores: keypoint_scores
    }

    if kp_params.predict_depth:
      postprocess_dict.update({
          fields.DetectionResultFields.detection_keypoint_depths:
              keypoint_depths
      })
    return postprocess_dict

  def _postprocess_embeddings(self, prediction_dict, y_indices, x_indices):
    """Performs postprocessing on embedding predictions.

    Args:
      prediction_dict: a dictionary holding predicted tensors, returned from the
        predict() method. This dictionary should contain embedding prediction
        feature maps for tracking task.
      y_indices: A [batch_size, max_detections] int tensor with y indices for
        all object centers.
      x_indices: A [batch_size, max_detections] int tensor with x indices for
        all object centers.

    Returns:
      embeddings: A [batch_size, max_detection, reid_embed_size] float32
        tensor with L2 normalized embeddings extracted from detection box
        centers.
    """
    embedding_predictions = prediction_dict[TRACK_REID][-1]
    embeddings = predicted_embeddings_at_object_centers(
        embedding_predictions, y_indices, x_indices)
    embeddings, _ = tf.linalg.normalize(embeddings, axis=-1)

    return embeddings

  def _scatter_keypoints_to_batch(self, num_ind, kpt_coords_for_example,
                                  kpt_scores_for_example,
                                  instance_inds_for_example, max_detections,
                                  total_num_keypoints):
    """Helper function to convert scattered keypoints into batch."""
    def left_fn(kpt_coords_for_example, kpt_scores_for_example,
                instance_inds_for_example):
      # Scatter into tensor where instances align with original detection
      # instances. New shape of keypoint coordinates and scores are
      # [1, max_detections, num_total_keypoints, 2] and
      # [1, max_detections, num_total_keypoints], respectively.
      return _pad_to_full_instance_dim(
          kpt_coords_for_example, kpt_scores_for_example,
          instance_inds_for_example,
          self._center_params.max_box_predictions)

    def right_fn():
      kpt_coords_for_example_all_det = tf.zeros(
          [1, max_detections, total_num_keypoints, 2], dtype=tf.float32)
      kpt_scores_for_example_all_det = tf.zeros(
          [1, max_detections, total_num_keypoints], dtype=tf.float32)
      return (kpt_coords_for_example_all_det,
              kpt_scores_for_example_all_det)

    left_fn = functools.partial(left_fn, kpt_coords_for_example,
                                kpt_scores_for_example,
                                instance_inds_for_example)

    # Use dimension values instead of tf.size for tf.lite compatibility.
    return tf.cond(num_ind[0] > 0, left_fn, right_fn)

  def _postprocess_keypoints_multi_class(self, prediction_dict, classes,
                                         y_indices, x_indices, boxes,
                                         num_detections):
    """Performs postprocessing on keypoint predictions.

    This is the most general keypoint postprocessing function which supports
    multiple keypoint tasks (e.g. human and dog keypoints) and multiple object
    detection classes. Note that it is the most expensive postprocessing logics
    and is currently not tf.lite/tf.js compatible. See
    _postprocess_keypoints_single_class if you plan to export the model in more
    portable format.

    Args:
      prediction_dict: a dictionary holding predicted tensors, returned from the
        predict() method. This dictionary should contain keypoint prediction
        feature maps for each keypoint task.
      classes: A [batch_size, max_detections] int tensor with class indices for
        all detected objects.
      y_indices: A [batch_size, max_detections] int tensor with y indices for
        all object centers.
      x_indices: A [batch_size, max_detections] int tensor with x indices for
        all object centers.
      boxes: A [batch_size, max_detections, 4] float32 tensor with bounding
        boxes in (un-normalized) output space.
      num_detections: A [batch_size] int tensor with the number of valid
        detections for each image.

    Returns:
      A tuple of
      keypoints: a [batch_size, max_detection, num_total_keypoints, 2] float32
        tensor with keypoints in the output (strided) coordinate frame.
      keypoint_scores: a [batch_size, max_detections, num_total_keypoints]
        float32 tensor with keypoint scores.
    """
    total_num_keypoints = sum(len(kp_dict.keypoint_indices) for kp_dict
                              in self._kp_params_dict.values())
    batch_size, max_detections = _get_shape(classes, 2)
    kpt_coords_for_example_list = []
    kpt_scores_for_example_list = []
    for ex_ind in range(batch_size):
      # The tensors that host the keypoint coordinates and scores for all
      # instances and all keypoints. They will be updated by scatter_nd_add for
      # each keypoint tasks.
      kpt_coords_for_example_all_det = tf.zeros(
          [max_detections, total_num_keypoints, 2])
      kpt_scores_for_example_all_det = tf.zeros(
          [max_detections, total_num_keypoints])
      for task_name, kp_params in self._kp_params_dict.items():
        keypoint_heatmap = prediction_dict[
            get_keypoint_name(task_name, KEYPOINT_HEATMAP)][-1]
        keypoint_offsets = prediction_dict[
            get_keypoint_name(task_name, KEYPOINT_OFFSET)][-1]
        keypoint_regression = prediction_dict[
            get_keypoint_name(task_name, KEYPOINT_REGRESSION)][-1]
        instance_inds = self._get_instance_indices(
            classes, num_detections, ex_ind, kp_params.class_id)

        # Gather the feature map locations corresponding to the object class.
        y_indices_for_kpt_class = tf.gather(y_indices, instance_inds, axis=1)
        x_indices_for_kpt_class = tf.gather(x_indices, instance_inds, axis=1)
        if boxes is None:
          boxes_for_kpt_class = None
        else:
          boxes_for_kpt_class = tf.gather(boxes, instance_inds, axis=1)

        # Postprocess keypoints and scores for class and single image. Shapes
        # are [1, num_instances_i, num_keypoints_i, 2] and
        # [1, num_instances_i, num_keypoints_i], respectively. Note that
        # num_instances_i and num_keypoints_i refers to the number of
        # instances and keypoints for class i, respectively.
        (kpt_coords_for_class, kpt_scores_for_class, _) = (
            self._postprocess_keypoints_for_class_and_image(
                keypoint_heatmap,
                keypoint_offsets,
                keypoint_regression,
                classes,
                y_indices_for_kpt_class,
                x_indices_for_kpt_class,
                boxes_for_kpt_class,
                ex_ind,
                kp_params,
            ))

        # Prepare the indices for scatter_nd. The resulting combined_inds has
        # the shape of [num_instances_i * num_keypoints_i, 2], where the first
        # column corresponds to the instance IDs and the second column
        # corresponds to the keypoint IDs.
        kpt_inds = tf.constant(kp_params.keypoint_indices, dtype=tf.int32)
        kpt_inds = tf.expand_dims(kpt_inds, axis=0)
        instance_inds_expand = tf.expand_dims(instance_inds, axis=-1)
        kpt_inds_expand = kpt_inds * tf.ones_like(instance_inds_expand)
        instance_inds_expand = instance_inds_expand * tf.ones_like(kpt_inds)
        combined_inds = tf.stack(
            [instance_inds_expand, kpt_inds_expand], axis=2)
        combined_inds = tf.reshape(combined_inds, [-1, 2])

        # Reshape the keypoint coordinates/scores to [num_instances_i *
        # num_keypoints_i, 2]/[num_instances_i * num_keypoints_i] to be used
        # by scatter_nd_add.
        kpt_coords_for_class = tf.reshape(kpt_coords_for_class, [-1, 2])
        kpt_scores_for_class = tf.reshape(kpt_scores_for_class, [-1])
        kpt_coords_for_example_all_det = tf.tensor_scatter_nd_add(
            kpt_coords_for_example_all_det,
            combined_inds, kpt_coords_for_class)
        kpt_scores_for_example_all_det = tf.tensor_scatter_nd_add(
            kpt_scores_for_example_all_det,
            combined_inds, kpt_scores_for_class)

      kpt_coords_for_example_list.append(
          tf.expand_dims(kpt_coords_for_example_all_det, axis=0))
      kpt_scores_for_example_list.append(
          tf.expand_dims(kpt_scores_for_example_all_det, axis=0))

    # Concatenate all keypoints and scores from all examples in the batch.
    # Shapes are [batch_size, max_detections, num_total_keypoints, 2] and
    # [batch_size, max_detections, num_total_keypoints], respectively.
    keypoints = tf.concat(kpt_coords_for_example_list, axis=0)
    keypoint_scores = tf.concat(kpt_scores_for_example_list, axis=0)

    return keypoints, keypoint_scores

  def _postprocess_keypoints_single_class(self, prediction_dict, classes,
                                          y_indices, x_indices, boxes,
                                          num_detections):
    """Performs postprocessing on keypoint predictions (single class only).

    This function handles the special case of keypoint task that the model
    predicts only one class of the bounding box/keypoint (e.g. person). By the
    assumption, the function uses only tf.lite supported ops and should run
    faster.

    Args:
      prediction_dict: a dictionary holding predicted tensors, returned from the
        predict() method. This dictionary should contain keypoint prediction
        feature maps for each keypoint task.
      classes: A [batch_size, max_detections] int tensor with class indices for
        all detected objects.
      y_indices: A [batch_size, max_detections] int tensor with y indices for
        all object centers.
      x_indices: A [batch_size, max_detections] int tensor with x indices for
        all object centers.
      boxes: A [batch_size, max_detections, 4] float32 tensor with bounding
        boxes in (un-normalized) output space.
      num_detections: A [batch_size] int tensor with the number of valid
        detections for each image.

    Returns:
      A tuple of
      keypoints: a [batch_size, max_detection, num_total_keypoints, 2] float32
        tensor with keypoints in the output (strided) coordinate frame.
      keypoint_scores: a [batch_size, max_detections, num_total_keypoints]
        float32 tensor with keypoint scores.
    """
    # This function only works when there is only one keypoint task and the
    # number of classes equal to one. For more general use cases, please use
    # _postprocess_keypoints instead.
    assert len(self._kp_params_dict) == 1 and self._num_classes == 1
    task_name, kp_params = next(iter(self._kp_params_dict.items()))
    keypoint_heatmap = prediction_dict[
        get_keypoint_name(task_name, KEYPOINT_HEATMAP)][-1]
    keypoint_offsets = prediction_dict[
        get_keypoint_name(task_name, KEYPOINT_OFFSET)][-1]
    keypoint_regression = prediction_dict[
        get_keypoint_name(task_name, KEYPOINT_REGRESSION)][-1]
    keypoint_depth_predictions = None
    if kp_params.predict_depth:
      keypoint_depth_predictions = prediction_dict[get_keypoint_name(
          task_name, KEYPOINT_DEPTH)][-1]

    batch_size, _ = _get_shape(classes, 2)
    kpt_coords_for_example_list = []
    kpt_scores_for_example_list = []
    kpt_depths_for_example_list = []
    for ex_ind in range(batch_size):
      # Postprocess keypoints and scores for class and single image. Shapes
      # are [1, max_detections, num_keypoints, 2] and
      # [1, max_detections, num_keypoints], respectively.
      (kpt_coords_for_class, kpt_scores_for_class, kpt_depths_for_class) = (
          self._postprocess_keypoints_for_class_and_image(
              keypoint_heatmap,
              keypoint_offsets,
              keypoint_regression,
              classes,
              y_indices,
              x_indices,
              boxes,
              ex_ind,
              kp_params,
              keypoint_depth_predictions=keypoint_depth_predictions))

      kpt_coords_for_example_list.append(kpt_coords_for_class)
      kpt_scores_for_example_list.append(kpt_scores_for_class)
      kpt_depths_for_example_list.append(kpt_depths_for_class)

    # Concatenate all keypoints and scores from all examples in the batch.
    # Shapes are [batch_size, max_detections, num_keypoints, 2] and
    # [batch_size, max_detections, num_keypoints], respectively.
    keypoints = tf.concat(kpt_coords_for_example_list, axis=0)
    keypoint_scores = tf.concat(kpt_scores_for_example_list, axis=0)

    keypoint_depths = None
    if kp_params.predict_depth:
      keypoint_depths = tf.concat(kpt_depths_for_example_list, axis=0)

    return keypoints, keypoint_scores, keypoint_depths

  def _get_instance_indices(self, classes, num_detections, batch_index,
                            class_id):
    """Gets the instance indices that match the target class ID.

    Args:
      classes: A [batch_size, max_detections] int tensor with class indices for
        all detected objects.
      num_detections: A [batch_size] int tensor with the number of valid
        detections for each image.
      batch_index: An integer specifying the index for an example in the batch.
      class_id: Class id

    Returns:
      instance_inds: A [num_instances] int32 tensor where each element indicates
        the instance location within the `classes` tensor. This is useful to
        associate the refined keypoints with the original detections (i.e.
        boxes)
    """
    classes = classes[batch_index:batch_index+1, ...]
    _, max_detections = shape_utils.combined_static_and_dynamic_shape(
        classes)
    # Get the detection indices corresponding to the target class.
    # Call tf.math.equal with matched tensor shape to make it tf.lite
    # compatible.
    valid_detections_with_kpt_class = tf.math.logical_and(
        tf.range(max_detections) < num_detections[batch_index],
        tf.math.equal(classes[0], tf.fill(classes[0].shape, class_id)))
    instance_inds = tf.where(valid_detections_with_kpt_class)[:, 0]
    # Cast the indices tensor to int32 for tf.lite compatibility.
    return tf.cast(instance_inds, tf.int32)

  def _postprocess_keypoints_for_class_and_image(
      self,
      keypoint_heatmap,
      keypoint_offsets,
      keypoint_regression,
      classes,
      y_indices,
      x_indices,
      boxes,
      batch_index,
      kp_params,
      keypoint_depth_predictions=None):
    """Postprocess keypoints for a single image and class.

    Args:
      keypoint_heatmap: A [batch_size, height, width, num_keypoints] float32
        tensor with keypoint heatmaps.
      keypoint_offsets: A [batch_size, height, width, 2] float32 tensor with
        local offsets to keypoint centers.
      keypoint_regression: A [batch_size, height, width, 2 * num_keypoints]
        float32 tensor with regressed offsets to all keypoints.
      classes: A [batch_size, max_detections] int tensor with class indices for
        all detected objects.
      y_indices: A [batch_size, max_detections] int tensor with y indices for
        all object centers.
      x_indices: A [batch_size, max_detections] int tensor with x indices for
        all object centers.
      boxes: A [batch_size, max_detections, 4] float32 tensor with detected
        boxes in the output (strided) frame.
      batch_index: An integer specifying the index for an example in the batch.
      kp_params: A `KeypointEstimationParams` object with parameters for a
        single keypoint class.
      keypoint_depth_predictions: (optional) A [batch_size, height, width, 1]
        float32 tensor representing the keypoint depth prediction.

    Returns:
      A tuple of
      refined_keypoints: A [1, num_instances, num_keypoints, 2] float32 tensor
        with refined keypoints for a single class in a single image, expressed
        in the output (strided) coordinate frame. Note that `num_instances` is a
        dynamic dimension, and corresponds to the number of valid detections
        for the specific class.
      refined_scores: A [1, num_instances, num_keypoints] float32 tensor with
        keypoint scores.
      refined_depths: A [1, num_instances, num_keypoints] float32 tensor with
        keypoint depths. Return None if the input keypoint_depth_predictions is
        None.
    """
    num_keypoints = len(kp_params.keypoint_indices)

    keypoint_heatmap = tf.nn.sigmoid(
        keypoint_heatmap[batch_index:batch_index+1, ...])
    keypoint_offsets = keypoint_offsets[batch_index:batch_index+1, ...]
    keypoint_regression = keypoint_regression[batch_index:batch_index+1, ...]
    keypoint_depths = None
    if keypoint_depth_predictions is not None:
      keypoint_depths = keypoint_depth_predictions[batch_index:batch_index + 1,
                                                   ...]
    y_indices = y_indices[batch_index:batch_index+1, ...]
    x_indices = x_indices[batch_index:batch_index+1, ...]
    if boxes is None:
      boxes_slice = None
    else:
      boxes_slice = boxes[batch_index:batch_index+1, ...]

    # Gather the regressed keypoints. Final tensor has shape
    # [1, num_instances, num_keypoints, 2].
    regressed_keypoints_for_objects = regressed_keypoints_at_object_centers(
        keypoint_regression, y_indices, x_indices)
    regressed_keypoints_for_objects = tf.reshape(
        regressed_keypoints_for_objects, [1, -1, num_keypoints, 2])

    # Get the candidate keypoints and scores.
    # The shape of keypoint_candidates and keypoint_scores is:
    # [1, num_candidates_per_keypoint, num_keypoints, 2] and
    #  [1, num_candidates_per_keypoint, num_keypoints], respectively.
    (keypoint_candidates, keypoint_scores, num_keypoint_candidates,
     keypoint_depth_candidates) = (
         prediction_tensors_to_keypoint_candidates(
             keypoint_heatmap,
             keypoint_offsets,
             keypoint_score_threshold=(
                 kp_params.keypoint_candidate_score_threshold),
             max_pool_kernel_size=kp_params.peak_max_pool_kernel_size,
             max_candidates=kp_params.num_candidates_per_keypoint,
             keypoint_depths=keypoint_depths))

    kpts_std_dev_postprocess = [
        s * kp_params.std_dev_multiplier for s in kp_params.keypoint_std_dev
    ]
    # Get the refined keypoints and scores, of shape
    # [1, num_instances, num_keypoints, 2] and
    # [1, num_instances, num_keypoints], respectively.
    (refined_keypoints, refined_scores, refined_depths) = refine_keypoints(
        regressed_keypoints_for_objects,
        keypoint_candidates,
        keypoint_scores,
        num_keypoint_candidates,
        bboxes=boxes_slice,
        unmatched_keypoint_score=kp_params.unmatched_keypoint_score,
        box_scale=kp_params.box_scale,
        candidate_search_scale=kp_params.candidate_search_scale,
        candidate_ranking_mode=kp_params.candidate_ranking_mode,
        score_distance_offset=kp_params.score_distance_offset,
        keypoint_depth_candidates=keypoint_depth_candidates,
        keypoint_score_threshold=(kp_params.keypoint_candidate_score_threshold),
        score_distance_multiplier=kp_params.score_distance_multiplier,
        keypoint_std_dev=kpts_std_dev_postprocess)

    return refined_keypoints, refined_scores, refined_depths

  def regularization_losses(self):
    return []

  def restore_map(self,
                  fine_tune_checkpoint_type='detection',
                  load_all_detection_checkpoint_vars=False):
    raise RuntimeError('CenterNetMetaArch not supported under TF1.x.')

  def restore_from_objects(self, fine_tune_checkpoint_type='detection'):
    """Returns a map of Trackable objects to load from a foreign checkpoint.

    Returns a dictionary of Tensorflow 2 Trackable objects (e.g. tf.Module
    or Checkpoint). This enables the model to initialize based on weights from
    another task. For example, the feature extractor variables from a
    classification model can be used to bootstrap training of an object
    detector. When loading from an object detection model, the checkpoint model
    should have the same parameters as this detection model with exception of
    the num_classes parameter.

    Note that this function is intended to be used to restore Keras-based
    models when running Tensorflow 2, whereas restore_map (not implemented
    in CenterNet) is intended to be used to restore Slim-based models when
    running Tensorflow 1.x.

    TODO(jonathanhuang): Make this function consistent with other
    meta-architectures.

    Args:
      fine_tune_checkpoint_type: whether to restore from a full detection
        checkpoint (with compatible variable names) or to restore from a
        classification checkpoint for initialization prior to training.
        Valid values: `detection`, `classification`, `fine_tune`.
        Default 'detection'.
        'detection': used when loading models pre-trained on other detection
          tasks. With this checkpoint type the weights of the feature extractor
          are expected under the attribute 'feature_extractor'.
        'classification': used when loading models pre-trained on an image
          classification task. Note that only the encoder section of the network
          is loaded and not the upsampling layers. With this checkpoint type,
          the weights of only the encoder section are expected under the
          attribute 'feature_extractor'.
        'fine_tune': used when loading the entire CenterNet feature extractor
          pre-trained on other tasks. The checkpoints saved during CenterNet
          model training can be directly loaded using this type. With this
          checkpoint type, the weights of the feature extractor are expected
          under the attribute 'model._feature_extractor'.
        For more details, see the tensorflow section on Loading mechanics.
        https://www.tensorflow.org/guide/checkpoint#loading_mechanics

    Returns:
      A dict mapping keys to Trackable objects (tf.Module or Checkpoint).
    """

    if fine_tune_checkpoint_type == 'detection':
      feature_extractor_model = tf.train.Checkpoint(
          _feature_extractor=self._feature_extractor)
      return {'model': feature_extractor_model}

    elif fine_tune_checkpoint_type == 'classification':
      return {
          'feature_extractor':
              self._feature_extractor.classification_backbone
      }
    elif fine_tune_checkpoint_type == 'full':
      return {'model': self}
    elif fine_tune_checkpoint_type == 'fine_tune':
      raise ValueError(('"fine_tune" is no longer supported for CenterNet. '
                        'Please set fine_tune_checkpoint_type to "detection"'
                        ' which has the same functionality. If you are using'
                        ' the ExtremeNet checkpoint, download the new version'
                        ' from the model zoo.'))

    else:
      raise ValueError('Unknown fine tune checkpoint type {}'.format(
          fine_tune_checkpoint_type))

  def updates(self):
    if tf_version.is_tf2():
      raise RuntimeError('This model is intended to be used with model_lib_v2 '
                         'which does not support updates()')
    else:
      update_ops = []
      slim_update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS)
      # Copy the slim ops to avoid modifying the collection
      if slim_update_ops:
        update_ops.extend(slim_update_ops)
      return update_ops
