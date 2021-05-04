# Copyright 2021 The TensorFlow Authors. All Rights Reserved.
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

"""class YT8MFrameFeatureReader(BaseReader).

  Reads TFRecords of SequenceExamples.

  The TFRecords must contain SequenceExamples with the sparse in64 'labels'
  context feature and a fixed length byte-quantized feature vector, obtained
  from the features in 'feature_names'. The quantized features will be mapped
  back into a range between min_quantized_value and max_quantized_value.
  link for details: https://research.google.com/youtube8m/download.html
"""

from typing import Dict

import tensorflow as tf
from official.vision.beta.configs import video_classification as exp_cfg
from official.vision.beta.dataloaders import decoder
from official.vision.beta.dataloaders import parser
from official.vision.beta.projects.yt8m.dataloaders import utils


def resize_axis(tensor, axis, new_size, fill_value=0):
  """Truncates or pads a tensor to new_size on on a given axis.

  Truncate or extend tensor such that tensor.shape[axis] == new_size. If the
  size increases, the padding will be performed at the end, using fill_value.

  Args:
    tensor: The tensor to be resized.
    axis: An integer representing the dimension to be sliced.
    new_size: An integer or 0d tensor representing the new value for
      tensor.shape[axis].
    fill_value: Value to use to fill any new entries in the tensor. Will be cast
      to the type of tensor.

  Returns:
    The resized tensor.
  """
  tensor = tf.convert_to_tensor(tensor)
  shape = tf.unstack(tf.shape(tensor))

  pad_shape = shape[:]
  pad_shape[axis] = tf.maximum(0, new_size - shape[axis])

  shape[axis] = tf.minimum(shape[axis], new_size)
  shape = tf.stack(shape)

  resized = tf.concat([
      tf.slice(tensor, tf.zeros_like(shape), shape),
      tf.fill(tf.stack(pad_shape), tf.cast(fill_value, tensor.dtype))
  ], axis)

  # Update shape.
  new_shape = tensor.shape.as_list()  # A copy is being made.
  new_shape[axis] = new_size
  resized = tf.ensure_shape(resized, new_shape)
  return resized


def _process_segment_and_label(video_matrix, num_frames, contexts,
                               segment_labels, segment_size,
                               num_classes) -> Dict[str, tf.Tensor]:
  """Processes a batched Tensor of frames.

  The same parameters used in process should be used here.
  Args:
    video_matrix: different features concatenated into one matrix
    num_frames: Number of frames per subclip.
    contexts: context information extracted from decoder
    segment_labels: if we read segment labels instead.
    segment_size: the segment_size used for reading segments.
    num_classes: a positive integer for the number of classes.

  Returns:
    output: dictionary containing batch information
  """
  # Partition frame-level feature matrix to segment-level feature matrix.
  if segment_labels:
    start_times = contexts["segment_start_times"].values
    # Here we assume all the segments that started at the same start time has
    # the same segment_size.
    uniq_start_times, seg_idxs = tf.unique(start_times, out_idx=tf.dtypes.int64)
    # Range gather matrix, e.g., [[0,1,2],[1,2,3]] for segment_size == 3.
    range_mtx = tf.expand_dims(
        uniq_start_times, axis=-1) + tf.expand_dims(
            tf.range(0, segment_size, dtype=tf.int64), axis=0)
    # Shape: [num_segment, segment_size, feature_dim].
    batch_video_matrix = tf.gather_nd(video_matrix,
                                      tf.expand_dims(range_mtx, axis=-1))
    num_segment = tf.shape(batch_video_matrix)[0]
    batch_video_ids = tf.reshape(
        tf.tile([contexts["id"]], [num_segment]), (num_segment,))
    batch_frames = tf.reshape(
        tf.tile([segment_size], [num_segment]), (num_segment,))
    batch_frames = tf.cast(tf.expand_dims(batch_frames, 1), tf.float32)

    # For segment labels, all labels are not exhaustively rated. So we only
    # evaluate the rated labels.

    # Label indices for each segment, shape: [num_segment, 2].
    label_indices = tf.stack([seg_idxs, contexts["segment_labels"].values],
                             axis=-1)
    label_values = contexts["segment_scores"].values
    sparse_labels = tf.sparse.SparseTensor(label_indices, label_values,
                                           (num_segment, num_classes))
    batch_labels = tf.sparse.to_dense(sparse_labels, validate_indices=False)

    sparse_label_weights = tf.sparse.SparseTensor(
        label_indices, tf.ones_like(label_values, dtype=tf.float32),
        (num_segment, num_classes))
    batch_label_weights = tf.sparse.to_dense(
        sparse_label_weights, validate_indices=False)
    # output_dict = utils.get_segments(batch_video_matrix, batch_frames, 5)
  else:
    # Process video-level labels.
    label_indices = contexts["labels"].values
    sparse_labels = tf.sparse.SparseTensor(
        tf.expand_dims(label_indices, axis=-1),
        tf.ones_like(contexts["labels"].values, dtype=tf.bool), (num_classes,))
    labels = tf.sparse.to_dense(
        sparse_labels, default_value=False, validate_indices=False)

    # convert to batch format.
    batch_video_ids = tf.expand_dims(contexts["id"], 0)
    batch_video_matrix = tf.expand_dims(video_matrix, 0)
    batch_labels = tf.expand_dims(labels, 0)
    batch_frames = tf.expand_dims(num_frames, 0)
    batch_label_weights = None

  output_dict = {
      "video_ids": batch_video_ids,
      "video_matrix": batch_video_matrix,
      "labels": batch_labels,
      "num_frames": batch_frames,
  }
  if batch_label_weights is not None:
    output_dict["label_weights"] = batch_label_weights

  return output_dict


def _get_video_matrix(features, feature_size, max_frames, max_quantized_value,
                      min_quantized_value):
  """Decodes features from an input string and quantizes it.

  Args:
    features: raw feature values
    feature_size: length of each frame feature vector
    max_frames: number of frames (rows) in the output feature_matrix
    max_quantized_value: the maximum of the quantized value.
    min_quantized_value: the minimum of the quantized value.

  Returns:
    feature_matrix: matrix of all frame-features
    num_frames: number of frames in the sequence
  """
  decoded_features = tf.reshape(
      tf.cast(tf.io.decode_raw(features, tf.uint8), tf.float32),
      [-1, feature_size])

  num_frames = tf.math.minimum(tf.shape(decoded_features)[0], max_frames)
  feature_matrix = utils.Dequantize(decoded_features, max_quantized_value,
                                    min_quantized_value)
  feature_matrix = resize_axis(feature_matrix, 0, max_frames)
  return feature_matrix, num_frames


def _concat_features(features, feature_names, feature_sizes, max_frames,
                     max_quantized_value, min_quantized_value):
  """Loads (potentially) different types of features and concatenates them.

  Args:
      features: raw feature values
      feature_names: list of feature names
      feature_sizes: list of features sizes
      max_frames: number of frames in the sequence
      max_quantized_value: the maximum of the quantized value.
      min_quantized_value: the minimum of the quantized value.

  Returns:
      video_matrix: different features concatenated into one matrix
      num_frames: the number of frames in the video
  """

  num_features = len(feature_names)
  assert num_features > 0, "No feature selected: feature_names is empty!"

  assert len(feature_names) == len(feature_sizes), (
      "length of feature_names (={}) != length of feature_sizes (={})".format(
          len(feature_names), len(feature_sizes)))

  num_frames = -1  # the number of frames in the video
  feature_matrices = [None] * num_features  # an array of different features
  for feature_index in range(num_features):
    feature_matrix, num_frames_in_this_feature = _get_video_matrix(
        features[feature_names[feature_index]], feature_sizes[feature_index],
        max_frames, max_quantized_value, min_quantized_value)
    if num_frames == -1:
      num_frames = num_frames_in_this_feature

    feature_matrices[feature_index] = feature_matrix

  # cap the number of frames at self.max_frames
  num_frames = tf.minimum(num_frames, max_frames)

  # concatenate different features
  video_matrix = tf.concat(feature_matrices, 1)

  return video_matrix, num_frames


class Decoder(decoder.Decoder):
  """A tf.Example decoder for classification task."""

  def __init__(
      self,
      input_params: exp_cfg.DataConfig,
  ):

    self._segment_labels = input_params.segment_labels
    self._feature_names = input_params.feature_names
    self._context_features = {
        "id": tf.io.FixedLenFeature([], tf.string),
    }
    if self._segment_labels:
      self._context_features.update({
          # There is no need to read end-time given we always assume the segment
          # has the same size.
          "segment_labels": tf.io.VarLenFeature(tf.int64),
          "segment_start_times": tf.io.VarLenFeature(tf.int64),
          "segment_scores": tf.io.VarLenFeature(tf.float32)
      })
    else:
      self._context_features.update({"labels": tf.io.VarLenFeature(tf.int64)})

    self._sequence_features = {
        feature_name: tf.io.FixedLenSequenceFeature([], dtype=tf.string)
        for feature_name in self._feature_names
    }

  def decode(self, serialized_example):
    """Parses a single tf.Example into image and label tensors."""

    contexts, features = tf.io.parse_single_sequence_example(
        serialized_example,
        context_features=self._context_features,
        sequence_features=self._sequence_features)

    return {"contexts": contexts, "features": features}


class Parser(parser.Parser):
  """Parses a video and label dataset.

    takes the decoded raw tensors dict
    and parse them into a dictionary of tensors
    that can be consumed by the model.
    It will be executed after decoder.
  """

  def __init__(
      self,
      input_params: exp_cfg.DataConfig,
      max_quantized_value=2,
      min_quantized_value=-2,
  ):
    self._num_classes = input_params.num_classes
    self._segment_size = input_params.segment_size
    self._segment_labels = input_params.segment_labels
    self._feature_names = input_params.feature_names
    self._feature_sizes = input_params.feature_sizes
    self.stride = input_params.temporal_stride
    self._max_frames = input_params.max_frames
    self._num_frames = input_params.num_frames
    self._seed = input_params.random_seed
    self._max_quantized_value = max_quantized_value
    self._min_quantized_value = min_quantized_value

  def _parse_train_data(self, decoded_tensors):
    """Parses data for training."""
    # loads (potentially) different types of features and concatenates them
    self.video_matrix, self.num_frames = _concat_features(
        decoded_tensors["features"], self._feature_names, self._feature_sizes,
        self._max_frames, self._max_quantized_value, self._min_quantized_value)
    output_dict = _process_segment_and_label(self.video_matrix, self.num_frames,
                                             decoded_tensors["contexts"],
                                             self._segment_labels,
                                             self._segment_size,
                                             self._num_classes)
    return output_dict

  def _parse_eval_data(self, decoded_tensors):
    """Parses data for evaluation."""
    # loads (potentially) different types of features and concatenates them
    self.video_matrix, self.num_frames = _concat_features(
        decoded_tensors["features"], self._feature_names, self._feature_sizes,
        self._max_frames, self._max_quantized_value, self._min_quantized_value)
    output_dict = _process_segment_and_label(self.video_matrix, self.num_frames,
                                             decoded_tensors["contexts"],
                                             self._segment_labels,
                                             self._segment_size,
                                             self._num_classes)
    return output_dict  # batched

  def parse_fn(self, is_training):
    """Returns a parse fn that reads and parses raw tensors from the decoder.

    Args:
      is_training: a `bool` to indicate whether it is in training mode.

    Returns:
      parse: a `callable` that takes the serialized example and generate the
        images, labels tuple where labels is a dict of Tensors that contains
        labels.
    """

    def parse(decoded_tensors):
      """Parses the serialized example data."""
      if is_training:
        return self._parse_train_data(decoded_tensors)
      else:
        return self._parse_eval_data(decoded_tensors)

    return parse


class PostBatchProcessor():
  """Processes a video and label dataset which is batched."""

  def __init__(self, input_params: exp_cfg.DataConfig):
    self.segment_labels = input_params.segment_labels
    self.num_classes = input_params.num_classes
    self.segment_size = input_params.segment_size

  def post_fn(self, batched_tensors):
    """Processes batched Tensors."""
    video_ids = batched_tensors["video_ids"]
    video_matrix = batched_tensors["video_matrix"]
    labels = batched_tensors["labels"]
    num_frames = batched_tensors["num_frames"]
    label_weights = None

    if self.segment_labels:
      # [batch x num_segment x segment_size x num_features]
      # -> [batch * num_segment x segment_size x num_features]
      video_ids = tf.reshape(video_ids, [-1])
      video_matrix = tf.reshape(video_matrix, [-1, self.segment_size, 1152])
      labels = tf.reshape(labels, [-1, self.num_classes])
      num_frames = tf.reshape(num_frames, [-1, 1])

      label_weights = tf.reshape(batched_tensors["label_weights"],
                                 [-1, self.num_classes])

    else:
      video_matrix = tf.squeeze(video_matrix)
      labels = tf.squeeze(labels)

    batched_tensors = {
        "video_ids": video_ids,
        "video_matrix": video_matrix,
        "labels": labels,
        "num_frames": num_frames,
    }

    if label_weights is not None:
      batched_tensors["label_weights"] = label_weights

    return batched_tensors


class TransformBatcher():
  """Performs manual batching on input dataset."""

  def __init__(self, input_params: exp_cfg.DataConfig):
    self._segment_labels = input_params.segment_labels
    self._global_batch_size = input_params.global_batch_size
    self._is_training = input_params.is_training

  def batch_fn(self, dataset, input_context):
    """Add padding when segment_labels is true."""
    per_replica_batch_size = input_context.get_per_replica_batch_size(
        self._global_batch_size) if input_context else self._global_batch_size
    if not self._segment_labels:
      dataset = dataset.batch(per_replica_batch_size, drop_remainder=True)
    else:
      # add padding
      pad_shapes = {
          "video_ids": [None],
          "video_matrix": [None, None, None],
          "labels": [None, None],
          "num_frames": [None, None],
          "label_weights": [None, None]
      }
      pad_values = {
          "video_ids": None,
          "video_matrix": 0.0,
          "labels": -1.0,
          "num_frames": 0.0,
          "label_weights": 0.0
      }
      dataset = dataset.padded_batch(
          per_replica_batch_size,
          padded_shapes=pad_shapes,
          drop_remainder=True,
          padding_values=pad_values)
    return dataset
