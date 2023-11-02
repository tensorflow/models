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

"""class YT8MFrameFeatureReader(BaseReader).

  Reads TFRecords of SequenceExamples.

  The TFRecords must contain SequenceExamples with the sparse in64 'labels'
  context feature and a fixed length byte-quantized feature vector, obtained
  from the features in 'feature_names'. The quantized features will be mapped
  back into a range between min_quantized_value and max_quantized_value.
  link for details: https://research.google.com/youtube8m/download.html
"""
from typing import Any, Dict

import tensorflow as tf, tf_keras
from official.projects.yt8m.dataloaders import utils
from official.vision.configs import video_classification as exp_cfg
from official.vision.dataloaders import decoder
from official.vision.dataloaders import parser


def resize_axis(tensor, axis, new_size, fill_value=0):
  """Truncates or pads a tensor to new_size on a given axis.

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
    segment_size: the segment_size used for reading segments. Segment length.
    num_classes: a positive integer for the number of classes.

  Returns:
    output: dictionary containing batch information
  """
  # Partition frame-level feature matrix to segment-level feature matrix.
  batch_video_ids = None
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
    if "id" in contexts:
      batch_video_ids = tf.reshape(
          tf.tile([contexts["id"]], [num_segment]), (num_segment,))
    batch_frames = tf.reshape(
        tf.tile([segment_size], [num_segment]), (num_segment, 1))
    batch_frames = tf.cast(batch_frames, tf.int32)

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

  else:
    # Process video-level labels.
    label_indices = contexts["labels"].values
    sparse_labels = tf.sparse.SparseTensor(
        tf.expand_dims(label_indices, axis=-1),
        tf.ones_like(contexts["labels"].values, dtype=tf.float32),
        (num_classes,),
    )
    labels = tf.sparse.to_dense(
        sparse_labels, validate_indices=False)

    # convert to batch format.
    if "id" in contexts:
      batch_video_ids = tf.expand_dims(contexts["id"], 0)
    batch_video_matrix = tf.expand_dims(video_matrix, 0)
    batch_labels = tf.expand_dims(labels, 0)
    batch_frames = tf.expand_dims(num_frames, 0)
    batch_label_weights = None

  output_dict = {
      "video_matrix": batch_video_matrix,
      "labels": batch_labels,
      "num_frames": batch_frames,
  }
  if batch_video_ids is not None:
    output_dict["video_ids"] = batch_video_ids
  if batch_label_weights is not None:
    output_dict["label_weights"] = batch_label_weights

  return output_dict


# TODO(allenyan, zhengxu): Adds a unit test for this function.
def _get_video_matrix(features, feature_size, dtype, max_frames,
                      max_quantized_value, min_quantized_value):
  """Decodes features from an input string and quantizes it.

  Args:
    features: raw feature values.
    feature_size: length of each frame feature vector.
    dtype: raw type of the feature.
    max_frames: number of frames (rows) in the output feature_matrix.
    max_quantized_value: the maximum of the quantized value.
    min_quantized_value: the minimum of the quantized value.

  Returns:
    feature_matrix: matrix of all frame-features
    num_frames: number of frames in the sequence
  """
  decoded_features = tf.reshape(features, [-1, feature_size])

  if dtype.is_integer:
    feature_matrix = utils.dequantize(decoded_features, max_quantized_value,
                                      min_quantized_value)
  else:
    feature_matrix = decoded_features

  num_frames = tf.math.minimum(tf.shape(decoded_features)[0], max_frames)
  feature_matrix = feature_matrix[:num_frames]

  return feature_matrix, num_frames


def _concat_features(
    features,
    feature_names,
    feature_sizes,
    feature_dtypes,
    max_frames,
    max_quantized_value,
    min_quantized_value,
    per_feature_l2_norm=False,
):
  """Loads (potentially) different types of features and concatenates them.

  Args:
      features: raw feature values
      feature_names: list of feature names
      feature_sizes: list of features sizes
      feature_dtypes: dtype of the feature.
      max_frames: number of frames in the sequence
      max_quantized_value: the maximum of the quantized value.
      min_quantized_value: the minimum of the quantized value.
      per_feature_l2_norm: whether to l2 normalize each feature.

  Returns:
      video_matrix: different features concatenated into one matrix
      num_frames: the number of frames in the video
  """

  num_features = len(feature_names)
  assert num_features > 0, "No feature selected: feature_names is empty!"

  assert len(feature_names) == len(feature_sizes), (
      "length of feature_names (={}) != length of feature_sizes (={})".format(
          len(feature_names), len(feature_sizes)))
  assert len(feature_names) == len(feature_dtypes), (
      "length of feature_names (={}) != length of feature_sizes (={})".format(
          len(feature_names), len(feature_dtypes)))

  # the number of common frames of all features in the video
  num_common_frames = 1080000  # set max to a 10-hour video at 30fps
  feature_matrices = [None] * num_features  # an array of different features
  for i in range(num_features):
    feature_matrix, num_frames_in_this_feature = _get_video_matrix(
        features[feature_names[i]], feature_sizes[i],
        tf.dtypes.as_dtype(feature_dtypes[i]), max_frames, max_quantized_value,
        min_quantized_value)
    num_common_frames = tf.math.minimum(num_frames_in_this_feature,
                                        num_common_frames)
    if per_feature_l2_norm:
      feature_matrix = tf.math.l2_normalize(feature_matrix, axis=-1)
    feature_matrices[i] = feature_matrix

  for i in range(num_features):
    feature_matrices[i] = feature_matrices[i][:num_common_frames]

  # Concatenate different features.
  video_matrix = tf.concat(feature_matrices, 1)

  return video_matrix, num_common_frames


class Decoder(decoder.Decoder):
  """A tf.train.SequeneExample decoder for classification task."""

  def __init__(
      self,
      input_params: exp_cfg.DataConfig,
  ):

    self._segment_labels = input_params.segment_labels
    self._feature_names = input_params.feature_names
    self._feature_sources = input_params.feature_sources
    self._feature_sizes = input_params.feature_sizes
    self._feature_dtypes = input_params.feature_dtypes
    self._feature_from_bytes = input_params.feature_from_bytes
    self._include_video_id = input_params.include_video_id
    self._label_field = input_params.label_field

    assert len(self._feature_names) == len(self._feature_sources), (
        "length of feature_names (={}) != length of feature_sizes (={})".format(
            len(self._feature_names), len(self._feature_sources)))

    self._context_features = {}
    self._sequence_features = {}
    if self._include_video_id:
      self._context_features["id"] = tf.io.FixedLenFeature([], tf.string)

    if self._segment_labels:
      self._context_features.update({
          # There is no need to read end-time given we always assume the segment
          # has the same size.
          "segment_labels": tf.io.VarLenFeature(tf.int64),
          "segment_start_times": tf.io.VarLenFeature(tf.int64),
          "segment_scores": tf.io.VarLenFeature(tf.float32)
      })
    else:
      self._add_labels_specification()

    for i, name in enumerate(self._feature_names):
      if self._feature_from_bytes[i]:
        feature_type = tf.io.FixedLenSequenceFeature([], dtype=tf.string)
      else:
        dtype = tf.dtypes.as_dtype(self._feature_dtypes[i])
        feature_shape = [self._feature_sizes[i]]
        if self._feature_sources[i] == "feature":
          feature_type = tf.io.FixedLenSequenceFeature(feature_shape, dtype)
        else:
          feature_type = tf.io.FixedLenFeature(feature_shape, dtype)
      if self._feature_sources[i] == "feature":
        self._sequence_features[name] = feature_type
      elif self._feature_sources[i] == "context":
        self._context_features[name] = feature_type
      else:
        raise ValueError(
            f"Unknown feature source {self._feature_sources[i]} for {name}")

  def _add_labels_specification(self):
    if not self._label_field:
      raise ValueError(f"Invalid label field: {self._label_field}!")
    self._context_features.update(
        {self._label_field: tf.io.VarLenFeature(tf.int64)})

  def decode(self,
             serialized_example: tf.train.SequenceExample) -> Dict[str, Any]:
    """Parses a single tf.train.SequenceExample into video and label tensors."""
    contexts, features = tf.io.parse_single_sequence_example(
        serialized_example,
        context_features=self._context_features,
        sequence_features=self._sequence_features)
    decoded_tensor = {**contexts, **features}
    for i, name in enumerate(self._feature_names):
      # Convert the VarLen feature to dense tensor.
      if self._feature_from_bytes[i]:
        dtype = tf.dtypes.as_dtype(self._feature_dtypes[i])
        decoded_tensor[name] = tf.cast(
            tf.io.decode_raw(decoded_tensor[name], out_type=dtype), tf.float32
        )
      else:
        if isinstance(decoded_tensor[name], tf.SparseTensor):
          decoded_tensor[name] = tf.sparse.to_dense(decoded_tensor[name])
    return decoded_tensor


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
    self._label_field = input_params.label_field
    self._segment_size = input_params.segment_size
    self._segment_labels = input_params.segment_labels
    self._include_video_id = input_params.include_video_id
    self._feature_names = input_params.feature_names
    self._feature_sources = input_params.feature_sources
    self._feature_sizes = input_params.feature_sizes
    self._feature_dtypes = input_params.feature_dtypes
    self._max_frames = input_params.max_frames
    self._sample_random_frames = input_params.sample_random_frames
    self._num_sample_frames = input_params.num_sample_frames
    self._max_quantized_value = max_quantized_value
    self._min_quantized_value = min_quantized_value
    self._input_per_feature_l2_norm = input_params.input_per_feature_l2_norm

  def _parse_train_data(self, decoded_tensors):
    """Parses data for training."""
    # loads (potentially) different types of features and concatenates them
    video_matrix, num_frames = _concat_features(
        decoded_tensors, self._feature_names, self._feature_sizes,
        self._feature_dtypes, self._max_frames, self._max_quantized_value,
        self._min_quantized_value, self._input_per_feature_l2_norm)
    if not self._include_video_id and "id" in decoded_tensors:
      del decoded_tensors["id"]

    # Valid `num_frames` comes from _concat_features().
    outputs = self._process_label(video_matrix, num_frames, decoded_tensors)
    if self._num_sample_frames is None:
      # Padding to max_frames.
      outputs["video_matrix"] = resize_axis(
          outputs["video_matrix"], 1, self._max_frames
      )
    else:
      outputs["video_matrix"] = utils.sample_video_frames(
          outputs["video_matrix"],
          tf.reshape(outputs["num_frames"], [-1, 1]),
          random_frames=self._sample_random_frames,
          num_sample_frames=self._num_sample_frames,
      )
      outputs["num_frames"] = (
          tf.ones_like(outputs["num_frames"]) * self._num_sample_frames
      )
    return outputs

  def _parse_eval_data(self, decoded_tensors):
    """Parses data for evaluation."""
    # loads (potentially) different types of features and concatenates them
    video_matrix, num_frames = _concat_features(
        decoded_tensors, self._feature_names, self._feature_sizes,
        self._feature_dtypes, self._max_frames, self._max_quantized_value,
        self._min_quantized_value, self._input_per_feature_l2_norm)
    if not self._include_video_id and "id" in decoded_tensors:
      del decoded_tensors["id"]

    outputs = self._process_label(video_matrix, num_frames, decoded_tensors)
    if self._num_sample_frames is None:
      # Padding to max_frames.
      outputs["video_matrix"] = resize_axis(
          outputs["video_matrix"], 1, self._max_frames
      )
    else:
      outputs["video_matrix"] = utils.sample_video_frames(
          outputs["video_matrix"],
          tf.reshape(outputs["num_frames"], [-1, 1]),
          random_frames=self._sample_random_frames,
          num_sample_frames=self._num_sample_frames,
      )
      outputs["num_frames"] = (
          tf.ones_like(outputs["num_frames"]) * self._num_sample_frames
      )
    return outputs

  def _process_label(self, video_matrix, num_frames, contexts):
    """Processes a batched Tensor of frames.

    Args:
      video_matrix: video feature matric.
      num_frames: number of frames in this video.
      contexts: context information extracted from decoder.

    Returns:
      output: dictionary containing batch information
    """
    if self._label_field and not self._segment_labels:
      contexts["labels"] = contexts[self._label_field]
    output_dict = _process_segment_and_label(video_matrix, num_frames, contexts,
                                             self._segment_labels,
                                             self._segment_size,
                                             self._num_classes)
    return output_dict

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

      # Concatenate video features to all frames if there are both video-level
      # (context) and frame-level (feature) features.
      if "feature" in self._feature_sources:
        # Take first frame feature matrix, any feature matrix should be fine
        # since assume all frame features have same number of frames.
        feature_idx = self._feature_sources.index("feature")
        num_frames = tf.shape(
            decoded_tensors[self._feature_names[feature_idx]]
        )[0]
        for feature_idx, feature_source in enumerate(self._feature_sources):
          if feature_source == "context":
            feature_name = self._feature_names[feature_idx]
            context_tensor = tf.reshape(
                decoded_tensors[feature_name],
                shape=(1, self._feature_sizes[feature_idx]),
            )
            decoded_tensors[feature_name] = tf.tile(
                context_tensor, [num_frames, 1]
            )

      if is_training:
        return self._parse_train_data(decoded_tensors)
      else:
        return self._parse_eval_data(decoded_tensors)

    return parse


class TransformBatcher():
  """Performs manual batching on input dataset."""

  def __init__(self, input_params: exp_cfg.DataConfig):
    self._segment_labels = input_params.segment_labels
    self._global_batch_size = input_params.global_batch_size
    self._is_training = input_params.is_training
    self._include_video_id = input_params.include_video_id
    self._drop_remainder = input_params.drop_remainder

  def batch_fn(self, dataset, input_context):
    """Add padding when segment_labels is true."""
    per_replica_batch_size = input_context.get_per_replica_batch_size(
        self._global_batch_size) if input_context else self._global_batch_size
    # Add padding specifications.
    pad_values = {
        "video_matrix": 0.0,
        "labels": -1.0,
        "num_frames": 0,
    }
    if self._include_video_id:
      pad_values["video_ids"] = None
    if self._segment_labels:
      pad_values["label_weights"] = 0.0
    dataset = dataset.padded_batch(
        per_replica_batch_size,
        padding_values=pad_values,
        drop_remainder=self._drop_remainder,
    )
    return dataset


class PostBatchProcessor():
  """Processes a video and label dataset which is batched."""

  def __init__(self, input_params: exp_cfg.DataConfig):
    self.segment_labels = input_params.segment_labels
    self.num_classes = input_params.num_classes
    self.num_batched_frames = (
        input_params.num_sample_frames or input_params.max_frames
    )
    self.num_features = sum(input_params.feature_sizes)

  def post_fn(self, batched_tensors: Dict[str,
                                          tf.Tensor]) -> Dict[str, tf.Tensor]:
    """Processes batched Tensors."""
    video_ids = batched_tensors.get("video_ids", None)
    video_matrix = batched_tensors["video_matrix"]
    labels = batched_tensors["labels"]
    num_frames = batched_tensors["num_frames"]

    if self.segment_labels:
      # [batch x num_segment x num_batched_frames x num_features]
      # -> [batch * num_segment x num_batched_frames x num_features]
      if video_ids is not None:
        video_ids = tf.reshape(video_ids, [-1])
      video_matrix = tf.reshape(
          video_matrix, [-1, self.num_batched_frames, self.num_features]
      )
      labels = tf.reshape(labels, [-1, self.num_classes])
      num_frames = tf.reshape(num_frames, [-1, 1])
      batched_tensors["label_weights"] = tf.reshape(
          batched_tensors["label_weights"], [-1, self.num_classes])
    else:
      # NOTE(b/237445211): Must provide axis argument to tf.squeeze.
      video_matrix = tf.squeeze(video_matrix, axis=1)
      labels = tf.squeeze(labels, axis=1)
      num_frames = tf.reshape(num_frames, [-1, 1])
      if "label_weights" in batched_tensors:
        batched_tensors["label_weights"] = tf.squeeze(
            batched_tensors["label_weights"], axis=1)

    batched_tensors.update({
        "video_matrix": video_matrix,
        "labels": labels,
        "num_frames": num_frames,
    })
    if video_ids is not None:
      batched_tensors["video_ids"] = video_ids

    return batched_tensors
