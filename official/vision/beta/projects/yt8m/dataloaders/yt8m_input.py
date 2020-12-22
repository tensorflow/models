'''class YT8MFrameFeatureReader(BaseReader): --> reimplement in tf2
  Reads TFRecords of SequenceExamples.

  The TFRecords must contain SequenceExamples with the sparse in64 'labels'
  context feature and a fixed length byte-quantized feature vector, obtained
  from the features in 'feature_names'. The quantized features will be mapped
  back into a range between min_quantized_value and max_quantized_value.
  link for details: https://research.google.com/youtube8m/download.html
  '''

from typing import Dict, Optional, Tuple
from absl import logging
import tensorflow as tf
# import utils
from random import seed
from official.vision.beta.projects.yt8m import utils
from official.vision.beta.configs import video_classification as exp_cfg
from official.vision.beta.dataloaders import decoder
from official.vision.beta.dataloaders import parser
from official.vision.beta.ops import preprocess_ops_3d


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

def sampler(video_matrix,
            num_frames: int = 32,
            stride: int = 1,
            seed: Optional[int] = None) -> tf.Tensor:
  """
  Args:
    video_matrix: different features concatenated into one matrix
                  [num_segment, segment_size, features]
    is_training: Whether or not in training mode. If True, random sample, crop
      and left right flip is used.
    num_frames: Number of frames per subclip.
    stride: Temporal stride to sample frames.
    seed: A deterministic seed to use when sampling.

  Returns:
    matrix of size: [num_segment, segment_size, features]
    maintained the same as video_matrix
  """
  # Sample random clip.
  sampled_video_matrix = []
  for image in video_matrix: #iterate over num segment / image: (segment_size, features)
    image = preprocess_ops_3d.sample_sequence(image, num_frames, True, stride,
                                            seed)
    sampled_video_matrix.append(image)

  sampled_video_matrix = tf.stack(sampled_video_matrix, axis=0)
  return sampled_video_matrix

def _process_segment_and_label(video_matrix,
                               num_frames,
                               contexts,
                               segment_labels,
                               segment_size,
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
    uniq_start_times, seg_idxs = tf.unique(start_times,
                                           out_idx=tf.dtypes.int64)
    # Range gather matrix, e.g., [[0,1,2],[1,2,3]] for segment_size == 3.
    range_mtx = tf.expand_dims(uniq_start_times, axis=-1) + tf.expand_dims(
      tf.range(0, segment_size, dtype=tf.int64), axis=0)
    # Shape: [num_segment, segment_size, feature_dim].
    batch_video_matrix = tf.gather_nd(video_matrix,
                                      tf.expand_dims(range_mtx, axis=-1))
    num_segment = tf.shape(batch_video_matrix)[0]
    batch_video_ids = tf.reshape(tf.tile([contexts["id"]], [num_segment]),
                                 (num_segment,))
    batch_frames = tf.reshape(tf.tile([segment_size], [num_segment]),
                              (num_segment,))

    # For segment labels, all labels are not exhausively rated. So we only
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
    batch_label_weights = tf.sparse.to_dense(sparse_label_weights,
                                             validate_indices=False)
  else:
    # Process video-level labels.
    label_indices = contexts["labels"].values
    sparse_labels = tf.sparse.SparseTensor(
      tf.expand_dims(label_indices, axis=-1),
      tf.ones_like(contexts["labels"].values, dtype=tf.bool),
      (num_classes,))
    labels = tf.sparse.to_dense(sparse_labels,
                                default_value=False,
                                validate_indices=False)

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


def _get_video_matrix(features, feature_size, max_frames,
                      max_quantized_value, min_quantized_value):
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
    tf.cast(tf.io.decode_raw(features, tf.uint8), tf.float32),  # tf.decode_raw -> tf.io.decode_raw
    [-1, feature_size])

  num_frames = tf.minimum(tf.shape(decoded_features)[0], max_frames)
  feature_matrix = utils.Dequantize(decoded_features, max_quantized_value,
                                    min_quantized_value)
  feature_matrix = resize_axis(feature_matrix, 0, max_frames)
  return feature_matrix, num_frames


def _concat_features(features, feature_names, feature_sizes,
                     max_frames, max_quantized_value, min_quantized_value):
  '''loads (potentially) different types of features and concatenates them

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
  '''

  num_features = len(feature_names)
  assert num_features > 0, "No feature selected: feature_names is empty!"

  assert len(feature_names) == len(feature_sizes), (
    "length of feature_names (={}) != length of feature_sizes (={})".format(
      len(feature_names), len(feature_sizes)))

  num_frames = -1  # the number of frames in the video
  feature_matrices = [None] * num_features  # an array of different features
  for feature_index in range(num_features):
    feature_matrix, num_frames_in_this_feature = _get_video_matrix(
      features[feature_names[feature_index]],
      feature_sizes[feature_index], max_frames,
      max_quantized_value, min_quantized_value)
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

  def __init__(self,
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

    return {'contexts':contexts, 'features':features}


class Parser(parser.Parser):
  """Parses a video and label dataset.
    takes the decoded raw tensors dict
    and parse them into a dictionary of tensors that can be consumed by the model.
    It will be executed after decoder.
  """

  def __init__(self,
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
    self._max_quantized_value = max_quantized_value
    self._min_quantized_value = min_quantized_value
    self.seed = seed


  def _parse_train_data(self, decoded_tensors):  # -> Tuple[Dict[str, tf.Tensor], tf.Tensor]
    """Parses data for training."""
    # loads (potentially) different types of features and concatenates them
    self.video_matrix, self.num_frames = _concat_features(decoded_tensors["features"], self._feature_names, self._feature_sizes,
                                                          self._max_frames, self._max_quantized_value,
                                                          self._min_quantized_value)
    # call sampler
    self.video_matrix = sampler(self.video_matrix, self.num_frames, self.stride, self.seed)
    output_dict = _process_segment_and_label(self.video_matrix, self.num_frames, decoded_tensors["contexts"], self._segment_labels,
                                             self._segment_size, self._num_classes)
    return output_dict

  def _parse_eval_data(self, decoded_tensors):  # -> Tuple[Dict[str, tf.Tensor], tf.Tensor]
    """Parses data for training."""
    # loads (potentially) different types of features and concatenates them
    self.video_matrix, self.num_frames = _concat_features(decoded_tensors["features"], self._feature_names, self._feature_sizes,
                                                          self._max_frames, self._max_quantized_value,
                                                          self._min_quantized_value)
    output_dict = _process_segment_and_label(self.video_matrix, self.num_frames, decoded_tensors["contexts"], self._segment_labels,
                                               self._segment_size, self._num_classes)

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
