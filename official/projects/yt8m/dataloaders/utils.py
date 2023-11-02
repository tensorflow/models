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

"""Contains a collection of util functions for training and evaluating."""

from absl import logging
import numpy as np
import tensorflow as tf, tf_keras
from official.vision.dataloaders import tfexample_utils


def dequantize(feat_vector, max_quantized_value=2, min_quantized_value=-2):
  """Dequantize the feature from the byte format to the float format.

  Args:
    feat_vector: the input 1-d vector.
    max_quantized_value: the maximum of the quantized value.
    min_quantized_value: the minimum of the quantized value.

  Returns:
    A float vector which has the same shape as feat_vector.
  """
  assert max_quantized_value > min_quantized_value
  quantized_range = max_quantized_value - min_quantized_value
  scalar = quantized_range / 255.0
  bias = (quantized_range / 512.0) + min_quantized_value
  return feat_vector * scalar + bias


def make_summary(name, value):
  """Creates a tf.Summary proto with the given name and value."""
  summary = tf.Summary()
  val = summary.value.add()
  val.tag = str(name)
  val.simple_value = float(value)
  return summary


def add_global_step_summary(summary_writer,
                            global_step_val,
                            global_step_info_dict,
                            summary_scope="Eval"):
  """Add the global_step summary to the Tensorboard.

  Args:
    summary_writer: Tensorflow summary_writer.
    global_step_val: a int value of the global step.
    global_step_info_dict: a dictionary of the evaluation metrics calculated for
      a mini-batch.
    summary_scope: Train or Eval.

  Returns:
    A string of this global_step summary
  """
  this_hit_at_one = global_step_info_dict["hit_at_one"]
  this_perr = global_step_info_dict["perr"]
  this_loss = global_step_info_dict["loss"]
  examples_per_second = global_step_info_dict.get("examples_per_second", -1)

  summary_writer.add_summary(
      make_summary("GlobalStep/" + summary_scope + "_Hit@1", this_hit_at_one),
      global_step_val)
  summary_writer.add_summary(
      make_summary("GlobalStep/" + summary_scope + "_Perr", this_perr),
      global_step_val)
  summary_writer.add_summary(
      make_summary("GlobalStep/" + summary_scope + "_Loss", this_loss),
      global_step_val)

  if examples_per_second != -1:
    summary_writer.add_summary(
        make_summary("GlobalStep/" + summary_scope + "_Example_Second",
                     examples_per_second), global_step_val)

  summary_writer.flush()
  info = (
      "global_step {0} | Batch Hit@1: {1:.3f} | Batch PERR: {2:.3f} | Batch "
      "Loss: {3:.3f} | Examples_per_sec: {4:.3f}").format(
          global_step_val, this_hit_at_one, this_perr, this_loss,
          examples_per_second)
  return info


def add_epoch_summary(summary_writer,
                      global_step_val,
                      epoch_info_dict,
                      summary_scope="Eval"):
  """Add the epoch summary to the Tensorboard.

  Args:
    summary_writer: Tensorflow summary_writer.
    global_step_val: a int value of the global step.
    epoch_info_dict: a dictionary of the evaluation metrics calculated for the
      whole epoch.
    summary_scope: Train or Eval.

  Returns:
    A string of this global_step summary
  """
  epoch_id = epoch_info_dict["epoch_id"]
  avg_hit_at_one = epoch_info_dict["avg_hit_at_one"]
  avg_perr = epoch_info_dict["avg_perr"]
  avg_loss = epoch_info_dict["avg_loss"]
  aps = epoch_info_dict["aps"]
  gap = epoch_info_dict["gap"]
  mean_ap = np.mean(aps)

  summary_writer.add_summary(
      make_summary("Epoch/" + summary_scope + "_Avg_Hit@1", avg_hit_at_one),
      global_step_val)
  summary_writer.add_summary(
      make_summary("Epoch/" + summary_scope + "_Avg_Perr", avg_perr),
      global_step_val)
  summary_writer.add_summary(
      make_summary("Epoch/" + summary_scope + "_Avg_Loss", avg_loss),
      global_step_val)
  summary_writer.add_summary(
      make_summary("Epoch/" + summary_scope + "_MAP", mean_ap), global_step_val)
  summary_writer.add_summary(
      make_summary("Epoch/" + summary_scope + "_GAP", gap), global_step_val)
  summary_writer.flush()

  info = ("epoch/eval number {0} | Avg_Hit@1: {1:.3f} | Avg_PERR: {2:.3f} "
          "| MAP: {3:.3f} | GAP: {4:.3f} | Avg_Loss: {5:3f} | num_classes: {6}"
         ).format(epoch_id, avg_hit_at_one, avg_perr, mean_ap, gap, avg_loss,
                  len(aps))
  return info


def get_list_of_feature_names_and_sizes(feature_names, feature_sizes):
  """Extract the list of feature names and the dimensionality.

  Args:
    feature_names: string containing comma separated list of feature names
    feature_sizes: string containing comma separated list of feature sizes

  Returns:
    List of the feature names and list of the dimensionality of each feature.
    Elements in the first/second list are strings/integers.
  """
  list_of_feature_names = [
      feature_names.strip() for feature_names in feature_names.split(",")
  ]
  list_of_feature_sizes = [
      int(feature_sizes) for feature_sizes in feature_sizes.split(",")
  ]
  if len(list_of_feature_names) != len(list_of_feature_sizes):
    logging.error(
        "length of the feature names (=%r) != length of feature "
        "sizes (=%r)", str(len(list_of_feature_names)),
        str(len(list_of_feature_sizes)))

  return list_of_feature_names, list_of_feature_sizes


def make_yt8m_example(
    num_segment: int = 5, num_frames: int = 120
) -> tf.train.SequenceExample:
  """Generate fake data for unit tests."""
  rgb = np.random.randint(low=256, size=1024, dtype=np.uint8)
  audio = np.random.randint(low=256, size=128, dtype=np.uint8)

  seq_example = tf.train.SequenceExample()
  seq_example.context.feature["id"].bytes_list.value[:] = [b"id001"]
  seq_example.context.feature["labels"].int64_list.value[:] = [1, 2, 3, 4]
  seq_example.context.feature["segment_labels"].int64_list.value[:] = (
      [4] * num_segment)
  seq_example.context.feature["segment_start_times"].int64_list.value[:] = [
      i * 5 for i in range(num_segment)
  ]
  seq_example.context.feature["segment_scores"].float_list.value[:] = (
      [0.5] * num_segment)
  tfexample_utils.put_bytes_list_to_feature(
      seq_example, rgb.tobytes(), key="rgb", repeat_num=num_frames)
  tfexample_utils.put_bytes_list_to_feature(
      seq_example, audio.tobytes(), key="audio", repeat_num=num_frames)

  return seq_example


# TODO(yeqing): Move the test related functions to test_utils.
def make_example_with_float_features(
    num_segment: int = 5) -> tf.train.SequenceExample:
  """Generate fake data for unit tests."""
  rgb = np.random.rand(1, 2048).astype(np.float32)
  audio = np.random.rand(256).astype(np.float32)

  seq_example = tf.train.SequenceExample()
  seq_example.context.feature["id"].bytes_list.value[:] = [b"id001"]
  seq_example.context.feature["clip/label/index"].int64_list.value[:] = [
      1, 2, 3, 4
  ]
  seq_example.context.feature["segment_labels"].int64_list.value[:] = (
      [4] * num_segment)
  seq_example.context.feature["segment_start_times"].int64_list.value[:] = [
      i * 5 for i in range(num_segment)
  ]
  seq_example.context.feature["segment_scores"].float_list.value[:] = (
      [0.] * num_segment)
  seq_example.context.feature[
      "VIDEO_EMBEDDING/context_feature/floats"].float_list.value[:] = (
          audio.tolist())

  tfexample_utils.put_float_list_to_feature(
      seq_example, rgb.tolist(), key="FEATURE/feature/floats")

  return seq_example


def sample_random_sequence(batch_video_matrix, num_frames, num_samples):
  """Samples a random sequence of frames of size num_samples.

  Args:
    batch_video_matrix: tensor of shape [batch_size x max_frames x feature_size]
    num_frames: tensor of shape [batch_size x 1]
    num_samples: a scalar indicating the number of samples

  Returns:
    reshaped batch_video_matrix in [batch_size x 'num_samples' x feature_size]
  """

  batch_size = tf.shape(batch_video_matrix)[0]
  frame_index_offset = tf.tile(
      tf.expand_dims(tf.range(num_samples), 0), [batch_size, 1])
  max_start_frame_index = tf.maximum(num_frames - num_samples, 0)
  start_frame_index = tf.cast(
      tf.multiply(
          tf.random.uniform([batch_size, 1]),
          tf.cast(max_start_frame_index + 1, tf.float32)), tf.int32)
  frame_index = tf.minimum(start_frame_index + frame_index_offset,
                           tf.cast(num_frames - 1, tf.int32))
  batch_index = tf.tile(
      tf.expand_dims(tf.range(batch_size), 1), [1, num_samples])
  index = tf.stack([batch_index, frame_index], 2)
  return tf.gather_nd(batch_video_matrix, index)


def sample_random_frames(batch_video_matrix, num_frames, num_samples):
  """Samples a random set of frames of size num_samples.

  Args:
    batch_video_matrix: tensor of shape [batch_size x max_frames x feature_size]
    num_frames: tensor of shape [batch_size x 1]
    num_samples (int): a scalar indicating the number of samples

  Returns:
    reshaped batch_video_matrix in [batch_size x 'num_samples' x feature_size]
  """
  batch_size = tf.shape(batch_video_matrix)[0]
  frame_index = tf.cast(
      tf.multiply(
          tf.random.uniform([batch_size, num_samples]),
          tf.tile(num_frames, [1, num_samples])), tf.int32)
  batch_index = tf.tile(
      tf.expand_dims(tf.range(batch_size), 1), [1, num_samples])
  index = tf.stack([batch_index, frame_index], 2)
  return tf.gather_nd(batch_video_matrix, index)


def sample_video_frames(
    batch_video_matrix: tf.Tensor,
    num_frames: tf.Tensor,
    random_frames: bool = True,
    num_sample_frames: int = 25,
):
  """Preprocesses input to sample frames."""

  # Sample random frames / random sequence.
  num_frames = tf.cast(num_frames, tf.float32)
  if random_frames:
    batch_video_matrix = sample_random_frames(
        batch_video_matrix, num_frames, num_sample_frames
    )
  else:
    batch_video_matrix = sample_random_sequence(
        batch_video_matrix, num_frames, num_sample_frames
    )
  return batch_video_matrix
