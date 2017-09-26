# Copyright 2017 The TensorFlow Authors. All Rights Reserved.
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

"""Builder function for post processing operations."""
import functools

import tensorflow as tf
from object_detection.core import post_processing
from object_detection.protos import post_processing_pb2


def build(post_processing_config):
  """Builds callables for post-processing operations.

  Builds callables for non-max suppression and score conversion based on the
  configuration.

  Non-max suppression callable takes `boxes`, `scores`, and optionally
  `clip_window`, `parallel_iterations` and `scope` as inputs. It returns
  `nms_boxes`, `nms_scores`, `nms_nms_classes` and `num_detections`. See
  post_processing.batch_multiclass_non_max_suppression for the type and shape
  of these tensors.

  Score converter callable should be called with `input` tensor. The callable
  returns the output from one of 3 tf operations based on the configuration -
  tf.identity, tf.sigmoid or tf.nn.softmax. See tensorflow documentation for
  argument and return value descriptions.

  Args:
    post_processing_config: post_processing.proto object containing the
      parameters for the post-processing operations.

  Returns:
    non_max_suppressor_fn: Callable for non-max suppression.
    score_converter_fn: Callable for score conversion.

  Raises:
    ValueError: if the post_processing_config is of incorrect type.
  """
  if not isinstance(post_processing_config, post_processing_pb2.PostProcessing):
    raise ValueError('post_processing_config not of type '
                     'post_processing_pb2.Postprocessing.')
  non_max_suppressor_fn = _build_non_max_suppressor(
      post_processing_config.batch_non_max_suppression)
  score_converter_fn = _build_score_converter(
      post_processing_config.score_converter)
  return non_max_suppressor_fn, score_converter_fn


def _build_non_max_suppressor(nms_config):
  """Builds non-max suppresson based on the nms config.

  Args:
    nms_config: post_processing_pb2.PostProcessing.BatchNonMaxSuppression proto.

  Returns:
    non_max_suppressor_fn: Callable non-max suppressor.

  Raises:
    ValueError: On incorrect iou_threshold or on incompatible values of
      max_total_detections and max_detections_per_class.
  """
  if nms_config.iou_threshold < 0 or nms_config.iou_threshold > 1.0:
    raise ValueError('iou_threshold not in [0, 1.0].')
  if nms_config.max_detections_per_class > nms_config.max_total_detections:
    raise ValueError('max_detections_per_class should be no greater than '
                     'max_total_detections.')

  non_max_suppressor_fn = functools.partial(
      post_processing.batch_multiclass_non_max_suppression,
      score_thresh=nms_config.score_threshold,
      iou_thresh=nms_config.iou_threshold,
      max_size_per_class=nms_config.max_detections_per_class,
      max_total_size=nms_config.max_total_detections)
  return non_max_suppressor_fn


def _build_score_converter(score_converter_config):
  """Builds score converter based on the config.

  Builds one of [tf.identity, tf.sigmoid, tf.softmax] score converters based on
  the config.

  Args:
    score_converter_config: post_processing_pb2.PostProcessing.score_converter.

  Returns:
    Callable score converter op.

  Raises:
    ValueError: On unknown score converter.
  """
  if score_converter_config == post_processing_pb2.PostProcessing.IDENTITY:
    return tf.identity
  if score_converter_config == post_processing_pb2.PostProcessing.SIGMOID:
    return tf.sigmoid
  if score_converter_config == post_processing_pb2.PostProcessing.SOFTMAX:
    return tf.nn.softmax
  raise ValueError('Unknown score converter.')
