# Copyright 2022 The TensorFlow Authors. All Rights Reserved.
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

"""2D detection evaluator for the Waymo Open Dataset."""
import pprint
from absl import logging

import tensorflow as tf
from official.vision.ops import box_ops
from waymo_open_dataset import label_pb2
from waymo_open_dataset.metrics.python import wod_detection_evaluator
from waymo_open_dataset.protos import breakdown_pb2
from waymo_open_dataset.protos import metrics_pb2


def get_2d_detection_default_config():
  """Returns the config proto for WOD 2D detection Evaluation."""
  config = metrics_pb2.Config()

  config.breakdown_generator_ids.append(breakdown_pb2.Breakdown.OBJECT_TYPE)
  difficulty = config.difficulties.add()
  difficulty.levels.append(label_pb2.Label.LEVEL_1)
  difficulty.levels.append(label_pb2.Label.LEVEL_2)
  config.breakdown_generator_ids.append(breakdown_pb2.Breakdown.ALL_BUT_SIGN)
  difficulty = config.difficulties.add()
  difficulty.levels.append(label_pb2.Label.LEVEL_1)
  difficulty.levels.append(label_pb2.Label.LEVEL_2)
  config.matcher_type = metrics_pb2.MatcherProto.TYPE_HUNGARIAN
  config.iou_thresholds.append(0.0)
  config.iou_thresholds.append(0.7)
  config.iou_thresholds.append(0.5)
  config.iou_thresholds.append(0.5)
  config.iou_thresholds.append(0.5)
  config.box_type = label_pb2.Label.Box.TYPE_2D

  for i in range(100):
    config.score_cutoffs.append(i * 0.01)
  config.score_cutoffs.append(1.0)

  return config


class WOD2dDetectionEvaluator(wod_detection_evaluator.WODDetectionEvaluator):
  """WOD 2D detection evaluation metric class."""

  def __init__(self, config=None):
    if config is None:
      config = get_2d_detection_default_config()
    super().__init__(config=config)

  def _remove_padding(self, tensor_dict, num_valid):
    """Remove the paddings of the prediction/groundtruth data."""
    result_tensor_dict = {}
    gather_indices = tf.range(num_valid)
    for k, v in tensor_dict.items():
      if 'frame_id' in k:
        result_tensor_dict[k] = tf.tile([v], [num_valid])
      else:
        result_tensor_dict[k] = tf.gather(v, gather_indices)
    return result_tensor_dict

  def update_state(self, groundtruths, predictions):
    """Update the metrics state with prediction and groundtruth data.

    Args:
      groundtruths: a dictionary of Tensors including the fields below.
        Required fields:
          - source_id: a numpy array of int or string of shape [batch_size].
          - num_detections: a numpy array of int of shape [batch_size].
          - boxes: a numpy array of float of shape [batch_size, K, 4].
          - classes: a numpy array of int of shape [batch_size, K].
          - difficulties: a numpy array of int of shape [batch_size, K].

      predictions: a dictionary of tensors including the fields below.
        Required fields:
          - source_id: a numpy array of int or string of shape [batch_size].
          - image_info: a numpy array of float of shape [batch_size, 4, 2].
          - num_detections: a numpy array of int of shape [batch_size].
          - detection_boxes: a numpy array of float of shape [batch_size, K, 4].
          - detection_classes: a numpy array of int of shape [batch_size, K].
          - detection_scores: a numpy array of float of shape [batch_size, K].
    """
    # Preprocess potentially aggregated tensors.
    for k, v in groundtruths.items():
      if isinstance(v, tuple):
        groundtruths[k] = tf.concat(v, axis=0)
    for k, v in predictions.items():
      if isinstance(v, tuple):
        predictions[k] = tf.concat(v, axis=0)

    # Change cyclists' type id from 3 to 4, where 3 is reserved for sign.
    groundtruth_type = tf.cast(groundtruths['classes'], tf.uint8)
    groundtruth_type = tf.where(
        tf.equal(groundtruth_type, 3),
        tf.ones_like(groundtruth_type) * 4, groundtruth_type)
    prediction_type = tf.cast(predictions['detection_classes'], tf.uint8)
    prediction_type = tf.where(
        tf.equal(prediction_type, 3),
        tf.ones_like(prediction_type) * 4, prediction_type)

    # Rescale the detection boxes back to original scale.
    image_scale = tf.tile(predictions['image_info'][:, 2:3, :], (1, 1, 2))
    prediction_bbox = predictions['detection_boxes'] / image_scale

    batch_size = tf.shape(groundtruths['source_id'])[0]

    for i in tf.range(batch_size):
      frame_groundtruths = {
          'ground_truth_frame_id':
              groundtruths['source_id'][i],
          'ground_truth_bbox':
              box_ops.yxyx_to_cycxhw(
                  tf.cast(groundtruths['boxes'][i], tf.float32)),
          'ground_truth_type':
              groundtruth_type[i],
          'ground_truth_difficulty':
              tf.cast(groundtruths['difficulties'][i], tf.uint8),
      }
      frame_groundtruths = self._remove_padding(
          frame_groundtruths, groundtruths['num_detections'][i])
      frame_predictions = {
          'prediction_frame_id':
              groundtruths['source_id'][i],
          'prediction_bbox':
              box_ops.yxyx_to_cycxhw(
                  tf.cast(prediction_bbox[i], tf.float32)),
          'prediction_type':
              prediction_type[i],
          'prediction_score':
              tf.cast(predictions['detection_scores'][i], tf.float32),
          'prediction_overlap_nlz':
              tf.zeros_like(predictions['detection_scores'][i], dtype=tf.bool)
      }
      frame_predictions = self._remove_padding(frame_predictions,
                                               predictions['num_detections'][i])
      super().update_state(frame_groundtruths, frame_predictions)

  def evaluate(self):
    """Compute the final metrics."""
    ap, _, _, _, _, _, _ = super().evaluate()
    metric_dict = {}
    for i, name in enumerate(self._breakdown_names):
      # Skip sign metrics in 2d detection task.
      if 'SIGN' in name:
        continue
      metric_dict['WOD metrics/{}/AP'.format(name)] = ap[i]
    pp = pprint.PrettyPrinter()
    logging.info('WOD Detection Metrics: \n %s', pp.pformat(metric_dict))

    return metric_dict
