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
"""Detection model evaluator.

This file provides a generic evaluation method that can be used to evaluate a
DetectionModel.
"""

import logging
import tensorflow as tf

from object_detection import eval_util
from object_detection.core import prefetcher
from object_detection.core import wsod_fields as fields
from object_detection.metrics import coco_evaluation
from object_detection.utils import object_detection_evaluation

flags = tf.app.flags
flags.DEFINE_integer('max_proposals', 100, 
    'Specify the max number of proposals to evaluate.')
flags.DEFINE_float('nms_iou_threshold', 0, 
    'If greater than 0, apply the nms using the threshold.')
FLAGS = flags.FLAGS


# A dictionary of metric names to classes that implement the metric. The classes
# in the dictionary must implement
# utils.object_detection_evaluation.DetectionEvaluator interface.
EVAL_METRICS_CLASS_DICT = {
    'pascal_voc_detection_metrics':
        object_detection_evaluation.PascalDetectionEvaluator,
    'weighted_pascal_voc_detection_metrics':
        object_detection_evaluation.WeightedPascalDetectionEvaluator,
    'pascal_voc_instance_segmentation_metrics':
        object_detection_evaluation.PascalInstanceSegmentationEvaluator,
    'weighted_pascal_voc_instance_segmentation_metrics':
        object_detection_evaluation.WeightedPascalInstanceSegmentationEvaluator,
    'open_images_V2_detection_metrics':
        object_detection_evaluation.OpenImagesDetectionEvaluator,
    'coco_detection_metrics':
        coco_evaluation.CocoDetectionEvaluator,
    'coco_mask_metrics':
        coco_evaluation.CocoMaskEvaluator,
    'oid_challenge_object_detection_metrics':
        object_detection_evaluation.OpenImagesDetectionChallengeEvaluator,
}

EVAL_DEFAULT_METRIC = 'coco_detection_metrics'
#EVAL_DEFAULT_METRIC = 'pascal_voc_detection_metrics'


def _extract_predictions_and_losses(model,
                                    create_input_dict_fn,
                                    ignore_groundtruth=False):
  """Constructs tensorflow detection graph and returns output tensors.

  Args:
    model: model to perform predictions with.
    create_input_dict_fn: function to create input tensor dictionaries.
    ignore_groundtruth: whether groundtruth should be ignored.

  Returns:
    prediction_groundtruth_dict: A dictionary with postprocessed tensors (keyed
      by standard_fields.DetectionResultsFields) and optional groundtruth
      tensors (keyed by standard_fields.InputDataFields).
    losses_dict: A dictionary containing detection losses. This is empty when
      ignore_groundtruth is true.
  """
  input_dict = create_input_dict_fn()
  prefetch_queue = prefetcher.prefetch(input_dict, capacity=500)
  input_dict = prefetch_queue.dequeue()
  original_image = tf.expand_dims(input_dict[fields.InputDataFields.image], 0)
  preprocessed_image, true_image_shapes = model.preprocess(
      tf.to_float(original_image))
  prediction_dict = model.predict(preprocessed_image, true_image_shapes)

  # Evaluate the proposals.

  # detections = model.postprocess(prediction_dict, true_image_shapes)

  model.provide_captions(
      [input_dict[fields.InputDataFields.groundtruth_caption]])

  (wsod_anchors, wsod_anchor_scores
   ) = model.provide_wsod_rpn_groundtruth( 
     prediction_dict['preprocessed_inputs'], 
     prediction_dict['image_shape'])

  batch = 1
  max_proposals = FLAGS.max_proposals

  if FLAGS.nms_iou_threshold < 1e-6:
    top_k_scores, top_k_anchors = tf.nn.top_k(wsod_anchor_scores, max_proposals)
    top_k_anchors = tf.stack([
        tf.tile(tf.expand_dims(
            tf.range(batch, dtype=tf.int32), axis=-1), [1, max_proposals]),
        top_k_anchors], axis=-1)
    top_k_anchors = tf.gather_nd(wsod_anchors, top_k_anchors)
    num_detections = tf.expand_dims(max_proposals, axis=0)
  else:
    wsod_anchors_squeezed = tf.squeeze(wsod_anchors, axis=0)
    wsod_anchor_scores_squeezed = tf.squeeze(wsod_anchor_scores, axis=0)

    selected = tf.image.non_max_suppression(
        wsod_anchors_squeezed, 
        wsod_anchor_scores_squeezed,
        max_output_size=max_proposals, 
        iou_threshold=FLAGS.nms_iou_threshold)
    num_detections = tf.expand_dims(tf.shape(selected)[0], axis=0)
    top_k_anchors = tf.expand_dims(
        tf.gather(wsod_anchors_squeezed, selected), axis=0)
    top_k_scores = tf.expand_dims(
        tf.gather(wsod_anchor_scores_squeezed, selected), axis=0)

  detections = {
    fields.DetectionResultFields.num_detections:
      num_detections,
    fields.DetectionResultFields.detection_boxes:
      top_k_anchors,
    fields.DetectionResultFields.detection_scores:
      top_k_scores,
  }

  #min_v = tf.reduce_min(wsod_anchor_scores, axis=1, keepdims=True)
  #wsod_anchor_scores = wsod_anchor_scores - min_v
  #(nmsed_wsod_anchors, nmsed_wsod_anchor_scores, _, _, _,
  # nmsed_num_wsod_anchors) = model._first_stage_nms_fn(
  #     tf.expand_dims(wsod_anchors, axis=2),
  #     tf.expand_dims(wsod_anchor_scores, axis=2) + min_v,
  #     clip_window=None)
  #detections = {
  #  fields.DetectionResultFields.num_detections:
  #    nmsed_num_wsod_anchors,
  #  fields.DetectionResultFields.detection_boxes:
  #    nmsed_wsod_anchors,
  #  fields.DetectionResultFields.detection_scores:
  #    nmsed_wsod_anchor_scores,
  #}

  groundtruth = None
  losses_dict = {}
  if not ignore_groundtruth:
    groundtruth = {
        fields.InputDataFields.groundtruth_boxes:
            input_dict[fields.InputDataFields.groundtruth_boxes],
        fields.InputDataFields.groundtruth_classes:
            input_dict[fields.InputDataFields.groundtruth_classes],
        fields.InputDataFields.groundtruth_area:
            input_dict[fields.InputDataFields.groundtruth_area],
        fields.InputDataFields.groundtruth_is_crowd:
            input_dict[fields.InputDataFields.groundtruth_is_crowd],
        fields.InputDataFields.groundtruth_difficult:
            input_dict[fields.InputDataFields.groundtruth_difficult]
    }
    if fields.InputDataFields.groundtruth_group_of in input_dict:
      groundtruth[fields.InputDataFields.groundtruth_group_of] = (
          input_dict[fields.InputDataFields.groundtruth_group_of])
    groundtruth_masks_list = None
    if fields.DetectionResultFields.detection_masks in detections:
      groundtruth[fields.InputDataFields.groundtruth_instance_masks] = (
          input_dict[fields.InputDataFields.groundtruth_instance_masks])
      groundtruth_masks_list = [
          input_dict[fields.InputDataFields.groundtruth_instance_masks]]
    groundtruth_keypoints_list = None
    if fields.DetectionResultFields.detection_keypoints in detections:
      groundtruth[fields.InputDataFields.groundtruth_keypoints] = (
          input_dict[fields.InputDataFields.groundtruth_keypoints])
      groundtruth_keypoints_list = [
          input_dict[fields.InputDataFields.groundtruth_keypoints]]
    label_id_offset = 1
    model.provide_groundtruth(
        [input_dict[fields.InputDataFields.groundtruth_boxes]],
        [tf.one_hot(input_dict[fields.InputDataFields.groundtruth_classes]
                    - label_id_offset, depth=model.num_classes)],
        groundtruth_masks_list, groundtruth_keypoints_list)
    #losses_dict.update(model.loss(prediction_dict, true_image_shapes))

  result_dict = eval_util.result_dict_for_single_example(
      original_image,
      input_dict[fields.InputDataFields.source_id],
      detections,
      groundtruth,
      class_agnostic=(
          fields.DetectionResultFields.detection_classes not in detections),
      scale_to_absolute=True)
  return result_dict, losses_dict


def get_evaluators(eval_config, categories):
  """Returns the evaluator class according to eval_config, valid for categories.

  Args:
    eval_config: evaluation configurations.
    categories: a list of categories to evaluate.
  Returns:
    An list of instances of DetectionEvaluator.

  Raises:
    ValueError: if metric is not in the metric class dictionary.
  """
  eval_metric_fn_keys = eval_config.metrics_set
  if not eval_metric_fn_keys:
    eval_metric_fn_keys = [EVAL_DEFAULT_METRIC]
  evaluators_list = []
  for eval_metric_fn_key in eval_metric_fn_keys:
    if eval_metric_fn_key not in EVAL_METRICS_CLASS_DICT:
      raise ValueError('Metric not found: {}'.format(eval_metric_fn_key))
    evaluators_list.append(
        EVAL_METRICS_CLASS_DICT[eval_metric_fn_key](categories=categories))
  return evaluators_list

global_num_boxes = 0

def evaluate(create_input_dict_fn, create_model_fn, eval_config, categories,
             checkpoint_dir, eval_dir, graph_hook_fn=None, evaluator_list=None):
  """Evaluation function for detection models.

  Args:
    create_input_dict_fn: a function to create a tensor input dictionary.
    create_model_fn: a function that creates a DetectionModel.
    eval_config: a eval_pb2.EvalConfig protobuf.
    categories: a list of category dictionaries. Each dict in the list should
                have an integer 'id' field and string 'name' field.
    checkpoint_dir: directory to load the checkpoints to evaluate from.
    eval_dir: directory to write evaluation metrics summary to.
    graph_hook_fn: Optional function that is called after the training graph is
      completely built. This is helpful to perform additional changes to the
      training graph such as optimizing batchnorm. The function should modify
      the default graph.
    evaluator_list: Optional list of instances of DetectionEvaluator. If not
      given, this list of metrics is created according to the eval_config.

  Returns:
    metrics: A dictionary containing metric names and values from the latest
      run.
  """

  model = create_model_fn()

  if eval_config.ignore_groundtruth and not eval_config.export_path:
    logging.fatal('If ignore_groundtruth=True then an export_path is '
                  'required. Aborting!!!')

  tensor_dict, losses_dict = _extract_predictions_and_losses(
      model=model,
      create_input_dict_fn=create_input_dict_fn,
      ignore_groundtruth=eval_config.ignore_groundtruth)

  def _process_batch(tensor_dict, sess, batch_index, counters,
                     losses_dict=None):
    """Evaluates tensors in tensor_dict, losses_dict and visualizes examples.

    This function calls sess.run on tensor_dict, evaluating the original_image
    tensor only on the first K examples and visualizing detections overlaid
    on this original_image.

    Args:
      tensor_dict: a dictionary of tensors
      sess: tensorflow session
      batch_index: the index of the batch amongst all batches in the run.
      counters: a dictionary holding 'success' and 'skipped' fields which can
        be updated to keep track of number of successful and failed runs,
        respectively.  If these fields are not updated, then the success/skipped
        counter values shown at the end of evaluation will be incorrect.
      losses_dict: Optional dictonary of scalar loss tensors.

    Returns:
      result_dict: a dictionary of numpy arrays
      result_losses_dict: a dictionary of scalar losses. This is empty if input
        losses_dict is None.
    """
    global global_num_boxes
    try:
      if not losses_dict:
        losses_dict = {}
      result_dict, result_losses_dict = sess.run([tensor_dict, losses_dict])
      if len(result_dict['detection_scores']) > 0:
        #import pdb
        #pdb.set_trace()
        j = 1
      global_num_boxes += len(result_dict['detection_scores'])
      counters['success'] += 1
    except tf.errors.InvalidArgumentError:
      logging.info('Skipping image')
      counters['skipped'] += 1
      return {}, {}
    global_step = tf.train.global_step(sess, tf.train.get_global_step())
    if batch_index < eval_config.num_visualizations:
      tag = 'image-{}'.format(batch_index)
      eval_util.visualize_detection_results(
          result_dict,
          tag,
          global_step,
          categories=categories,
          summary_dir=eval_dir,
          export_dir=eval_config.visualization_export_dir,
          show_groundtruth=eval_config.visualize_groundtruth_boxes,
          groundtruth_box_visualization_color=eval_config.
          groundtruth_box_visualization_color,
          min_score_thresh=eval_config.min_score_threshold,
          max_num_predictions=eval_config.max_num_boxes_to_visualize,
          skip_scores=eval_config.skip_scores,
          skip_labels=eval_config.skip_labels,
          keep_image_id_for_visualization_export=eval_config.
          keep_image_id_for_visualization_export)
    return result_dict, result_losses_dict

  if graph_hook_fn: graph_hook_fn()

  variables_to_restore = tf.global_variables()
  global_step = tf.train.get_or_create_global_step()
  variables_to_restore.append(global_step)

  if eval_config.use_moving_averages:
    variable_averages = tf.train.ExponentialMovingAverage(0.0)
    variables_to_restore = variable_averages.variables_to_restore()
  saver = tf.train.Saver(variables_to_restore)

  def _restore_latest_checkpoint(sess):
    if checkpoint_dir:
      latest_checkpoint = tf.train.latest_checkpoint(checkpoint_dir)
      saver.restore(sess, latest_checkpoint)

  if not evaluator_list:
    evaluator_list = get_evaluators(eval_config, categories)

  metrics = eval_util.repeated_checkpoint_run(
      tensor_dict=tensor_dict,
      summary_dir=eval_dir,
      evaluators=evaluator_list,
      batch_processor=_process_batch,
      checkpoint_dirs=[checkpoint_dir],
      variables_to_restore=None,
      restore_fn=_restore_latest_checkpoint,
      num_batches=eval_config.num_examples,
      eval_interval_secs=eval_config.eval_interval_secs,
      max_number_of_evaluations=(1 if eval_config.ignore_groundtruth else
                                 eval_config.max_evals
                                 if eval_config.max_evals else None),
      master=eval_config.eval_master,
      save_graph=eval_config.save_graph,
      save_graph_dir=(eval_dir if eval_config.save_graph else ''),
      losses_dict=losses_dict,
      eval_export_path=eval_config.export_path)

  print('evaluated %i boxes' % (global_num_boxes))
  return metrics

