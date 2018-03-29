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
"""Common utility functions for evaluation."""
import collections
import logging
import os
import time

import numpy as np
import tensorflow as tf

from object_detection.core import box_list
from object_detection.core import box_list_ops
from object_detection.core import keypoint_ops
from object_detection.core import standard_fields as fields
from object_detection.metrics import coco_evaluation
from object_detection.utils import label_map_util
from object_detection.utils import ops
from object_detection.utils import visualization_utils as vis_utils

slim = tf.contrib.slim


def write_metrics(metrics, global_step, summary_dir):
  """Write metrics to a summary directory.

  Args:
    metrics: A dictionary containing metric names and values.
    global_step: Global step at which the metrics are computed.
    summary_dir: Directory to write tensorflow summaries to.
  """
  logging.info('Writing metrics to tf summary.')
  summary_writer = tf.summary.FileWriterCache.get(summary_dir)
  for key in sorted(metrics):
    summary = tf.Summary(value=[
        tf.Summary.Value(tag=key, simple_value=metrics[key]),
    ])
    summary_writer.add_summary(summary, global_step)
    logging.info('%s: %f', key, metrics[key])
  logging.info('Metrics written to tf summary.')


# TODO(rathodv): Add tests.
def visualize_detection_results(result_dict,
                                tag,
                                global_step,
                                categories,
                                summary_dir='',
                                export_dir='',
                                agnostic_mode=False,
                                show_groundtruth=False,
                                groundtruth_box_visualization_color='black',
                                min_score_thresh=.5,
                                max_num_predictions=20,
                                skip_scores=False,
                                skip_labels=False,
                                keep_image_id_for_visualization_export=False):
  """Visualizes detection results and writes visualizations to image summaries.

  This function visualizes an image with its detected bounding boxes and writes
  to image summaries which can be viewed on tensorboard.  It optionally also
  writes images to a directory. In the case of missing entry in the label map,
  unknown class name in the visualization is shown as "N/A".

  Args:
    result_dict: a dictionary holding groundtruth and detection
      data corresponding to each image being evaluated.  The following keys
      are required:
        'original_image': a numpy array representing the image with shape
          [1, height, width, 3]
        'detection_boxes': a numpy array of shape [N, 4]
        'detection_scores': a numpy array of shape [N]
        'detection_classes': a numpy array of shape [N]
      The following keys are optional:
        'groundtruth_boxes': a numpy array of shape [N, 4]
        'groundtruth_keypoints': a numpy array of shape [N, num_keypoints, 2]
      Detections are assumed to be provided in decreasing order of score and for
      display, and we assume that scores are probabilities between 0 and 1.
    tag: tensorboard tag (string) to associate with image.
    global_step: global step at which the visualization are generated.
    categories: a list of dictionaries representing all possible categories.
      Each dict in this list has the following keys:
          'id': (required) an integer id uniquely identifying this category
          'name': (required) string representing category name
            e.g., 'cat', 'dog', 'pizza'
          'supercategory': (optional) string representing the supercategory
            e.g., 'animal', 'vehicle', 'food', etc
    summary_dir: the output directory to which the image summaries are written.
    export_dir: the output directory to which images are written.  If this is
      empty (default), then images are not exported.
    agnostic_mode: boolean (default: False) controlling whether to evaluate in
      class-agnostic mode or not.
    show_groundtruth: boolean (default: False) controlling whether to show
      groundtruth boxes in addition to detected boxes
    groundtruth_box_visualization_color: box color for visualizing groundtruth
      boxes
    min_score_thresh: minimum score threshold for a box to be visualized
    max_num_predictions: maximum number of detections to visualize
    skip_scores: whether to skip score when drawing a single detection
    skip_labels: whether to skip label when drawing a single detection
    keep_image_id_for_visualization_export: whether to keep image identifier in
      filename when exported to export_dir
  Raises:
    ValueError: if result_dict does not contain the expected keys (i.e.,
      'original_image', 'detection_boxes', 'detection_scores',
      'detection_classes')
  """
  detection_fields = fields.DetectionResultFields
  input_fields = fields.InputDataFields
  if not set([
      input_fields.original_image,
      detection_fields.detection_boxes,
      detection_fields.detection_scores,
      detection_fields.detection_classes,
  ]).issubset(set(result_dict.keys())):
    raise ValueError('result_dict does not contain all expected keys.')
  if show_groundtruth and input_fields.groundtruth_boxes not in result_dict:
    raise ValueError('If show_groundtruth is enabled, result_dict must contain '
                     'groundtruth_boxes.')
  logging.info('Creating detection visualizations.')
  category_index = label_map_util.create_category_index(categories)

  image = np.squeeze(result_dict[input_fields.original_image], axis=0)
  detection_boxes = result_dict[detection_fields.detection_boxes]
  detection_scores = result_dict[detection_fields.detection_scores]
  detection_classes = np.int32((result_dict[
      detection_fields.detection_classes]))
  detection_keypoints = result_dict.get(detection_fields.detection_keypoints)
  detection_masks = result_dict.get(detection_fields.detection_masks)
  detection_boundaries = result_dict.get(detection_fields.detection_boundaries)

  # Plot groundtruth underneath detections
  if show_groundtruth:
    groundtruth_boxes = result_dict[input_fields.groundtruth_boxes]
    groundtruth_keypoints = result_dict.get(input_fields.groundtruth_keypoints)
    vis_utils.visualize_boxes_and_labels_on_image_array(
        image=image,
        boxes=groundtruth_boxes,
        classes=None,
        scores=None,
        category_index=category_index,
        keypoints=groundtruth_keypoints,
        use_normalized_coordinates=False,
        max_boxes_to_draw=None,
        groundtruth_box_visualization_color=groundtruth_box_visualization_color)
  vis_utils.visualize_boxes_and_labels_on_image_array(
      image,
      detection_boxes,
      detection_classes,
      detection_scores,
      category_index,
      instance_masks=detection_masks,
      instance_boundaries=detection_boundaries,
      keypoints=detection_keypoints,
      use_normalized_coordinates=False,
      max_boxes_to_draw=max_num_predictions,
      min_score_thresh=min_score_thresh,
      agnostic_mode=agnostic_mode,
      skip_scores=skip_scores,
      skip_labels=skip_labels)

  if export_dir:
    if keep_image_id_for_visualization_export and result_dict[fields.
                                                              InputDataFields()
                                                              .key]:
      export_path = os.path.join(export_dir, 'export-{}-{}.png'.format(
          tag, result_dict[fields.InputDataFields().key]))
    else:
      export_path = os.path.join(export_dir, 'export-{}.png'.format(tag))
    vis_utils.save_image_array_as_png(image, export_path)

  summary = tf.Summary(value=[
      tf.Summary.Value(
          tag=tag,
          image=tf.Summary.Image(
              encoded_image_string=vis_utils.encode_image_array_as_png_str(
                  image)))
  ])
  summary_writer = tf.summary.FileWriterCache.get(summary_dir)
  summary_writer.add_summary(summary, global_step)

  logging.info('Detection visualizations written to summary with tag %s.', tag)


def _run_checkpoint_once(tensor_dict,
                         evaluators=None,
                         batch_processor=None,
                         checkpoint_dirs=None,
                         variables_to_restore=None,
                         restore_fn=None,
                         num_batches=1,
                         master='',
                         save_graph=False,
                         save_graph_dir='',
                         losses_dict=None):
  """Evaluates metrics defined in evaluators and returns summaries.

  This function loads the latest checkpoint in checkpoint_dirs and evaluates
  all metrics defined in evaluators. The metrics are processed in batch by the
  batch_processor.

  Args:
    tensor_dict: a dictionary holding tensors representing a batch of detections
      and corresponding groundtruth annotations.
    evaluators: a list of object of type DetectionEvaluator to be used for
      evaluation. Note that the metric names produced by different evaluators
      must be unique.
    batch_processor: a function taking four arguments:
      1. tensor_dict: the same tensor_dict that is passed in as the first
        argument to this function.
      2. sess: a tensorflow session
      3. batch_index: an integer representing the index of the batch amongst
        all batches
      By default, batch_processor is None, which defaults to running:
        return sess.run(tensor_dict)
      To skip an image, it suffices to return an empty dictionary in place of
      result_dict.
    checkpoint_dirs: list of directories to load into an EnsembleModel. If it
      has only one directory, EnsembleModel will not be used --
        a DetectionModel
      will be instantiated directly. Not used if restore_fn is set.
    variables_to_restore: None, or a dictionary mapping variable names found in
      a checkpoint to model variables. The dictionary would normally be
      generated by creating a tf.train.ExponentialMovingAverage object and
      calling its variables_to_restore() method. Not used if restore_fn is set.
    restore_fn: None, or a function that takes a tf.Session object and correctly
      restores all necessary variables from the correct checkpoint file. If
      None, attempts to restore from the first directory in checkpoint_dirs.
    num_batches: the number of batches to use for evaluation.
    master: the location of the Tensorflow session.
    save_graph: whether or not the Tensorflow graph is stored as a pbtxt file.
    save_graph_dir: where to store the Tensorflow graph on disk. If save_graph
      is True this must be non-empty.
    losses_dict: optional dictionary of scalar detection losses.

  Returns:
    global_step: the count of global steps.
    all_evaluator_metrics: A dictionary containing metric names and values.

  Raises:
    ValueError: if restore_fn is None and checkpoint_dirs doesn't have at least
      one element.
    ValueError: if save_graph is True and save_graph_dir is not defined.
  """
  if save_graph and not save_graph_dir:
    raise ValueError('`save_graph_dir` must be defined.')
  sess = tf.Session(master, graph=tf.get_default_graph())
  sess.run(tf.global_variables_initializer())
  sess.run(tf.local_variables_initializer())
  sess.run(tf.tables_initializer())
  if restore_fn:
    restore_fn(sess)
  else:
    if not checkpoint_dirs:
      raise ValueError('`checkpoint_dirs` must have at least one entry.')
    checkpoint_file = tf.train.latest_checkpoint(checkpoint_dirs[0])
    saver = tf.train.Saver(variables_to_restore)
    saver.restore(sess, checkpoint_file)

  if save_graph:
    tf.train.write_graph(sess.graph_def, save_graph_dir, 'eval.pbtxt')

  counters = {'skipped': 0, 'success': 0}
  aggregate_result_losses_dict = collections.defaultdict(list)
  with tf.contrib.slim.queues.QueueRunners(sess):
    try:
      for batch in range(int(num_batches)):
        if (batch + 1) % 100 == 0:
          logging.info('Running eval ops batch %d/%d', batch + 1, num_batches)
        if not batch_processor:
          try:
            if not losses_dict:
              losses_dict = {}
            result_dict, result_losses_dict = sess.run([tensor_dict,
                                                        losses_dict])
            counters['success'] += 1
          except tf.errors.InvalidArgumentError:
            logging.info('Skipping image')
            counters['skipped'] += 1
            result_dict = {}
        else:
          result_dict, result_losses_dict = batch_processor(
              tensor_dict, sess, batch, counters, losses_dict=losses_dict)
        if not result_dict:
          continue
        for key, value in iter(result_losses_dict.items()):
          aggregate_result_losses_dict[key].append(value)
        for evaluator in evaluators:
          # TODO(b/65130867): Use image_id tensor once we fix the input data
          # decoders to return correct image_id.
          # TODO(akuznetsa): result_dict contains batches of images, while
          # add_single_ground_truth_image_info expects a single image. Fix
          evaluator.add_single_ground_truth_image_info(
              image_id=batch, groundtruth_dict=result_dict)
          evaluator.add_single_detected_image_info(
              image_id=batch, detections_dict=result_dict)
      logging.info('Running eval batches done.')
    except tf.errors.OutOfRangeError:
      logging.info('Done evaluating -- epoch limit reached')
    finally:
      # When done, ask the threads to stop.
      logging.info('# success: %d', counters['success'])
      logging.info('# skipped: %d', counters['skipped'])
      all_evaluator_metrics = {}
      for evaluator in evaluators:
        metrics = evaluator.evaluate()
        evaluator.clear()
        if any(key in all_evaluator_metrics for key in metrics):
          raise ValueError('Metric names between evaluators must not collide.')
        all_evaluator_metrics.update(metrics)
      global_step = tf.train.global_step(sess, tf.train.get_global_step())

      for key, value in iter(aggregate_result_losses_dict.items()):
        all_evaluator_metrics['Losses/' + key] = np.mean(value)
  sess.close()
  return (global_step, all_evaluator_metrics)


# TODO(rathodv): Add tests.
def repeated_checkpoint_run(tensor_dict,
                            summary_dir,
                            evaluators,
                            batch_processor=None,
                            checkpoint_dirs=None,
                            variables_to_restore=None,
                            restore_fn=None,
                            num_batches=1,
                            eval_interval_secs=120,
                            max_number_of_evaluations=None,
                            master='',
                            save_graph=False,
                            save_graph_dir='',
                            losses_dict=None):
  """Periodically evaluates desired tensors using checkpoint_dirs or restore_fn.

  This function repeatedly loads a checkpoint and evaluates a desired
  set of tensors (provided by tensor_dict) and hands the resulting numpy
  arrays to a function result_processor which can be used to further
  process/save/visualize the results.

  Args:
    tensor_dict: a dictionary holding tensors representing a batch of detections
      and corresponding groundtruth annotations.
    summary_dir: a directory to write metrics summaries.
    evaluators: a list of object of type DetectionEvaluator to be used for
      evaluation. Note that the metric names produced by different evaluators
      must be unique.
    batch_processor: a function taking three arguments:
      1. tensor_dict: the same tensor_dict that is passed in as the first
        argument to this function.
      2. sess: a tensorflow session
      3. batch_index: an integer representing the index of the batch amongst
        all batches
      By default, batch_processor is None, which defaults to running:
        return sess.run(tensor_dict)
    checkpoint_dirs: list of directories to load into a DetectionModel or an
      EnsembleModel if restore_fn isn't set. Also used to determine when to run
      next evaluation. Must have at least one element.
    variables_to_restore: None, or a dictionary mapping variable names found in
      a checkpoint to model variables. The dictionary would normally be
      generated by creating a tf.train.ExponentialMovingAverage object and
      calling its variables_to_restore() method. Not used if restore_fn is set.
    restore_fn: a function that takes a tf.Session object and correctly restores
      all necessary variables from the correct checkpoint file.
    num_batches: the number of batches to use for evaluation.
    eval_interval_secs: the number of seconds between each evaluation run.
    max_number_of_evaluations: the max number of iterations of the evaluation.
      If the value is left as None the evaluation continues indefinitely.
    master: the location of the Tensorflow session.
    save_graph: whether or not the Tensorflow graph is saved as a pbtxt file.
    save_graph_dir: where to save on disk the Tensorflow graph. If store_graph
      is True this must be non-empty.
    losses_dict: optional dictionary of scalar detection losses.

  Returns:
    metrics: A dictionary containing metric names and values in the latest
      evaluation.

  Raises:
    ValueError: if max_num_of_evaluations is not None or a positive number.
    ValueError: if checkpoint_dirs doesn't have at least one element.
  """
  if max_number_of_evaluations and max_number_of_evaluations <= 0:
    raise ValueError(
        '`number_of_steps` must be either None or a positive number.')

  if not checkpoint_dirs:
    raise ValueError('`checkpoint_dirs` must have at least one entry.')

  last_evaluated_model_path = None
  number_of_evaluations = 0
  while True:
    start = time.time()
    logging.info('Starting evaluation at ' + time.strftime(
        '%Y-%m-%d-%H:%M:%S', time.gmtime()))
    model_path = tf.train.latest_checkpoint(checkpoint_dirs[0])
    if not model_path:
      logging.info('No model found in %s. Will try again in %d seconds',
                   checkpoint_dirs[0], eval_interval_secs)
    elif model_path == last_evaluated_model_path:
      logging.info('Found already evaluated checkpoint. Will try again in %d '
                   'seconds', eval_interval_secs)
    else:
      last_evaluated_model_path = model_path
      global_step, metrics = _run_checkpoint_once(tensor_dict, evaluators,
                                                  batch_processor,
                                                  checkpoint_dirs,
                                                  variables_to_restore,
                                                  restore_fn, num_batches,
                                                  master, save_graph,
                                                  save_graph_dir,
                                                  losses_dict=losses_dict)
      write_metrics(metrics, global_step, summary_dir)
    number_of_evaluations += 1

    if (max_number_of_evaluations and
        number_of_evaluations >= max_number_of_evaluations):
      logging.info('Finished evaluation!')
      break
    time_to_next_eval = start + eval_interval_secs - time.time()
    if time_to_next_eval > 0:
      time.sleep(time_to_next_eval)

  return metrics


def result_dict_for_single_example(image,
                                   key,
                                   detections,
                                   groundtruth=None,
                                   class_agnostic=False,
                                   scale_to_absolute=False):
  """Merges all detection and groundtruth information for a single example.

  Note that evaluation tools require classes that are 1-indexed, and so this
  function performs the offset. If `class_agnostic` is True, all output classes
  have label 1.

  Args:
    image: A single 4D uint8 image tensor of shape [1, H, W, C].
    key: A single string tensor identifying the image.
    detections: A dictionary of detections, returned from
      DetectionModel.postprocess().
    groundtruth: (Optional) Dictionary of groundtruth items, with fields:
      'groundtruth_boxes': [num_boxes, 4] float32 tensor of boxes, in
        normalized coordinates.
      'groundtruth_classes': [num_boxes] int64 tensor of 1-indexed classes.
      'groundtruth_area': [num_boxes] float32 tensor of bbox area. (Optional)
      'groundtruth_is_crowd': [num_boxes] int64 tensor. (Optional)
      'groundtruth_difficult': [num_boxes] int64 tensor. (Optional)
      'groundtruth_group_of': [num_boxes] int64 tensor. (Optional)
      'groundtruth_instance_masks': 3D int64 tensor of instance masks
        (Optional).
    class_agnostic: Boolean indicating whether the detections are class-agnostic
      (i.e. binary). Default False.
    scale_to_absolute: Boolean indicating whether boxes and keypoints should be
      scaled to absolute coordinates. Note that for IoU based evaluations, it
      does not matter whether boxes are expressed in absolute or relative
      coordinates. Default False.

  Returns:
    A dictionary with:
    'original_image': A [1, H, W, C] uint8 image tensor.
    'key': A string tensor with image identifier.
    'detection_boxes': [max_detections, 4] float32 tensor of boxes, in
      normalized or absolute coordinates, depending on the value of
      `scale_to_absolute`.
    'detection_scores': [max_detections] float32 tensor of scores.
    'detection_classes': [max_detections] int64 tensor of 1-indexed classes.
    'detection_masks': [max_detections, H, W] float32 tensor of binarized
      masks, reframed to full image masks.
    'groundtruth_boxes': [num_boxes, 4] float32 tensor of boxes, in
      normalized or absolute coordinates, depending on the value of
      `scale_to_absolute`. (Optional)
    'groundtruth_classes': [num_boxes] int64 tensor of 1-indexed classes.
      (Optional)
    'groundtruth_area': [num_boxes] float32 tensor of bbox area. (Optional)
    'groundtruth_is_crowd': [num_boxes] int64 tensor. (Optional)
    'groundtruth_difficult': [num_boxes] int64 tensor. (Optional)
    'groundtruth_group_of': [num_boxes] int64 tensor. (Optional)
    'groundtruth_instance_masks': 3D int64 tensor of instance masks
      (Optional).

  """
  label_id_offset = 1  # Applying label id offset (b/63711816)

  input_data_fields = fields.InputDataFields
  output_dict = {
      input_data_fields.original_image: image,
      input_data_fields.key: key,
  }

  detection_fields = fields.DetectionResultFields
  detection_boxes = detections[detection_fields.detection_boxes][0]
  image_shape = tf.shape(image)
  detection_scores = detections[detection_fields.detection_scores][0]

  if class_agnostic:
    detection_classes = tf.ones_like(detection_scores, dtype=tf.int64)
  else:
    detection_classes = (
        tf.to_int64(detections[detection_fields.detection_classes][0]) +
        label_id_offset)

  num_detections = tf.to_int32(detections[detection_fields.num_detections][0])
  detection_boxes = tf.slice(
      detection_boxes, begin=[0, 0], size=[num_detections, -1])
  detection_classes = tf.slice(
      detection_classes, begin=[0], size=[num_detections])
  detection_scores = tf.slice(
      detection_scores, begin=[0], size=[num_detections])

  if scale_to_absolute:
    absolute_detection_boxlist = box_list_ops.to_absolute_coordinates(
        box_list.BoxList(detection_boxes), image_shape[1], image_shape[2])
    output_dict[detection_fields.detection_boxes] = (
        absolute_detection_boxlist.get())
  else:
    output_dict[detection_fields.detection_boxes] = detection_boxes
  output_dict[detection_fields.detection_classes] = detection_classes
  output_dict[detection_fields.detection_scores] = detection_scores

  if detection_fields.detection_masks in detections:
    detection_masks = detections[detection_fields.detection_masks][0]
    # TODO(rathodv): This should be done in model's postprocess
    # function ideally.
    detection_masks = tf.slice(
        detection_masks, begin=[0, 0, 0], size=[num_detections, -1, -1])
    detection_masks_reframed = ops.reframe_box_masks_to_image_masks(
        detection_masks, detection_boxes, image_shape[1], image_shape[2])
    detection_masks_reframed = tf.cast(
        tf.greater(detection_masks_reframed, 0.5), tf.uint8)
    output_dict[detection_fields.detection_masks] = detection_masks_reframed
  if detection_fields.detection_keypoints in detections:
    detection_keypoints = detections[detection_fields.detection_keypoints][0]
    output_dict[detection_fields.detection_keypoints] = detection_keypoints
    if scale_to_absolute:
      absolute_detection_keypoints = keypoint_ops.scale(
          detection_keypoints, image_shape[1], image_shape[2])
      output_dict[detection_fields.detection_keypoints] = (
          absolute_detection_keypoints)

  if groundtruth:
    if input_data_fields.groundtruth_instance_masks in groundtruth:
      groundtruth[input_data_fields.groundtruth_instance_masks] = tf.cast(
          groundtruth[input_data_fields.groundtruth_instance_masks], tf.uint8)
    output_dict.update(groundtruth)
    if scale_to_absolute:
      groundtruth_boxes = groundtruth[input_data_fields.groundtruth_boxes]
      absolute_gt_boxlist = box_list_ops.to_absolute_coordinates(
          box_list.BoxList(groundtruth_boxes), image_shape[1], image_shape[2])
      output_dict[input_data_fields.groundtruth_boxes] = (
          absolute_gt_boxlist.get())
    # For class-agnostic models, groundtruth classes all become 1.
    if class_agnostic:
      groundtruth_classes = groundtruth[input_data_fields.groundtruth_classes]
      groundtruth_classes = tf.ones_like(groundtruth_classes, dtype=tf.int64)
      output_dict[input_data_fields.groundtruth_classes] = groundtruth_classes

  return output_dict


def get_eval_metric_ops_for_evaluators(evaluation_metrics,
                                       categories,
                                       eval_dict,
                                       include_metrics_per_category=False):
  """Returns a dictionary of eval metric ops to use with `tf.EstimatorSpec`.

  Args:
    evaluation_metrics: List of evaluation metric names. Current options are
      'coco_detection_metrics' and 'coco_mask_metrics'.
    categories: A list of dicts, each of which has the following keys -
        'id': (required) an integer id uniquely identifying this category.
        'name': (required) string representing category name e.g., 'cat', 'dog'.
    eval_dict: An evaluation dictionary, returned from
      result_dict_for_single_example().
    include_metrics_per_category: If True, include metrics for each category.

  Returns:
    A dictionary of metric names to tuple of value_op and update_op that can be
    used as eval metric ops in tf.EstimatorSpec.

  Raises:
    ValueError: If any of the metrics in `evaluation_metric` is not
    'coco_detection_metrics' or 'coco_mask_metrics'.
  """
  evaluation_metrics = list(set(evaluation_metrics))

  input_data_fields = fields.InputDataFields
  detection_fields = fields.DetectionResultFields
  eval_metric_ops = {}
  for metric in evaluation_metrics:
    if metric == 'coco_detection_metrics':
      coco_evaluator = coco_evaluation.CocoDetectionEvaluator(
          categories, include_metrics_per_category=include_metrics_per_category)
      eval_metric_ops.update(
          coco_evaluator.get_estimator_eval_metric_ops(
              image_id=eval_dict[input_data_fields.key],
              groundtruth_boxes=eval_dict[input_data_fields.groundtruth_boxes],
              groundtruth_classes=eval_dict[
                  input_data_fields.groundtruth_classes],
              detection_boxes=eval_dict[detection_fields.detection_boxes],
              detection_scores=eval_dict[detection_fields.detection_scores],
              detection_classes=eval_dict[detection_fields.detection_classes]))
    elif metric == 'coco_mask_metrics':
      coco_mask_evaluator = coco_evaluation.CocoMaskEvaluator(
          categories, include_metrics_per_category=include_metrics_per_category)
      eval_metric_ops.update(
          coco_mask_evaluator.get_estimator_eval_metric_ops(
              image_id=eval_dict[input_data_fields.key],
              groundtruth_boxes=eval_dict[input_data_fields.groundtruth_boxes],
              groundtruth_classes=eval_dict[
                  input_data_fields.groundtruth_classes],
              groundtruth_instance_masks=eval_dict[
                  input_data_fields.groundtruth_instance_masks],
              detection_scores=eval_dict[detection_fields.detection_scores],
              detection_classes=eval_dict[detection_fields.detection_classes],
              detection_masks=eval_dict[detection_fields.detection_masks]))
    else:
      raise ValueError('The only evaluation metrics supported are '
                       '"coco_detection_metrics" and "coco_mask_metrics". '
                       'Found {} in the evaluation metrics'.format(metric))

  return eval_metric_ops


