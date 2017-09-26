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

"""Common functions for repeatedly evaluating a checkpoint.
"""
import copy
import logging
import os
import time

import numpy as np
import tensorflow as tf

from object_detection.utils import label_map_util
from object_detection.utils import object_detection_evaluation
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
  summary_writer = tf.summary.FileWriter(summary_dir)
  for key in sorted(metrics):
    summary = tf.Summary(value=[
        tf.Summary.Value(tag=key, simple_value=metrics[key]),
    ])
    summary_writer.add_summary(summary, global_step)
    logging.info('%s: %f', key, metrics[key])
  summary_writer.close()
  logging.info('Metrics written to tf summary.')


def evaluate_detection_results_pascal_voc(result_lists,
                                          categories,
                                          label_id_offset=0,
                                          iou_thres=0.5,
                                          corloc_summary=False):
  """Computes Pascal VOC detection metrics given groundtruth and detections.

  This function computes Pascal VOC metrics. This function by default
  takes detections and groundtruth boxes encoded in result_lists and writes
  evaluation results to tf summaries which can be viewed on tensorboard.

  Args:
    result_lists: a dictionary holding lists of groundtruth and detection
      data corresponding to each image being evaluated.  The following keys
      are required:
        'image_id': a list of string ids
        'detection_boxes': a list of float32 numpy arrays of shape [N, 4]
        'detection_scores': a list of float32 numpy arrays of shape [N]
        'detection_classes': a list of int32 numpy arrays of shape [N]
        'groundtruth_boxes': a list of float32 numpy arrays of shape [M, 4]
        'groundtruth_classes': a list of int32 numpy arrays of shape [M]
      and the remaining fields below are optional:
        'difficult': a list of boolean arrays of shape [M] indicating the
          difficulty of groundtruth boxes. Some datasets like PASCAL VOC provide
          this information and it is used to remove difficult examples from eval
          in order to not penalize the models on them.
      Note that it is okay to have additional fields in result_lists --- they
      are simply ignored.
    categories: a list of dictionaries representing all possible categories.
      Each dict in this list has the following keys:
          'id': (required) an integer id uniquely identifying this category
          'name': (required) string representing category name
            e.g., 'cat', 'dog', 'pizza'
    label_id_offset: an integer offset for the label space.
    iou_thres: float determining the IoU threshold at which a box is considered
        correct. Defaults to the standard 0.5.
    corloc_summary: boolean. If True, also outputs CorLoc metrics.

  Returns:
    A dictionary of metric names to scalar values.

  Raises:
    ValueError: if the set of keys in result_lists is not a superset of the
      expected list of keys.  Unexpected keys are ignored.
    ValueError: if the lists in result_lists have inconsistent sizes.
  """
  # check for expected keys in result_lists
  expected_keys = [
      'detection_boxes', 'detection_scores', 'detection_classes', 'image_id'
  ]
  expected_keys += ['groundtruth_boxes', 'groundtruth_classes']
  if not set(expected_keys).issubset(set(result_lists.keys())):
    raise ValueError('result_lists does not have expected key set.')
  num_results = len(result_lists[expected_keys[0]])
  for key in expected_keys:
    if len(result_lists[key]) != num_results:
      raise ValueError('Inconsistent list sizes in result_lists')

  # Pascal VOC evaluator assumes foreground index starts from zero.
  categories = copy.deepcopy(categories)
  for idx in range(len(categories)):
    categories[idx]['id'] -= label_id_offset

  # num_classes (maybe encoded as categories)
  num_classes = max([cat['id'] for cat in categories]) + 1
  logging.info('Computing Pascal VOC metrics on results.')
  if all(image_id.isdigit() for image_id in result_lists['image_id']):
    image_ids = [int(image_id) for image_id in result_lists['image_id']]
  else:
    image_ids = range(num_results)

  evaluator = object_detection_evaluation.ObjectDetectionEvaluation(
      num_classes, matching_iou_threshold=iou_thres)

  difficult_lists = None
  if 'difficult' in result_lists and result_lists['difficult']:
    difficult_lists = result_lists['difficult']
  for idx, image_id in enumerate(image_ids):
    difficult = None
    if difficult_lists is not None and difficult_lists[idx].size:
      difficult = difficult_lists[idx].astype(np.bool)
    evaluator.add_single_ground_truth_image_info(
        image_id, result_lists['groundtruth_boxes'][idx],
        result_lists['groundtruth_classes'][idx] - label_id_offset,
        difficult)
    evaluator.add_single_detected_image_info(
        image_id, result_lists['detection_boxes'][idx],
        result_lists['detection_scores'][idx],
        result_lists['detection_classes'][idx] - label_id_offset)
  per_class_ap, mean_ap, _, _, per_class_corloc, mean_corloc = (
      evaluator.evaluate())

  metrics = {'Precision/mAP@{}IOU'.format(iou_thres): mean_ap}
  category_index = label_map_util.create_category_index(categories)
  for idx in range(per_class_ap.size):
    if idx in category_index:
      display_name = ('PerformanceByCategory/mAP@{}IOU/{}'
                      .format(iou_thres, category_index[idx]['name']))
      metrics[display_name] = per_class_ap[idx]

  if corloc_summary:
    metrics['CorLoc/CorLoc@{}IOU'.format(iou_thres)] = mean_corloc
    for idx in range(per_class_corloc.size):
      if idx in category_index:
        display_name = (
            'PerformanceByCategory/CorLoc@{}IOU/{}'.format(
                iou_thres, category_index[idx]['name']))
        metrics[display_name] = per_class_corloc[idx]
  return metrics


# TODO: Add tests.
def visualize_detection_results(result_dict,
                                tag,
                                global_step,
                                categories,
                                summary_dir='',
                                export_dir='',
                                agnostic_mode=False,
                                show_groundtruth=False,
                                min_score_thresh=.5,
                                max_num_predictions=20):
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
    min_score_thresh: minimum score threshold for a box to be visualized
    max_num_predictions: maximum number of detections to visualize
  Raises:
    ValueError: if result_dict does not contain the expected keys (i.e.,
      'original_image', 'detection_boxes', 'detection_scores',
      'detection_classes')
  """
  if not set([
      'original_image', 'detection_boxes', 'detection_scores',
      'detection_classes'
  ]).issubset(set(result_dict.keys())):
    raise ValueError('result_dict does not contain all expected keys.')
  if show_groundtruth and 'groundtruth_boxes' not in result_dict:
    raise ValueError('If show_groundtruth is enabled, result_dict must contain '
                     'groundtruth_boxes.')
  logging.info('Creating detection visualizations.')
  category_index = label_map_util.create_category_index(categories)

  image = np.squeeze(result_dict['original_image'], axis=0)
  detection_boxes = result_dict['detection_boxes']
  detection_scores = result_dict['detection_scores']
  detection_classes = np.int32((result_dict['detection_classes']))
  detection_keypoints = result_dict.get('detection_keypoints', None)
  detection_masks = result_dict.get('detection_masks', None)

  # Plot groundtruth underneath detections
  if show_groundtruth:
    groundtruth_boxes = result_dict['groundtruth_boxes']
    groundtruth_keypoints = result_dict.get('groundtruth_keypoints', None)
    vis_utils.visualize_boxes_and_labels_on_image_array(
        image,
        groundtruth_boxes,
        None,
        None,
        category_index,
        keypoints=groundtruth_keypoints,
        use_normalized_coordinates=False,
        max_boxes_to_draw=None)
  vis_utils.visualize_boxes_and_labels_on_image_array(
      image,
      detection_boxes,
      detection_classes,
      detection_scores,
      category_index,
      instance_masks=detection_masks,
      keypoints=detection_keypoints,
      use_normalized_coordinates=False,
      max_boxes_to_draw=max_num_predictions,
      min_score_thresh=min_score_thresh,
      agnostic_mode=agnostic_mode)

  if export_dir:
    export_path = os.path.join(export_dir, 'export-{}.png'.format(tag))
    vis_utils.save_image_array_as_png(image, export_path)

  summary = tf.Summary(value=[
      tf.Summary.Value(tag=tag, image=tf.Summary.Image(
          encoded_image_string=vis_utils.encode_image_array_as_png_str(
              image)))
  ])
  summary_writer = tf.summary.FileWriter(summary_dir)
  summary_writer.add_summary(summary, global_step)
  summary_writer.close()

  logging.info('Detection visualizations written to summary with tag %s.', tag)


# TODO: Add tests.
# TODO: Have an argument called `aggregated_processor_tensor_keys` that contains
# a whitelist of tensors used by the `aggregated_result_processor` instead of a
# blacklist. This will prevent us from inadvertently adding any evaluated
# tensors into the `results_list` data structure that are not needed by
# `aggregated_result_preprocessor`.
def run_checkpoint_once(tensor_dict,
                        update_op,
                        summary_dir,
                        aggregated_result_processor=None,
                        batch_processor=None,
                        checkpoint_dirs=None,
                        variables_to_restore=None,
                        restore_fn=None,
                        num_batches=1,
                        master='',
                        save_graph=False,
                        save_graph_dir='',
                        metric_names_to_values=None,
                        keys_to_exclude_from_results=()):
  """Evaluates both python metrics and tensorflow slim metrics.

  Python metrics are processed in batch by the aggregated_result_processor,
  while tensorflow slim metrics statistics are computed by running
  metric_names_to_updates tensors and aggregated using metric_names_to_values
  tensor.

  Args:
    tensor_dict: a dictionary holding tensors representing a batch of detections
      and corresponding groundtruth annotations.
    update_op: a tensorflow update op that will run for each batch along with
      the tensors in tensor_dict..
    summary_dir: a directory to write metrics summaries.
    aggregated_result_processor: a function taking one arguments:
      1. result_lists: a dictionary with keys matching those in tensor_dict
        and corresponding values being the list of results for each tensor
        in tensor_dict.  The length of each such list is num_batches.
    batch_processor: a function taking four arguments:
      1. tensor_dict: the same tensor_dict that is passed in as the first
        argument to this function.
      2. sess: a tensorflow session
      3. batch_index: an integer representing the index of the batch amongst
        all batches
      4. update_op: a tensorflow update op that will run for each batch.
      and returns result_dict, a dictionary of results for that batch.
      By default, batch_processor is None, which defaults to running:
        return sess.run(tensor_dict)
      To skip an image, it suffices to return an empty dictionary in place of
      result_dict.
    checkpoint_dirs: list of directories to load into an EnsembleModel. If it
      has only one directory, EnsembleModel will not be used -- a DetectionModel
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
    metric_names_to_values: A dictionary containing metric names to tensors
      which will be evaluated after processing all batches
      of [tensor_dict, update_op]. If any metrics depend on statistics computed
      during each batch ensure that `update_op` tensor has a control dependency
      on the update ops that compute the statistics.
    keys_to_exclude_from_results: keys in tensor_dict that will be excluded
      from results_list. Note that the tensors corresponding to these keys will
      still be evaluated for each batch, but won't be added to results_list.

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

  valid_keys = list(set(tensor_dict.keys()) - set(keys_to_exclude_from_results))
  result_lists = {key: [] for key in valid_keys}
  counters = {'skipped': 0, 'success': 0}
  other_metrics = None
  with tf.contrib.slim.queues.QueueRunners(sess):
    try:
      for batch in range(int(num_batches)):
        if (batch + 1) % 100 == 0:
          logging.info('Running eval ops batch %d/%d', batch + 1, num_batches)
        if not batch_processor:
          try:
            (result_dict, _) = sess.run([tensor_dict, update_op])
            counters['success'] += 1
          except tf.errors.InvalidArgumentError:
            logging.info('Skipping image')
            counters['skipped'] += 1
            result_dict = {}
        else:
          result_dict = batch_processor(
              tensor_dict, sess, batch, counters, update_op)
        for key in result_dict:
          if key in valid_keys:
            result_lists[key].append(result_dict[key])
      if metric_names_to_values is not None:
        other_metrics = sess.run(metric_names_to_values)
      logging.info('Running eval batches done.')
    except tf.errors.OutOfRangeError:
      logging.info('Done evaluating -- epoch limit reached')
    finally:
      # When done, ask the threads to stop.
      metrics = aggregated_result_processor(result_lists)
      if other_metrics is not None:
        metrics.update(other_metrics)
      global_step = tf.train.global_step(sess, slim.get_global_step())
      write_metrics(metrics, global_step, summary_dir)
      logging.info('# success: %d', counters['success'])
      logging.info('# skipped: %d', counters['skipped'])
  sess.close()


# TODO: Add tests.
def repeated_checkpoint_run(tensor_dict,
                            update_op,
                            summary_dir,
                            aggregated_result_processor=None,
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
                            metric_names_to_values=None,
                            keys_to_exclude_from_results=()):
  """Periodically evaluates desired tensors using checkpoint_dirs or restore_fn.

  This function repeatedly loads a checkpoint and evaluates a desired
  set of tensors (provided by tensor_dict) and hands the resulting numpy
  arrays to a function result_processor which can be used to further
  process/save/visualize the results.

  Args:
    tensor_dict: a dictionary holding tensors representing a batch of detections
      and corresponding groundtruth annotations.
    update_op: a tensorflow update op that will run for each batch along with
      the tensors in tensor_dict.
    summary_dir: a directory to write metrics summaries.
    aggregated_result_processor: a function taking one argument:
      1. result_lists: a dictionary with keys matching those in tensor_dict
        and corresponding values being the list of results for each tensor
        in tensor_dict.  The length of each such list is num_batches.
    batch_processor: a function taking three arguments:
      1. tensor_dict: the same tensor_dict that is passed in as the first
        argument to this function.
      2. sess: a tensorflow session
      3. batch_index: an integer representing the index of the batch amongst
        all batches
      4. update_op: a tensorflow update op that will run for each batch.
      and returns result_dict, a dictionary of results for that batch.
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
    metric_names_to_values: A dictionary containing metric names to tensors
      which will be evaluated after processing all batches
      of [tensor_dict, update_op]. If any metrics depend on statistics computed
      during each batch ensure that `update_op` tensor has a control dependency
      on the update ops that compute the statistics.
    keys_to_exclude_from_results: keys in tensor_dict that will be excluded
      from results_list. Note that the tensors corresponding to these keys will
      still be evaluated for each batch, but won't be added to results_list.

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
    logging.info('Starting evaluation at ' + time.strftime('%Y-%m-%d-%H:%M:%S',
                                                           time.gmtime()))
    model_path = tf.train.latest_checkpoint(checkpoint_dirs[0])
    if not model_path:
      logging.info('No model found in %s. Will try again in %d seconds',
                   checkpoint_dirs[0], eval_interval_secs)
    elif model_path == last_evaluated_model_path:
      logging.info('Found already evaluated checkpoint. Will try again in %d '
                   'seconds', eval_interval_secs)
    else:
      last_evaluated_model_path = model_path
      run_checkpoint_once(tensor_dict, update_op, summary_dir,
                          aggregated_result_processor,
                          batch_processor, checkpoint_dirs,
                          variables_to_restore, restore_fn, num_batches, master,
                          save_graph, save_graph_dir, metric_names_to_values,
                          keys_to_exclude_from_results)
    number_of_evaluations += 1

    if (max_number_of_evaluations and
        number_of_evaluations >= max_number_of_evaluations):
      logging.info('Finished evaluation!')
      break
    time_to_next_eval = start + eval_interval_secs - time.time()
    if time_to_next_eval > 0:
      time.sleep(time_to_next_eval)
