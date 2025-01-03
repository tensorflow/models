# Copyright 2018 The TensorFlow Authors. All Rights Reserved.
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

import tensorflow.compat.v1 as tf
from tensorflow.contrib import tfprof as contrib_tfprof
from lstm_object_detection.metrics import coco_evaluation_all_frames
from object_detection import eval_util
from object_detection.core import prefetcher
from object_detection.core import standard_fields as fields
from object_detection.metrics import coco_evaluation
from object_detection.utils import object_detection_evaluation


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
    'open_images_detection_metrics':
        object_detection_evaluation.OpenImagesDetectionEvaluator,
    'coco_detection_metrics':
        coco_evaluation.CocoDetectionEvaluator,
    'coco_mask_metrics':
        coco_evaluation.CocoMaskEvaluator,
    'coco_evaluation_all_frames':
        coco_evaluation_all_frames.CocoEvaluationAllFrames,
}

EVAL_DEFAULT_METRIC = 'pascal_voc_detection_metrics'


def _create_detection_op(model, input_dict, batch):
  """Create detection ops.

  Args:
    model: model to perform predictions with.
    input_dict: A dict holds input data.
    batch: batch size for evaluation.

  Returns:
    Detection tensor ops.
  """
  video_tensor = tf.stack(list(input_dict[fields.InputDataFields.image]))
  preprocessed_video, true_image_shapes = model.preprocess(
      tf.to_float(video_tensor))
  if batch is not None:
    prediction_dict = model.predict(preprocessed_video, true_image_shapes,
                                    batch)
  else:
    prediction_dict = model.predict(preprocessed_video, true_image_shapes)

  return model.postprocess(prediction_dict, true_image_shapes)


def _extract_prediction_tensors(model,
                                create_input_dict_fn,
                                ignore_groundtruth=False):
  """Restores the model in a tensorflow session.

  Args:
    model: model to perform predictions with.
    create_input_dict_fn: function to create input tensor dictionaries.
    ignore_groundtruth: whether groundtruth should be ignored.


  Returns:
    tensor_dict: A tensor dictionary with evaluations.
  """
  input_dict = create_input_dict_fn()
  batch = None
  if 'batch' in input_dict:
    batch = input_dict.pop('batch')
  else:
    prefetch_queue = prefetcher.prefetch(input_dict, capacity=500)
    input_dict = prefetch_queue.dequeue()
    # consistent format for images and videos
    for key, value in input_dict.iteritems():
      input_dict[key] = (value,)

  detections = _create_detection_op(model, input_dict, batch)

  # Print out anaylsis of the model.
  contrib_tfprof.model_analyzer.print_model_analysis(
      tf.get_default_graph(),
      tfprof_options=contrib_tfprof.model_analyzer
      .TRAINABLE_VARS_PARAMS_STAT_OPTIONS)
  contrib_tfprof.model_analyzer.print_model_analysis(
      tf.get_default_graph(),
      tfprof_options=contrib_tfprof.model_analyzer.FLOAT_OPS_OPTIONS)

  num_frames = len(input_dict[fields.InputDataFields.image])
  ret = []
  for i in range(num_frames):
    original_image = tf.expand_dims(input_dict[fields.InputDataFields.image][i],
                                    0)
    groundtruth = None
    if not ignore_groundtruth:
      groundtruth = {
          fields.InputDataFields.groundtruth_boxes:
              input_dict[fields.InputDataFields.groundtruth_boxes][i],
          fields.InputDataFields.groundtruth_classes:
              input_dict[fields.InputDataFields.groundtruth_classes][i],
      }
      optional_keys = (
          fields.InputDataFields.groundtruth_area,
          fields.InputDataFields.groundtruth_is_crowd,
          fields.InputDataFields.groundtruth_difficult,
          fields.InputDataFields.groundtruth_group_of,
      )
      for opt_key in optional_keys:
        if opt_key in input_dict:
          groundtruth[opt_key] = input_dict[opt_key][i]
      if fields.DetectionResultFields.detection_masks in detections:
        groundtruth[fields.InputDataFields.groundtruth_instance_masks] = (
            input_dict[fields.InputDataFields.groundtruth_instance_masks][i])

    detections_frame = {
        key: tf.expand_dims(value[i], 0)
        for key, value in detections.iteritems()
    }

    source_id = (
        batch.key[0] if batch is not None else
        input_dict[fields.InputDataFields.source_id][i])
    ret.append(
        eval_util.result_dict_for_single_example(
            original_image,
            source_id,
            detections_frame,
            groundtruth,
            class_agnostic=(fields.DetectionResultFields.detection_classes
                            not in detections),
            scale_to_absolute=True))
  return ret


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
    else:
      evaluators_list.append(
          EVAL_METRICS_CLASS_DICT[eval_metric_fn_key](categories=categories))
  return evaluators_list


def evaluate(create_input_dict_fn,
             create_model_fn,
             eval_config,
             categories,
             checkpoint_dir,
             eval_dir,
             graph_hook_fn=None):
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

  Returns:
    metrics: A dictionary containing metric names and values from the latest
      run.
  """

  model = create_model_fn()

  if eval_config.ignore_groundtruth and not eval_config.export_path:
    tf.logging.fatal('If ignore_groundtruth=True then an export_path is '
                  'required. Aborting!!!')

  tensor_dicts = _extract_prediction_tensors(
      model=model,
      create_input_dict_fn=create_input_dict_fn,
      ignore_groundtruth=eval_config.ignore_groundtruth)

  def _process_batch(tensor_dicts,
                     sess,
                     batch_index,
                     counters,
                     losses_dict=None):
    """Evaluates tensors in tensor_dicts, visualizing the first K examples.

    This function calls sess.run on tensor_dicts, evaluating the original_image
    tensor only on the first K examples and visualizing detections overlaid
    on this original_image.

    Args:
      tensor_dicts: a dictionary of tensors
      sess: tensorflow session
      batch_index: the index of the batch amongst all batches in the run.
      counters: a dictionary holding 'success' and 'skipped' fields which can
        be updated to keep track of number of successful and failed runs,
        respectively.  If these fields are not updated, then the success/skipped
        counter values shown at the end of evaluation will be incorrect.
      losses_dict: Optional dictonary of scalar loss tensors. Necessary only
        for matching function signiture in third_party eval_util.py.

    Returns:
      result_dict: a dictionary of numpy arrays
      result_losses_dict: a dictionary of scalar losses. This is empty if input
        losses_dict is None. Necessary only for matching function signiture in
        third_party eval_util.py.
    """
    if batch_index % 10 == 0:
      tf.logging.info('Running eval ops batch %d', batch_index)
    if not losses_dict:
      losses_dict = {}
    try:
      result_dicts, result_losses_dict = sess.run([tensor_dicts, losses_dict])
      counters['success'] += 1
    except tf.errors.InvalidArgumentError:
      tf.logging.info('Skipping image')
      counters['skipped'] += 1
      return {}
    num_images = len(tensor_dicts)
    for i in range(num_images):
      result_dict = result_dicts[i]
      global_step = tf.train.global_step(sess, tf.train.get_global_step())
      tag = 'image-%d' % (batch_index * num_images + i)
      if batch_index < eval_config.num_visualizations / num_images:
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
    if num_images > 1:
      return result_dicts, result_losses_dict
    else:
      return result_dicts[0], result_losses_dict

  variables_to_restore = tf.global_variables()
  global_step = tf.train.get_or_create_global_step()
  variables_to_restore.append(global_step)

  if graph_hook_fn:
    graph_hook_fn()

  if eval_config.use_moving_averages:
    variable_averages = tf.train.ExponentialMovingAverage(0.0)
    variables_to_restore = variable_averages.variables_to_restore()
    for key in variables_to_restore.keys():
      if 'moving_mean' in key:
        variables_to_restore[key.replace(
            'moving_mean', 'moving_mean/ExponentialMovingAverage')] = (
                variables_to_restore[key])
        del variables_to_restore[key]
      if 'moving_variance' in key:
        variables_to_restore[key.replace(
            'moving_variance', 'moving_variance/ExponentialMovingAverage')] = (
                variables_to_restore[key])
        del variables_to_restore[key]

  saver = tf.train.Saver(variables_to_restore)

  def _restore_latest_checkpoint(sess):
    latest_checkpoint = tf.train.latest_checkpoint(checkpoint_dir)
    saver.restore(sess, latest_checkpoint)

  metrics = eval_util.repeated_checkpoint_run(
      tensor_dict=tensor_dicts,
      summary_dir=eval_dir,
      evaluators=get_evaluators(eval_config, categories),
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
      save_graph_dir=(eval_dir if eval_config.save_graph else ''))

  return metrics
