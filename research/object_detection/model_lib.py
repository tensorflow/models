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
r"""Constructs model, inputs, and training environment."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import functools
import os

import tensorflow as tf

from object_detection import eval_util
from object_detection import inputs
from object_detection.builders import graph_rewriter_builder
from object_detection.builders import model_builder
from object_detection.builders import optimizer_builder
from object_detection.core import standard_fields as fields
from object_detection.utils import config_util
from object_detection.utils import label_map_util
from object_detection.utils import shape_utils
from object_detection.utils import variables_helper
from object_detection.utils import visualization_utils as vis_utils

# A map of names to methods that help build the model.
MODEL_BUILD_UTIL_MAP = {
    'get_configs_from_pipeline_file':
        config_util.get_configs_from_pipeline_file,
    'create_pipeline_proto_from_configs':
        config_util.create_pipeline_proto_from_configs,
    'merge_external_params_with_configs':
        config_util.merge_external_params_with_configs,
    'create_train_input_fn': inputs.create_train_input_fn,
    'create_eval_input_fn': inputs.create_eval_input_fn,
    'create_predict_input_fn': inputs.create_predict_input_fn,
}


def _prepare_groundtruth_for_eval(detection_model, class_agnostic):
  """Extracts groundtruth data from detection_model and prepares it for eval.

  Args:
    detection_model: A `DetectionModel` object.
    class_agnostic: Whether the detections are class_agnostic.

  Returns:
    A tuple of:
    groundtruth: Dictionary with the following fields:
      'groundtruth_boxes': [num_boxes, 4] float32 tensor of boxes, in
        normalized coordinates.
      'groundtruth_classes': [num_boxes] int64 tensor of 1-indexed classes.
      'groundtruth_masks': 3D float32 tensor of instance masks (if provided in
        groundtruth)
      'groundtruth_is_crowd': [num_boxes] bool tensor indicating is_crowd
        annotations (if provided in groundtruth).
    class_agnostic: Boolean indicating whether detections are class agnostic.
  """
  input_data_fields = fields.InputDataFields()
  groundtruth_boxes = detection_model.groundtruth_lists(
      fields.BoxListFields.boxes)[0]
  # For class-agnostic models, groundtruth one-hot encodings collapse to all
  # ones.
  if class_agnostic:
    groundtruth_boxes_shape = tf.shape(groundtruth_boxes)
    groundtruth_classes_one_hot = tf.ones([groundtruth_boxes_shape[0], 1])
  else:
    groundtruth_classes_one_hot = detection_model.groundtruth_lists(
        fields.BoxListFields.classes)[0]
  label_id_offset = 1  # Applying label id offset (b/63711816)
  groundtruth_classes = (
      tf.argmax(groundtruth_classes_one_hot, axis=1) + label_id_offset)
  groundtruth = {
      input_data_fields.groundtruth_boxes: groundtruth_boxes,
      input_data_fields.groundtruth_classes: groundtruth_classes
  }
  if detection_model.groundtruth_has_field(fields.BoxListFields.masks):
    groundtruth[input_data_fields.groundtruth_instance_masks] = (
        detection_model.groundtruth_lists(fields.BoxListFields.masks)[0])
  if detection_model.groundtruth_has_field(fields.BoxListFields.is_crowd):
    groundtruth[input_data_fields.groundtruth_is_crowd] = (
        detection_model.groundtruth_lists(fields.BoxListFields.is_crowd)[0])
  return groundtruth


def unstack_batch(tensor_dict, unpad_groundtruth_tensors=True):
  """Unstacks all tensors in `tensor_dict` along 0th dimension.

  Unstacks tensor from the tensor dict along 0th dimension and returns a
  tensor_dict containing values that are lists of unstacked tensors.

  Tensors in the `tensor_dict` are expected to be of one of the three shapes:
  1. [batch_size]
  2. [batch_size, height, width, channels]
  3. [batch_size, num_boxes, d1, d2, ... dn]

  When unpad_groundtruth_tensors is set to true, unstacked tensors of form 3
  above are sliced along the `num_boxes` dimension using the value in tensor
  field.InputDataFields.num_groundtruth_boxes.

  Note that this function has a static list of input data fields and has to be
  kept in sync with the InputDataFields defined in core/standard_fields.py

  Args:
    tensor_dict: A dictionary of batched groundtruth tensors.
    unpad_groundtruth_tensors: Whether to remove padding along `num_boxes`
      dimension of the groundtruth tensors.

  Returns:
    A dictionary where the keys are from fields.InputDataFields and values are
    a list of unstacked (optionally unpadded) tensors.

  Raises:
    ValueError: If unpad_tensors is True and `tensor_dict` does not contain
      `num_groundtruth_boxes` tensor.
  """
  unbatched_tensor_dict = {key: tf.unstack(tensor)
                           for key, tensor in tensor_dict.items()}
  if unpad_groundtruth_tensors:
    if (fields.InputDataFields.num_groundtruth_boxes not in
        unbatched_tensor_dict):
      raise ValueError('`num_groundtruth_boxes` not found in tensor_dict. '
                       'Keys available: {}'.format(
                           unbatched_tensor_dict.keys()))
    unbatched_unpadded_tensor_dict = {}
    unpad_keys = set([
        # List of input data fields that are padded along the num_boxes
        # dimension. This list has to be kept in sync with InputDataFields in
        # standard_fields.py.
        fields.InputDataFields.groundtruth_instance_masks,
        fields.InputDataFields.groundtruth_classes,
        fields.InputDataFields.groundtruth_boxes,
        fields.InputDataFields.groundtruth_keypoints,
        fields.InputDataFields.groundtruth_group_of,
        fields.InputDataFields.groundtruth_difficult,
        fields.InputDataFields.groundtruth_is_crowd,
        fields.InputDataFields.groundtruth_area,
        fields.InputDataFields.groundtruth_weights
    ]).intersection(set(unbatched_tensor_dict.keys()))

    for key in unpad_keys:
      unpadded_tensor_list = []
      for num_gt, padded_tensor in zip(
          unbatched_tensor_dict[fields.InputDataFields.num_groundtruth_boxes],
          unbatched_tensor_dict[key]):
        tensor_shape = shape_utils.combined_static_and_dynamic_shape(
            padded_tensor)
        slice_begin = tf.zeros([len(tensor_shape)], dtype=tf.int32)
        slice_size = tf.stack(
            [num_gt] + [-1 if dim is None else dim for dim in tensor_shape[1:]])
        unpadded_tensor = tf.slice(padded_tensor, slice_begin, slice_size)
        unpadded_tensor_list.append(unpadded_tensor)
      unbatched_unpadded_tensor_dict[key] = unpadded_tensor_list
    unbatched_tensor_dict.update(unbatched_unpadded_tensor_dict)

  return unbatched_tensor_dict


def create_model_fn(detection_model_fn, configs, hparams, use_tpu=False):
  """Creates a model function for `Estimator`.

  Args:
    detection_model_fn: Function that returns a `DetectionModel` instance.
    configs: Dictionary of pipeline config objects.
    hparams: `HParams` object.
    use_tpu: Boolean indicating whether model should be constructed for
        use on TPU.

  Returns:
    `model_fn` for `Estimator`.
  """
  train_config = configs['train_config']
  eval_input_config = configs['eval_input_config']
  eval_config = configs['eval_config']

  def model_fn(features, labels, mode, params=None):
    """Constructs the object detection model.

    Args:
      features: Dictionary of feature tensors, returned from `input_fn`.
      labels: Dictionary of groundtruth tensors if mode is TRAIN or EVAL,
        otherwise None.
      mode: Mode key from tf.estimator.ModeKeys.
      params: Parameter dictionary passed from the estimator.

    Returns:
      An `EstimatorSpec` that encapsulates the model and its serving
        configurations.
    """
    params = params or {}
    total_loss, train_op, detections, export_outputs = None, None, None, None
    is_training = mode == tf.estimator.ModeKeys.TRAIN
    detection_model = detection_model_fn(is_training=is_training,
                                         add_summaries=(not use_tpu))
    scaffold_fn = None

    if mode == tf.estimator.ModeKeys.TRAIN:
      labels = unstack_batch(
          labels,
          unpad_groundtruth_tensors=train_config.unpad_groundtruth_tensors)
    elif mode == tf.estimator.ModeKeys.EVAL:
      # For evaling on train data, it is necessary to check whether groundtruth
      # must be unpadded.
      boxes_shape = (
          labels[fields.InputDataFields.groundtruth_boxes].get_shape()
          .as_list())
      unpad_groundtruth_tensors = True if boxes_shape[1] is not None else False
      labels = unstack_batch(
          labels, unpad_groundtruth_tensors=unpad_groundtruth_tensors)

    if mode in (tf.estimator.ModeKeys.TRAIN, tf.estimator.ModeKeys.EVAL):
      gt_boxes_list = labels[fields.InputDataFields.groundtruth_boxes]
      gt_classes_list = labels[fields.InputDataFields.groundtruth_classes]
      gt_masks_list = None
      if fields.InputDataFields.groundtruth_instance_masks in labels:
        gt_masks_list = labels[
            fields.InputDataFields.groundtruth_instance_masks]
      gt_keypoints_list = None
      if fields.InputDataFields.groundtruth_keypoints in labels:
        gt_keypoints_list = labels[fields.InputDataFields.groundtruth_keypoints]
      if fields.InputDataFields.groundtruth_is_crowd in labels:
        gt_is_crowd_list = labels[fields.InputDataFields.groundtruth_is_crowd]
      detection_model.provide_groundtruth(
          groundtruth_boxes_list=gt_boxes_list,
          groundtruth_classes_list=gt_classes_list,
          groundtruth_masks_list=gt_masks_list,
          groundtruth_keypoints_list=gt_keypoints_list,
          groundtruth_weights_list=labels[
              fields.InputDataFields.groundtruth_weights],
          groundtruth_is_crowd_list=gt_is_crowd_list)

    preprocessed_images = features[fields.InputDataFields.image]
    prediction_dict = detection_model.predict(
        preprocessed_images, features[fields.InputDataFields.true_image_shape])
    detections = detection_model.postprocess(
        prediction_dict, features[fields.InputDataFields.true_image_shape])

    if mode == tf.estimator.ModeKeys.TRAIN:
      if train_config.fine_tune_checkpoint and hparams.load_pretrained:
        if not train_config.fine_tune_checkpoint_type:
          # train_config.from_detection_checkpoint field is deprecated. For
          # backward compatibility, set train_config.fine_tune_checkpoint_type
          # based on train_config.from_detection_checkpoint.
          if train_config.from_detection_checkpoint:
            train_config.fine_tune_checkpoint_type = 'detection'
          else:
            train_config.fine_tune_checkpoint_type = 'classification'
        asg_map = detection_model.restore_map(
            fine_tune_checkpoint_type=train_config.fine_tune_checkpoint_type,
            load_all_detection_checkpoint_vars=(
                train_config.load_all_detection_checkpoint_vars))
        available_var_map = (
            variables_helper.get_variables_available_in_checkpoint(
                asg_map, train_config.fine_tune_checkpoint,
                include_global_step=False))
        if use_tpu:
          def tpu_scaffold():
            tf.train.init_from_checkpoint(train_config.fine_tune_checkpoint,
                                          available_var_map)
            return tf.train.Scaffold()
          scaffold_fn = tpu_scaffold
        else:
          tf.train.init_from_checkpoint(train_config.fine_tune_checkpoint,
                                        available_var_map)

    if mode in (tf.estimator.ModeKeys.TRAIN, tf.estimator.ModeKeys.EVAL):
      losses_dict = detection_model.loss(
          prediction_dict, features[fields.InputDataFields.true_image_shape])
      losses = [loss_tensor for loss_tensor in losses_dict.itervalues()]
      if train_config.add_regularization_loss:
        regularization_losses = tf.get_collection(
            tf.GraphKeys.REGULARIZATION_LOSSES)
        if regularization_losses:
          regularization_loss = tf.add_n(regularization_losses,
                                         name='regularization_loss')
          losses.append(regularization_loss)
          losses_dict['Loss/regularization_loss'] = regularization_loss
      total_loss = tf.add_n(losses, name='total_loss')
      losses_dict['Loss/total_loss'] = total_loss

      if 'graph_rewriter_config' in configs:
        graph_rewriter_fn = graph_rewriter_builder.build(
            configs['graph_rewriter_config'], is_training=is_training)
        graph_rewriter_fn()

      # TODO(rathodv): Stop creating optimizer summary vars in EVAL mode once we
      # can write learning rate summaries on TPU without host calls.
      global_step = tf.train.get_or_create_global_step()
      training_optimizer, optimizer_summary_vars = optimizer_builder.build(
          train_config.optimizer)

    if mode == tf.estimator.ModeKeys.TRAIN:
      if use_tpu:
        training_optimizer = tf.contrib.tpu.CrossShardOptimizer(
            training_optimizer)

      # Optionally freeze some layers by setting their gradients to be zero.
      trainable_variables = None
      if train_config.freeze_variables:
        trainable_variables = tf.contrib.framework.filter_variables(
            tf.trainable_variables(),
            exclude_patterns=train_config.freeze_variables)

      clip_gradients_value = None
      if train_config.gradient_clipping_by_norm > 0:
        clip_gradients_value = train_config.gradient_clipping_by_norm

      if not use_tpu:
        for var in optimizer_summary_vars:
          tf.summary.scalar(var.op.name, var)
      summaries = [] if use_tpu else None
      train_op = tf.contrib.layers.optimize_loss(
          loss=total_loss,
          global_step=global_step,
          learning_rate=None,
          clip_gradients=clip_gradients_value,
          optimizer=training_optimizer,
          variables=trainable_variables,
          summaries=summaries,
          name='')  # Preventing scope prefix on all variables.

    if mode == tf.estimator.ModeKeys.PREDICT:
      export_outputs = {
          tf.saved_model.signature_constants.PREDICT_METHOD_NAME:
              tf.estimator.export.PredictOutput(detections)
      }

    eval_metric_ops = None
    scaffold = None
    if mode == tf.estimator.ModeKeys.EVAL:
      class_agnostic = (fields.DetectionResultFields.detection_classes
                        not in detections)
      groundtruth = _prepare_groundtruth_for_eval(
          detection_model, class_agnostic)
      use_original_images = fields.InputDataFields.original_image in features
      eval_images = (
          features[fields.InputDataFields.original_image] if use_original_images
          else features[fields.InputDataFields.image])
      eval_dict = eval_util.result_dict_for_single_example(
          eval_images[0:1],
          features[inputs.HASH_KEY][0],
          detections,
          groundtruth,
          class_agnostic=class_agnostic,
          scale_to_absolute=True)

      if class_agnostic:
        category_index = label_map_util.create_class_agnostic_category_index()
      else:
        category_index = label_map_util.create_category_index_from_labelmap(
            eval_input_config.label_map_path)
      img_summary = None
      if not use_tpu and use_original_images:
        detection_and_groundtruth = (
            vis_utils.draw_side_by_side_evaluation_image(
                eval_dict, category_index, max_boxes_to_draw=20,
                min_score_thresh=0.2,
                use_normalized_coordinates=False))
        img_summary = tf.summary.image('Detections_Left_Groundtruth_Right',
                                       detection_and_groundtruth)

      # Eval metrics on a single example.
      eval_metrics = eval_config.metrics_set
      if not eval_metrics:
        eval_metrics = ['coco_detection_metrics']
      eval_metric_ops = eval_util.get_eval_metric_ops_for_evaluators(
          eval_metrics,
          category_index.values(),
          eval_dict,
          include_metrics_per_category=eval_config.include_metrics_per_category)
      for loss_key, loss_tensor in iter(losses_dict.items()):
        eval_metric_ops[loss_key] = tf.metrics.mean(loss_tensor)
      for var in optimizer_summary_vars:
        eval_metric_ops[var.op.name] = (var, tf.no_op())
      if img_summary is not None:
        eval_metric_ops['Detections_Left_Groundtruth_Right'] = (
            img_summary, tf.no_op())
      eval_metric_ops = {str(k): v for k, v in eval_metric_ops.iteritems()}

      if eval_config.use_moving_averages:
        variable_averages = tf.train.ExponentialMovingAverage(0.0)
        variables_to_restore = variable_averages.variables_to_restore()
        keep_checkpoint_every_n_hours = (
            train_config.keep_checkpoint_every_n_hours)
        saver = tf.train.Saver(
            variables_to_restore,
            keep_checkpoint_every_n_hours=keep_checkpoint_every_n_hours)
        scaffold = tf.train.Scaffold(saver=saver)

    if use_tpu:
      return tf.contrib.tpu.TPUEstimatorSpec(
          mode=mode,
          scaffold_fn=scaffold_fn,
          predictions=detections,
          loss=total_loss,
          train_op=train_op,
          eval_metrics=eval_metric_ops,
          export_outputs=export_outputs)
    else:
      return tf.estimator.EstimatorSpec(
          mode=mode,
          predictions=detections,
          loss=total_loss,
          train_op=train_op,
          eval_metric_ops=eval_metric_ops,
          export_outputs=export_outputs,
          scaffold=scaffold)

  return model_fn


def create_estimator_and_inputs(run_config,
                                hparams,
                                pipeline_config_path,
                                train_steps=None,
                                eval_steps=None,
                                model_fn_creator=create_model_fn,
                                use_tpu_estimator=False,
                                use_tpu=False,
                                num_shards=1,
                                params=None,
                                **kwargs):
  """Creates `Estimator`, input functions, and steps.

  Args:
    run_config: A `RunConfig`.
    hparams: A `HParams`.
    pipeline_config_path: A path to a pipeline config file.
    train_steps: Number of training steps. If None, the number of training steps
      is set from the `TrainConfig` proto.
    eval_steps: Number of evaluation steps per evaluation cycle. If None, the
      number of evaluation steps is set from the `EvalConfig` proto.
    model_fn_creator: A function that creates a `model_fn` for `Estimator`.
      Follows the signature:

      * Args:
        * `detection_model_fn`: Function that returns `DetectionModel` instance.
        * `configs`: Dictionary of pipeline config objects.
        * `hparams`: `HParams` object.
      * Returns:
        `model_fn` for `Estimator`.

    use_tpu_estimator: Whether a `TPUEstimator` should be returned. If False,
      an `Estimator` will be returned.
    use_tpu: Boolean, whether training and evaluation should run on TPU. Only
      used if `use_tpu_estimator` is True.
    num_shards: Number of shards (TPU cores). Only used if `use_tpu_estimator`
      is True.
    params: Parameter dictionary passed from the estimator. Only used if
      `use_tpu_estimator` is True.
    **kwargs: Additional keyword arguments for configuration override.

  Returns:
    A dictionary with the following fields:
    'estimator': An `Estimator` or `TPUEstimator`.
    'train_input_fn': A training input function.
    'eval_input_fn': An evaluation input function.
    'eval_on_train_input_fn': An evaluation-on-train input function.
    'predict_input_fn': A prediction input function.
    'train_steps': Number of training steps. Either directly from input or from
      configuration.
    'eval_steps': Number of evaluation steps. Either directly from input or from
      configuration.
  """
  get_configs_from_pipeline_file = MODEL_BUILD_UTIL_MAP[
      'get_configs_from_pipeline_file']
  merge_external_params_with_configs = MODEL_BUILD_UTIL_MAP[
      'merge_external_params_with_configs']
  create_pipeline_proto_from_configs = MODEL_BUILD_UTIL_MAP[
      'create_pipeline_proto_from_configs']
  create_train_input_fn = MODEL_BUILD_UTIL_MAP['create_train_input_fn']
  create_eval_input_fn = MODEL_BUILD_UTIL_MAP['create_eval_input_fn']
  create_predict_input_fn = MODEL_BUILD_UTIL_MAP['create_predict_input_fn']

  configs = get_configs_from_pipeline_file(pipeline_config_path)
  configs = merge_external_params_with_configs(
      configs,
      hparams,
      train_steps=train_steps,
      eval_steps=eval_steps,
      **kwargs)
  model_config = configs['model']
  train_config = configs['train_config']
  train_input_config = configs['train_input_config']
  eval_config = configs['eval_config']
  eval_input_config = configs['eval_input_config']

  if train_steps is None:
    train_steps = configs['train_config'].num_steps

  if eval_steps is None:
    eval_steps = configs['eval_config'].num_examples

  detection_model_fn = functools.partial(
      model_builder.build, model_config=model_config)

  # Create the input functions for TRAIN/EVAL/PREDICT.
  train_input_fn = create_train_input_fn(
      train_config=train_config,
      train_input_config=train_input_config,
      model_config=model_config)
  eval_input_fn = create_eval_input_fn(
      eval_config=eval_config,
      eval_input_config=eval_input_config,
      model_config=model_config)
  eval_on_train_input_fn = create_eval_input_fn(
      eval_config=eval_config,
      eval_input_config=train_input_config,
      model_config=model_config)
  predict_input_fn = create_predict_input_fn(model_config=model_config)

  model_fn = model_fn_creator(detection_model_fn, configs, hparams, use_tpu)
  if use_tpu_estimator:
    estimator = tf.contrib.tpu.TPUEstimator(
        model_fn=model_fn,
        train_batch_size=train_config.batch_size,
        # For each core, only batch size 1 is supported for eval.
        eval_batch_size=num_shards * 1 if use_tpu else 1,
        use_tpu=use_tpu,
        config=run_config,
        params=params if params else {})
  else:
    estimator = tf.estimator.Estimator(model_fn=model_fn, config=run_config)

  # Write the as-run pipeline config to disk.
  if run_config.is_chief:
    pipeline_config_final = create_pipeline_proto_from_configs(
        configs)
    config_util.save_pipeline_config(pipeline_config_final, estimator.model_dir)

  return dict(
      estimator=estimator,
      train_input_fn=train_input_fn,
      eval_input_fn=eval_input_fn,
      eval_on_train_input_fn=eval_on_train_input_fn,
      predict_input_fn=predict_input_fn,
      train_steps=train_steps,
      eval_steps=eval_steps)


def create_train_and_eval_specs(train_input_fn,
                                eval_input_fn,
                                eval_on_train_input_fn,
                                predict_input_fn,
                                train_steps,
                                eval_steps,
                                eval_on_train_data=False,
                                eval_on_train_steps=None,
                                final_exporter_name='Servo',
                                eval_spec_name='eval'):
  """Creates a `TrainSpec` and `EvalSpec`s.

  Args:
    train_input_fn: Function that produces features and labels on train data.
    eval_input_fn: Function that produces features and labels on eval data.
    eval_on_train_input_fn: Function that produces features and labels for
      evaluation on train data.
    predict_input_fn: Function that produces features for inference.
    train_steps: Number of training steps.
    eval_steps: Number of eval steps.
    eval_on_train_data: Whether to evaluate model on training data. Default is
      False.
    eval_on_train_steps: Number of eval steps for training data. If not given,
      uses eval_steps.
    final_exporter_name: String name given to `FinalExporter`.
    eval_spec_name: String name given to main `EvalSpec`.

  Returns:
    Tuple of `TrainSpec` and list of `EvalSpecs`. The first `EvalSpec` is for
    evaluation data. If `eval_on_train_data` is True, the second `EvalSpec` in
    the list will correspond to training data.
  """

  exporter = tf.estimator.FinalExporter(
      name=final_exporter_name, serving_input_receiver_fn=predict_input_fn)

  train_spec = tf.estimator.TrainSpec(
      input_fn=train_input_fn, max_steps=train_steps)

  eval_specs = [
      tf.estimator.EvalSpec(
          name=eval_spec_name,
          input_fn=eval_input_fn,
          steps=eval_steps,
          exporters=exporter)
  ]

  if eval_on_train_data:
    eval_specs.append(
        tf.estimator.EvalSpec(
            name='eval_on_train', input_fn=eval_on_train_input_fn,
            steps=eval_on_train_steps or eval_steps))

  return train_spec, eval_specs


def continuous_eval(estimator, model_dir, input_fn, eval_steps, train_steps,
                    name):
  """Perform continuous evaluation on checkpoints written to a model directory.

  Args:
    estimator: Estimator object to use for evaluation.
    model_dir: Model directory to read checkpoints for continuous evaluation.
    input_fn: Input function to use for evaluation.
    eval_steps: Number of steps to run during each evaluation.
    train_steps: Number of training steps. This is used to infer the last
      checkpoint and stop evaluation loop.
    name: Namescope for eval summary.
  """
  def terminate_eval():
    tf.logging.info('Terminating eval after 180 seconds of no checkpoints')
    return True

  for ckpt in tf.contrib.training.checkpoints_iterator(
      model_dir, min_interval_secs=180, timeout=None,
      timeout_fn=terminate_eval):

    tf.logging.info('Starting Evaluation.')
    try:
      eval_results = estimator.evaluate(
          input_fn=input_fn,
          steps=eval_steps,
          checkpoint_path=ckpt,
          name=name)
      tf.logging.info('Eval results: %s' % eval_results)

      # Terminate eval job when final checkpoint is reached
      current_step = int(os.path.basename(ckpt).split('-')[1])
      if current_step >= train_steps:
        tf.logging.info(
            'Evaluation finished after training step %d' % current_step)
        break

    except tf.errors.NotFoundError:
      tf.logging.info(
          'Checkpoint %s no longer exists, skipping checkpoint' % ckpt)


def populate_experiment(run_config,
                        hparams,
                        pipeline_config_path,
                        train_steps=None,
                        eval_steps=None,
                        model_fn_creator=create_model_fn,
                        **kwargs):
  """Populates an `Experiment` object.

  EXPERIMENT CLASS IS DEPRECATED. Please switch to
  tf.estimator.train_and_evaluate. As an example, see model_main.py.

  Args:
    run_config: A `RunConfig`.
    hparams: A `HParams`.
    pipeline_config_path: A path to a pipeline config file.
    train_steps: Number of training steps. If None, the number of training steps
      is set from the `TrainConfig` proto.
    eval_steps: Number of evaluation steps per evaluation cycle. If None, the
      number of evaluation steps is set from the `EvalConfig` proto.
    model_fn_creator: A function that creates a `model_fn` for `Estimator`.
      Follows the signature:

      * Args:
        * `detection_model_fn`: Function that returns `DetectionModel` instance.
        * `configs`: Dictionary of pipeline config objects.
        * `hparams`: `HParams` object.
      * Returns:
        `model_fn` for `Estimator`.

    **kwargs: Additional keyword arguments for configuration override.

  Returns:
    An `Experiment` that defines all aspects of training, evaluation, and
    export.
  """
  tf.logging.warning('Experiment is being deprecated. Please use '
                     'tf.estimator.train_and_evaluate(). See model_main.py for '
                     'an example.')
  train_and_eval_dict = create_estimator_and_inputs(
      run_config,
      hparams,
      pipeline_config_path,
      train_steps=train_steps,
      eval_steps=eval_steps,
      model_fn_creator=model_fn_creator,
      **kwargs)
  estimator = train_and_eval_dict['estimator']
  train_input_fn = train_and_eval_dict['train_input_fn']
  eval_input_fn = train_and_eval_dict['eval_input_fn']
  predict_input_fn = train_and_eval_dict['predict_input_fn']
  train_steps = train_and_eval_dict['train_steps']
  eval_steps = train_and_eval_dict['eval_steps']

  export_strategies = [
      tf.contrib.learn.utils.saved_model_export_utils.make_export_strategy(
          serving_input_fn=predict_input_fn)
  ]

  return tf.contrib.learn.Experiment(
      estimator=estimator,
      train_input_fn=train_input_fn,
      eval_input_fn=eval_input_fn,
      train_steps=train_steps,
      eval_steps=eval_steps,
      export_strategies=export_strategies,
      eval_delay_secs=120,)
