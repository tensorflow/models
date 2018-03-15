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
r"""Creates and runs `Experiment` for object detection model.

This uses the TF.learn framework to define and run an object detection model
wrapped in an `Estimator`.
Note that this module is only compatible with SSD Meta architecture at the
moment.
"""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import functools
import os

import tensorflow as tf

from google.protobuf import text_format
from tensorflow.contrib.learn.python.learn import learn_runner
from tensorflow.contrib.tpu.python.tpu import tpu_optimizer
from tensorflow.python.lib.io import file_io
from object_detection import eval_util
from object_detection import inputs
from object_detection import model_hparams
from object_detection.builders import model_builder
from object_detection.builders import optimizer_builder
from object_detection.core import standard_fields as fields
from object_detection.utils import config_util
from object_detection.utils import label_map_util
from object_detection.utils import shape_utils
from object_detection.utils import variables_helper
from object_detection.utils import visualization_utils as vis_utils

tf.flags.DEFINE_string('model_dir', None, 'Path to output model directory '
                       'where event and checkpoint files will be written.')
tf.flags.DEFINE_string('pipeline_config_path', None, 'Path to pipeline config '
                       'file.')
tf.flags.DEFINE_integer('num_train_steps', 500000, 'Number of train steps.')
tf.flags.DEFINE_integer('num_eval_steps', 10000, 'Number of train steps.')
FLAGS = tf.flags.FLAGS


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


def _get_groundtruth_data(detection_model, class_agnostic):
  """Extracts groundtruth data from detection_model.

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
      labels = unstack_batch(labels, unpad_groundtruth_tensors=False)

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
      detection_model.provide_groundtruth(
          groundtruth_boxes_list=gt_boxes_list,
          groundtruth_classes_list=gt_classes_list,
          groundtruth_masks_list=gt_masks_list,
          groundtruth_keypoints_list=gt_keypoints_list)

    preprocessed_images = features[fields.InputDataFields.image]
    prediction_dict = detection_model.predict(
        preprocessed_images, features[fields.InputDataFields.true_image_shape])
    detections = detection_model.postprocess(
        prediction_dict, features[fields.InputDataFields.true_image_shape])

    if mode == tf.estimator.ModeKeys.TRAIN:
      if not train_config.fine_tune_checkpoint_type:
        # train_config.from_detection_checkpoint field is deprecated. For
        # backward compatibility, sets finetune_checkpoint_type based on
        # from_detection_checkpoint.
        if train_config.from_detection_checkpoint:
          train_config.fine_tune_checkpoint_type = 'detection'
        else:
          train_config.fine_tune_checkpoint_type = 'classification'
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
          if not use_tpu:
            tf.summary.scalar('regularization_loss', regularization_loss)
      total_loss = tf.add_n(losses, name='total_loss')

    if mode == tf.estimator.ModeKeys.TRAIN:
      global_step = tf.train.get_or_create_global_step()
      training_optimizer, optimizer_summary_vars = optimizer_builder.build(
          train_config.optimizer)

      if use_tpu:
        training_optimizer = tpu_optimizer.CrossShardOptimizer(
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
    if mode == tf.estimator.ModeKeys.EVAL:
      # Detection summaries during eval.
      class_agnostic = (fields.DetectionResultFields.detection_classes
                        not in detections)
      groundtruth = _get_groundtruth_data(detection_model, class_agnostic)
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
          scale_to_absolute=False)

      if class_agnostic:
        category_index = label_map_util.create_class_agnostic_category_index()
      else:
        category_index = label_map_util.create_category_index_from_labelmap(
            eval_input_config.label_map_path)
      if not use_tpu and use_original_images:
        detection_and_groundtruth = (
            vis_utils.draw_side_by_side_evaluation_image(
                eval_dict, category_index, max_boxes_to_draw=20,
                min_score_thresh=0.2))
        tf.summary.image('Detections_Left_Groundtruth_Right',
                         detection_and_groundtruth)

      # Eval metrics on a single image.
      eval_metrics = eval_config.metrics_set
      if not eval_metrics:
        eval_metrics = ['coco_detection_metrics']
      eval_metric_ops = eval_util.get_eval_metric_ops_for_evaluators(
          eval_metrics, category_index.values(), eval_dict,
          include_metrics_per_category=False)

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
          export_outputs=export_outputs)

  return model_fn


def build_experiment_fn(train_steps, eval_steps):
  """Returns a function that creates an `Experiment`."""

  def build_experiment(run_config, hparams):
    """Builds an `Experiment` from configuration and hyperparameters.

    Args:
      run_config: A `RunConfig`.
      hparams: A `HParams`.

    Returns:
      An `Experiment` object.
    """
    return populate_experiment(run_config, hparams, FLAGS.pipeline_config_path,
                               train_steps, eval_steps)

  return build_experiment


def populate_experiment(run_config,
                        hparams,
                        pipeline_config_path,
                        train_steps=None,
                        eval_steps=None,
                        model_fn_creator=create_model_fn,
                        **kwargs):
  """Populates an `Experiment` object.

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
  get_configs_from_pipeline_file = MODEL_BUILD_UTIL_MAP[
      'get_configs_from_pipeline_file']
  create_pipeline_proto_from_configs = MODEL_BUILD_UTIL_MAP[
      'create_pipeline_proto_from_configs']
  merge_external_params_with_configs = MODEL_BUILD_UTIL_MAP[
      'merge_external_params_with_configs']
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

  if train_steps is None and train_config.num_steps:
    train_steps = train_config.num_steps

  if eval_steps is None and eval_config.num_examples:
    eval_steps = eval_config.num_examples

  detection_model_fn = functools.partial(
      model_builder.build, model_config=model_config)

  # Create the input functions for TRAIN/EVAL.
  train_input_fn = create_train_input_fn(
      train_config=train_config,
      train_input_config=train_input_config,
      model_config=model_config)
  eval_input_fn = create_eval_input_fn(
      eval_config=eval_config,
      eval_input_config=eval_input_config,
      model_config=model_config)

  export_strategies = [
      tf.contrib.learn.utils.saved_model_export_utils.make_export_strategy(
          serving_input_fn=create_predict_input_fn(
              model_config=model_config))
  ]

  estimator = tf.estimator.Estimator(
      model_fn=model_fn_creator(detection_model_fn, configs, hparams),
      config=run_config)

  if run_config.is_chief:
    # Store the final pipeline config for traceability.
    pipeline_config_final = create_pipeline_proto_from_configs(
        configs)
    if not file_io.file_exists(estimator.model_dir):
      file_io.recursive_create_dir(estimator.model_dir)
    pipeline_config_final_path = os.path.join(estimator.model_dir,
                                              'pipeline.config')
    config_text = text_format.MessageToString(pipeline_config_final)
    with tf.gfile.Open(pipeline_config_final_path, 'wb') as f:
      tf.logging.info('Writing as-run pipeline config file to %s',
                      pipeline_config_final_path)
      f.write(config_text)

  return tf.contrib.learn.Experiment(
      estimator=estimator,
      train_input_fn=train_input_fn,
      eval_input_fn=eval_input_fn,
      train_steps=train_steps,
      eval_steps=eval_steps,
      export_strategies=export_strategies,
      eval_delay_secs=120,)


def main(unused_argv):
  tf.flags.mark_flag_as_required('model_dir')
  tf.flags.mark_flag_as_required('pipeline_config_path')
  config = tf.contrib.learn.RunConfig(model_dir=FLAGS.model_dir)
  learn_runner.run(
      experiment_fn=build_experiment_fn(FLAGS.num_train_steps,
                                        FLAGS.num_eval_steps),
      run_config=config,
      hparams=model_hparams.create_hparams())


if __name__ == '__main__':
  tf.app.run()
