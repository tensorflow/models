# Copyright 2019 The TensorFlow Authors. All Rights Reserved.
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

import copy
import os
import time

import tensorflow.compat.v1 as tf
import tensorflow.compat.v2 as tf2

from object_detection import eval_util
from object_detection import inputs
from object_detection import model_lib
from object_detection.builders import optimizer_builder
from object_detection.core import standard_fields as fields
from object_detection.protos import train_pb2
from object_detection.utils import config_util
from object_detection.utils import label_map_util
from object_detection.utils import ops
from object_detection.utils import visualization_utils as vutils

# pylint: disable=g-import-not-at-top
try:
  from tensorflow.contrib import tpu as contrib_tpu
except ImportError:
  # TF 2.0 doesn't ship with contrib.
  pass
# pylint: enable=g-import-not-at-top

MODEL_BUILD_UTIL_MAP = model_lib.MODEL_BUILD_UTIL_MAP


RESTORE_MAP_ERROR_TEMPLATE = (
    'Since we are restoring a v2 style checkpoint'
    ' restore_map was expected to return a (str -> Model) mapping,'
    ' but we received a ({} -> {}) mapping instead.'
)


def _compute_losses_and_predictions_dicts(
    model, features, labels,
    add_regularization_loss=True):
  """Computes the losses dict and predictions dict for a model on inputs.

  Args:
    model: a DetectionModel (based on Keras).
    features: Dictionary of feature tensors from the input dataset.
      Should be in the format output by `inputs.train_input` and
      `inputs.eval_input`.
        features[fields.InputDataFields.image] is a [batch_size, H, W, C]
          float32 tensor with preprocessed images.
        features[HASH_KEY] is a [batch_size] int32 tensor representing unique
          identifiers for the images.
        features[fields.InputDataFields.true_image_shape] is a [batch_size, 3]
          int32 tensor representing the true image shapes, as preprocessed
          images could be padded.
        features[fields.InputDataFields.original_image] (optional) is a
          [batch_size, H, W, C] float32 tensor with original images.
    labels: A dictionary of groundtruth tensors post-unstacking. The original
      labels are of the form returned by `inputs.train_input` and
      `inputs.eval_input`. The shapes may have been modified by unstacking with
      `model_lib.unstack_batch`. However, the dictionary includes the following
      fields.
        labels[fields.InputDataFields.num_groundtruth_boxes] is a
          int32 tensor indicating the number of valid groundtruth boxes
          per image.
        labels[fields.InputDataFields.groundtruth_boxes] is a float32 tensor
          containing the corners of the groundtruth boxes.
        labels[fields.InputDataFields.groundtruth_classes] is a float32
          one-hot tensor of classes.
        labels[fields.InputDataFields.groundtruth_weights] is a float32 tensor
          containing groundtruth weights for the boxes.
        -- Optional --
        labels[fields.InputDataFields.groundtruth_instance_masks] is a
          float32 tensor containing only binary values, which represent
          instance masks for objects.
        labels[fields.InputDataFields.groundtruth_keypoints] is a
          float32 tensor containing keypoints for each box.
        labels[fields.InputDataFields.groundtruth_dp_num_points] is an int32
          tensor with the number of sampled DensePose points per object.
        labels[fields.InputDataFields.groundtruth_dp_part_ids] is an int32
          tensor with the DensePose part ids (0-indexed) per object.
        labels[fields.InputDataFields.groundtruth_dp_surface_coords] is a
          float32 tensor with the DensePose surface coordinates.
        labels[fields.InputDataFields.groundtruth_group_of] is a tf.bool tensor
          containing group_of annotations.
        labels[fields.InputDataFields.groundtruth_labeled_classes] is a float32
          k-hot tensor of classes.
        labels[fields.InputDataFields.groundtruth_track_ids] is a int32
          tensor of track IDs.
    add_regularization_loss: Whether or not to include the model's
      regularization loss in the losses dictionary.

  Returns:
    A tuple containing the losses dictionary (with the total loss under
    the key 'Loss/total_loss'), and the predictions dictionary produced by
    `model.predict`.

  """
  model_lib.provide_groundtruth(model, labels)
  preprocessed_images = features[fields.InputDataFields.image]

  prediction_dict = model.predict(
      preprocessed_images,
      features[fields.InputDataFields.true_image_shape],
      **model.get_side_inputs(features))
  prediction_dict = ops.bfloat16_to_float32_nested(prediction_dict)

  losses_dict = model.loss(
      prediction_dict, features[fields.InputDataFields.true_image_shape])
  losses = [loss_tensor for loss_tensor in losses_dict.values()]
  if add_regularization_loss:
    # TODO(kaftan): As we figure out mixed precision & bfloat 16, we may
    ## need to convert these regularization losses from bfloat16 to float32
    ## as well.
    regularization_losses = model.regularization_losses()
    if regularization_losses:
      regularization_losses = ops.bfloat16_to_float32_nested(
          regularization_losses)
      regularization_loss = tf.add_n(
          regularization_losses, name='regularization_loss')
      losses.append(regularization_loss)
      losses_dict['Loss/regularization_loss'] = regularization_loss

  total_loss = tf.add_n(losses, name='total_loss')
  losses_dict['Loss/total_loss'] = total_loss

  return losses_dict, prediction_dict


# TODO(kaftan): Explore removing learning_rate from this method & returning
## The full losses dict instead of just total_loss, then doing all summaries
## saving in a utility method called by the outer training loop.
# TODO(kaftan): Explore adding gradient summaries
def eager_train_step(detection_model,
                     features,
                     labels,
                     unpad_groundtruth_tensors,
                     optimizer,
                     learning_rate,
                     add_regularization_loss=True,
                     clip_gradients_value=None,
                     global_step=None,
                     num_replicas=1.0):
  """Process a single training batch.

  This method computes the loss for the model on a single training batch,
  while tracking the gradients with a gradient tape. It then updates the
  model variables with the optimizer, clipping the gradients if
  clip_gradients_value is present.

  This method can run eagerly or inside a tf.function.

  Args:
    detection_model: A DetectionModel (based on Keras) to train.
    features: Dictionary of feature tensors from the input dataset.
      Should be in the format output by `inputs.train_input.
        features[fields.InputDataFields.image] is a [batch_size, H, W, C]
          float32 tensor with preprocessed images.
        features[HASH_KEY] is a [batch_size] int32 tensor representing unique
          identifiers for the images.
        features[fields.InputDataFields.true_image_shape] is a [batch_size, 3]
          int32 tensor representing the true image shapes, as preprocessed
          images could be padded.
        features[fields.InputDataFields.original_image] (optional, not used
          during training) is a
          [batch_size, H, W, C] float32 tensor with original images.
    labels: A dictionary of groundtruth tensors. This method unstacks
      these labels using model_lib.unstack_batch. The stacked labels are of
      the form returned by `inputs.train_input` and `inputs.eval_input`.
        labels[fields.InputDataFields.num_groundtruth_boxes] is a [batch_size]
          int32 tensor indicating the number of valid groundtruth boxes
          per image.
        labels[fields.InputDataFields.groundtruth_boxes] is a
          [batch_size, num_boxes, 4] float32 tensor containing the corners of
          the groundtruth boxes.
        labels[fields.InputDataFields.groundtruth_classes] is a
          [batch_size, num_boxes, num_classes] float32 one-hot tensor of
          classes. num_classes includes the background class.
        labels[fields.InputDataFields.groundtruth_weights] is a
          [batch_size, num_boxes] float32 tensor containing groundtruth weights
          for the boxes.
        -- Optional --
        labels[fields.InputDataFields.groundtruth_instance_masks] is a
          [batch_size, num_boxes, H, W] float32 tensor containing only binary
          values, which represent instance masks for objects.
        labels[fields.InputDataFields.groundtruth_keypoints] is a
          [batch_size, num_boxes, num_keypoints, 2] float32 tensor containing
          keypoints for each box.
        labels[fields.InputDataFields.groundtruth_dp_num_points] is a
          [batch_size, num_boxes] int32 tensor with the number of DensePose
          sampled points per instance.
        labels[fields.InputDataFields.groundtruth_dp_part_ids] is a
          [batch_size, num_boxes, max_sampled_points] int32 tensor with the
          part ids (0-indexed) for each instance.
        labels[fields.InputDataFields.groundtruth_dp_surface_coords] is a
          [batch_size, num_boxes, max_sampled_points, 4] float32 tensor with the
          surface coordinates for each point. Each surface coordinate is of the
          form (y, x, v, u) where (y, x) are normalized image locations and
          (v, u) are part-relative normalized surface coordinates.
        labels[fields.InputDataFields.groundtruth_labeled_classes] is a float32
          k-hot tensor of classes.
        labels[fields.InputDataFields.groundtruth_track_ids] is a int32
          tensor of track IDs.
    unpad_groundtruth_tensors: A parameter passed to unstack_batch.
    optimizer: The training optimizer that will update the variables.
    learning_rate: The learning rate tensor for the current training step.
      This is used only for TensorBoard logging purposes, it does not affect
       model training.
    add_regularization_loss: Whether or not to include the model's
      regularization loss in the losses dictionary.
    clip_gradients_value: If this is present, clip the gradients global norm
      at this value using `tf.clip_by_global_norm`.
    global_step: The current training step. Used for TensorBoard logging
      purposes. This step is not updated by this function and must be
      incremented separately.
    num_replicas: The number of replicas in the current distribution strategy.
      This is used to scale the total loss so that training in a distribution
      strategy works correctly.

  Returns:
    The total loss observed at this training step
  """
  # """Execute a single training step in the TF v2 style loop."""
  is_training = True

  detection_model._is_training = is_training  # pylint: disable=protected-access
  tf.keras.backend.set_learning_phase(is_training)

  labels = model_lib.unstack_batch(
      labels, unpad_groundtruth_tensors=unpad_groundtruth_tensors)

  with tf.GradientTape() as tape:
    losses_dict, _ = _compute_losses_and_predictions_dicts(
        detection_model, features, labels, add_regularization_loss)

    total_loss = losses_dict['Loss/total_loss']

    # Normalize loss for num replicas
    total_loss = tf.math.divide(total_loss,
                                tf.constant(num_replicas, dtype=tf.float32))
    losses_dict['Loss/normalized_total_loss'] = total_loss

  for loss_type in losses_dict:
    tf.compat.v2.summary.scalar(
        loss_type, losses_dict[loss_type], step=global_step)

  trainable_variables = detection_model.trainable_variables

  gradients = tape.gradient(total_loss, trainable_variables)

  if clip_gradients_value:
    gradients, _ = tf.clip_by_global_norm(gradients, clip_gradients_value)
  optimizer.apply_gradients(zip(gradients, trainable_variables))
  tf.compat.v2.summary.scalar('learning_rate', learning_rate, step=global_step)
  tf.compat.v2.summary.image(
      name='train_input_images',
      step=global_step,
      data=features[fields.InputDataFields.image],
      max_outputs=3)
  return total_loss


def validate_tf_v2_checkpoint_restore_map(checkpoint_restore_map):
  """Ensure that given dict is a valid TF v2 style restore map.

  Args:
    checkpoint_restore_map: A nested dict mapping strings to
      tf.keras.Model objects.

  Raises:
    ValueError: If they keys in checkpoint_restore_map are not strings or if
      the values are not keras Model objects.

  """

  for key, value in checkpoint_restore_map.items():
    if not (isinstance(key, str) and
            (isinstance(value, tf.Module)
             or isinstance(value, tf.train.Checkpoint))):
      if isinstance(key, str) and isinstance(value, dict):
        validate_tf_v2_checkpoint_restore_map(value)
      else:
        raise TypeError(
            RESTORE_MAP_ERROR_TEMPLATE.format(key.__class__.__name__,
                                              value.__class__.__name__))


def is_object_based_checkpoint(checkpoint_path):
  """Returns true if `checkpoint_path` points to an object-based checkpoint."""
  var_names = [var[0] for var in tf.train.list_variables(checkpoint_path)]
  return '_CHECKPOINTABLE_OBJECT_GRAPH' in var_names


def load_fine_tune_checkpoint(
    model, checkpoint_path, checkpoint_type, checkpoint_version, input_dataset,
    unpad_groundtruth_tensors):
  """Load a fine tuning classification or detection checkpoint.

  To make sure the model variables are all built, this method first executes
  the model by computing a dummy loss. (Models might not have built their
  variables before their first execution)

  It then loads an object-based classification or detection checkpoint.

  This method updates the model in-place and does not return a value.

  Args:
    model: A DetectionModel (based on Keras) to load a fine-tuning
      checkpoint for.
    checkpoint_path: Directory with checkpoints file or path to checkpoint.
    checkpoint_type: Whether to restore from a full detection
      checkpoint (with compatible variable names) or to restore from a
      classification checkpoint for initialization prior to training.
      Valid values: `detection`, `classification`.
    checkpoint_version: train_pb2.CheckpointVersion.V1 or V2 enum indicating
      whether to load checkpoints in V1 style or V2 style.  In this binary
      we only support V2 style (object-based) checkpoints.
    input_dataset: The tf.data Dataset the model is being trained on. Needed
      to get the shapes for the dummy loss computation.
    unpad_groundtruth_tensors: A parameter passed to unstack_batch.

  Raises:
    IOError: if `checkpoint_path` does not point at a valid object-based
      checkpoint
    ValueError: if `checkpoint_version` is not train_pb2.CheckpointVersion.V2
  """
  if not is_object_based_checkpoint(checkpoint_path):
    raise IOError('Checkpoint is expected to be an object-based checkpoint.')
  if checkpoint_version == train_pb2.CheckpointVersion.V1:
    raise ValueError('Checkpoint version should be V2')

  features, labels = iter(input_dataset).next()

  @tf.function
  def _dummy_computation_fn(features, labels):
    model._is_training = False  # pylint: disable=protected-access
    tf.keras.backend.set_learning_phase(False)

    labels = model_lib.unstack_batch(
        labels, unpad_groundtruth_tensors=unpad_groundtruth_tensors)

    return _compute_losses_and_predictions_dicts(
        model,
        features,
        labels)

  strategy = tf.compat.v2.distribute.get_strategy()
  if hasattr(tf.distribute.Strategy, 'run'):
    strategy.run(
        _dummy_computation_fn, args=(
            features,
            labels,
        ))
  else:
    strategy.experimental_run_v2(
        _dummy_computation_fn, args=(
            features,
            labels,
        ))

  restore_from_objects_dict = model.restore_from_objects(
      fine_tune_checkpoint_type=checkpoint_type)
  validate_tf_v2_checkpoint_restore_map(restore_from_objects_dict)
  ckpt = tf.train.Checkpoint(**restore_from_objects_dict)
  ckpt.restore(checkpoint_path).assert_existing_objects_matched()


def get_filepath(strategy, filepath):
  """Get appropriate filepath for worker.

  Args:
    strategy: A tf.distribute.Strategy object.
    filepath: A path to where the Checkpoint object is stored.

  Returns:
    A temporary filepath for non-chief workers to use or the original filepath
    for the chief.
  """
  if strategy.extended.should_checkpoint:
    return filepath
  else:
    # TODO(vighneshb) Replace with the public API when TF exposes it.
    task_id = strategy.extended._task_id  # pylint:disable=protected-access
    return os.path.join(filepath, 'temp_worker_{:03d}'.format(task_id))


def clean_temporary_directories(strategy, filepath):
  """Temporary directory clean up for MultiWorker Mirrored Strategy.

  This is needed for all non-chief workers.

  Args:
    strategy: A tf.distribute.Strategy object.
    filepath: The filepath for the temporary directory.
  """
  if not strategy.extended.should_checkpoint:
    if tf.io.gfile.exists(filepath) and tf.io.gfile.isdir(filepath):
      tf.io.gfile.rmtree(filepath)


def train_loop(
    pipeline_config_path,
    model_dir,
    config_override=None,
    train_steps=None,
    use_tpu=False,
    save_final_config=False,
    checkpoint_every_n=1000,
    checkpoint_max_to_keep=7,
    record_summaries=True,
    **kwargs):
  """Trains a model using eager + functions.

  This method:
    1. Processes the pipeline configs
    2. (Optionally) saves the as-run config
    3. Builds the model & optimizer
    4. Gets the training input data
    5. Loads a fine-tuning detection or classification checkpoint if requested
    6. Loops over the train data, executing distributed training steps inside
       tf.functions.
    7. Checkpoints the model every `checkpoint_every_n` training steps.
    8. Logs the training metrics as TensorBoard summaries.

  Args:
    pipeline_config_path: A path to a pipeline config file.
    model_dir:
      The directory to save checkpoints and summaries to.
    config_override: A pipeline_pb2.TrainEvalPipelineConfig text proto to
      override the config from `pipeline_config_path`.
    train_steps: Number of training steps. If None, the number of training steps
      is set from the `TrainConfig` proto.
    use_tpu: Boolean, whether training and evaluation should run on TPU.
    save_final_config: Whether to save final config (obtained after applying
      overrides) to `model_dir`.
    checkpoint_every_n:
      Checkpoint every n training steps.
    checkpoint_max_to_keep:
      int, the number of most recent checkpoints to keep in the model directory.
    record_summaries: Boolean, whether or not to record summaries.
    **kwargs: Additional keyword arguments for configuration override.
  """
  ## Parse the configs
  get_configs_from_pipeline_file = MODEL_BUILD_UTIL_MAP[
      'get_configs_from_pipeline_file']
  merge_external_params_with_configs = MODEL_BUILD_UTIL_MAP[
      'merge_external_params_with_configs']
  create_pipeline_proto_from_configs = MODEL_BUILD_UTIL_MAP[
      'create_pipeline_proto_from_configs']

  configs = get_configs_from_pipeline_file(
      pipeline_config_path, config_override=config_override)
  kwargs.update({
      'train_steps': train_steps,
      'use_bfloat16': configs['train_config'].use_bfloat16 and use_tpu
  })
  configs = merge_external_params_with_configs(
      configs, None, kwargs_dict=kwargs)
  model_config = configs['model']
  train_config = configs['train_config']
  train_input_config = configs['train_input_config']

  unpad_groundtruth_tensors = train_config.unpad_groundtruth_tensors
  add_regularization_loss = train_config.add_regularization_loss
  clip_gradients_value = None
  if train_config.gradient_clipping_by_norm > 0:
    clip_gradients_value = train_config.gradient_clipping_by_norm

  # update train_steps from config but only when non-zero value is provided
  if train_steps is None and train_config.num_steps != 0:
    train_steps = train_config.num_steps

  if kwargs['use_bfloat16']:
    tf.compat.v2.keras.mixed_precision.experimental.set_policy('mixed_bfloat16')

  if train_config.load_all_detection_checkpoint_vars:
    raise ValueError('train_pb2.load_all_detection_checkpoint_vars '
                     'unsupported in TF2')

  config_util.update_fine_tune_checkpoint_type(train_config)
  fine_tune_checkpoint_type = train_config.fine_tune_checkpoint_type
  fine_tune_checkpoint_version = train_config.fine_tune_checkpoint_version

  # Write the as-run pipeline config to disk.
  if save_final_config:
    pipeline_config_final = create_pipeline_proto_from_configs(configs)
    config_util.save_pipeline_config(pipeline_config_final, model_dir)

  # Build the model, optimizer, and training input
  strategy = tf.compat.v2.distribute.get_strategy()
  with strategy.scope():
    detection_model = MODEL_BUILD_UTIL_MAP['detection_model_fn_base'](
        model_config=model_config, is_training=True)

    def train_dataset_fn(input_context):
      """Callable to create train input."""
      # Create the inputs.
      train_input = inputs.train_input(
          train_config=train_config,
          train_input_config=train_input_config,
          model_config=model_config,
          model=detection_model,
          input_context=input_context)
      train_input = train_input.repeat()
      return train_input

    train_input = strategy.experimental_distribute_datasets_from_function(
        train_dataset_fn)


    global_step = tf.Variable(
        0, trainable=False, dtype=tf.compat.v2.dtypes.int64, name='global_step',
        aggregation=tf.compat.v2.VariableAggregation.ONLY_FIRST_REPLICA)
    optimizer, (learning_rate,) = optimizer_builder.build(
        train_config.optimizer, global_step=global_step)

    if callable(learning_rate):
      learning_rate_fn = learning_rate
    else:
      learning_rate_fn = lambda: learning_rate

  ## Train the model
  # Get the appropriate filepath (temporary or not) based on whether the worker
  # is the chief.
  summary_writer_filepath = get_filepath(strategy,
                                         os.path.join(model_dir, 'train'))
  if record_summaries:
    summary_writer = tf.compat.v2.summary.create_file_writer(
        summary_writer_filepath)
  else:
    summary_writer = tf2.summary.create_noop_writer()

  if use_tpu:
    num_steps_per_iteration = 100
  else:
    # TODO(b/135933080) Explore setting to 100 when GPU performance issues
    # are fixed.
    num_steps_per_iteration = 1

  with summary_writer.as_default():
    with strategy.scope():
      with tf.compat.v2.summary.record_if(
          lambda: global_step % num_steps_per_iteration == 0):
        # Load a fine-tuning checkpoint.
        if train_config.fine_tune_checkpoint:
          load_fine_tune_checkpoint(detection_model,
                                    train_config.fine_tune_checkpoint,
                                    fine_tune_checkpoint_type,
                                    fine_tune_checkpoint_version,
                                    train_input,
                                    unpad_groundtruth_tensors)

        ckpt = tf.compat.v2.train.Checkpoint(
            step=global_step, model=detection_model, optimizer=optimizer)

        manager_dir = get_filepath(strategy, model_dir)
        if not strategy.extended.should_checkpoint:
          checkpoint_max_to_keep = 1
        manager = tf.compat.v2.train.CheckpointManager(
            ckpt, manager_dir, max_to_keep=checkpoint_max_to_keep)

        # We use the following instead of manager.latest_checkpoint because
        # manager_dir does not point to the model directory when we are running
        # in a worker.
        latest_checkpoint = tf.train.latest_checkpoint(model_dir)
        ckpt.restore(latest_checkpoint)

        def train_step_fn(features, labels):
          """Single train step."""
          loss = eager_train_step(
              detection_model,
              features,
              labels,
              unpad_groundtruth_tensors,
              optimizer,
              learning_rate=learning_rate_fn(),
              add_regularization_loss=add_regularization_loss,
              clip_gradients_value=clip_gradients_value,
              global_step=global_step,
              num_replicas=strategy.num_replicas_in_sync)
          global_step.assign_add(1)
          return loss

        def _sample_and_train(strategy, train_step_fn, data_iterator):
          features, labels = data_iterator.next()
          if hasattr(tf.distribute.Strategy, 'run'):
            per_replica_losses = strategy.run(
                train_step_fn, args=(features, labels))
          else:
            per_replica_losses = strategy.experimental_run_v2(
                train_step_fn, args=(features, labels))
          # TODO(anjalisridhar): explore if it is safe to remove the
          ## num_replicas scaling of the loss and switch this to a ReduceOp.Mean
          return strategy.reduce(tf.distribute.ReduceOp.SUM,
                                 per_replica_losses, axis=None)

        @tf.function
        def _dist_train_step(data_iterator):
          """A distributed train step."""

          if num_steps_per_iteration > 1:
            for _ in tf.range(num_steps_per_iteration - 1):
              # Following suggestion on yaqs/5402607292645376
              with tf.name_scope(''):
                _sample_and_train(strategy, train_step_fn, data_iterator)

          return _sample_and_train(strategy, train_step_fn, data_iterator)

        train_input_iter = iter(train_input)

        if int(global_step.value()) == 0:
          manager.save()

        checkpointed_step = int(global_step.value())
        logged_step = global_step.value()

        last_step_time = time.time()
        for _ in range(global_step.value(), train_steps,
                       num_steps_per_iteration):

          loss = _dist_train_step(train_input_iter)

          time_taken = time.time() - last_step_time
          last_step_time = time.time()

          tf.compat.v2.summary.scalar(
              'steps_per_sec', num_steps_per_iteration * 1.0 / time_taken,
              step=global_step)

          if global_step.value() - logged_step >= 100:
            tf.logging.info(
                'Step {} per-step time {:.3f}s loss={:.3f}'.format(
                    global_step.value(), time_taken / num_steps_per_iteration,
                    loss))
            logged_step = global_step.value()

          if ((int(global_step.value()) - checkpointed_step) >=
              checkpoint_every_n):
            manager.save()
            checkpointed_step = int(global_step.value())

  # Remove the checkpoint directories of the non-chief workers that
  # MultiWorkerMirroredStrategy forces us to save during sync distributed
  # training.
  clean_temporary_directories(strategy, manager_dir)
  clean_temporary_directories(strategy, summary_writer_filepath)


def eager_eval_loop(
    detection_model,
    configs,
    eval_dataset,
    use_tpu=False,
    postprocess_on_cpu=False,
    global_step=None):
  """Evaluate the model eagerly on the evaluation dataset.

  This method will compute the evaluation metrics specified in the configs on
  the entire evaluation dataset, then return the metrics. It will also log
  the metrics to TensorBoard.

  Args:
    detection_model: A DetectionModel (based on Keras) to evaluate.
    configs: Object detection configs that specify the evaluators that should
      be used, as well as whether regularization loss should be included and
      if bfloat16 should be used on TPUs.
    eval_dataset: Dataset containing evaluation data.
    use_tpu: Whether a TPU is being used to execute the model for evaluation.
    postprocess_on_cpu: Whether model postprocessing should happen on
      the CPU when using a TPU to execute the model.
    global_step: A variable containing the training step this model was trained
      to. Used for logging purposes.

  Returns:
    A dict of evaluation metrics representing the results of this evaluation.
  """
  train_config = configs['train_config']
  eval_input_config = configs['eval_input_config']
  eval_config = configs['eval_config']
  add_regularization_loss = train_config.add_regularization_loss

  is_training = False
  detection_model._is_training = is_training  # pylint: disable=protected-access
  tf.keras.backend.set_learning_phase(is_training)

  evaluator_options = eval_util.evaluator_options_from_eval_config(
      eval_config)

  class_agnostic_category_index = (
      label_map_util.create_class_agnostic_category_index())
  class_agnostic_evaluators = eval_util.get_evaluators(
      eval_config,
      list(class_agnostic_category_index.values()),
      evaluator_options)

  class_aware_evaluators = None
  if eval_input_config.label_map_path:
    class_aware_category_index = (
        label_map_util.create_category_index_from_labelmap(
            eval_input_config.label_map_path))
    class_aware_evaluators = eval_util.get_evaluators(
        eval_config,
        list(class_aware_category_index.values()),
        evaluator_options)

  evaluators = None
  loss_metrics = {}

  @tf.function
  def compute_eval_dict(features, labels):
    """Compute the evaluation result on an image."""
    # For evaling on train data, it is necessary to check whether groundtruth
    # must be unpadded.
    boxes_shape = (
        labels[fields.InputDataFields.groundtruth_boxes].get_shape().as_list())
    unpad_groundtruth_tensors = boxes_shape[1] is not None and not use_tpu
    labels = model_lib.unstack_batch(
        labels, unpad_groundtruth_tensors=unpad_groundtruth_tensors)

    losses_dict, prediction_dict = _compute_losses_and_predictions_dicts(
        detection_model, features, labels, add_regularization_loss)

    def postprocess_wrapper(args):
      return detection_model.postprocess(args[0], args[1])

    # TODO(kaftan): Depending on how postprocessing will work for TPUS w/
    ## TPUStrategy, may be good to move wrapping to a utility method
    if use_tpu and postprocess_on_cpu:
      detections = contrib_tpu.outside_compilation(
          postprocess_wrapper,
          (prediction_dict, features[fields.InputDataFields.true_image_shape]))
    else:
      detections = postprocess_wrapper(
          (prediction_dict, features[fields.InputDataFields.true_image_shape]))

    class_agnostic = (
        fields.DetectionResultFields.detection_classes not in detections)
    # TODO(kaftan) (or anyone): move `_prepare_groundtruth_for_eval to eval_util
    ## and call this from there.
    groundtruth = model_lib._prepare_groundtruth_for_eval(  # pylint: disable=protected-access
        detection_model, class_agnostic, eval_input_config.max_number_of_boxes)
    use_original_images = fields.InputDataFields.original_image in features
    if use_original_images:
      eval_images = features[fields.InputDataFields.original_image]
      true_image_shapes = tf.slice(
          features[fields.InputDataFields.true_image_shape], [0, 0], [-1, 3])
      original_image_spatial_shapes = features[
          fields.InputDataFields.original_image_spatial_shape]
    else:
      eval_images = features[fields.InputDataFields.image]
      true_image_shapes = None
      original_image_spatial_shapes = None

    eval_dict = eval_util.result_dict_for_batched_example(
        eval_images,
        features[inputs.HASH_KEY],
        detections,
        groundtruth,
        class_agnostic=class_agnostic,
        scale_to_absolute=True,
        original_image_spatial_shapes=original_image_spatial_shapes,
        true_image_shapes=true_image_shapes)

    return eval_dict, losses_dict, class_agnostic

  agnostic_categories = label_map_util.create_class_agnostic_category_index()
  per_class_categories = label_map_util.create_category_index_from_labelmap(
      eval_input_config.label_map_path)
  keypoint_edges = [
      (kp.start, kp.end) for kp in eval_config.keypoint_edge]

  for i, (features, labels) in enumerate(eval_dataset):
    eval_dict, losses_dict, class_agnostic = compute_eval_dict(features, labels)

    if class_agnostic:
      category_index = agnostic_categories
    else:
      category_index = per_class_categories

    if i % 100 == 0:
      tf.logging.info('Finished eval step %d', i)

    use_original_images = fields.InputDataFields.original_image in features
    if use_original_images and i < eval_config.num_visualizations:
      sbys_image_list = vutils.draw_side_by_side_evaluation_image(
          eval_dict,
          category_index=category_index,
          max_boxes_to_draw=eval_config.max_num_boxes_to_visualize,
          min_score_thresh=eval_config.min_score_threshold,
          use_normalized_coordinates=False,
          keypoint_edges=keypoint_edges or None)
      sbys_images = tf.concat(sbys_image_list, axis=0)
      tf.compat.v2.summary.image(
          name='eval_side_by_side_' + str(i),
          step=global_step,
          data=sbys_images,
          max_outputs=eval_config.num_visualizations)
      if eval_util.has_densepose(eval_dict):
        dp_image_list = vutils.draw_densepose_visualizations(
            eval_dict)
        dp_images = tf.concat(dp_image_list, axis=0)
        tf.compat.v2.summary.image(
            name='densepose_detections_' + str(i),
            step=global_step,
            data=dp_images,
            max_outputs=eval_config.num_visualizations)

    if evaluators is None:
      if class_agnostic:
        evaluators = class_agnostic_evaluators
      else:
        evaluators = class_aware_evaluators

    for evaluator in evaluators:
      evaluator.add_eval_dict(eval_dict)

    for loss_key, loss_tensor in iter(losses_dict.items()):
      if loss_key not in loss_metrics:
        loss_metrics[loss_key] = tf.keras.metrics.Mean()
      # Skip the loss with value equal or lower than 0.0 when calculating the
      # average loss since they don't usually reflect the normal loss values
      # causing spurious average loss value.
      if loss_tensor <= 0.0:
        continue
      loss_metrics[loss_key].update_state(loss_tensor)

  eval_metrics = {}

  for evaluator in evaluators:
    eval_metrics.update(evaluator.evaluate())
  for loss_key in loss_metrics:
    eval_metrics[loss_key] = loss_metrics[loss_key].result()

  eval_metrics = {str(k): v for k, v in eval_metrics.items()}
  tf.logging.info('Eval metrics at step %d', global_step)
  for k in eval_metrics:
    tf.compat.v2.summary.scalar(k, eval_metrics[k], step=global_step)
    tf.logging.info('\t+ %s: %f', k, eval_metrics[k])

  return eval_metrics


def eval_continuously(
    pipeline_config_path,
    config_override=None,
    train_steps=None,
    sample_1_of_n_eval_examples=1,
    sample_1_of_n_eval_on_train_examples=1,
    use_tpu=False,
    override_eval_num_epochs=True,
    postprocess_on_cpu=False,
    model_dir=None,
    checkpoint_dir=None,
    wait_interval=180,
    timeout=3600,
    eval_index=None,
    **kwargs):
  """Run continuous evaluation of a detection model eagerly.

  This method builds the model, and continously restores it from the most
  recent training checkpoint in the checkpoint directory & evaluates it
  on the evaluation data.

  Args:
    pipeline_config_path: A path to a pipeline config file.
    config_override: A pipeline_pb2.TrainEvalPipelineConfig text proto to
      override the config from `pipeline_config_path`.
    train_steps: Number of training steps. If None, the number of training steps
      is set from the `TrainConfig` proto.
    sample_1_of_n_eval_examples: Integer representing how often an eval example
      should be sampled. If 1, will sample all examples.
    sample_1_of_n_eval_on_train_examples: Similar to
      `sample_1_of_n_eval_examples`, except controls the sampling of training
      data for evaluation.
    use_tpu: Boolean, whether training and evaluation should run on TPU.
    override_eval_num_epochs: Whether to overwrite the number of epochs to 1 for
      eval_input.
    postprocess_on_cpu: When use_tpu and postprocess_on_cpu are true,
      postprocess is scheduled on the host cpu.
    model_dir: Directory to output resulting evaluation summaries to.
    checkpoint_dir: Directory that contains the training checkpoints.
    wait_interval: The mimmum number of seconds to wait before checking for a
      new checkpoint.
    timeout: The maximum number of seconds to wait for a checkpoint. Execution
      will terminate if no new checkpoints are found after these many seconds.
    eval_index: int, optional If give, only evaluate the dataset at the given
      index.

    **kwargs: Additional keyword arguments for configuration override.
  """
  get_configs_from_pipeline_file = MODEL_BUILD_UTIL_MAP[
      'get_configs_from_pipeline_file']
  merge_external_params_with_configs = MODEL_BUILD_UTIL_MAP[
      'merge_external_params_with_configs']

  configs = get_configs_from_pipeline_file(
      pipeline_config_path, config_override=config_override)
  kwargs.update({
      'sample_1_of_n_eval_examples': sample_1_of_n_eval_examples,
      'use_bfloat16': configs['train_config'].use_bfloat16 and use_tpu
  })
  if train_steps is not None:
    kwargs['train_steps'] = train_steps
  if override_eval_num_epochs:
    kwargs.update({'eval_num_epochs': 1})
    tf.logging.warning(
        'Forced number of epochs for all eval validations to be 1.')
  configs = merge_external_params_with_configs(
      configs, None, kwargs_dict=kwargs)
  model_config = configs['model']
  train_input_config = configs['train_input_config']
  eval_config = configs['eval_config']
  eval_input_configs = configs['eval_input_configs']
  eval_on_train_input_config = copy.deepcopy(train_input_config)
  eval_on_train_input_config.sample_1_of_n_examples = (
      sample_1_of_n_eval_on_train_examples)
  if override_eval_num_epochs and eval_on_train_input_config.num_epochs != 1:
    tf.logging.warning('Expected number of evaluation epochs is 1, but '
                       'instead encountered `eval_on_train_input_config'
                       '.num_epochs` = '
                       '{}. Overwriting `num_epochs` to 1.'.format(
                           eval_on_train_input_config.num_epochs))
    eval_on_train_input_config.num_epochs = 1

  if kwargs['use_bfloat16']:
    tf.compat.v2.keras.mixed_precision.experimental.set_policy('mixed_bfloat16')

  detection_model = MODEL_BUILD_UTIL_MAP['detection_model_fn_base'](
      model_config=model_config, is_training=True)

  # Create the inputs.
  eval_inputs = []
  for eval_input_config in eval_input_configs:
    next_eval_input = inputs.eval_input(
        eval_config=eval_config,
        eval_input_config=eval_input_config,
        model_config=model_config,
        model=detection_model)
    eval_inputs.append((eval_input_config.name, next_eval_input))

  if eval_index is not None:
    eval_inputs = [eval_inputs[eval_index]]

  global_step = tf.compat.v2.Variable(
      0, trainable=False, dtype=tf.compat.v2.dtypes.int64)

  for latest_checkpoint in tf.train.checkpoints_iterator(
      checkpoint_dir, timeout=timeout, min_interval_secs=wait_interval):
    ckpt = tf.compat.v2.train.Checkpoint(
        step=global_step, model=detection_model)

    ckpt.restore(latest_checkpoint).expect_partial()

    for eval_name, eval_input in eval_inputs:
      summary_writer = tf.compat.v2.summary.create_file_writer(
          os.path.join(model_dir, 'eval', eval_name))
      with summary_writer.as_default():
        eager_eval_loop(
            detection_model,
            configs,
            eval_input,
            use_tpu=use_tpu,
            postprocess_on_cpu=postprocess_on_cpu,
            global_step=global_step)
