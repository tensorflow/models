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

"""Detection model trainer.

This file provides a generic training method that can be used to train a
DetectionModel.
"""

import functools

import tensorflow as tf

from object_detection.builders import optimizer_builder
from object_detection.builders import preprocessor_builder
from object_detection.core import batcher
from object_detection.core import preprocessor
from object_detection.core import standard_fields as fields
from object_detection.utils import ops as util_ops
from object_detection.utils import variables_helper
from deployment import model_deploy

slim = tf.contrib.slim


def create_input_queue(batch_size_per_clone, create_tensor_dict_fn,
                       batch_queue_capacity, num_batch_queue_threads,
                       prefetch_queue_capacity, data_augmentation_options):
  """Sets up reader, prefetcher and returns input queue.

  Args:
    batch_size_per_clone: batch size to use per clone.
    create_tensor_dict_fn: function to create tensor dictionary.
    batch_queue_capacity: maximum number of elements to store within a queue.
    num_batch_queue_threads: number of threads to use for batching.
    prefetch_queue_capacity: maximum capacity of the queue used to prefetch
                             assembled batches.
    data_augmentation_options: a list of tuples, where each tuple contains a
      data augmentation function and a dictionary containing arguments and their
      values (see preprocessor.py).

  Returns:
    input queue: a batcher.BatchQueue object holding enqueued tensor_dicts
      (which hold images, boxes and targets).  To get a batch of tensor_dicts,
      call input_queue.Dequeue().
  """
  tensor_dict = create_tensor_dict_fn()

  tensor_dict[fields.InputDataFields.image] = tf.expand_dims(
      tensor_dict[fields.InputDataFields.image], 0)

  images = tensor_dict[fields.InputDataFields.image]
  float_images = tf.to_float(images)
  tensor_dict[fields.InputDataFields.image] = float_images

  include_instance_masks = (fields.InputDataFields.groundtruth_instance_masks
                            in tensor_dict)
  include_keypoints = (fields.InputDataFields.groundtruth_keypoints
                       in tensor_dict)
  include_multiclass_scores = (fields.InputDataFields.multiclass_scores
                               in tensor_dict)
  if data_augmentation_options:
    tensor_dict = preprocessor.preprocess(
        tensor_dict, data_augmentation_options,
        func_arg_map=preprocessor.get_default_func_arg_map(
            include_multiclass_scores=include_multiclass_scores,
            include_instance_masks=include_instance_masks,
            include_keypoints=include_keypoints))

  input_queue = batcher.BatchQueue(
      tensor_dict,
      batch_size=batch_size_per_clone,
      batch_queue_capacity=batch_queue_capacity,
      num_batch_queue_threads=num_batch_queue_threads,
      prefetch_queue_capacity=prefetch_queue_capacity)
  return input_queue


def get_inputs(input_queue,
               num_classes,
               merge_multiple_label_boxes=False,
               use_multiclass_scores=False):
  """Dequeues batch and constructs inputs to object detection model.

  Args:
    input_queue: BatchQueue object holding enqueued tensor_dicts.
    num_classes: Number of classes.
    merge_multiple_label_boxes: Whether to merge boxes with multiple labels
      or not. Defaults to false. Merged boxes are represented with a single
      box and a k-hot encoding of the multiple labels associated with the
      boxes.
    use_multiclass_scores: Whether to use multiclass scores instead of
      groundtruth_classes.

  Returns:
    images: a list of 3-D float tensor of images.
    image_keys: a list of string keys for the images.
    locations_list: a list of tensors of shape [num_boxes, 4]
      containing the corners of the groundtruth boxes.
    classes_list: a list of padded one-hot (or K-hot) float32 tensors containing
      target classes.
    masks_list: a list of 3-D float tensors of shape [num_boxes, image_height,
      image_width] containing instance masks for objects if present in the
      input_queue. Else returns None.
    keypoints_list: a list of 3-D float tensors of shape [num_boxes,
      num_keypoints, 2] containing keypoints for objects if present in the
      input queue. Else returns None.
    weights_lists: a list of 1-D float32 tensors of shape [num_boxes]
      containing groundtruth weight for each box.
  """
  read_data_list = input_queue.dequeue()
  label_id_offset = 1
  def extract_images_and_targets(read_data):
    """Extract images and targets from the input dict."""
    image = read_data[fields.InputDataFields.image]
    key = ''
    if fields.InputDataFields.source_id in read_data:
      key = read_data[fields.InputDataFields.source_id]
    location_gt = read_data[fields.InputDataFields.groundtruth_boxes]
    classes_gt = tf.cast(read_data[fields.InputDataFields.groundtruth_classes],
                         tf.int32)
    classes_gt -= label_id_offset

    if merge_multiple_label_boxes and use_multiclass_scores:
      raise ValueError(
          'Using both merge_multiple_label_boxes and use_multiclass_scores is'
          'not supported'
      )

    if merge_multiple_label_boxes:
      location_gt, classes_gt, _ = util_ops.merge_boxes_with_multiple_labels(
          location_gt, classes_gt, num_classes)
      classes_gt = tf.cast(classes_gt, tf.float32)
    elif use_multiclass_scores:
      classes_gt = tf.cast(read_data[fields.InputDataFields.multiclass_scores],
                           tf.float32)
    else:
      classes_gt = util_ops.padded_one_hot_encoding(
          indices=classes_gt, depth=num_classes, left_pad=0)
    masks_gt = read_data.get(fields.InputDataFields.groundtruth_instance_masks)
    keypoints_gt = read_data.get(fields.InputDataFields.groundtruth_keypoints)
    if (merge_multiple_label_boxes and (
        masks_gt is not None or keypoints_gt is not None)):
      raise NotImplementedError('Multi-label support is only for boxes.')
    weights_gt = read_data.get(
        fields.InputDataFields.groundtruth_weights)
    return (image, key, location_gt, classes_gt, masks_gt, keypoints_gt,
            weights_gt)

  return zip(*map(extract_images_and_targets, read_data_list))


def _create_losses(input_queue, create_model_fn, train_config):
  """Creates loss function for a DetectionModel.

  Args:
    input_queue: BatchQueue object holding enqueued tensor_dicts.
    create_model_fn: A function to create the DetectionModel.
    train_config: a train_pb2.TrainConfig protobuf.
  """
  detection_model = create_model_fn()
  (images, _, groundtruth_boxes_list, groundtruth_classes_list,
   groundtruth_masks_list, groundtruth_keypoints_list, _) = get_inputs(
       input_queue,
       detection_model.num_classes,
       train_config.merge_multiple_label_boxes,
       train_config.use_multiclass_scores)

  preprocessed_images = []
  true_image_shapes = []
  for image in images:
    resized_image, true_image_shape = detection_model.preprocess(image)
    preprocessed_images.append(resized_image)
    true_image_shapes.append(true_image_shape)

  images = tf.concat(preprocessed_images, 0)
  true_image_shapes = tf.concat(true_image_shapes, 0)

  if any(mask is None for mask in groundtruth_masks_list):
    groundtruth_masks_list = None
  if any(keypoints is None for keypoints in groundtruth_keypoints_list):
    groundtruth_keypoints_list = None

  detection_model.provide_groundtruth(groundtruth_boxes_list,
                                      groundtruth_classes_list,
                                      groundtruth_masks_list,
                                      groundtruth_keypoints_list)
  prediction_dict = detection_model.predict(images, true_image_shapes)

  losses_dict = detection_model.loss(prediction_dict, true_image_shapes)
  for loss_tensor in losses_dict.values():
    tf.losses.add_loss(loss_tensor)


def train(create_tensor_dict_fn,
          create_model_fn,
          train_config,
          master,
          task,
          num_clones,
          worker_replicas,
          clone_on_cpu,
          ps_tasks,
          worker_job_name,
          is_chief,
          train_dir,
          graph_hook_fn=None):
  """Training function for detection models.

  Args:
    create_tensor_dict_fn: a function to create a tensor input dictionary.
    create_model_fn: a function that creates a DetectionModel and generates
                     losses.
    train_config: a train_pb2.TrainConfig protobuf.
    master: BNS name of the TensorFlow master to use.
    task: The task id of this training instance.
    num_clones: The number of clones to run per machine.
    worker_replicas: The number of work replicas to train with.
    clone_on_cpu: True if clones should be forced to run on CPU.
    ps_tasks: Number of parameter server tasks.
    worker_job_name: Name of the worker job.
    is_chief: Whether this replica is the chief replica.
    train_dir: Directory to write checkpoints and training summaries to.
    graph_hook_fn: Optional function that is called after the inference graph is
      built (before optimization). This is helpful to perform additional changes
      to the training graph such as adding FakeQuant ops. The function should
      modify the default graph.

  Raises:
    ValueError: If both num_clones > 1 and train_config.sync_replicas is true.
  """

  detection_model = create_model_fn()
  data_augmentation_options = [
      preprocessor_builder.build(step)
      for step in train_config.data_augmentation_options]

  with tf.Graph().as_default():
    # Build a configuration specifying multi-GPU and multi-replicas.
    deploy_config = model_deploy.DeploymentConfig(
        num_clones=num_clones,
        clone_on_cpu=clone_on_cpu,
        replica_id=task,
        num_replicas=worker_replicas,
        num_ps_tasks=ps_tasks,
        worker_job_name=worker_job_name)

    # Place the global step on the device storing the variables.
    with tf.device(deploy_config.variables_device()):
      global_step = slim.create_global_step()

    if num_clones != 1 and train_config.sync_replicas:
      raise ValueError('In Synchronous SGD mode num_clones must ',
                       'be 1. Found num_clones: {}'.format(num_clones))
    batch_size = train_config.batch_size // num_clones
    if train_config.sync_replicas:
      batch_size //= train_config.replicas_to_aggregate

    with tf.device(deploy_config.inputs_device()):
      input_queue = create_input_queue(
          batch_size, create_tensor_dict_fn,
          train_config.batch_queue_capacity,
          train_config.num_batch_queue_threads,
          train_config.prefetch_queue_capacity, data_augmentation_options)

    # Gather initial summaries.
    # TODO(rathodv): See if summaries can be added/extracted from global tf
    # collections so that they don't have to be passed around.
    summaries = set(tf.get_collection(tf.GraphKeys.SUMMARIES))
    global_summaries = set([])

    model_fn = functools.partial(_create_losses,
                                 create_model_fn=create_model_fn,
                                 train_config=train_config)
    clones = model_deploy.create_clones(deploy_config, model_fn, [input_queue])
    first_clone_scope = clones[0].scope

    if graph_hook_fn:
      with tf.device(deploy_config.variables_device()):
        graph_hook_fn()

    # Gather update_ops from the first clone. These contain, for example,
    # the updates for the batch_norm variables created by model_fn.
    update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS, first_clone_scope)

    with tf.device(deploy_config.optimizer_device()):
      training_optimizer, optimizer_summary_vars = optimizer_builder.build(
          train_config.optimizer)
      for var in optimizer_summary_vars:
        tf.summary.scalar(var.op.name, var, family='LearningRate')

    sync_optimizer = None
    if train_config.sync_replicas:
      training_optimizer = tf.train.SyncReplicasOptimizer(
          training_optimizer,
          replicas_to_aggregate=train_config.replicas_to_aggregate,
          total_num_replicas=worker_replicas)
      sync_optimizer = training_optimizer

    with tf.device(deploy_config.optimizer_device()):
      regularization_losses = (None if train_config.add_regularization_loss
                               else [])
      total_loss, grads_and_vars = model_deploy.optimize_clones(
          clones, training_optimizer,
          regularization_losses=regularization_losses)
      total_loss = tf.check_numerics(total_loss, 'LossTensor is inf or nan.')

      # Optionally multiply bias gradients by train_config.bias_grad_multiplier.
      if train_config.bias_grad_multiplier:
        biases_regex_list = ['.*/biases']
        grads_and_vars = variables_helper.multiply_gradients_matching_regex(
            grads_and_vars,
            biases_regex_list,
            multiplier=train_config.bias_grad_multiplier)

      # Optionally freeze some layers by setting their gradients to be zero.
      if train_config.freeze_variables:
        grads_and_vars = variables_helper.freeze_gradients_matching_regex(
            grads_and_vars, train_config.freeze_variables)

      # Optionally clip gradients
      if train_config.gradient_clipping_by_norm > 0:
        with tf.name_scope('clip_grads'):
          grads_and_vars = slim.learning.clip_gradient_norms(
              grads_and_vars, train_config.gradient_clipping_by_norm)

      # Create gradient updates.
      grad_updates = training_optimizer.apply_gradients(grads_and_vars,
                                                        global_step=global_step)
      update_ops.append(grad_updates)
      update_op = tf.group(*update_ops, name='update_barrier')
      with tf.control_dependencies([update_op]):
        train_tensor = tf.identity(total_loss, name='train_op')

    # Add summaries.
    for model_var in slim.get_model_variables():
      global_summaries.add(tf.summary.histogram('ModelVars/' +
                                                model_var.op.name, model_var))
    for loss_tensor in tf.losses.get_losses():
      global_summaries.add(tf.summary.scalar('Losses/' + loss_tensor.op.name,
                                             loss_tensor))
    global_summaries.add(
        tf.summary.scalar('Losses/TotalLoss', tf.losses.get_total_loss()))

    # Add the summaries from the first clone. These contain the summaries
    # created by model_fn and either optimize_clones() or _gather_clone_loss().
    summaries |= set(tf.get_collection(tf.GraphKeys.SUMMARIES,
                                       first_clone_scope))
    summaries |= global_summaries

    # Merge all summaries together.
    summary_op = tf.summary.merge(list(summaries), name='summary_op')

    # Soft placement allows placing on CPU ops without GPU implementation.
    session_config = tf.ConfigProto(allow_soft_placement=True,
                                    log_device_placement=False)

    # Save checkpoints regularly.
    keep_checkpoint_every_n_hours = train_config.keep_checkpoint_every_n_hours
    saver = tf.train.Saver(
        keep_checkpoint_every_n_hours=keep_checkpoint_every_n_hours)

    # Create ops required to initialize the model from a given checkpoint.
    init_fn = None
    if train_config.fine_tune_checkpoint:
      if not train_config.fine_tune_checkpoint_type:
        # train_config.from_detection_checkpoint field is deprecated. For
        # backward compatibility, fine_tune_checkpoint_type is set based on
        # from_detection_checkpoint.
        if train_config.from_detection_checkpoint:
          train_config.fine_tune_checkpoint_type = 'detection'
        else:
          train_config.fine_tune_checkpoint_type = 'classification'
      var_map = detection_model.restore_map(
          fine_tune_checkpoint_type=train_config.fine_tune_checkpoint_type,
          load_all_detection_checkpoint_vars=(
              train_config.load_all_detection_checkpoint_vars))
      available_var_map = (variables_helper.
                           get_variables_available_in_checkpoint(
                               var_map, train_config.fine_tune_checkpoint,
                               include_global_step=False))
      init_saver = tf.train.Saver(available_var_map)
      def initializer_fn(sess):
        init_saver.restore(sess, train_config.fine_tune_checkpoint)
      init_fn = initializer_fn

    slim.learning.train(
        train_tensor,
        logdir=train_dir,
        master=master,
        is_chief=is_chief,
        session_config=session_config,
        startup_delay_steps=train_config.startup_delay_steps,
        init_fn=init_fn,
        summary_op=summary_op,
        number_of_steps=(
            train_config.num_steps if train_config.num_steps else None),
        save_summaries_secs=120,
        sync_optimizer=sync_optimizer,
        saver=saver)
