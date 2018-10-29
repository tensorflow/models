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

"""Detection model trainer.

This file provides a generic training method that can be used to train a
DetectionModel.
"""

import functools
import tensorflow as tf
from google3.pyglib import logging

from object_detection.builders import optimizer_builder
from object_detection.core import standard_fields as fields
from object_detection.utils import ops as util_ops
from object_detection.utils import variables_helper
from deployment import model_deploy

slim = tf.contrib.slim


def create_input_queue(create_tensor_dict_fn):
  """Sets up reader, prefetcher and returns input queue.

  Args:
    create_tensor_dict_fn: function to create tensor dictionary.

  Returns:
    all_dict: A dictionary holds tensors for images, boxes, and targets.
  """
  tensor_dict = create_tensor_dict_fn()
  all_dict = {}

  num_images = len(tensor_dict[fields.InputDataFields.image])
  all_dict['batch'] = tensor_dict['batch']
  del tensor_dict['batch']

  for i in range(num_images):
    suffix = str(i)
    for key, val in tensor_dict.items():
      all_dict[key + suffix] = val[i]

    all_dict[fields.InputDataFields.image + suffix] = tf.to_float(
        tf.expand_dims(all_dict[fields.InputDataFields.image + suffix], 0))

  return all_dict


def get_inputs(input_queue, num_classes, merge_multiple_label_boxes=False):
  """Dequeues batch and constructs inputs to object detection model.

  Args:
    input_queue: BatchQueue object holding enqueued tensor_dicts.
    num_classes: Number of classes.
    merge_multiple_label_boxes: Whether to merge boxes with multiple labels
      or not. Defaults to false. Merged boxes are represented with a single
      box and a k-hot encoding of the multiple labels associated with the
      boxes.

  Returns:
    images: a list of 3-D float tensor of images.
    image_keys: a list of string keys for the images.
    locations: a list of tensors of shape [num_boxes, 4] containing the corners
      of the groundtruth boxes.
    classes: a list of padded one-hot tensors containing target classes.
    masks: a list of 3-D float tensors of shape [num_boxes, image_height,
      image_width] containing instance masks for objects if present in the
      input_queue. Else returns None.
    keypoints: a list of 3-D float tensors of shape [num_boxes, num_keypoints,
      2] containing keypoints for objects if present in the
      input queue. Else returns None.
  """
  read_data_list = input_queue
  label_id_offset = 1

  def extract_images_and_targets(read_data):
    """Extract images and targets from the input dict."""
    suffix = 0

    images = []
    keys = []
    locations = []
    classes = []
    masks = []
    keypoints = []

    while fields.InputDataFields.image + str(suffix) in read_data:
      image = read_data[fields.InputDataFields.image + str(suffix)]
      key = ''
      if fields.InputDataFields.source_id in read_data:
        key = read_data[fields.InputDataFields.source_id + str(suffix)]
      location_gt = (
          read_data[fields.InputDataFields.groundtruth_boxes + str(suffix)])
      classes_gt = tf.cast(
          read_data[fields.InputDataFields.groundtruth_classes + str(suffix)],
          tf.int32)
      classes_gt -= label_id_offset
      masks_gt = read_data.get(
          fields.InputDataFields.groundtruth_instance_masks + str(suffix))
      keypoints_gt = read_data.get(
          fields.InputDataFields.groundtruth_keypoints + str(suffix))

      if merge_multiple_label_boxes:
        location_gt, classes_gt, _ = util_ops.merge_boxes_with_multiple_labels(
            location_gt, classes_gt, num_classes)
      else:
        classes_gt = util_ops.padded_one_hot_encoding(
            indices=classes_gt, depth=num_classes, left_pad=0)

      # Batch read input data and groundtruth. Images and locations, classes by
      # default should have the same number of items.
      images.append(image)
      keys.append(key)
      locations.append(location_gt)
      classes.append(classes_gt)
      masks.append(masks_gt)
      keypoints.append(keypoints_gt)

      suffix += 1

    return (images, keys, locations, classes, masks, keypoints)

  return extract_images_and_targets(read_data_list)


def _create_losses(input_queue, create_model_fn, train_config):
  """Creates loss function for a DetectionModel.

  Args:
    input_queue: BatchQueue object holding enqueued tensor_dicts.
    create_model_fn: A function to create the DetectionModel.
    train_config: a train_pb2.TrainConfig protobuf.
  """

  detection_model = create_model_fn()
  (images, _, groundtruth_boxes_list, groundtruth_classes_list,
   groundtruth_masks_list, groundtruth_keypoints_list) = get_inputs(
       input_queue, detection_model.num_classes,
       train_config.merge_multiple_label_boxes)

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

  detection_model.provide_groundtruth(
      groundtruth_boxes_list, groundtruth_classes_list, groundtruth_masks_list,
      groundtruth_keypoints_list)
  prediction_dict = detection_model.predict(images, true_image_shapes,
                                            input_queue['batch'])

  losses_dict = detection_model.loss(prediction_dict, true_image_shapes)
  for loss_tensor in losses_dict.values():
    tf.losses.add_loss(loss_tensor)


def get_restore_checkpoint_ops(restore_checkpoints, detection_model,
                               train_config):
  """Restore checkpoint from saved checkpoints.

  Args:
    restore_checkpoints: loaded checkpoints.
    detection_model: Object detection model built from config file.
    train_config: a train_pb2.TrainConfig protobuf.

  Returns:
    restorers: A list ops to init the model from checkpoints.

  """
  restorers = []
  vars_restored = []
  for restore_checkpoint in restore_checkpoints:
    var_map = detection_model.restore_map(
        fine_tune_checkpoint_type=train_config.fine_tune_checkpoint_type)
    available_var_map = (
        variables_helper.get_variables_available_in_checkpoint(
            var_map, restore_checkpoint))
    for var_name, var in available_var_map.iteritems():
      if var in vars_restored:
        logging.info('Variable %s contained in multiple checkpoints',
                     var.op.name)
        del available_var_map[var_name]
      else:
        vars_restored.append(var)

    # Initialize from ExponentialMovingAverages if possible.
    available_ema_var_map = {}
    ckpt_reader = tf.train.NewCheckpointReader(restore_checkpoint)
    ckpt_vars_to_shape_map = ckpt_reader.get_variable_to_shape_map()
    for var_name, var in available_var_map.iteritems():
      var_name_ema = var_name + '/ExponentialMovingAverage'
      if var_name_ema in ckpt_vars_to_shape_map:
        available_ema_var_map[var_name_ema] = var
      else:
        available_ema_var_map[var_name] = var
    available_var_map = available_ema_var_map
    init_saver = tf.train.Saver(available_var_map)
    if available_var_map.keys():
      restorers.append(init_saver)
    else:
      logging.info('WARNING: Checkpoint %s has no restorable variables',
                   restore_checkpoint)

    return restorers


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
    graph_hook_fn: Optional function that is called after the training graph is
      completely built. This is helpful to perform additional changes to the
      training graph such as optimizing batchnorm. The function should modify
      the default graph.
  """

  detection_model = create_model_fn()

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

    with tf.device(deploy_config.inputs_device()):
      input_queue = create_input_queue(create_tensor_dict_fn)

    # Gather initial summaries.
    # TODO(rathodv): See if summaries can be added/extracted from global tf
    # collections so that they don't have to be passed around.
    summaries = set(tf.get_collection(tf.GraphKeys.SUMMARIES))
    global_summaries = set([])

    model_fn = functools.partial(
        _create_losses,
        create_model_fn=create_model_fn,
        train_config=train_config)
    clones = model_deploy.create_clones(deploy_config, model_fn, [input_queue])
    first_clone_scope = clones[0].scope

    # Gather update_ops from the first clone. These contain, for example,
    # the updates for the batch_norm variables created by model_fn.
    update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS, first_clone_scope)

    with tf.device(deploy_config.optimizer_device()):
      training_optimizer, optimizer_summary_vars = optimizer_builder.build(
          train_config.optimizer)
      for var in optimizer_summary_vars:
        tf.summary.scalar(var.op.name, var)

    sync_optimizer = None
    if train_config.sync_replicas:
      training_optimizer = tf.train.SyncReplicasOptimizer(
          training_optimizer,
          replicas_to_aggregate=train_config.replicas_to_aggregate,
          total_num_replicas=train_config.worker_replicas)
      sync_optimizer = training_optimizer

    # Create ops required to initialize the model from a given checkpoint.
    init_fn = None
    if train_config.fine_tune_checkpoint:
      restore_checkpoints = [
          path.strip() for path in train_config.fine_tune_checkpoint.split(',')
      ]

      restorers = get_restore_checkpoint_ops(restore_checkpoints,
                                             detection_model, train_config)

      def initializer_fn(sess):
        for i, restorer in enumerate(restorers):
          restorer.restore(sess, restore_checkpoints[i])

      init_fn = initializer_fn

    with tf.device(deploy_config.optimizer_device()):
      regularization_losses = (
          None if train_config.add_regularization_loss else [])
      total_loss, grads_and_vars = model_deploy.optimize_clones(
          clones,
          training_optimizer,
          regularization_losses=regularization_losses)
      total_loss = tf.check_numerics(total_loss, 'LossTensor is inf or nan.')

      # Optionally multiply bias gradients by train_config.bias_grad_multiplier.
      if train_config.bias_grad_multiplier:
        biases_regex_list = ['.*/biases']
        grads_and_vars = variables_helper.multiply_gradients_matching_regex(
            grads_and_vars,
            biases_regex_list,
            multiplier=train_config.bias_grad_multiplier)

      # Optionally clip gradients
      if train_config.gradient_clipping_by_norm > 0:
        with tf.name_scope('clip_grads'):
          grads_and_vars = slim.learning.clip_gradient_norms(
              grads_and_vars, train_config.gradient_clipping_by_norm)

      moving_average_variables = slim.get_model_variables()
      variable_averages = tf.train.ExponentialMovingAverage(0.9999, global_step)
      update_ops.append(variable_averages.apply(moving_average_variables))

      # Create gradient updates.
      grad_updates = training_optimizer.apply_gradients(
          grads_and_vars, global_step=global_step)
      update_ops.append(grad_updates)
      update_op = tf.group(*update_ops, name='update_barrier')
      with tf.control_dependencies([update_op]):
        train_tensor = tf.identity(total_loss, name='train_op')

    if graph_hook_fn:
      with tf.device(deploy_config.variables_device()):
        graph_hook_fn()

    # Add summaries.
    for model_var in slim.get_model_variables():
      global_summaries.add(tf.summary.histogram(model_var.op.name, model_var))
    for loss_tensor in tf.losses.get_losses():
      global_summaries.add(tf.summary.scalar(loss_tensor.op.name, loss_tensor))
    global_summaries.add(
        tf.summary.scalar('TotalLoss', tf.losses.get_total_loss()))

    # Add the summaries from the first clone. These contain the summaries
    # created by model_fn and either optimize_clones() or _gather_clone_loss().
    summaries |= set(
        tf.get_collection(tf.GraphKeys.SUMMARIES, first_clone_scope))
    summaries |= set(tf.get_collection(tf.GraphKeys.SUMMARIES, 'critic_loss'))
    summaries |= global_summaries

    # Merge all summaries together.
    summary_op = tf.summary.merge(list(summaries), name='summary_op')

    # Soft placement allows placing on CPU ops without GPU implementation.
    session_config = tf.ConfigProto(
        allow_soft_placement=True, log_device_placement=False)

    # Save checkpoints regularly.
    keep_checkpoint_every_n_hours = train_config.keep_checkpoint_every_n_hours
    saver = tf.train.Saver(
        keep_checkpoint_every_n_hours=keep_checkpoint_every_n_hours)

    slim.learning.train(
        train_tensor,
        logdir=train_dir,
        master=master,
        is_chief=is_chief,
        session_config=session_config,
        startup_delay_steps=train_config.startup_delay_steps,
        init_fn=init_fn,
        summary_op=summary_op,
        number_of_steps=(train_config.num_steps
                         if train_config.num_steps else None),
        save_summaries_secs=120,
        sync_optimizer=sync_optimizer,
        saver=saver)
