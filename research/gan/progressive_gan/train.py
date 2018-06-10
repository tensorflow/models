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
"""Train a progressive GAN model.

See https://arxiv.org/abs/1710.10196 for details about the model.

See https://github.com/tkarras/progressive_growing_of_gans for the original
theano implementation.
"""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os


from absl import logging
import numpy as np
import tensorflow as tf

import networks

tfgan = tf.contrib.gan


def make_train_sub_dir(stage_id, **kwargs):
  """Returns the log directory for training stage `stage_id`."""
  return os.path.join(kwargs['train_root_dir'], 'stage_{:05d}'.format(stage_id))


def make_resolution_schedule(**kwargs):
  """Returns an object of `ResolutionSchedule`."""
  return networks.ResolutionSchedule(
      start_resolutions=(kwargs['start_height'], kwargs['start_width']),
      scale_base=kwargs['scale_base'],
      num_resolutions=kwargs['num_resolutions'])


def get_stage_ids(**kwargs):
  """Returns a list of stage ids.

  Args:
    **kwargs: A dictionary of
        'train_root_dir': A string of root directory of training logs.
        'num_resolutions': An integer of number of progressive resolutions.
  """
  train_sub_dirs = [
      sub_dir for sub_dir in tf.gfile.ListDirectory(kwargs['train_root_dir'])
      if sub_dir.startswith('stage_')
  ]

  # If fresh start, start with start_stage_id = 0
  # If has been trained for n = len(train_sub_dirs) stages, start with the last
  # stage, i.e. start_stage_id = n - 1.
  start_stage_id = max(0, len(train_sub_dirs) - 1)

  return range(start_stage_id, get_total_num_stages(**kwargs))


def get_total_num_stages(**kwargs):
  """Returns total number of training stages."""
  return 2 * kwargs['num_resolutions'] - 1


def get_stage_info(stage_id, **kwargs):
  """Returns information for a training stage.

  Args:
    stage_id: An integer of training stage index.
    **kwargs: A dictionary of
        'num_resolutions': An integer of number of progressive resolutions.
        'stable_stage_num_images': An integer of number of training images in
            the stable stage.
        'transition_stage_num_images': An integer of number of training images
            in the transition stage.
        'total_num_images': An integer of total number of training images.

  Returns:
    A tuple of integers. The first entry is the number of blocks. The second
    entry is the accumulated total number of training images when stage
    `stage_id` is finished.

  Raises:
    ValueError: If `stage_id` is not in [0, total number of stages).
  """
  total_num_stages = get_total_num_stages(**kwargs)
  if not (stage_id >= 0 and stage_id < total_num_stages):
    raise ValueError(
        '`stage_id` must be in [0, {0}), but instead was {1}'.format(
            total_num_stages, stage_id))

  # Even stage_id: stable training stage.
  # Odd stage_id: transition training stage.
  num_blocks = (stage_id + 1) // 2 + 1
  num_images = ((stage_id // 2 + 1) * kwargs['stable_stage_num_images'] + (
      (stage_id + 1) // 2) * kwargs['transition_stage_num_images'])

  total_num_images = kwargs['total_num_images']
  if stage_id >= total_num_stages - 1:
    num_images = total_num_images
  num_images = min(num_images, total_num_images)

  return num_blocks, num_images


def make_latent_vectors(num, **kwargs):
  """Returns a batch of `num` random latent vectors."""
  return tf.random_normal([num, kwargs['latent_vector_size']], dtype=tf.float32)


def make_interpolated_latent_vectors(num_rows, num_columns, **kwargs):
  """Returns a batch of linearly interpolated latent vectors.

  Given two randomly generated latent vector za and zb, it can generate
  a row of `num_columns` interpolated latent vectors, i.e.
  [..., za + (zb - za) * i / (num_columns - 1), ...] where
  i = 0, 1, ..., `num_columns` - 1.

  This function produces `num_rows` such rows and returns a (flattened)
  batch of latent vectors with batch size `num_rows * num_columns`.

  Args:
    num_rows: An integer. Number of rows of interpolated latent vectors.
    num_columns: An integer. Number of interpolated latent vectors in each row.
    **kwargs: A dictionary of
        'latent_vector_size': An integer of latent vector size.

  Returns:
    A `Tensor` of shape `[num_rows * num_columns, latent_vector_size]`.
  """
  ans = []
  for _ in range(num_rows):
    z = tf.random_normal([2, kwargs['latent_vector_size']])
    r = tf.reshape(
        tf.to_float(tf.range(num_columns)) / (num_columns - 1), [-1, 1])
    dz = z[1] - z[0]
    ans.append(z[0] + tf.stack([dz] * num_columns) * r)
  return tf.concat(ans, axis=0)


def define_loss(gan_model, **kwargs):
  """Defines progressive GAN losses.

  The generator and discriminator both use wasserstein loss. In addition,
  a small penalty term is added to the discriminator loss to prevent it getting
  too large.

  Args:
    gan_model: A `GANModel` namedtuple.
    **kwargs: A dictionary of
        'gradient_penalty_weight': A float of gradient norm target for
            wasserstein loss.
        'gradient_penalty_target': A float of gradient penalty weight for
            wasserstein loss.
        'real_score_penalty_weight': A float of Additional penalty to keep
            the scores from drifting too far from zero.

  Returns:
    A `GANLoss` namedtuple.
  """
  gan_loss = tfgan.gan_loss(
      gan_model,
      generator_loss_fn=tfgan.losses.wasserstein_generator_loss,
      discriminator_loss_fn=tfgan.losses.wasserstein_discriminator_loss,
      gradient_penalty_weight=kwargs['gradient_penalty_weight'],
      gradient_penalty_target=kwargs['gradient_penalty_target'],
      gradient_penalty_epsilon=0.0)

  real_score_penalty = tf.reduce_mean(
      tf.square(gan_model.discriminator_real_outputs))
  tf.summary.scalar('real_score_penalty', real_score_penalty)

  return gan_loss._replace(
      discriminator_loss=(
          gan_loss.discriminator_loss +
          kwargs['real_score_penalty_weight'] * real_score_penalty))


def define_train_ops(gan_model, gan_loss, **kwargs):
  """Defines progressive GAN train ops.

  Args:
    gan_model: A `GANModel` namedtuple.
    gan_loss: A `GANLoss` namedtuple.
    **kwargs: A dictionary of
        'adam_beta1': A float of Adam optimizer beta1.
        'adam_beta2': A float of Adam optimizer beta2.
        'generator_learning_rate': A float of generator learning rate.
        'discriminator_learning_rate': A float of discriminator learning rate.

  Returns:
    A tuple of `GANTrainOps` namedtuple and a list variables tracking the state
    of optimizers.
  """
  with tf.variable_scope('progressive_gan_train_ops') as var_scope:
    beta1, beta2 = kwargs['adam_beta1'], kwargs['adam_beta2']
    gen_opt = tf.train.AdamOptimizer(kwargs['generator_learning_rate'], beta1,
                                     beta2)
    dis_opt = tf.train.AdamOptimizer(kwargs['discriminator_learning_rate'],
                                     beta1, beta2)
    gan_train_ops = tfgan.gan_train_ops(gan_model, gan_loss, gen_opt, dis_opt)
  return gan_train_ops, tf.get_collection(
      tf.GraphKeys.GLOBAL_VARIABLES, scope=var_scope.name)


def add_generator_smoothing_ops(generator_ema, gan_model, gan_train_ops):
  """Adds generator smoothing ops."""
  with tf.control_dependencies([gan_train_ops.generator_train_op]):
    new_generator_train_op = generator_ema.apply(gan_model.generator_variables)

  gan_train_ops = gan_train_ops._replace(
      generator_train_op=new_generator_train_op)
  generator_vars_to_restore = generator_ema.variables_to_restore(
      gan_model.generator_variables)
  return gan_train_ops, generator_vars_to_restore


def build_model(stage_id, real_images, **kwargs):
  """Builds progressive GAN model.

  Args:
    stage_id: An integer of training stage index.
    real_images: A 4D `Tensor` of NHWC format.
    **kwargs: A dictionary of
        'batch_size': Number of training images in each minibatch.
        'start_height': An integer of start image height.
        'start_width': An integer of start image width.
        'scale_base': An integer of resolution multiplier.
        'num_resolutions': An integer of number of progressive resolutions.
        'stable_stage_num_images': An integer of number of training images in
            the stable stage.
        'transition_stage_num_images': An integer of number of training images
            in the transition stage.
        'total_num_images': An integer of total number of training images.
        'kernel_size': Convolution kernel size.
        'colors': Number of image channels.
        'to_rgb_use_tanh_activation': Whether to apply tanh activation when
            output rgb.
        'fmap_base': Base number of filters.
        'fmap_decay': Decay of number of filters.
        'fmap_max': Max number of filters.
        'latent_vector_size': An integer of latent vector size.
        'gradient_penalty_weight': A float of gradient norm target for
            wasserstein loss.
        'gradient_penalty_target': A float of gradient penalty weight for
            wasserstein loss.
        'real_score_penalty_weight': A float of Additional penalty to keep
            the scores from drifting too far from zero.
        'adam_beta1': A float of Adam optimizer beta1.
        'adam_beta2': A float of Adam optimizer beta2.
        'generator_learning_rate': A float of generator learning rate.
        'discriminator_learning_rate': A float of discriminator learning rate.

  Returns:
    An inernal object that wraps all information about the model.
  """
  batch_size = kwargs['batch_size']
  kernel_size = kwargs['kernel_size']
  colors = kwargs['colors']
  resolution_schedule = make_resolution_schedule(**kwargs)

  num_blocks, num_images = get_stage_info(stage_id, **kwargs)

  global_step = tf.train.get_or_create_global_step()
  current_image_id = global_step * batch_size
  tf.summary.scalar('current_image_id', current_image_id)

  progress = networks.compute_progress(
      current_image_id, kwargs['stable_stage_num_images'],
      kwargs['transition_stage_num_images'], num_blocks)
  tf.summary.scalar('progress', progress)

  real_images = networks.blend_images(
      real_images, progress, resolution_schedule, num_blocks=num_blocks)

  def _num_filters_fn(block_id):
    """Computes number of filters of block `block_id`."""
    return networks.num_filters(block_id, kwargs['fmap_base'],
                                kwargs['fmap_decay'], kwargs['fmap_max'])

  def _generator_fn(z):
    """Builds generator network."""
    return networks.generator(
        z,
        progress,
        _num_filters_fn,
        resolution_schedule,
        num_blocks=num_blocks,
        kernel_size=kernel_size,
        colors=colors,
        to_rgb_activation=(tf.tanh
                           if kwargs['to_rgb_use_tanh_activation'] else None))

  def _discriminator_fn(x):
    """Builds discriminator network."""
    return networks.discriminator(
        x,
        progress,
        _num_filters_fn,
        resolution_schedule,
        num_blocks=num_blocks,
        kernel_size=kernel_size)

  ########## Define model.
  z = make_latent_vectors(batch_size, **kwargs)

  gan_model = tfgan.gan_model(
      generator_fn=lambda z: _generator_fn(z)[0],
      discriminator_fn=lambda x, unused_z: _discriminator_fn(x)[0],
      real_data=real_images,
      generator_inputs=z)

  ########## Define loss.
  gan_loss = define_loss(gan_model, **kwargs)

  ########## Define train ops.
  gan_train_ops, optimizer_var_list = define_train_ops(gan_model, gan_loss,
                                                       **kwargs)

  ########## Generator smoothing.
  generator_ema = tf.train.ExponentialMovingAverage(decay=0.999)
  gan_train_ops, generator_vars_to_restore = add_generator_smoothing_ops(
      generator_ema, gan_model, gan_train_ops)

  class Model(object):
    pass

  model = Model()
  model.resolution_schedule = resolution_schedule
  model.stage_id = stage_id
  model.num_images = num_images
  model.num_blocks = num_blocks
  model.global_step = global_step
  model.current_image_id = current_image_id
  model.progress = progress
  model.num_filters_fn = _num_filters_fn
  model.generator_fn = _generator_fn
  model.discriminator_fn = _discriminator_fn
  model.gan_model = gan_model
  model.gan_loss = gan_loss
  model.gan_train_ops = gan_train_ops
  model.optimizer_var_list = optimizer_var_list
  model.generator_ema = generator_ema
  model.generator_vars_to_restore = generator_vars_to_restore
  return model


def make_var_scope_custom_getter_for_ema(ema):
  """Makes variable scope custom getter."""
  def _custom_getter(getter, name, *args, **kwargs):
    var = getter(name, *args, **kwargs)
    ema_var = ema.average(var)
    return ema_var if ema_var else var
  return _custom_getter


def add_model_summaries(model, **kwargs):
  """Adds model summaries.

  This function adds several useful summaries during training:
  - fake_images: A grid of fake images based on random latent vectors.
  - interp_images: A grid of fake images based on interpolated latent vectors.
  - real_images_blend: A grid of real images.
  - summaries for `gan_model` losses, variable distributions etc.

  Args:
    model: An model object having all information of progressive GAN model,
        e.g. the return of build_model().
    **kwargs: A dictionary of
      'batch_size': Number of training images in each minibatch.
      'fake_grid_size': The fake image grid size for summaries.
      'interp_grid_size': The latent space interpolated image grid size for
          summaries.
      'colors': Number of image channels.
      'latent_vector_size': An integer of latent vector size.
  """
  fake_grid_size = kwargs['fake_grid_size']
  interp_grid_size = kwargs['interp_grid_size']
  colors = kwargs['colors']

  image_shape = list(model.resolution_schedule.final_resolutions)

  fake_batch_size = fake_grid_size**2
  fake_images_shape = [fake_batch_size] + image_shape + [colors]

  interp_batch_size = interp_grid_size**2
  interp_images_shape = [interp_batch_size] + image_shape + [colors]

  # When making prediction, use the ema smoothed generator vars.
  with tf.variable_scope(
      model.gan_model.generator_scope,
      reuse=True,
      custom_getter=make_var_scope_custom_getter_for_ema(model.generator_ema)):
    z_fake = make_latent_vectors(fake_batch_size, **kwargs)
    fake_images = model.gan_model.generator_fn(z_fake)
    fake_images.set_shape(fake_images_shape)

    z_interp = make_interpolated_latent_vectors(interp_grid_size,
                                                interp_grid_size, **kwargs)
    interp_images = model.gan_model.generator_fn(z_interp)
    interp_images.set_shape(interp_images_shape)

  tf.summary.image(
      'fake_images',
      tfgan.eval.eval_utils.image_grid(
          fake_images,
          grid_shape=[fake_grid_size] * 2,
          image_shape=image_shape,
          num_channels=colors),
      max_outputs=1)

  tf.summary.image(
      'interp_images',
      tfgan.eval.eval_utils.image_grid(
          interp_images,
          grid_shape=[interp_grid_size] * 2,
          image_shape=image_shape,
          num_channels=colors),
      max_outputs=1)

  real_grid_size = int(np.sqrt(kwargs['batch_size']))
  tf.summary.image(
      'real_images_blend',
      tfgan.eval.eval_utils.image_grid(
          model.gan_model.real_data[:real_grid_size**2],
          grid_shape=(real_grid_size, real_grid_size),
          image_shape=image_shape,
          num_channels=colors),
      max_outputs=1)

  tfgan.eval.add_gan_model_summaries(model.gan_model)


def make_scaffold(stage_id, optimizer_var_list, **kwargs):
  """Makes a custom scaffold.

  The scaffold
  - restores variables from the last training stage.
  - initializes new variables in the new block.

  Args:
    stage_id: An integer of stage id.
    optimizer_var_list: A list of optimizer variables.
    **kwargs: A dictionary of
        'train_root_dir': A string of root directory of training logs.
        'num_resolutions': An integer of number of progressive resolutions.
        'stable_stage_num_images': An integer of number of training images in
            the stable stage.
        'transition_stage_num_images': An integer of number of training images
            in the transition stage.
        'total_num_images': An integer of total number of training images.

  Returns:
    A `Scaffold` object.
  """
  # Holds variables that from the previous stage and need to be restored.
  restore_var_list = []
  prev_ckpt = None
  curr_ckpt = tf.train.latest_checkpoint(make_train_sub_dir(stage_id, **kwargs))
  if stage_id > 0 and curr_ckpt is None:
    prev_ckpt = tf.train.latest_checkpoint(
        make_train_sub_dir(stage_id - 1, **kwargs))

    num_blocks, _ = get_stage_info(stage_id, **kwargs)
    prev_num_blocks, _ = get_stage_info(stage_id - 1, **kwargs)

    # Holds variables created in the new block of the current stage. If the
    # current stage is a stable stage (except the initial stage), this list
    # will be empty.
    new_block_var_list = []
    for block_id in range(prev_num_blocks + 1, num_blocks + 1):
      new_block_var_list.extend(
          tf.get_collection(
              tf.GraphKeys.GLOBAL_VARIABLES,
              scope='.*/{}/'.format(networks.block_name(block_id))))

    # Every variables that are 1) not for optimizers and 2) from the new block
    # need to be restored.
    restore_var_list = [
        var for var in tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES)
        if var not in set(optimizer_var_list + new_block_var_list)
    ]

  # Add saver op to graph. This saver is used to restore variables from the
  # previous stage.
  saver_for_restore = tf.train.Saver(
      var_list=restore_var_list, allow_empty=True)
  # Add the op to graph that initializes all global variables.
  init_op = tf.global_variables_initializer()

  def _init_fn(unused_scaffold, sess):
    # First initialize every variables.
    sess.run(init_op)
    logging.info('\n'.join([var.name for var in restore_var_list]))
    # Then overwrite variables saved in previous stage.
    if prev_ckpt is not None:
      saver_for_restore.restore(sess, prev_ckpt)

  # Use a dummy init_op here as all initialization is done in init_fn.
  return tf.train.Scaffold(init_op=tf.constant([]), init_fn=_init_fn)


def make_status_message(model):
  """Makes a string `Tensor` of training status."""
  return tf.string_join(
      [
          'Starting train step: ',
          tf.as_string(model.global_step), ', current_image_id: ',
          tf.as_string(model.current_image_id), ', progress: ',
          tf.as_string(model.progress), ', num_blocks: {}'.format(
              model.num_blocks)
      ],
      name='status_message')


def train(model, **kwargs):
  """Trains progressive GAN for stage `stage_id`.

  Args:
    model: An model object having all information of progressive GAN model,
        e.g. the return of build_model().
    **kwargs: A dictionary of
        'train_root_dir': A string of root directory of training logs.
        'master': Name of the TensorFlow master to use.
        'task': The Task ID. This value is used when training with multiple
            workers to identify each worker.
        'save_summaries_num_images': Save summaries in this number of images.
  Returns:
    None.
  """
  batch_size = kwargs['batch_size']

  logging.info('stage_id=%d, num_blocks=%d, num_images=%d', model.stage_id,
               model.num_blocks, model.num_images)

  scaffold = make_scaffold(model.stage_id, model.optimizer_var_list, **kwargs)

  tfgan.gan_train(
      model.gan_train_ops,
      logdir=make_train_sub_dir(model.stage_id, **kwargs),
      get_hooks_fn=tfgan.get_sequential_train_hooks(tfgan.GANTrainSteps(1, 1)),
      hooks=[
          tf.train.StopAtStepHook(last_step=model.num_images // batch_size),
          tf.train.LoggingTensorHook(
              [make_status_message(model)], every_n_iter=10)
      ],
      master=kwargs['master'],
      is_chief=(kwargs['task'] == 0),
      scaffold=scaffold,
      save_checkpoint_secs=600,
      save_summaries_steps=(kwargs['save_summaries_num_images'] // batch_size))
