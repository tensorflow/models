# Copyright 2018 The TensorFlow Authors All Rights Reserved.
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

"""Training script for the FEELVOS model.

See model.py for more details and usage.
"""
import six
import tensorflow as tf

from feelvos import common
from feelvos import model
from feelvos.datasets import video_dataset
from feelvos.utils import embedding_utils
from feelvos.utils import train_utils
from feelvos.utils import video_input_generator
from deployment import model_deploy

slim = tf.contrib.slim
prefetch_queue = slim.prefetch_queue
flags = tf.app.flags
FLAGS = flags.FLAGS

# Settings for multi-GPUs/multi-replicas training.

flags.DEFINE_integer('num_clones', 1, 'Number of clones to deploy.')

flags.DEFINE_boolean('clone_on_cpu', False, 'Use CPUs to deploy clones.')

flags.DEFINE_integer('num_replicas', 1, 'Number of worker replicas.')

flags.DEFINE_integer('startup_delay_steps', 15,
                     'Number of training steps between replicas startup.')

flags.DEFINE_integer('num_ps_tasks', 0,
                     'The number of parameter servers. If the value is 0, then '
                     'the parameters are handled locally by the worker.')

flags.DEFINE_string('master', '', 'BNS name of the tensorflow server')

flags.DEFINE_integer('task', 0, 'The task ID.')

# Settings for logging.

flags.DEFINE_string('train_logdir', None,
                    'Where the checkpoint and logs are stored.')

flags.DEFINE_integer('log_steps', 10,
                     'Display logging information at every log_steps.')

flags.DEFINE_integer('save_interval_secs', 1200,
                     'How often, in seconds, we save the model to disk.')

flags.DEFINE_integer('save_summaries_secs', 600,
                     'How often, in seconds, we compute the summaries.')

# Settings for training strategy.

flags.DEFINE_enum('learning_policy', 'poly', ['poly', 'step'],
                  'Learning rate policy for training.')

flags.DEFINE_float('base_learning_rate', 0.0007,
                   'The base learning rate for model training.')

flags.DEFINE_float('learning_rate_decay_factor', 0.1,
                   'The rate to decay the base learning rate.')

flags.DEFINE_integer('learning_rate_decay_step', 2000,
                     'Decay the base learning rate at a fixed step.')

flags.DEFINE_float('learning_power', 0.9,
                   'The power value used in the poly learning policy.')

flags.DEFINE_integer('training_number_of_steps', 200000,
                     'The number of steps used for training')

flags.DEFINE_float('momentum', 0.9, 'The momentum value to use')

flags.DEFINE_integer('train_batch_size', 6,
                     'The number of images in each batch during training.')

flags.DEFINE_integer('train_num_frames_per_video', 3,
                     'The number of frames used per video during training')

flags.DEFINE_float('weight_decay', 0.00004,
                   'The value of the weight decay for training.')

flags.DEFINE_multi_integer('train_crop_size', [465, 465],
                           'Image crop size [height, width] during training.')

flags.DEFINE_float('last_layer_gradient_multiplier', 1.0,
                   'The gradient multiplier for last layers, which is used to '
                   'boost the gradient of last layers if the value > 1.')

flags.DEFINE_boolean('upsample_logits', True,
                     'Upsample logits during training.')

flags.DEFINE_integer('batch_capacity_factor', 16, 'Batch capacity factor.')

flags.DEFINE_integer('num_readers', 1, 'Number of readers for data provider.')

flags.DEFINE_integer('batch_num_threads', 1, 'Batch number of threads.')

flags.DEFINE_integer('prefetch_queue_capacity_factor', 32,
                     'Prefetch queue capacity factor.')

flags.DEFINE_integer('prefetch_queue_num_threads', 1,
                     'Prefetch queue number of threads.')

flags.DEFINE_integer('train_max_neighbors_per_object', 1024,
                     'The maximum number of candidates for the nearest '
                     'neighbor query per object after subsampling')

# Settings for fine-tuning the network.

flags.DEFINE_string('tf_initial_checkpoint', None,
                    'The initial checkpoint in tensorflow format.')

flags.DEFINE_boolean('initialize_last_layer', False,
                     'Initialize the last layer.')

flags.DEFINE_boolean('last_layers_contain_logits_only', False,
                     'Only consider logits as last layers or not.')

flags.DEFINE_integer('slow_start_step', 0,
                     'Training model with small learning rate for few steps.')

flags.DEFINE_float('slow_start_learning_rate', 1e-4,
                   'Learning rate employed during slow start.')

flags.DEFINE_boolean('fine_tune_batch_norm', False,
                     'Fine tune the batch norm parameters or not.')

flags.DEFINE_float('min_scale_factor', 1.,
                   'Mininum scale factor for data augmentation.')

flags.DEFINE_float('max_scale_factor', 1.3,
                   'Maximum scale factor for data augmentation.')

flags.DEFINE_float('scale_factor_step_size', 0,
                   'Scale factor step size for data augmentation.')

flags.DEFINE_multi_integer('atrous_rates', None,
                           'Atrous rates for atrous spatial pyramid pooling.')

flags.DEFINE_integer('output_stride', 8,
                     'The ratio of input to output spatial resolution.')

flags.DEFINE_boolean('sample_only_first_frame_for_finetuning', False,
                     'Whether to only sample the first frame during '
                     'fine-tuning. This should be False when using lucid data, '
                     'but True when fine-tuning on the first frame only. Only '
                     'has an effect if first_frame_finetuning is True.')

flags.DEFINE_multi_integer('first_frame_finetuning', [0],
                           'Whether to only sample the first frame for '
                           'fine-tuning.')

# Dataset settings.

flags.DEFINE_multi_string('dataset', [], 'Name of the segmentation datasets.')

flags.DEFINE_multi_float('dataset_sampling_probabilities', [],
                         'A list of probabilities to sample each of the '
                         'datasets.')

flags.DEFINE_string('train_split', 'train',
                    'Which split of the dataset to be used for training')

flags.DEFINE_multi_string('dataset_dir', [], 'Where the datasets reside.')

flags.DEFINE_multi_integer('three_frame_dataset', [0],
                           'Whether the dataset has exactly three frames per '
                           'video of which the first is to be used as reference'
                           ' and the two others are consecutive frames to be '
                           'used as query  frames.'
                           'Set true for pascal lucid data.')

flags.DEFINE_boolean('damage_initial_previous_frame_mask', False,
                     'Whether to artificially damage the initial previous '
                     'frame mask. Only has an effect if '
                     'also_attend_to_previous_frame is True.')

flags.DEFINE_float('top_k_percent_pixels', 0.15, 'Float in [0.0, 1.0].'
                   'When its value < 1.0, only compute the loss for the top k'
                   'percent pixels (e.g., the top 20% pixels). This is useful'
                   'for hard pixel mining.')

flags.DEFINE_integer('hard_example_mining_step', 100000,
                     'The training step in which the hard exampling mining '
                     'kicks off. Note that we gradually reduce the mining '
                     'percent to the top_k_percent_pixels. For example, if '
                     'hard_example_mining_step=100K and '
                     'top_k_percent_pixels=0.25, then mining percent will '
                     'gradually reduce from 100% to 25% until 100K steps '
                     'after which we only mine top 25% pixels. Only has an '
                     'effect if top_k_percent_pixels < 1.0')


def _build_deeplab(inputs_queue_or_samples, outputs_to_num_classes,
                   ignore_label):
  """Builds a clone of DeepLab.

  Args:
    inputs_queue_or_samples: A prefetch queue for images and labels, or
      directly a dict of the samples.
    outputs_to_num_classes: A map from output type to the number of classes.
      For example, for the task of semantic segmentation with 21 semantic
      classes, we would have outputs_to_num_classes['semantic'] = 21.
    ignore_label: Ignore label.

  Returns:
    A map of maps from output_type (e.g., semantic prediction) to a
      dictionary of multi-scale logits names to logits. For each output_type,
      the dictionary has keys which correspond to the scales and values which
      correspond to the logits. For example, if `scales` equals [1.0, 1.5],
      then the keys would include 'merged_logits', 'logits_1.00' and
      'logits_1.50'.

  Raises:
    ValueError: If classification_loss is not softmax, softmax_with_attention,
      or triplet.
  """
  if hasattr(inputs_queue_or_samples, 'dequeue'):
    samples = inputs_queue_or_samples.dequeue()
  else:
    samples = inputs_queue_or_samples
  train_crop_size = (None if 0 in FLAGS.train_crop_size else
                     FLAGS.train_crop_size)

  model_options = common.VideoModelOptions(
      outputs_to_num_classes=outputs_to_num_classes,
      crop_size=train_crop_size,
      atrous_rates=FLAGS.atrous_rates,
      output_stride=FLAGS.output_stride)

  if model_options.classification_loss == 'softmax_with_attention':
    clone_batch_size = FLAGS.train_batch_size // FLAGS.num_clones

    # Create summaries of ground truth labels.
    for n in range(clone_batch_size):
      tf.summary.image(
          'gt_label_%d' % n,
          tf.cast(samples[common.LABEL][
              n * FLAGS.train_num_frames_per_video:
              (n + 1) * FLAGS.train_num_frames_per_video],
                  tf.uint8) * 32, max_outputs=FLAGS.train_num_frames_per_video)

    if common.PRECEDING_FRAME_LABEL in samples:
      preceding_frame_label = samples[common.PRECEDING_FRAME_LABEL]
      init_softmax = []
      for n in range(clone_batch_size):
        init_softmax_n = embedding_utils.create_initial_softmax_from_labels(
            preceding_frame_label[n, tf.newaxis],
            samples[common.LABEL][n * FLAGS.train_num_frames_per_video,
                                  tf.newaxis],
            common.parse_decoder_output_stride(),
            reduce_labels=True)
        init_softmax_n = tf.squeeze(init_softmax_n, axis=0)
        init_softmax.append(init_softmax_n)
        tf.summary.image('preceding_frame_label',
                         tf.cast(preceding_frame_label[n, tf.newaxis],
                                 tf.uint8) * 32)
    else:
      init_softmax = None

    outputs_to_scales_to_logits = (
        model.multi_scale_logits_with_nearest_neighbor_matching(
            samples[common.IMAGE],
            model_options=model_options,
            image_pyramid=FLAGS.image_pyramid,
            weight_decay=FLAGS.weight_decay,
            is_training=True,
            fine_tune_batch_norm=FLAGS.fine_tune_batch_norm,
            reference_labels=samples[common.LABEL],
            clone_batch_size=FLAGS.train_batch_size // FLAGS.num_clones,
            num_frames_per_video=FLAGS.train_num_frames_per_video,
            embedding_dimension=FLAGS.embedding_dimension,
            max_neighbors_per_object=FLAGS.train_max_neighbors_per_object,
            k_nearest_neighbors=FLAGS.k_nearest_neighbors,
            use_softmax_feedback=FLAGS.use_softmax_feedback,
            initial_softmax_feedback=init_softmax,
            embedding_seg_feature_dimension=
            FLAGS.embedding_seg_feature_dimension,
            embedding_seg_n_layers=FLAGS.embedding_seg_n_layers,
            embedding_seg_kernel_size=FLAGS.embedding_seg_kernel_size,
            embedding_seg_atrous_rates=FLAGS.embedding_seg_atrous_rates,
            normalize_nearest_neighbor_distances=
            FLAGS.normalize_nearest_neighbor_distances,
            also_attend_to_previous_frame=FLAGS.also_attend_to_previous_frame,
            damage_initial_previous_frame_mask=
            FLAGS.damage_initial_previous_frame_mask,
            use_local_previous_frame_attention=
            FLAGS.use_local_previous_frame_attention,
            previous_frame_attention_window_size=
            FLAGS.previous_frame_attention_window_size,
            use_first_frame_matching=FLAGS.use_first_frame_matching
        ))
  else:
    outputs_to_scales_to_logits = model.multi_scale_logits_v2(
        samples[common.IMAGE],
        model_options=model_options,
        image_pyramid=FLAGS.image_pyramid,
        weight_decay=FLAGS.weight_decay,
        is_training=True,
        fine_tune_batch_norm=FLAGS.fine_tune_batch_norm)

  if model_options.classification_loss == 'softmax':
    for output, num_classes in six.iteritems(outputs_to_num_classes):
      train_utils.add_softmax_cross_entropy_loss_for_each_scale(
          outputs_to_scales_to_logits[output],
          samples[common.LABEL],
          num_classes,
          ignore_label,
          loss_weight=1.0,
          upsample_logits=FLAGS.upsample_logits,
          scope=output)
  elif model_options.classification_loss == 'triplet':
    for output, _ in six.iteritems(outputs_to_num_classes):
      train_utils.add_triplet_loss_for_each_scale(
          FLAGS.train_batch_size // FLAGS.num_clones,
          FLAGS.train_num_frames_per_video,
          FLAGS.embedding_dimension, outputs_to_scales_to_logits[output],
          samples[common.LABEL], scope=output)
  elif model_options.classification_loss == 'softmax_with_attention':
    labels = samples[common.LABEL]
    batch_size = FLAGS.train_batch_size // FLAGS.num_clones
    num_frames_per_video = FLAGS.train_num_frames_per_video
    h, w = train_utils.resolve_shape(labels)[1:3]
    labels = tf.reshape(labels, tf.stack(
        [batch_size, num_frames_per_video, h, w, 1]))
    # Strip the reference labels off.
    if FLAGS.also_attend_to_previous_frame or FLAGS.use_softmax_feedback:
      n_ref_frames = 2
    else:
      n_ref_frames = 1
    labels = labels[:, n_ref_frames:]
    # Merge batch and time dimensions.
    labels = tf.reshape(labels, tf.stack(
        [batch_size * (num_frames_per_video - n_ref_frames), h, w, 1]))

    for output, num_classes in six.iteritems(outputs_to_num_classes):
      train_utils.add_dynamic_softmax_cross_entropy_loss_for_each_scale(
          outputs_to_scales_to_logits[output],
          labels,
          ignore_label,
          loss_weight=1.0,
          upsample_logits=FLAGS.upsample_logits,
          scope=output,
          top_k_percent_pixels=FLAGS.top_k_percent_pixels,
          hard_example_mining_step=FLAGS.hard_example_mining_step)
  else:
    raise ValueError('Only support softmax, softmax_with_attention'
                     ' or triplet for classification_loss.')

  return outputs_to_scales_to_logits


def main(unused_argv):
  # Set up deployment (i.e., multi-GPUs and/or multi-replicas).
  config = model_deploy.DeploymentConfig(
      num_clones=FLAGS.num_clones,
      clone_on_cpu=FLAGS.clone_on_cpu,
      replica_id=FLAGS.task,
      num_replicas=FLAGS.num_replicas,
      num_ps_tasks=FLAGS.num_ps_tasks)

  with tf.Graph().as_default():
    with tf.device(config.inputs_device()):
      train_crop_size = (None if 0 in FLAGS.train_crop_size else
                         FLAGS.train_crop_size)
      assert FLAGS.dataset
      assert len(FLAGS.dataset) == len(FLAGS.dataset_dir)
      if len(FLAGS.first_frame_finetuning) == 1:
        first_frame_finetuning = (list(FLAGS.first_frame_finetuning)
                                  * len(FLAGS.dataset))
      else:
        first_frame_finetuning = FLAGS.first_frame_finetuning
      if len(FLAGS.three_frame_dataset) == 1:
        three_frame_dataset = (list(FLAGS.three_frame_dataset)
                               * len(FLAGS.dataset))
      else:
        three_frame_dataset = FLAGS.three_frame_dataset
      assert len(FLAGS.dataset) == len(first_frame_finetuning)
      assert len(FLAGS.dataset) == len(three_frame_dataset)
      datasets, samples_list = zip(
          *[_get_dataset_and_samples(config, train_crop_size, dataset,
                                     dataset_dir, bool(first_frame_finetuning_),
                                     bool(three_frame_dataset_))
            for dataset, dataset_dir, first_frame_finetuning_,
            three_frame_dataset_ in zip(FLAGS.dataset, FLAGS.dataset_dir,
                                        first_frame_finetuning,
                                        three_frame_dataset)])
      # Note that this way of doing things is wasteful since it will evaluate
      # all branches but just use one of them. But let's do it anyway for now,
      # since it's easy and will probably be fast enough.
      dataset = datasets[0]
      if len(samples_list) == 1:
        samples = samples_list[0]
      else:
        probabilities = FLAGS.dataset_sampling_probabilities
        if probabilities:
          assert len(probabilities) == len(samples_list)
        else:
          # Default to uniform probabilities.
          probabilities = [1.0 / len(samples_list) for _ in samples_list]
        probabilities = tf.constant(probabilities)
        logits = tf.log(probabilities[tf.newaxis])
        rand_idx = tf.squeeze(tf.multinomial(logits, 1, output_dtype=tf.int32),
                              axis=[0, 1])

        def wrap(x):
          def f():
            return x
          return f

        samples = tf.case({tf.equal(rand_idx, idx): wrap(s)
                           for idx, s in enumerate(samples_list)},
                          exclusive=True)

      # Prefetch_queue requires the shape to be known at graph creation time.
      # So we only use it if we crop to a fixed size.
      if train_crop_size is None:
        inputs_queue = samples
      else:
        inputs_queue = prefetch_queue.prefetch_queue(
            samples,
            capacity=FLAGS.prefetch_queue_capacity_factor*config.num_clones,
            num_threads=FLAGS.prefetch_queue_num_threads)

    # Create the global step on the device storing the variables.
    with tf.device(config.variables_device()):
      global_step = tf.train.get_or_create_global_step()

      # Define the model and create clones.
      model_fn = _build_deeplab
      if FLAGS.classification_loss == 'triplet':
        embedding_dim = FLAGS.embedding_dimension
        output_type_to_dim = {'embedding': embedding_dim}
      else:
        output_type_to_dim = {common.OUTPUT_TYPE: dataset.num_classes}
      model_args = (inputs_queue, output_type_to_dim, dataset.ignore_label)
      clones = model_deploy.create_clones(config, model_fn, args=model_args)

      # Gather update_ops from the first clone. These contain, for example,
      # the updates for the batch_norm variables created by model_fn.
      first_clone_scope = config.clone_scope(0)
      update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS, first_clone_scope)

    # Gather initial summaries.
    summaries = set(tf.get_collection(tf.GraphKeys.SUMMARIES))

    # Add summaries for model variables.
    for model_var in tf.contrib.framework.get_model_variables():
      summaries.add(tf.summary.histogram(model_var.op.name, model_var))

    # Add summaries for losses.
    for loss in tf.get_collection(tf.GraphKeys.LOSSES, first_clone_scope):
      summaries.add(tf.summary.scalar('losses/%s' % loss.op.name, loss))

    # Build the optimizer based on the device specification.
    with tf.device(config.optimizer_device()):
      learning_rate = train_utils.get_model_learning_rate(
          FLAGS.learning_policy,
          FLAGS.base_learning_rate,
          FLAGS.learning_rate_decay_step,
          FLAGS.learning_rate_decay_factor,
          FLAGS.training_number_of_steps,
          FLAGS.learning_power,
          FLAGS.slow_start_step,
          FLAGS.slow_start_learning_rate)
      optimizer = tf.train.MomentumOptimizer(learning_rate, FLAGS.momentum)
      summaries.add(tf.summary.scalar('learning_rate', learning_rate))

    startup_delay_steps = FLAGS.task * FLAGS.startup_delay_steps

    with tf.device(config.variables_device()):
      total_loss, grads_and_vars = model_deploy.optimize_clones(
          clones, optimizer)
      total_loss = tf.check_numerics(total_loss, 'Loss is inf or nan.')
      summaries.add(tf.summary.scalar('total_loss', total_loss))

      # Modify the gradients for biases and last layer variables.
      last_layers = model.get_extra_layer_scopes(
          FLAGS.last_layers_contain_logits_only)
      grad_mult = train_utils.get_model_gradient_multipliers(
          last_layers, FLAGS.last_layer_gradient_multiplier)
      if grad_mult:
        grads_and_vars = slim.learning.multiply_gradients(grads_and_vars,
                                                          grad_mult)

      with tf.name_scope('grad_clipping'):
        grads_and_vars = slim.learning.clip_gradient_norms(grads_and_vars, 5.0)

      # Create histogram summaries for the gradients.
      # We have too many summaries for mldash, so disable this one for now.
      # for grad, var in grads_and_vars:
      #   summaries.add(tf.summary.histogram(
      #       var.name.replace(':0', '_0') + '/gradient', grad))

      # Create gradient update op.
      grad_updates = optimizer.apply_gradients(grads_and_vars,
                                               global_step=global_step)
      update_ops.append(grad_updates)
      update_op = tf.group(*update_ops)
      with tf.control_dependencies([update_op]):
        train_tensor = tf.identity(total_loss, name='train_op')

    # Add the summaries from the first clone. These contain the summaries
    # created by model_fn and either optimize_clones() or _gather_clone_loss().
    summaries |= set(tf.get_collection(tf.GraphKeys.SUMMARIES,
                                       first_clone_scope))

    # Merge all summaries together.
    summary_op = tf.summary.merge(list(summaries))

    # Soft placement allows placing on CPU ops without GPU implementation.
    session_config = tf.ConfigProto(allow_soft_placement=True,
                                    log_device_placement=False)

    # Start the training.
    slim.learning.train(
        train_tensor,
        logdir=FLAGS.train_logdir,
        log_every_n_steps=FLAGS.log_steps,
        master=FLAGS.master,
        number_of_steps=FLAGS.training_number_of_steps,
        is_chief=(FLAGS.task == 0),
        session_config=session_config,
        startup_delay_steps=startup_delay_steps,
        init_fn=train_utils.get_model_init_fn(FLAGS.train_logdir,
                                              FLAGS.tf_initial_checkpoint,
                                              FLAGS.initialize_last_layer,
                                              last_layers,
                                              ignore_missing_vars=True),
        summary_op=summary_op,
        save_summaries_secs=FLAGS.save_summaries_secs,
        save_interval_secs=FLAGS.save_interval_secs)


def _get_dataset_and_samples(config, train_crop_size, dataset_name,
                             dataset_dir, first_frame_finetuning,
                             three_frame_dataset):
  """Creates dataset object and samples dict of tensor.

  Args:
    config: A DeploymentConfig.
    train_crop_size: Integer, the crop size used for training.
    dataset_name: String, the name of the dataset.
    dataset_dir: String, the directory of the dataset.
    first_frame_finetuning: Boolean, whether the used dataset is a dataset
      for first frame fine-tuning.
    three_frame_dataset: Boolean, whether the dataset has exactly three frames
      per video of which the first is to be used as reference and the two
      others are consecutive frames to be used as query frames.

  Returns:
    dataset: An instance of slim Dataset.
    samples: A dictionary of tensors for semantic segmentation.
  """

  # Split the batch across GPUs.
  assert FLAGS.train_batch_size % config.num_clones == 0, (
      'Training batch size not divisble by number of clones (GPUs).')

  clone_batch_size = FLAGS.train_batch_size / config.num_clones

  if first_frame_finetuning:
    train_split = 'val'
  else:
    train_split = FLAGS.train_split

  data_type = 'tf_sequence_example'
  # Get dataset-dependent information.
  dataset = video_dataset.get_dataset(
      dataset_name,
      train_split,
      dataset_dir=dataset_dir,
      data_type=data_type)

  tf.gfile.MakeDirs(FLAGS.train_logdir)
  tf.logging.info('Training on %s set', train_split)

  samples = video_input_generator.get(
      dataset,
      FLAGS.train_num_frames_per_video,
      train_crop_size,
      clone_batch_size,
      num_readers=FLAGS.num_readers,
      num_threads=FLAGS.batch_num_threads,
      min_resize_value=FLAGS.min_resize_value,
      max_resize_value=FLAGS.max_resize_value,
      resize_factor=FLAGS.resize_factor,
      min_scale_factor=FLAGS.min_scale_factor,
      max_scale_factor=FLAGS.max_scale_factor,
      scale_factor_step_size=FLAGS.scale_factor_step_size,
      dataset_split=FLAGS.train_split,
      is_training=True,
      model_variant=FLAGS.model_variant,
      batch_capacity_factor=FLAGS.batch_capacity_factor,
      decoder_output_stride=common.parse_decoder_output_stride(),
      first_frame_finetuning=first_frame_finetuning,
      sample_only_first_frame_for_finetuning=
      FLAGS.sample_only_first_frame_for_finetuning,
      sample_adjacent_and_consistent_query_frames=
      FLAGS.sample_adjacent_and_consistent_query_frames or
      FLAGS.use_softmax_feedback,
      remap_labels_to_reference_frame=True,
      three_frame_dataset=three_frame_dataset,
      add_prev_frame_label=not FLAGS.also_attend_to_previous_frame
  )
  return dataset, samples


if __name__ == '__main__':
  flags.mark_flag_as_required('train_logdir')
  tf.logging.set_verbosity(tf.logging.INFO)
  tf.app.run()
