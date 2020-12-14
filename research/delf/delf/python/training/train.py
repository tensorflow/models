# Lint as: python3
# Copyright 2020 The TensorFlow Authors. All Rights Reserved.
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
"""Training script for DELF/G on Google Landmarks Dataset.

Uses classification loss, with MirroredStrategy, to support running on multiple
GPUs.
"""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os
import time

from absl import app
from absl import flags
from absl import logging
import tensorflow as tf
import tensorflow_probability as tfp

# Placeholder for internal import. Do not remove this line.
from delf.python.training.datasets import googlelandmarks as gld
from delf.python.training.model import delf_model
from delf.python.training.model import delg_model

FLAGS = flags.FLAGS

flags.DEFINE_boolean('debug', False, 'Debug mode.')
flags.DEFINE_string('logdir', '/tmp/delf', 'WithTensorBoard logdir.')
flags.DEFINE_string(
    'train_file_pattern', '/tmp/data/train*',
    'File pattern of training dataset files.')
flags.DEFINE_string(
    'validation_file_pattern', '/tmp/data/validation*',
    'File pattern of validation dataset files.')
flags.DEFINE_enum(
    'dataset_version', 'gld_v1', ['gld_v1', 'gld_v2', 'gld_v2_clean'],
    'Google Landmarks dataset version, used to determine the number of '
    'classes.')
flags.DEFINE_integer('seed', 0, 'Seed to training dataset.')
flags.DEFINE_float('initial_lr', 0.01, 'Initial learning rate.')
flags.DEFINE_integer('batch_size', 32, 'Global batch size.')
flags.DEFINE_integer('max_iters', 500000, 'Maximum iterations.')
flags.DEFINE_boolean('block3_strides', True, 'Whether to use block3_strides.')
flags.DEFINE_boolean(
    'use_augmentation', True, 'Whether to use ImageNet style augmentation.')
flags.DEFINE_string(
    'imagenet_checkpoint', None,
    'ImageNet checkpoint for ResNet backbone. If None, no checkpoint is used.')
flags.DEFINE_float(
    'attention_loss_weight', 1.0,
    'Weight to apply to the attention loss when calculating the '
    'total loss of the model.')
flags.DEFINE_boolean(
    'delg_global_features', False, 'Whether to train a DELG model.')
flags.DEFINE_float(
    'delg_gem_power', 3.0,
    'Power for Generalized Mean pooling. Used only if '
    'delg_global_features=True.')
flags.DEFINE_integer(
    'delg_embedding_layer_dim', 2048,
    'Size of the FC whitening layer (embedding layer). Used only if'
    'delg_global_features:True.')
flags.DEFINE_float(
    'delg_scale_factor_init', 45.25,
    'Initial value of the scaling factor of the cosine logits. The default '
    'value is sqrt(2048). Used only if delg_global_features=True.')
flags.DEFINE_float(
    'delg_arcface_margin', 0.1,
    'ArcFace margin. Used only if delg_global_features=True.')
flags.DEFINE_integer('image_size', 321, 'Size of each image side to use.')
flags.DEFINE_boolean(
    'use_autoencoder', True, 'Whether to train an autoencoder.')
flags.DEFINE_float(
    'reconstruction_loss_weight', 10.0,
    'Weight to apply to the reconstruction loss from the autoencoder when'
    'calculating total loss of the model. Used only if use_autoencoder=True.')
flags.DEFINE_float(
    'autoencoder_dimensions', 128,
    'Number of dimensions of the autoencoder. Used only if'
    'use_autoencoder=True.')
flags.DEFINE_float(
    'local_feature_map_channels', 1024,
    'Number of channels at backbone layer used for local feature extraction. '
    'Default value 1024 is the number of channels of block3. Used only if'
    'use_autoencoder=True.')


def _record_accuracy(metric, logits, labels):
  """Record accuracy given predicted logits and ground-truth labels."""
  softmax_probabilities = tf.keras.layers.Softmax()(logits)
  metric.update_state(labels, softmax_probabilities)


def _attention_summaries(scores, global_step):
  """Record statistics of the attention score."""
  tf.summary.image(
      'batch_attention',
      scores / tf.reduce_max(scores + 1e-3),
      step=global_step)
  tf.summary.scalar('attention/max', tf.reduce_max(scores), step=global_step)
  tf.summary.scalar('attention/min', tf.reduce_min(scores), step=global_step)
  tf.summary.scalar('attention/mean', tf.reduce_mean(scores), step=global_step)
  tf.summary.scalar(
      'attention/percent_25',
      tfp.stats.percentile(scores, 25.0),
      step=global_step)
  tf.summary.scalar(
      'attention/percent_50',
      tfp.stats.percentile(scores, 50.0),
      step=global_step)
  tf.summary.scalar(
      'attention/percent_75',
      tfp.stats.percentile(scores, 75.0),
      step=global_step)


def create_model(num_classes):
  """Define DELF model, and initialize classifiers."""
  if FLAGS.delg_global_features:
    model = delg_model.Delg(
        block3_strides=FLAGS.block3_strides,
        name='DELG',
        gem_power=FLAGS.delg_gem_power,
        embedding_layer_dim=FLAGS.delg_embedding_layer_dim,
        scale_factor_init=FLAGS.delg_scale_factor_init,
        arcface_margin=FLAGS.delg_arcface_margin,
        use_dim_reduction=FLAGS.use_autoencoder,
        reduced_dimension=FLAGS.autoencoder_dimensions,
        dim_expand_channels=FLAGS.local_feature_map_channels)
  else:
    model = delf_model.Delf(
        block3_strides=FLAGS.block3_strides,
        name='DELF',
        use_dim_reduction=FLAGS.use_autoencoder,
        reduced_dimension=FLAGS.autoencoder_dimensions,
        dim_expand_channels=FLAGS.local_feature_map_channels)
  model.init_classifiers(num_classes)
  return model


def _learning_rate_schedule(global_step_value, max_iters, initial_lr):
  """Calculates learning_rate with linear decay.

  Args:
    global_step_value: int, global step.
    max_iters: int, maximum iterations.
    initial_lr: float, initial learning rate.

  Returns:
    lr: float, learning rate.
  """
  lr = initial_lr * (1.0 - global_step_value / max_iters)
  return lr


def main(argv):
  if len(argv) > 1:
    raise app.UsageError('Too many command-line arguments.')

  #-------------------------------------------------------------
  # Log flags used.
  logging.info('Running training script with\n')
  logging.info('logdir= %s', FLAGS.logdir)
  logging.info('initial_lr= %f', FLAGS.initial_lr)
  logging.info('block3_strides= %s', str(FLAGS.block3_strides))

  # ------------------------------------------------------------
  # Create the strategy.
  strategy = tf.distribute.MirroredStrategy()
  logging.info('Number of devices: %d', strategy.num_replicas_in_sync)
  if FLAGS.debug:
    print('Number of devices:', strategy.num_replicas_in_sync)

  max_iters = FLAGS.max_iters
  global_batch_size = FLAGS.batch_size
  image_size = FLAGS.image_size
  num_eval_batches = int(50000 / global_batch_size)
  report_interval = 100
  eval_interval = 1000
  save_interval = 1000

  initial_lr = FLAGS.initial_lr

  clip_val = tf.constant(10.0)

  if FLAGS.debug:
    tf.config.run_functions_eagerly(True)
    global_batch_size = 4
    max_iters = 100
    num_eval_batches = 1
    save_interval = 1
    report_interval = 10

  # Determine the number of classes based on the version of the dataset.
  gld_info = gld.GoogleLandmarksInfo()
  num_classes = gld_info.num_classes[FLAGS.dataset_version]

  # ------------------------------------------------------------
  # Create the distributed train/validation sets.
  train_dataset = gld.CreateDataset(
      file_pattern=FLAGS.train_file_pattern,
      batch_size=global_batch_size,
      image_size=image_size,
      augmentation=FLAGS.use_augmentation,
      seed=FLAGS.seed)
  validation_dataset = gld.CreateDataset(
      file_pattern=FLAGS.validation_file_pattern,
      batch_size=global_batch_size,
      image_size=image_size,
      augmentation=False,
      seed=FLAGS.seed)

  train_dist_dataset = strategy.experimental_distribute_dataset(train_dataset)
  validation_dist_dataset = strategy.experimental_distribute_dataset(
      validation_dataset)

  train_iter = iter(train_dist_dataset)
  validation_iter = iter(validation_dist_dataset)

  # Create a checkpoint directory to store the checkpoints.
  checkpoint_prefix = os.path.join(FLAGS.logdir, 'delf_tf2-ckpt')

  # ------------------------------------------------------------
  # Finally, we do everything in distributed scope.
  with strategy.scope():
    # Compute loss.
    # Set reduction to `none` so we can do the reduction afterwards and divide
    # by global batch size.
    loss_object = tf.keras.losses.SparseCategoricalCrossentropy(
        from_logits=True, reduction=tf.keras.losses.Reduction.NONE)

    def compute_loss(labels, predictions):
      per_example_loss = loss_object(labels, predictions)
      return tf.nn.compute_average_loss(
          per_example_loss, global_batch_size=global_batch_size)

    # Set up metrics.
    desc_validation_loss = tf.keras.metrics.Mean(name='desc_validation_loss')
    attn_validation_loss = tf.keras.metrics.Mean(name='attn_validation_loss')
    desc_train_accuracy = tf.keras.metrics.SparseCategoricalAccuracy(
        name='desc_train_accuracy')
    attn_train_accuracy = tf.keras.metrics.SparseCategoricalAccuracy(
        name='attn_train_accuracy')
    desc_validation_accuracy = tf.keras.metrics.SparseCategoricalAccuracy(
        name='desc_validation_accuracy')
    attn_validation_accuracy = tf.keras.metrics.SparseCategoricalAccuracy(
        name='attn_validation_accuracy')

    # ------------------------------------------------------------
    # Setup DELF model and optimizer.
    model = create_model(num_classes)
    logging.info('Model, datasets loaded.\nnum_classes= %d', num_classes)

    optimizer = tf.keras.optimizers.SGD(learning_rate=initial_lr, momentum=0.9)

    # Setup summary writer.
    summary_writer = tf.summary.create_file_writer(
        os.path.join(FLAGS.logdir, 'train_logs'), flush_millis=10000)

    # Setup checkpoint directory.
    checkpoint = tf.train.Checkpoint(optimizer=optimizer, model=model)
    manager = tf.train.CheckpointManager(
        checkpoint,
        checkpoint_prefix,
        max_to_keep=10,
        keep_checkpoint_every_n_hours=3)
    # Restores the checkpoint, if existing.
    checkpoint.restore(manager.latest_checkpoint)

    # ------------------------------------------------------------
    # Train step to run on one GPU.
    def train_step(inputs):
      """Train one batch."""
      images, labels = inputs
      # Temporary workaround to avoid some corrupted labels.
      labels = tf.clip_by_value(labels, 0, model.num_classes)

      def _backprop_loss(tape, loss, weights):
        """Backpropogate losses using clipped gradients.

        Args:
          tape: gradient tape.
          loss: scalar Tensor, loss value.
          weights: keras model weights.
        """
        gradients = tape.gradient(loss, weights)
        clipped, _ = tf.clip_by_global_norm(gradients, clip_norm=clip_val)
        optimizer.apply_gradients(zip(clipped, weights))

      # Record gradients and loss through backbone.
      with tf.GradientTape() as gradient_tape:
        # Make a forward pass to calculate prelogits.
        (desc_prelogits, attn_prelogits, attn_scores, backbone_blocks,
         dim_expanded_features, _) = model.global_and_local_forward_pass(images)

        # Calculate global loss by applying the descriptor classifier.
        if FLAGS.delg_global_features:
          desc_logits = model.desc_classification(desc_prelogits, labels)
        else:
          desc_logits = model.desc_classification(desc_prelogits)
        desc_loss = compute_loss(labels, desc_logits)

        # Calculate attention loss by applying the attention block classifier.
        attn_logits = model.attn_classification(attn_prelogits)
        attn_loss = compute_loss(labels, attn_logits)

        # Calculate reconstruction loss between the attention prelogits and the
        # backbone.
        if FLAGS.use_autoencoder:
          block3 = tf.stop_gradient(backbone_blocks['block3'])
          reconstruction_loss = tf.math.reduce_mean(
              tf.keras.losses.MSE(block3, dim_expanded_features))
        else:
          reconstruction_loss = 0

        # Cumulate global loss, attention loss and reconstruction loss.
        total_loss = (desc_loss
                      + FLAGS.attention_loss_weight * attn_loss
                      + FLAGS.reconstruction_loss_weight * reconstruction_loss)

      # Perform backpropagation through the descriptor and attention layers
      # together. Note that this will increment the number of iterations of
      # "optimizer".
      _backprop_loss(gradient_tape, total_loss, model.trainable_weights)

      # Step number, for summary purposes.
      global_step = optimizer.iterations

      # Input image-related summaries.
      tf.summary.image('batch_images', (images + 1.0) / 2.0, step=global_step)
      tf.summary.scalar(
          'image_range/max', tf.reduce_max(images), step=global_step)
      tf.summary.scalar(
          'image_range/min', tf.reduce_min(images), step=global_step)

      # Attention and sparsity summaries.
      _attention_summaries(attn_scores, global_step)
      activations_zero_fractions = {
          'sparsity/%s' % k: tf.nn.zero_fraction(v)
          for k, v in backbone_blocks.items()
      }
      for k, v in activations_zero_fractions.items():
        tf.summary.scalar(k, v, step=global_step)

      # Scaling factor summary for cosine logits for a DELG model.
      if FLAGS.delg_global_features:
        tf.summary.scalar(
            'desc/scale_factor', model.scale_factor, step=global_step)

      # Record train accuracies.
      _record_accuracy(desc_train_accuracy, desc_logits, labels)
      _record_accuracy(attn_train_accuracy, attn_logits, labels)

      return desc_loss, attn_loss, reconstruction_loss

    # ------------------------------------------------------------
    def validation_step(inputs):
      """Validate one batch."""
      images, labels = inputs
      labels = tf.clip_by_value(labels, 0, model.num_classes)

      # Get descriptor predictions.
      blocks = {}
      prelogits = model.backbone(
          images, intermediates_dict=blocks, training=False)
      if FLAGS.delg_global_features:
        logits = model.desc_classification(prelogits, labels, training=False)
      else:
        logits = model.desc_classification(prelogits, training=False)
      softmax_probabilities = tf.keras.layers.Softmax()(logits)

      validation_loss = loss_object(labels, logits)
      desc_validation_loss.update_state(validation_loss)
      desc_validation_accuracy.update_state(labels, softmax_probabilities)

      # Get attention predictions.
      block3 = blocks['block3']  # pytype: disable=key-error
      prelogits, _, _ = model.attention(block3, training=False)

      logits = model.attn_classification(prelogits, training=False)
      softmax_probabilities = tf.keras.layers.Softmax()(logits)

      validation_loss = loss_object(labels, logits)
      attn_validation_loss.update_state(validation_loss)
      attn_validation_accuracy.update_state(labels, softmax_probabilities)

      return desc_validation_accuracy.result(), attn_validation_accuracy.result(
      )

    # `run` replicates the provided computation and runs it
    # with the distributed input.
    @tf.function
    def distributed_train_step(dataset_inputs):
      """Get the actual losses."""
      # Each (desc, attn) is a list of 3 losses - crossentropy, reg, total.
      desc_per_replica_loss, attn_per_replica_loss, recon_per_replica_loss = (
          strategy.run(train_step, args=(dataset_inputs,)))

      # Reduce over the replicas.
      desc_global_loss = strategy.reduce(
          tf.distribute.ReduceOp.SUM, desc_per_replica_loss, axis=None)
      attn_global_loss = strategy.reduce(
          tf.distribute.ReduceOp.SUM, attn_per_replica_loss, axis=None)
      recon_global_loss = strategy.reduce(
          tf.distribute.ReduceOp.SUM, recon_per_replica_loss, axis=None)

      return desc_global_loss, attn_global_loss, recon_global_loss

    @tf.function
    def distributed_validation_step(dataset_inputs):
      return strategy.run(validation_step, args=(dataset_inputs,))

    # ------------------------------------------------------------
    # *** TRAIN LOOP ***
    with summary_writer.as_default():
      record_cond = lambda: tf.equal(optimizer.iterations % report_interval, 0)
      with tf.summary.record_if(record_cond):
        global_step_value = optimizer.iterations.numpy()

        # TODO(dananghel): try to load pretrained weights at backbone creation.
        # Load pretrained weights for ResNet50 trained on ImageNet.
        if (FLAGS.imagenet_checkpoint is not None) and (not global_step_value):
          logging.info('Attempting to load ImageNet pretrained weights.')
          input_batch = next(train_iter)
          _, _, _ = distributed_train_step(input_batch)
          model.backbone.restore_weights(FLAGS.imagenet_checkpoint)
          logging.info('Done.')
        else:
          logging.info('Skip loading ImageNet pretrained weights.')
        if FLAGS.debug:
          model.backbone.log_weights()

        last_summary_step_value = None
        last_summary_time = None
        while global_step_value < max_iters:
          # input_batch : images(b, h, w, c), labels(b,).
          try:
            input_batch = next(train_iter)
          except tf.errors.OutOfRangeError:
            # Break if we run out of data in the dataset.
            logging.info('Stopping training at global step %d, no more data',
                         global_step_value)
            break

          # Set learning rate and run the training step over num_gpu gpus.
          optimizer.learning_rate = _learning_rate_schedule(
              optimizer.iterations.numpy(), max_iters, initial_lr)
          desc_dist_loss, attn_dist_loss, recon_dist_loss = (
              distributed_train_step(input_batch))

          # Step number, to be used for summary/logging.
          global_step = optimizer.iterations
          global_step_value = global_step.numpy()

          # LR, losses and accuracies summaries.
          tf.summary.scalar(
              'learning_rate', optimizer.learning_rate, step=global_step)
          tf.summary.scalar(
              'loss/desc/crossentropy', desc_dist_loss, step=global_step)
          tf.summary.scalar(
              'loss/attn/crossentropy', attn_dist_loss, step=global_step)
          if FLAGS.use_autoencoder:
            tf.summary.scalar(
                'loss/recon/mse', recon_dist_loss, step=global_step)

          tf.summary.scalar(
              'train_accuracy/desc',
              desc_train_accuracy.result(),
              step=global_step)
          tf.summary.scalar(
              'train_accuracy/attn',
              attn_train_accuracy.result(),
              step=global_step)

          # Summary for number of global steps taken per second.
          current_time = time.time()
          if (last_summary_step_value is not None and
              last_summary_time is not None):
            tf.summary.scalar(
                'global_steps_per_sec',
                (global_step_value - last_summary_step_value) /
                (current_time - last_summary_time),
                step=global_step)
          if tf.summary.should_record_summaries().numpy():
            last_summary_step_value = global_step_value
            last_summary_time = current_time

          # Print to console if running locally.
          if FLAGS.debug:
            if global_step_value % report_interval == 0:
              print(global_step.numpy())
              print('desc:', desc_dist_loss.numpy())
              print('attn:', attn_dist_loss.numpy())

          # Validate once in {eval_interval*n, n \in N} steps.
          if global_step_value % eval_interval == 0:
            for i in range(num_eval_batches):
              try:
                validation_batch = next(validation_iter)
                desc_validation_result, attn_validation_result = (
                    distributed_validation_step(validation_batch))
              except tf.errors.OutOfRangeError:
                logging.info('Stopping eval at batch %d, no more data', i)
                break

            # Log validation results to tensorboard.
            tf.summary.scalar(
                'validation/desc', desc_validation_result, step=global_step)
            tf.summary.scalar(
                'validation/attn', attn_validation_result, step=global_step)

            logging.info('\nValidation(%f)\n', global_step_value)
            logging.info(': desc: %f\n', desc_validation_result.numpy())
            logging.info(': attn: %f\n', attn_validation_result.numpy())
            # Print to console.
            if FLAGS.debug:
              print('Validation: desc:', desc_validation_result.numpy())
              print('          : attn:', attn_validation_result.numpy())

          # Save checkpoint once (each save_interval*n, n \in N) steps, or if
          # this is the last iteration.
          # TODO(andrearaujo): save only in one of the two ways. They are
          # identical, the only difference is that the manager adds some extra
          # prefixes and variables (eg, optimizer variables).
          if (global_step_value %
              save_interval == 0) or (global_step_value >= max_iters):
            save_path = manager.save(checkpoint_number=global_step_value)
            logging.info('Saved (%d) at %s', global_step_value, save_path)

            file_path = '%s/delf_weights' % FLAGS.logdir
            model.save_weights(file_path, save_format='tf')
            logging.info('Saved weights (%d) at %s', global_step_value,
                         file_path)

          # Reset metrics for next step.
          desc_train_accuracy.reset_states()
          attn_train_accuracy.reset_states()
          desc_validation_loss.reset_states()
          attn_validation_loss.reset_states()
          desc_validation_accuracy.reset_states()
          attn_validation_accuracy.reset_states()

    logging.info('Finished training for %d steps.', max_iters)


if __name__ == '__main__':
  app.run(main)
