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
"""Training script for DELF on Google Landmarks Dataset.

Script to train DELF using classification loss on Google Landmarks Dataset
using MirroredStrategy to so it can run on multiple GPUs.
"""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os

from absl import app
from absl import flags
from absl import logging
import tensorflow as tf
import tensorflow_probability as tfp

# Placeholder for internal import. Do not remove this line.
from delf.python.training.datasets import googlelandmarks as gld
from delf.python.training.model import delf_model

FLAGS = flags.FLAGS

flags.DEFINE_boolean('debug', False, 'Debug mode.')
flags.DEFINE_string('logdir', '/tmp/delf', 'WithTensorBoard logdir.')
flags.DEFINE_string('train_file_pattern', '/tmp/data/train*',
                    'File pattern of training dataset files.')
flags.DEFINE_string('validation_file_pattern', '/tmp/data/validation*',
                    'File pattern of validation dataset files.')
flags.DEFINE_integer('seed', 0, 'Seed to training dataset.')
flags.DEFINE_float('initial_lr', 0.001, 'Initial learning rate.')
flags.DEFINE_integer('batch_size', 32, 'Global batch size.')
flags.DEFINE_integer('max_iters', 500000, 'Maximum iterations.')
flags.DEFINE_boolean('block3_strides', False, 'Whether to use block3_strides.')
flags.DEFINE_boolean('use_augmentation', True,
                     'Whether to use ImageNet style augmentation.')


def _record_accuracy(metric, logits, labels):
  """Record accuracy given predicted logits and ground-truth labels."""
  softmax_probabilities = tf.keras.layers.Softmax()(logits)
  metric.update_state(labels, softmax_probabilities)


def _attention_summaries(scores, global_step):
  """Record statistics of the attention score."""
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
  model = delf_model.Delf(block3_strides=FLAGS.block3_strides, name='DELF')
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
  image_size = 321
  num_eval = 1000
  report_interval = 100
  eval_interval = 1000
  save_interval = 20000

  initial_lr = FLAGS.initial_lr

  clip_val = tf.constant(10.0)

  if FLAGS.debug:
    global_batch_size = 4
    max_iters = 4
    num_eval = 1
    save_interval = 1
    report_interval = 1

  # TODO(andrearaujo): Using placeholder, replace with actual value using
  # GoogleLandmarksInfo() from datasets/googlelandmarks.py.
  num_classes = 14951

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

  train_iterator = strategy.make_dataset_iterator(train_dataset)
  validation_iterator = strategy.make_dataset_iterator(validation_dataset)

  train_iterator.initialize()
  validation_iterator.initialize()

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
        checkpoint, checkpoint_prefix, max_to_keep=3)

    # ------------------------------------------------------------
    # Train step to run on one GPU.
    def train_step(inputs):
      """Train one batch."""
      images, labels = inputs
      # Temporary workaround to avoid some corrupted labels.
      labels = tf.clip_by_value(labels, 0, model.num_classes)

      global_step = optimizer.iterations
      tf.summary.scalar(
          'image_range/max', tf.reduce_max(images), step=global_step)
      tf.summary.scalar(
          'image_range/min', tf.reduce_min(images), step=global_step)

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
      with tf.GradientTape() as desc_tape:

        blocks = {}
        prelogits = model.backbone(
            images, intermediates_dict=blocks, training=True)

        # Report sparsity.
        activations_zero_fractions = {
            'sparsity/%s' % k: tf.nn.zero_fraction(v)
            for k, v in blocks.items()
        }
        for k, v in activations_zero_fractions.items():
          tf.summary.scalar(k, v, step=global_step)

        # Apply descriptor classifier.
        logits = model.desc_classification(prelogits)

        desc_loss = compute_loss(labels, logits)

      # Backprop only through backbone weights.
      _backprop_loss(desc_tape, desc_loss, model.desc_trainable_weights)

      # Record descriptor train accuracy.
      _record_accuracy(desc_train_accuracy, logits, labels)

      # Record gradients and loss through attention block.
      with tf.GradientTape() as attn_tape:
        block3 = blocks['block3']  # pytype: disable=key-error

        # Stopping gradients according to DELG paper:
        # (https://arxiv.org/abs/2001.05027).
        block3 = tf.stop_gradient(block3)

        prelogits, scores, _ = model.attention(block3, training=True)
        _attention_summaries(scores, global_step)

        # Apply attention block classifier.
        logits = model.attn_classification(prelogits)

        attn_loss = compute_loss(labels, logits)

      # Backprop only through attention weights.
      _backprop_loss(attn_tape, attn_loss, model.attn_trainable_weights)

      # Record attention train accuracy.
      _record_accuracy(attn_train_accuracy, logits, labels)

      return desc_loss, attn_loss

    # ------------------------------------------------------------
    def validation_step(inputs):
      """Validate one batch."""
      images, labels = inputs
      labels = tf.clip_by_value(labels, 0, model.num_classes)

      # Get descriptor predictions.
      blocks = {}
      prelogits = model.backbone(
          images, intermediates_dict=blocks, training=False)
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
      desc_per_replica_loss, attn_per_replica_loss = (
          strategy.run(train_step, args=(dataset_inputs,)))

      # Reduce over the replicas.
      desc_global_loss = strategy.reduce(
          tf.distribute.ReduceOp.SUM, desc_per_replica_loss, axis=None)
      attn_global_loss = strategy.reduce(
          tf.distribute.ReduceOp.SUM, attn_per_replica_loss, axis=None)

      return desc_global_loss, attn_global_loss

    @tf.function
    def distributed_validation_step(dataset_inputs):
      return strategy.run(validation_step, args=(dataset_inputs,))

    # ------------------------------------------------------------
    # *** TRAIN LOOP ***
    with summary_writer.as_default():
      with tf.summary.record_if(
          tf.math.equal(0, optimizer.iterations % report_interval)):

        global_step_value = optimizer.iterations.numpy()
        while global_step_value < max_iters:

          # input_batch : images(b, h, w, c), labels(b,).
          try:
            input_batch = train_iterator.get_next()
          except tf.errors.OutOfRangeError:
            # Break if we run out of data in the dataset.
            logging.info('Stopping training at global step %d, no more data',
                         global_step_value)
            break

          # Set learning rate for optimizer to use.
          global_step = optimizer.iterations
          global_step_value = global_step.numpy()

          learning_rate = _learning_rate_schedule(global_step_value, max_iters,
                                                  initial_lr)
          optimizer.learning_rate = learning_rate
          tf.summary.scalar(
              'learning_rate', optimizer.learning_rate, step=global_step)

          # Run the training step over num_gpu gpus.
          desc_dist_loss, attn_dist_loss = distributed_train_step(input_batch)

          # Log losses and accuracies to tensorboard.
          tf.summary.scalar(
              'loss/desc/crossentropy', desc_dist_loss, step=global_step)
          tf.summary.scalar(
              'loss/attn/crossentropy', attn_dist_loss, step=global_step)
          tf.summary.scalar(
              'train_accuracy/desc',
              desc_train_accuracy.result(),
              step=global_step)
          tf.summary.scalar(
              'train_accuracy/attn',
              attn_train_accuracy.result(),
              step=global_step)

          # Print to console if running locally.
          if FLAGS.debug:
            if global_step_value % report_interval == 0:
              print(global_step.numpy())
              print('desc:', desc_dist_loss.numpy())
              print('attn:', attn_dist_loss.numpy())

          # Validate once in {eval_interval*n, n \in N} steps.
          if global_step_value % eval_interval == 0:
            for i in range(num_eval):
              try:
                validation_batch = validation_iterator.get_next()
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

          # Save checkpoint once (each save_interval*n, n \in N) steps.
          if global_step_value % save_interval == 0:
            save_path = manager.save()
            logging.info('Saved({global_step_value}) at %s', save_path)

            file_path = '%s/delf_weights' % FLAGS.logdir
            model.save_weights(file_path, save_format='tf')
            logging.info('Saved weights({global_step_value}) at %s', file_path)

          # Reset metrics for next step.
          desc_train_accuracy.reset_states()
          attn_train_accuracy.reset_states()
          desc_validation_loss.reset_states()
          attn_validation_loss.reset_states()
          desc_validation_accuracy.reset_states()
          attn_validation_accuracy.reset_states()

          if global_step.numpy() > max_iters:
            break

    logging.info('Finished training for %d steps.', max_iters)


if __name__ == '__main__':
  app.run(main)
