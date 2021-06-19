# Lint as: python3
# Copyright 2021 The TensorFlow Authors. All Rights Reserved.
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
"""Training script for Global Features model."""

import math
import os

from absl import app
from absl import flags
from absl import logging
import numpy as np
import tensorflow as tf
import tensorflow_addons as tfa

from delf.python.datasets.sfm120k import dataset_download
from delf.python.datasets.sfm120k import sfm120k
from delf.python.training import global_features_utils
from delf.python.training import tensorboard_utils
from delf.python.training.global_features import train_utils
from delf.python.training.losses import ranking_losses
from delf.python.training.model import global_model

_LOSS_NAMES = ['contrastive', 'triplet']
_MODEL_NAMES = global_features_utils.get_standard_keras_models()
_OPTIMIZER_NAMES = ['sgd', 'adam']
_POOL_NAMES = ['mac', 'spoc', 'gem']
_PRECOMPUTE_WHITEN_NAMES = ['retrieval-SfM-30k', 'retrieval-SfM-120k']
_TEST_DATASET_NAMES = ['roxford5k', 'rparis6k']
_TRAINING_DATASET_NAMES = ['retrieval-SfM-120k']
_VALIDATION_TYPES = ['standard', 'eccv2020']

FLAGS = flags.FLAGS

flags.DEFINE_boolean('debug', False, 'Debug mode.')

# Export directory, training and val datasets, test datasets.
flags.DEFINE_string('data_root', "data",
                    'Absolute path to the folder containing training data.')
flags.DEFINE_string('directory', "data",
                    'Destination where trained network should be saved.')
flags.DEFINE_enum('training_dataset', 'retrieval-SfM-120k',
                  _TRAINING_DATASET_NAMES, 'Training dataset: ' +
                  ' | '.join(_TRAINING_DATASET_NAMES) + '.')
flags.DEFINE_enum('validation_type', None, _VALIDATION_TYPES,
                  'Type of the evaluation to use. Either `None`, `standard` '
                  'or `eccv2020`.')
flags.DEFINE_list('test_datasets', 'roxford5k,rparis6k',
                  'Comma separated list of test datasets: ' +
                  ' | '.join(_TEST_DATASET_NAMES) + '.')
flags.DEFINE_enum('precompute_whitening', None, _PRECOMPUTE_WHITEN_NAMES,
                  'Dataset used to learn whitening: ' +
                  ' | '.join(_PRECOMPUTE_WHITEN_NAMES) + '.')
flags.DEFINE_integer('test_freq', 5,
                     'Run test evaluation every N epochs.')
flags.DEFINE_list('multiscale', [1.],
                  'Use multiscale vectors for testing, ' +
                  ' examples: 1 | 1,1/2**(1/2),1/2 | 1,2**(1/2),1/2**(1/2)]. '
                  'Pass as a string of comma separated values.')

# Network architecture and initialization options.
flags.DEFINE_enum('arch', 'ResNet101', _MODEL_NAMES,
                  'Model architecture: ' + ' | '.join(_MODEL_NAMES) + '.')
flags.DEFINE_enum('pool', 'gem', _POOL_NAMES,
                  'Pooling options: ' + ' | '.join(_POOL_NAMES) + '.')
flags.DEFINE_bool('whitening', False,
                  'Whether to train model with learnable whitening ('
                  'linear layer) after the pooling.')
flags.DEFINE_bool('pretrained', True,
                  'Whether to initialize model with random weights ('
                  'default: pretrained on imagenet).')
flags.DEFINE_enum('loss', 'contrastive', _LOSS_NAMES,
                  'Training loss options: ' + ' | '.join(_LOSS_NAMES) + '.')
flags.DEFINE_float('loss_margin', 0.7, 'Loss margin.')

# train/val options specific for image retrieval learning.
flags.DEFINE_integer('image_size', 1024,
                     'Maximum size of longer image side used for training.')
flags.DEFINE_integer('neg_num', 5, 'Number of negative images per train/val '
                                   'tuple.')
flags.DEFINE_integer('query_size', 2000,
                     'Number of queries randomly drawn per one training epoch.')
flags.DEFINE_integer('pool_size', 20000,
                     'Size of the pool for hard negative mining.')

# Standard training/validation options.
flags.DEFINE_string('gpu_id', '0', 'GPU id used for training.')
flags.DEFINE_integer('epochs', 100, 'Number of total epochs to run.')
flags.DEFINE_integer('batch_size', 5,
                     'Number of (q,p,n1,...,nN) tuples in a mini-batch.')
flags.DEFINE_integer('update_every', 1,
                     'Update model weights every N batches, used to handle '
                     'relatively large batches, batch_size effectively '
                     'becomes update_every `x` batch_size.')
flags.DEFINE_enum('optimizer', 'adam', _OPTIMIZER_NAMES,
                  'Optimizer options: ' + ' | '.join(_OPTIMIZER_NAMES) + '.')
flags.DEFINE_float('lr', 1e-6, 'Initial learning rate.')
flags.DEFINE_float('momentum', 0.9, 'Momentum.')
flags.DEFINE_float('weight_decay', 1e-6, 'Weight decay.')
flags.DEFINE_bool('resume', False,
                  'Whether to start from the latest checkpoint in the logdir.')
flags.DEFINE_bool('launch_tensorboard', False, 'Whether to launch tensorboard.')


def main(argv):
  if len(argv) > 1:
    raise RuntimeError('Too many command-line arguments.')

  # Manually check if there are unknown test datasets and if the dataset
  # ground truth files are downloaded.
  for dataset in FLAGS.test_datasets:
    if dataset not in _TEST_DATASET_NAMES:
      raise ValueError('Unsupported or unknown test dataset: {}.'.format(
              dataset))

    test_data_config = os.path.join(FLAGS.data_root,
                                    'gnd_{}.pkl'.format(dataset))
    if not tf.io.gfile.exists(test_data_config):
      raise ValueError(
              '{} ground truth file at {} not found. Please download it '
              'according to '
              'the DELG instructions.'.format(dataset, FLAGS.data_root))

  # Check if train dataset is downloaded and download it if not found.
  dataset_download.download_train(FLAGS.data_root)

  # Creating model export directory if it does not exist.
  model_directory = global_features_utils.create_model_directory(
          FLAGS.training_dataset, FLAGS.arch, FLAGS.pool, FLAGS.whitening,
          FLAGS.pretrained, FLAGS.loss, FLAGS.loss_margin, FLAGS.optimizer,
          FLAGS.lr, FLAGS.weight_decay, FLAGS.neg_num, FLAGS.query_size,
          FLAGS.pool_size, FLAGS.batch_size, FLAGS.update_every,
          FLAGS.image_size, FLAGS.directory)

  # Setting up logging directory, same as where the model is stored.
  logging.get_absl_handler().use_absl_log_file('absl_logging', model_directory)

  # Set cuda visible device.
  os.environ['CUDA_VISIBLE_DEVICES'] = FLAGS.gpu_id
  global_features_utils.debug_and_log('>> Num GPUs Available: {}'.format(
          len(tf.config.experimental.list_physical_devices('GPU'))),
          FLAGS.debug)

  # Set random seeds.
  tf.random.set_seed(0)
  np.random.seed(0)

  # Initialize the model.
  if FLAGS.pretrained:
    global_features_utils.debug_and_log(
            '>> Using pre-trained model \'{}\''.format(FLAGS.arch))
  else:
    global_features_utils.debug_and_log(
            '>> Using model from scratch (random weights) \'{}\'.'.format(
                    FLAGS.arch))

  model_params = {'architecture': FLAGS.arch, 'pooling': FLAGS.pool,
                  'whitening': FLAGS.whitening, 'pretrained': FLAGS.pretrained,
                  'data_root': FLAGS.data_root}
  model = global_model.GlobalFeatureNet(**model_params)

  # Freeze running mean and std in batch normalization layers.
  # We do training one image at a time to improve memory requirements of
  # the network; therefore, the computed statistics would not be per a
  # batch. Instead, we choose freezing - setting the parameters of all
  # batch norm layers in the network to non-trainable (i.e., using original
  # imagenet statistics).
  for layer in model.feature_extractor.layers:
    if isinstance(layer, tf.keras.layers.BatchNormalization):
      layer.trainable = False

  global_features_utils.debug_and_log('>> Network initialized.')

  global_features_utils.debug_and_log('>> Loss: {}.'.format(FLAGS.loss))
  # Define the loss function.
  if FLAGS.loss == 'contrastive':
    criterion = ranking_losses.ContrastiveLoss(margin=FLAGS.loss_margin)
  elif FLAGS.loss == 'triplet':
    criterion = ranking_losses.TripletLoss(margin=FLAGS.loss_margin)
  else:
    raise ValueError('Loss {} not available.'.format(FLAGS.loss))

  # Defining parameters for the training.
  # When pre-computing whitening, we run evaluation before the network training
  # and the `start_epoch` is set to 0. In other cases, we start from epoch 1.
  start_epoch = 1
  exp_decay = math.exp(-0.01)
  decay_steps = FLAGS.query_size / FLAGS.batch_size

  # Define learning rate decay schedule.
  lr_scheduler = tf.keras.optimizers.schedules.ExponentialDecay(
          initial_learning_rate=FLAGS.lr,
          decay_steps=decay_steps,
          decay_rate=exp_decay)

  # Define the optimizer.
  if FLAGS.optimizer == 'sgd':
    opt = tfa.optimizers.extend_with_decoupled_weight_decay(
            tf.keras.optimizers.SGD)
    optimizer = opt(weight_decay=FLAGS.weight_decay,
                    learning_rate=lr_scheduler, momentum=FLAGS.momentum)
  elif FLAGS.optimizer == 'adam':
    opt = tfa.optimizers.extend_with_decoupled_weight_decay(
            tf.keras.optimizers.Adam)
    optimizer = opt(weight_decay=FLAGS.weight_decay, learning_rate=lr_scheduler)
  else:
    raise ValueError('Optimizer {} not available.'.format(FLAGS.optimizer))

  # Initializing logging.
  writer = tf.summary.create_file_writer(model_directory)
  tf.summary.experimental.set_step(1)

  # Setting up the checkpoint manager.
  checkpoint = tf.train.Checkpoint(optimizer=optimizer, model=model)
  manager = tf.train.CheckpointManager(
          checkpoint,
          model_directory,
          max_to_keep=10,
          keep_checkpoint_every_n_hours=3)
  if FLAGS.resume:
    # Restores the checkpoint, if existing.
    global_features_utils.debug_and_log('>> Continuing from a checkpoint.')
    checkpoint.restore(manager.latest_checkpoint)

  # Launching tensorboard if required.
  if FLAGS.launch_tensorboard:
    tensorboard = tf.keras.callbacks.TensorBoard(model_directory)
    tensorboard.set_model(model=model)
    tensorboard_utils.launch_tensorboard(log_dir=model_directory)

  # Log flags used.
  global_features_utils.debug_and_log('>> Running training script with:')
  global_features_utils.debug_and_log('>> logdir = {}'.format(model_directory))

  if FLAGS.training_dataset.startswith('retrieval-SfM-120k'):
    train_dataset = sfm120k.CreateDataset(
            data_root=FLAGS.data_root,
            mode='train',
            imsize=FLAGS.image_size,
            num_negatives=FLAGS.neg_num,
            num_queries=FLAGS.query_size,
            pool_size=FLAGS.pool_size
    )
    if FLAGS.validation_type is not None:
      val_dataset = sfm120k.CreateDataset(
              data_root=FLAGS.data_root,
              mode='val',
              imsize=FLAGS.image_size,
              num_negatives=FLAGS.neg_num,
              num_queries=float('Inf'),
              pool_size=float('Inf'),
              eccv2020=True if FLAGS.validation_type == 'eccv2020' else False
      )

  train_dataset_output_types = [tf.float32 for i in range(2 + FLAGS.neg_num)]
  train_dataset_output_types.append(tf.int32)

  global_features_utils.debug_and_log(
          '>> Training the {} network'.format(model_directory))
  global_features_utils.debug_and_log('>> GPU ids: {}'.format(FLAGS.gpu_id))

  with writer.as_default():

    # Precompute whitening if needed.
    if FLAGS.precompute_whitening is not None:
      epoch = 0
      train_utils.test_retrieval(
              FLAGS.test_datasets, model, writer=writer,
              epoch=epoch, model_directory=model_directory,
              precompute_whitening=FLAGS.precompute_whitening,
              data_root=FLAGS.data_root,
              multiscale=FLAGS.multiscale)

    for epoch in range(start_epoch, FLAGS.epochs + 1):
      # Set manual seeds per epoch.
      np.random.seed(epoch)
      tf.random.set_seed(epoch)

      # Find hard-negatives.
      # While hard-positive examples are fixed during the whole training
      # process and are randomly chosen from every epoch; hard-negatives
      # depend on the current CNN parameters and are re-mined once per epoch.
      avg_neg_distance = train_dataset.create_epoch_tuples(model)

      def _train_gen():
        return (inst for inst in train_dataset)

      train_loader = tf.data.Dataset.from_generator(
              _train_gen,
              output_types=tuple(train_dataset_output_types))

      loss = train_utils.train_val_one_epoch(
              loader=iter(train_loader), model=model,
              criterion=criterion, optimizer=optimizer, epoch=epoch,
              batch_size=FLAGS.batch_size, query_size=FLAGS.query_size,
              neg_num=FLAGS.neg_num, update_every=FLAGS.update_every,
              debug=FLAGS.debug)

      # Write a scalar summary.
      tf.summary.scalar('train_epoch_loss', loss, step=epoch)
      # Forces summary writer to send any buffered data to storage.
      writer.flush()

      # Evaluate on validation set.
      if FLAGS.validation_type is not None and (epoch % FLAGS.test_freq == 0 or
                                                epoch == 1):
        avg_neg_distance = val_dataset.create_epoch_tuples(model,
                                                           model_directory)

        def _val_gen():
          return (inst for inst in val_dataset)

        val_loader = tf.data.Dataset.from_generator(
                _val_gen, output_types=tuple(train_dataset_output_types))

        loss = train_utils.train_val_one_epoch(
                loader=iter(val_loader), model=model,
                criterion=criterion, optimizer=None,
                epoch=epoch, train=False, batch_size=FLAGS.batch_size,
                query_size=FLAGS.query_size, neg_num=FLAGS.neg_num,
                update_every=FLAGS.update_every, debug=FLAGS.debug)
        tf.summary.scalar('val_epoch_loss', loss, step=epoch)
        writer.flush()

      # Evaluate on test datasets every test_freq epochs.
      if epoch == 1 or epoch % FLAGS.test_freq == 0:
        train_utils.test_retrieval(
                FLAGS.test_datasets, model, writer=writer, epoch=epoch,
                model_directory=model_directory,
                precompute_whitening=FLAGS.precompute_whitening,
                data_root=FLAGS.data_root, multiscale=FLAGS.multiscale)

      # Saving checkpoints and model weights.
      try:
        save_path = manager.save(checkpoint_number=epoch)
        global_features_utils.debug_and_log(
                'Saved ({}) at {}'.format(epoch, save_path))

        filename = os.path.join(model_directory,
                                'checkpoint_epoch_{}.h5'.format(epoch))
        model.save_weights(filename, save_format='h5')
        global_features_utils.debug_and_log(
                'Saved weights ({}) at {}'.format(epoch, filename))
      except Exception as ex:
        global_features_utils.debug_and_log(
                'Could not save checkpoint: {}'.format(ex))


if __name__ == '__main__':
  app.run(main)
