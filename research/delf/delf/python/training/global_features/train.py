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
import time

from absl import flags
from absl import app
from absl import logging

import numpy as np
import tensorflow as tf
import tensorflow_addons as tfa
import pickle

from delf.python.datasets import tuples_dataset
from delf.python.datasets.revisited_op import dataset as testdataset
from delf.python.training.losses import ranking_losses
from delf.python.training import global_features_utils
from delf.python.training.model import global_model
from delf.python.datasets.sfm120k import sfm120k
from delf.python import whiten

_TRAINING_DATASET_NAMES = ['retrieval-SfM-120k']
_TEST_DATASET_NAMES = ['roxford5k', 'rparis6k']
_PRECOMPUTE_WHITEN_NAMES = ['retrieval-SfM-30k', 'retrieval-SfM-120k']
_POOL_NAMES = ['mac', 'spoc', 'gem']
_LOSS_NAMES = ['contrastive', 'triplet']
_OPTIMIZER_NAMES = ['sgd', 'adam']
_MODEL_NAMES = global_features_utils.get_standard_keras_models()

FLAGS = flags.FLAGS

flags.DEFINE_boolean('debug', default=False, help='Debug mode.')

# Export directory, training and val datasets, test datasets.
flags.DEFINE_string('data_root', default="data", help='Path to the data.')
flags.DEFINE_string('directory', default="data",
                    help='Destination where trained network should be saved.')
flags.DEFINE_enum('training_dataset', default='retrieval-SfM-120k',
                  enum_values=_TRAINING_DATASET_NAMES,
                  help='Training dataset: ' +
                       ' | '.join(_TRAINING_DATASET_NAMES) +
                       ' (default: retrieval-SfM-120k).')
flags.DEFINE_bool('val', default=True, help='Whether to run validation.')
flags.DEFINE_bool('val_eccv2020', default=False,
                  help='New validation dataset used with ECCV 2020 paper.')
flags.DEFINE_string('test_datasets', default='roxford5k,rparis6k',
                    help='Comma separated list of test datasets: ' +
                         ' | '.join(_TEST_DATASET_NAMES) +
                         ' (default: roxford5k,rparis6k).')
flags.DEFINE_enum('precompute_whitening', default=None,
                  enum_values=_PRECOMPUTE_WHITEN_NAMES,
                  help='Dataset used to learn whitening: ' +
                       ' | '.join(_PRECOMPUTE_WHITEN_NAMES) +
                       ' (default: None).')
flags.DEFINE_integer('test_freq', default=5,
                     help='Run test evaluation every N epochs (default: 1).')
flags.DEFINE_string('multiscale', default='[1.]',
                    help='Use multiscale vectors for testing, ' +
                         ' examples: \'[1]\' | \'[1, 1/2**(1/2), 1/2]\' | \'['
                         '1, 2**(1/2), 1/2**(1/2)]\' (default: \'[1]\').')

# Network architecture and initialization options.
flags.DEFINE_enum('arch', default='ResNet101', enum_values=_MODEL_NAMES,
                  help='Model architecture: ' +
                       ' | '.join(_MODEL_NAMES) +
                       ' (default: ResNet101).')
flags.DEFINE_enum('pool', default='gem', enum_values=_POOL_NAMES,
                  help='Pooling options: ' +
                       ' | '.join(_POOL_NAMES) +
                       ' (default: gem).')
flags.DEFINE_bool('whitening', False,
                  help='Whether to train model with learnable whitening ('
                       'linear layer) after the pooling.')
flags.DEFINE_bool('pretrained', True,
                  help='Whether to initialize model with random weights ('
                       'default: pretrained on imagenet).')
flags.DEFINE_enum('loss', default='contrastive', enum_values=_LOSS_NAMES,
                  help='Training loss options: ' +
                       ' | '.join(_LOSS_NAMES) +
                       ' (default: contrastive).')
flags.DEFINE_float('loss_margin', default=0.7,
                   help='Loss margin: (default: 0.7).')

# train/val options specific for image retrieval learning.
flags.DEFINE_integer('image_size', default=1024,
                     help='Maximum size of longer image side used for '
                          'training (default: 1024).')
flags.DEFINE_integer('neg_num', default=5,
                     help='Number of negative images per train/val tuple ('
                          'default: 5).')
flags.DEFINE_integer('query_size', default=2000,
                     help='Number of queries randomly drawn per one training '
                          'epoch (default: 2000).')
flags.DEFINE_integer('pool_size', default=20000,
                     help='Size of the pool for hard negative mining ('
                          'default: 20000).')

# Standard train/val options.
flags.DEFINE_string('gpu_ids', default='0',
                    help='GPU id used for training (default: 0).')
flags.DEFINE_integer('workers', default=8,
                     help='Number of data loading workers (default: 8).')
flags.DEFINE_integer('epochs', default=100,
                     help='Number of total epochs to run (default: 100).')
flags.DEFINE_integer('batch_size', default=5,
                     help='Number of (q,p,n1,...,nN) tuples in a mini-batch ('
                          'default: 5).')
flags.DEFINE_integer('update_every', default=1,
                     help='Update model weights every N batches, used to '
                          'handle relatively large batches, ' +
                          'batch_size effectively becomes update_every x '
                          'batch_size (default: 1).')
flags.DEFINE_enum('optimizer', default='adam', enum_values=_OPTIMIZER_NAMES,
                  help='Optimizer options: ' +
                       ' | '.join(_OPTIMIZER_NAMES) +
                       ' (default: adam).')
flags.DEFINE_float('lr', default=1e-6,
                   help='Initial learning rate (default: 1e-6).')
flags.DEFINE_float('momentum', default=0.9, help='Momentum.')
flags.DEFINE_float('weight_decay', default=1e-6,
                   help='Weight decay (default: 1e-6).')
flags.DEFINE_integer('print_freq', default=10,
                     help='Print frequency (default: 10).')
flags.DEFINE_bool('resume', default=False,
                  help='Whether to start from the latest checkpoint in the '
                       'logdir.')
flags.DEFINE_bool('launch_tensorboard', False,
                  help='Whether to launch tensorboard.')


def main(argv):
  # Manually check if there are unknown test datasets and if the dataset
  # ground truth files are downloaded.
  for dataset in FLAGS.test_datasets.split(','):
    if dataset not in _TEST_DATASET_NAMES:
      raise ValueError('Unsupported or unknown test dataset: {}.'.format(
        dataset))

    data_dir = os.path.join(FLAGS.data_root, 'gnd_{}.pkl'.format(dataset))
    if not os.path.isfile(data_dir):
      raise ValueError(
        '{} ground truth file at {} not found. Please download it according to '
        'the DELG instructions.'.format(dataset, FLAGS.data_root))

  # Check if train dataset is downloaded and dowload it if not found.
  sfm120k.download_train(FLAGS.data_root)

  # Creating model export directory if it does not exist.
  model_directory = global_features_utils.create_model_directory(
    FLAGS.training_dataset, FLAGS.arch, FLAGS.pool, FLAGS.whitening,
    FLAGS.pretrained, FLAGS.loss, FLAGS.loss_margin, FLAGS.optimizer, FLAGS.lr,
    FLAGS.weight_decay, FLAGS.neg_num, FLAGS.query_size, FLAGS.pool_size,
    FLAGS.batch_size, FLAGS.update_every, FLAGS.image_size, FLAGS.directory)

  # Setting up logging directory, same as where the model is stored.
  logging.get_absl_handler().use_absl_log_file('absl_logging', model_directory)

  # Set cuda visible device.
  os.environ['CUDA_VISIBLE_DEVICES'] = FLAGS.gpu_ids
  global_features_utils.debug_and_log('>> Num GPUs Available: {}'.format(
    len(tf.config.experimental.list_physical_devices('GPU'))), FLAGS.debug)

  # Allow memory growth.
  gpus = tf.config.experimental.list_physical_devices('GPU')
  if gpus:
    try:
      for gpu in gpus:
        tf.config.experimental.set_memory_growth(gpu, True)
    except RuntimeError as e:
      print(e)

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
  global_features_utils.debug_and_log('>> Network initialized.')

  global_features_utils.debug_and_log('>> Loss: {}.'.format(FLAGS.loss))
  # Define the loss function.
  if FLAGS.loss == 'contrastive':
    criterion = ranking_losses.ContrastiveLoss(margin=FLAGS.loss_margin)
  elif FLAGS.loss == 'triplet':
    criterion = ranking_losses.TripletLoss(margin=FLAGS.loss_margin)
  else:
    raise (RuntimeError('Loss {} not available!'.format(FLAGS.loss)))

  # Defining parameters for the training.
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
    optimizer = opt(weight_decay=FLAGS.weight_decay,
                    learning_rate=lr_scheduler)

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
    global_features_utils.launch_tensorboard(log_dir=model_directory)

  # Log flags used.
  global_features_utils.debug_and_log('>> Running training script with:')
  global_features_utils.debug_and_log('>> logdir = {}'.format(model_directory))

  if FLAGS.training_dataset.startswith('retrieval-SfM-120k'):
    train_dataset = sfm120k.CreateDataset(
      data_root=FLAGS.data_root,
      mode='train',
      imsize=FLAGS.image_size,
      nnum=FLAGS.neg_num,
      qsize=FLAGS.query_size,
      poolsize=FLAGS.pool_size
    )
    if FLAGS.val:
      val_dataset = sfm120k.CreateDataset(
        data_root=FLAGS.data_root,
        mode='val',
        imsize=FLAGS.image_size,
        nnum=FLAGS.neg_num,
        qsize=float('Inf'),
        poolsize=float('Inf'),
        eccv2020=FLAGS.val_eccv2020
      )
  else:
    train_dataset = tuples_dataset.TuplesDataset(
      name=FLAGS.training_dataset,
      data_root=FLAGS.data_root,
      mode='train',
      imsize=FLAGS.image_size,
      nnum=FLAGS.neg_num,
      qsize=FLAGS.query_size,
      poolsize=FLAGS.pool_size
    )
    if FLAGS.val:
      val_dataset = tuples_dataset.TuplesDataset(
        name=FLAGS.training_dataset,
        data_root=FLAGS.data_root,
        mode='val',
        imsize=FLAGS.image_size,
        nnum=FLAGS.neg_num,
        qsize=float('Inf'),
        poolsize=float('Inf')
      )

  output_types = [tf.float32 for i in range(2 + FLAGS.neg_num)]
  output_types.append(tf.int32)

  global_features_utils.debug_and_log(
    '>> Training the {} network'.format(model_directory))
  global_features_utils.debug_and_log('>> GPU ids: {}'.format(FLAGS.gpu_ids))

  with writer.as_default():

    # Precompute whitening if needed.
    if FLAGS.precompute_whitening:
      epoch = 0
      test(FLAGS.test_datasets, model, writer=writer, epoch=epoch,
           model_directory=model_directory)

    for epoch in range(start_epoch, FLAGS.epochs + 1):
      # Set manual seeds per epoch.
      np.random.seed(epoch)
      tf.random.set_seed(epoch)

      # Find hard-negatives.
      # While hard-positive examples are fixed during the whole training
      # process and are randomly chosen from every epoch; hard-negatives
      # depend on the current CNN parameters and are re-mined once per epoch.
      avg_neg_distance = train_dataset.create_epoch_tuples(model,
                                                           model_directory)

      def train_gen():
        return (inst for inst in train_dataset)

      train_loader = tf.data.Dataset.from_generator(train_gen,
                                                    output_types=tuple(
                                                      output_types))

      loss = train_val(loader=iter(train_loader), model=model,
                       criterion=criterion, optimizer=optimizer,
                       epoch=epoch)

      # Write a scalar summary.
      tf.summary.scalar('train_epoch_loss', loss, step=epoch)
      # Forces summary writer to send any buffered data to storage.
      writer.flush()

      # Evaluate on validation set.
      if FLAGS.val and (epoch % FLAGS.test_freq == 0 or epoch == 1):
        avg_neg_distance = val_dataset.create_epoch_tuples(model,
                                                           model_directory)

        def val_gen():
          return (inst for inst in val_dataset)

        val_loader = tf.data.Dataset.from_generator(val_gen,
                                                    output_types=tuple(
                                                      output_types))

        loss = train_val(loader=iter(val_loader), model=model,
                         criterion=criterion, optimizer=None,
                         epoch=epoch, train=False)
        tf.summary.scalar('val_epoch_loss', loss, step=epoch)
        writer.flush()

      # Evaluate on test datasets every test_freq epochs.
      if epoch == 1 or epoch % FLAGS.test_freq == 0:
        test(FLAGS.test_datasets, model, writer=writer, epoch=epoch,
             model_directory=model_directory)

      # Saving checkpoints and model weights.
      try:
        save_path = manager.save(checkpoint_number=epoch)
        global_features_utils.debug_and_log(
          'Saved ({}) at {}'.format(epoch, save_path))

        filename = os.path.join(model_directory,
                                'checkpoint_epoch_{}.h5'.format(
                                  epoch))
        model.save_weights(filename, save_format='h5')
        global_features_utils.debug_and_log(
          'Saved weights ({}) at {}'.format(epoch, filename))
      except Exception as ex:
        global_features_utils.debug_and_log(
          'Could not save checkpoint: {}'.format(ex))


def grad(criterion, model, input, target):
  """Records gradients and loss through the network..

  Args:
    criterion: Loss function.
    model: Network for the gradient computation.
    input: Tuple of query, positive and negative images.
    target: List of indexes to specify queries (-1), positives(1), negatives(0).

  Returns:
    loss: Loss for the training step.
    gradients: Computed gradients for the network trainable variables.
  """
  # Record gradients and loss through the network.
  with tf.GradientTape() as tape:
    output = tf.Variable(
      tf.zeros(shape=(0, model.meta['outputdim']), dtype=tf.float32))
    for img in input:
      # Compute descriptor vector for each image.
      o = model(tf.expand_dims(img, axis=0), training=True)
      output = tf.concat([output, o], 0)

    queries = tf.boolean_mask(output, target == -1, axis=0)
    positives = tf.boolean_mask(output, target == 1, axis=0)
    negatives = tf.boolean_mask(output, target == 0, axis=0)
    negatives = tf.reshape(negatives, [tf.shape(queries)[0], FLAGS.neg_num,
                                       model.meta['outputdim']])
    # Loss calculation.
    loss = criterion(queries, positives, negatives)

  return loss, tape.gradient(loss, model.trainable_variables)


def train_val(loader, model, criterion, optimizer, epoch, train=True):
  """Executes either training or validation step based on `train` value.

  Args:
    loader: Training/validation iterable dataset.
    model: Network to train/validate.
    criterion: Loss function.
    optimizer: Network optimizer.
    epoch: Integer, epoch number.
    train: Bool, specifies training or validation phase.

  Returns:
    average_epoch_loss: Average epoch loss.
  """

  batch_time = global_features_utils.AverageMeter()
  data_time = global_features_utils.AverageMeter()
  losses = global_features_utils.AverageMeter()

  # Retrieve all trainable variables we defined in the graph.
  tvs = model.trainable_variables
  accum_grads = [tf.Variable(tf.zeros_like(tv.read_value()), trainable=False)
                 for tv in tvs]

  end = time.time()
  batch_num = 0
  all_batch_num = FLAGS.query_size // FLAGS.batch_size
  state = 'Train' if train else 'Val'
  global_features_utils.debug_and_log('>> {} step:'.format(state))

  # For every batch in the dataset; Stops when all batches in the dataset have
  # been processed.
  while True:
    data_time.update(time.time() - end)

    if train:
      try:
        # Train on one batch.
        # We load batches into memory consequently.
        for _ in range(FLAGS.batch_size):
          # Because the images are not necessarily of the same size, we can't
          # set the batch size with .batch().
          batch = loader.get_next()
          input_tuple = batch[0:-1]
          target_tuple = batch[-1]

          loss_value, grads = grad(criterion, model, input_tuple, target_tuple)
          losses.update(loss_value)
          # Adds to each element from the list you initialized earlier
          # with zeros its gradient (works because accum_vars and gvs
          # are in the same order).
          accum_grads = [accum_grads[i].assign_add(gv) for i, gv in
                         enumerate(grads)]

        if (batch_num + 1) % FLAGS.update_every == 0 or (
                batch_num + 1) == all_batch_num:
          # Do one step for multiple batches. Accumulated gradients are
          # used.
          optimizer.apply_gradients(
            zip(accum_grads, model.trainable_variables))
          accum_grads = [
            tf.Variable(tf.zeros_like(tv.read_value()), trainable=False)
            for tv in tvs]
      except Exception as ex:
        global_features_utils.debug_and_log(ex)
        break

    else:
      # Validate one batch.
      # We load full batch into memory.
      input = []
      target = []
      try:
        for _ in range(FLAGS.batch_size):
          # Because the images are not necessarily of the same size, we can't
          # set the batch size with .batch().
          batch = loader.get_next()
          input.append(batch[0:-1])
          target.append(batch[-1])
      except Exception as ex:
        global_features_utils.debug_and_log(ex)
        break

      output = tf.zeros(shape=(0, model.meta['outputdim']), dtype=tf.float32)
      for input_tuple in input:
        for img in input_tuple:
          # Compute the global descriptor vector.
          model_out = model(tf.expand_dims(img, axis=0), training=False)
          output = tf.concat([output, model_out], 0)

      # No need to reduce memory consumption (no backward pass):
      # Compute loss for the full batch.
      tmp_target = tf.concat(target, axis=0)
      queries = tf.boolean_mask(output, tmp_target == -1, axis=0)
      positives = tf.boolean_mask(output, tmp_target == 1, axis=0)
      negatives = tf.boolean_mask(output, tmp_target == 0, axis=0)
      negatives = tf.reshape(negatives, [tf.shape(queries)[0], FLAGS.neg_num,
                                         model.meta['outputdim']])
      loss = criterion(queries, positives, negatives)

      # Record loss.
      losses.update(loss / FLAGS.batch_size, FLAGS.batch_size)

    # Measure elapsed time.
    batch_time.update(time.time() - end)
    end = time.time()

    # Record immediate loss and elapsed time.
    if FLAGS.debug and ((batch_num + 1) % FLAGS.print_freq == 0 or
                        batch_num == 0 or (batch_num + 1) == all_batch_num):
      global_features_utils.debug_and_log(
        '>> {0}: [{1} epoch][{2}/{3} batch]\t Time val: {batch_time.val:.3f} '
        '(Batch Time avg: {batch_time.avg:.3f})\t Data {data_time.val:.3f} ('
        'Time avg: {data_time.avg:.3f})\t Immediate loss value: {loss.val:.4f} '
        '(Loss avg: {loss.avg:.4f})'.format(
          state, epoch, batch_num + 1, all_batch_num, batch_time=batch_time,
          data_time=data_time, loss=losses), debug=True, log=False)
    batch_num += 1

  return losses.avg


def test(datasets, net, epoch, writer=None, model_directory=None):
  """Testing step.

  Evaluates the network on the provided test datasets by computing single-scale
  mAP for easy/medium/hard cases. If `writer` is specified, saves the mAP
  values in a tensorboard supported format.

  Args:
    datasets: List of dataset names for model testing (from
      `_TEST_DATASET_NAMES`).
    net: Network to evaluate.
    epoch: Integer, epoch number.
    writer: Tensorboard writer.
    model_directory: String, path to the model directory.
  """
  global_features_utils.debug_and_log(">> Testing step:")
  global_features_utils.debug_and_log(
    '>> Evaluating network on test datasets...')

  # For testing we use image size of max 1024.
  image_size = 1024

  # Precompute whitening.
  if FLAGS.precompute_whitening:

    # If whitening already precomputed, load it and skip the computations.
    filename = os.path.join(
      model_directory, 'learned_whitening_mP_{}_epoch.pkl'.format(epoch))
    filename_layer = os.path.join(
      model_directory, 'learned_whitening_layer_config_{}_epoch.pkl'.format(
        epoch))

    if os.path.isfile(filename):
      global_features_utils.debug_and_log(
        '>> {}: Whitening for this epoch is already precomputed. '
        'Loading...'.format(FLAGS.precompute_whitening))
      with tf.io.gfile.GFile(filename, 'rb') as learned_whitening_file:
        learned_whitening = pickle.load(learned_whitening_file)

    else:
      start = time.time()
      global_features_utils.debug_and_log(
        '>> {}: Learning whitening...'.format(FLAGS.precompute_whitening))

      # Loading db.
      db_root = os.path.join(FLAGS.data_root, 'train',
                             FLAGS.precompute_whitening)
      ims_root = os.path.join(db_root, 'ims')
      db_fn = os.path.join(db_root,
                           '{}-whiten.pkl'.format(FLAGS.precompute_whitening))
      with tf.io.gfile.GFile(db_fn, 'rb') as f:
        db = pickle.load(f)
      images = [sfm120k.id2filename(db['cids'][i], ims_root) for i in
                range(len(db['cids']))]

      # Extract whitening vectors.
      global_features_utils.debug_and_log(
        '>> {}: Extracting...'.format(FLAGS.precompute_whitening))
      wvecs = global_model.extract_global_descriptors_from_list(net, images,
                                                                image_size)

      # Learning whitening.
      global_features_utils.debug_and_log(
        '>> {}: Learning...'.format(FLAGS.precompute_whitening))
      wvecs = wvecs.numpy()
      m, P = whiten.whitenlearn(wvecs, db['qidxs'], db['pidxs'])
      learned_whitening = {'m': m, 'P': P}

      global_features_utils.debug_and_log(
        '>> {}: Elapsed time: {}'.format(FLAGS.precompute_whitening,
                                         global_features_utils.htime(
                                           time.time() - start)))
      # Save learned_whitening parameters for a later use.
      with tf.io.gfile.GFile(filename, 'wb') as learned_whitening_file:
        pickle.dump(learned_whitening, learned_whitening_file)

      # Saving whitening as a layer.
      bias = -np.dot(m.T, P.T)
      whitening_layer = tf.keras.layers.Dense(
        net.meta['outputdim'],
        activation=None,
        use_bias=True,
        kernel_initializer=tf.keras.initializers.Constant(P.T),
        bias_initializer=tf.keras.initializers.Constant(bias)
      )
      with tf.io.gfile.GFile(filename_layer, 'wb') as learned_whitening_file:
        pickle.dump(whitening_layer.get_config(), learned_whitening_file)
  else:
    learned_whitening = None

  # Evaluate on test datasets.
  datasets = datasets.split(',')
  for dataset in datasets:
    start = time.time()

    # Prepare config structure for the test dataset.
    cfg = testdataset.create_config_for_test_dataset(dataset, os.path.join(
      FLAGS.data_root))
    images = [cfg['im_fname'](cfg, i) for i in range(cfg['n'])]
    qimages = [cfg['qim_fname'](cfg, i) for i in range(cfg['nq'])]
    bounding_boxes = [tuple(cfg['gnd'][i]['bbx']) for i in range(cfg['nq'])]

    # Extract database and query vectors.
    global_features_utils.debug_and_log(
      '>> {}: Extracting database images...'.format(dataset))
    vecs = global_model.extract_global_descriptors_from_list(
      net, images, image_size, ms=list(eval(FLAGS.multiscale)))
    global_features_utils.debug_and_log(
      '>> {}: Extracting query images...'.format(dataset))
    qvecs = global_model.extract_global_descriptors_from_list(
      net, qimages, image_size, bounding_boxes, ms=list(eval(FLAGS.multiscale)))

    global_features_utils.debug_and_log('>> {}: Evaluating...'.format(dataset))

    # Convert the obtained descriptors to numpy.
    vecs = vecs.numpy()
    qvecs = qvecs.numpy()

    # Search, rank and print test set metrics.
    scores = np.dot(vecs.T, qvecs)
    ranks = np.transpose(np.argsort(-scores, axis=0))
    metrics = global_features_utils.compute_metrics_and_print(dataset, ranks,
                                                              cfg['gnd'])
    # Save calculated metrics in a tensorboard format.
    if writer:
      tf.summary.scalar('test_accuracy_{}_E'.format(dataset), metrics[0][0],
                        step=epoch)
      tf.summary.scalar('test_accuracy_{}_M'.format(dataset), metrics[1][0],
                        step=epoch)
      tf.summary.scalar('test_accuracy_{}_H'.format(dataset), metrics[2][0],
                        step=epoch)
      writer.flush()

    if learned_whitening is not None:

      # Whiten the vectors.
      m = learned_whitening['m']
      P = learned_whitening['P']
      vecs_lw = whiten.whitenapply(vecs, m, P)
      qvecs_lw = whiten.whitenapply(qvecs, m, P)

      # Search, rank, and print.
      scores = np.dot(vecs_lw.T, qvecs_lw)
      ranks = np.transpose(np.argsort(-scores, axis=0))
      metrics = global_features_utils.compute_metrics_and_print(
        dataset + ' + whiten', ranks, cfg['gnd'])

      if writer:
        tf.summary.scalar('test_accuracy_whiten_{}_E'.format(dataset),
                          metrics[0][0], step=epoch)
        tf.summary.scalar('test_accuracy_whiten_{}_M'.format(dataset),
                          metrics[1][0], step=epoch)
        tf.summary.scalar('test_accuracy_whiten_{}_H'.format(dataset),
                          metrics[2][0], step=epoch)
        writer.flush()

    global_features_utils.debug_and_log(
      '>> {}: Elapsed time: {}'.format(dataset, global_features_utils.htime(
        time.time() - start)))


if __name__ == '__main__':
  app.run(main)
