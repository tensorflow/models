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
"""Training utilities for Global Features model."""

import os
import pickle
import time

import numpy as np
import tensorflow as tf

from delf.python import whiten
from delf.python.datasets.revisited_op import dataset as test_dataset
from delf.python.datasets.sfm120k import sfm120k
from delf.python.training import global_features_utils
from delf.python.training.model import global_model


def _compute_loss_and_gradient(criterion, model, input, target, neg_num=5):
  """Records gradients and loss through the network.

  Args:
    criterion: Loss function.
    model: Network for the gradient computation.
    input: Tuple of query, positive and negative images.
    target: List of indexes to specify queries (-1), positives(1), negatives(0).
    neg_num: Integer, number of negatives per a tuple.

  Returns:
    loss: Loss for the training step.
    gradients: Computed gradients for the network trainable variables.
  """
  # Record gradients and loss through the network.
  with tf.GradientTape() as tape:
    descriptors = tf.zeros(shape=(0, model.meta['outputdim']), dtype=tf.float32)
    for img in input:
      # Compute descriptor vector for each image.
      o = model(tf.expand_dims(img, axis=0), training=True)
      descriptors = tf.concat([descriptors, o], 0)

    queries = descriptors[target == -1]
    positives = descriptors[target == 1]
    negatives = descriptors[target == 0]

    negatives = tf.reshape(negatives, [tf.shape(queries)[0], neg_num,
                                       model.meta['outputdim']])
    # Loss calculation.
    loss = criterion(queries, positives, negatives)

  return loss, tape.gradient(loss, model.trainable_variables)


def train_val_one_epoch(
        loader, model, criterion, optimizer, epoch, train=True, batch_size=5,
        query_size=2000, neg_num=5, update_every=1, debug=False):
  """Executes either training or validation step based on `train` value.

  Args:
    loader: Training/validation iterable dataset.
    model: Network to train/validate.
    criterion: Loss function.
    optimizer: Network optimizer.
    epoch: Integer, epoch number.
    train: Bool, specifies training or validation phase.
    batch_size: Integer, number of (q,p,n1,...,nN) tuples in a mini-batch.
    query_size: Integer, number of queries randomly drawn per one training
      epoch.
    neg_num: Integer, number of negatives per a tuple.
    update_every: Integer, update model weights every N batches, used to
      handle relatively large batches batch_size effectively becomes
      update_every x batch_size.
    debug: Bool, whether debug mode is used.

  Returns:
    average_epoch_loss: Average epoch loss.
  """
  batch_time = global_features_utils.AverageMeter()
  data_time = global_features_utils.AverageMeter()
  losses = global_features_utils.AverageMeter()

  # Retrieve all trainable variables we defined in the graph.
  tvs = model.trainable_variables
  accum_grads = [tf.zeros_like(tv.read_value()) for tv in tvs]

  end = time.time()
  batch_num = 0
  print_frequency = 10
  all_batch_num = query_size // batch_size
  state = 'Train' if train else 'Val'
  global_features_utils.debug_and_log('>> {} step:'.format(state))

  # For every batch in the dataset; Stops when all batches in the dataset have
  # been processed.
  while True:
    data_time.update(time.time() - end)

    if train:
      try:
        # Train on one batch.
        # Each image in the batch is loaded into memory consecutively.
        for _ in range(batch_size):
          # Because the images are not necessarily of the same size, we can't
          # set the batch size with .batch().
          batch = loader.get_next()
          input_tuple = batch[0:-1]
          target_tuple = batch[-1]

          loss_value, grads = _compute_loss_and_gradient(
                  criterion, model, input_tuple, target_tuple, neg_num)
          losses.update(loss_value)
          # Accumulate gradients.
          accum_grads += grads

        # Perform weight update if required.
        if (batch_num + 1) % update_every == 0 or (
                batch_num + 1) == all_batch_num:
          # Do one step for multiple batches. Accumulated gradients are
          # used.
          optimizer.apply_gradients(
                  zip(accum_grads, model.trainable_variables))
          accum_grads = [tf.zeros_like(tv.read_value()) for tv in tvs]
      # We break when we run out of range, i.e., we exhausted all dataset
      # images.
      except tf.errors.OutOfRangeError:
        break

    else:
      # Validate one batch.
      # We load full batch into memory.
      input = []
      target = []
      try:
        for _ in range(batch_size):
          # Because the images are not necessarily of the same size, we can't
          # set the batch size with .batch().
          batch = loader.get_next()
          input.append(batch[0:-1])
          target.append(batch[-1])
      # We break when we run out of range, i.e., we exhausted all dataset
      # images.
      except tf.errors.OutOfRangeError:
        break

      descriptors = tf.zeros(shape=(0, model.meta['outputdim']),
                             dtype=tf.float32)

      for input_tuple in input:
        for img in input_tuple:
          # Compute the global descriptor vector.
          model_out = model(tf.expand_dims(img, axis=0), training=False)
          descriptors = tf.concat([descriptors, model_out], 0)

      # No need to reduce memory consumption (no backward pass):
      # Compute loss for the full batch.
      queries = descriptors[target == -1]
      positives = descriptors[target == 1]
      negatives = descriptors[target == 0]
      negatives = tf.reshape(negatives, [tf.shape(queries)[0], neg_num,
                                         model.meta['outputdim']])
      loss = criterion(queries, positives, negatives)

      # Record loss.
      losses.update(loss / batch_size, batch_size)

    # Measure elapsed time.
    batch_time.update(time.time() - end)
    end = time.time()

    # Record immediate loss and elapsed time.
    if debug and ((batch_num + 1) % print_frequency == 0 or
                  batch_num == 0 or (batch_num + 1) == all_batch_num):
      global_features_utils.debug_and_log(
              '>> {0}: [{1} epoch][{2}/{3} batch]\t Time val: {'
              'batch_time.val:.3f} '
              '(Batch Time avg: {batch_time.avg:.3f})\t Data {'
              'data_time.val:.3f} ('
              'Time avg: {data_time.avg:.3f})\t Immediate loss value: {'
              'loss.val:.4f} '
              '(Loss avg: {loss.avg:.4f})'.format(
                      state, epoch, batch_num + 1, all_batch_num,
                      batch_time=batch_time,
                      data_time=data_time, loss=losses), debug=True, log=False)
    batch_num += 1

  return losses.avg


def test_retrieval(datasets, net, epoch, writer=None, model_directory=None,
                   precompute_whitening=None, data_root='data', multiscale=[1.],
                   test_image_size=1024):
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
    precompute_whitening: Dataset used to learn whitening. If no
      precomputation required, then `None`. Only 'retrieval-SfM-30k' and
      'retrieval-SfM-120k' datasets are supported for whitening pre-computation.
    data_root: Absolute path to the data folder.
    multiscale: List of scales for multiscale testing.
    test_image_size: Integer, maximum size of the test images.
  """
  global_features_utils.debug_and_log(">> Testing step:")
  global_features_utils.debug_and_log(
          '>> Evaluating network on test datasets...')

  # Precompute whitening.
  if precompute_whitening is not None:

    # If whitening already precomputed, load it and skip the computations.
    filename = os.path.join(
            model_directory, 'learned_whitening_mP_{}_epoch.pkl'.format(epoch))
    filename_layer = os.path.join(
            model_directory,
            'learned_whitening_layer_config_{}_epoch.pkl'.format(
                    epoch))

    if tf.io.gfile.exists(filename):
      global_features_utils.debug_and_log(
              '>> {}: Whitening for this epoch is already precomputed. '
              'Loading...'.format(precompute_whitening))
      with tf.io.gfile.GFile(filename, 'rb') as learned_whitening_file:
        learned_whitening = pickle.load(learned_whitening_file)

    else:
      start = time.time()
      global_features_utils.debug_and_log(
              '>> {}: Learning whitening...'.format(precompute_whitening))

      # Loading db.
      db_root = os.path.join(data_root, 'train', precompute_whitening)
      ims_root = os.path.join(db_root, 'ims')
      db_filename = os.path.join(db_root,
                                 '{}-whiten.pkl'.format(precompute_whitening))
      with tf.io.gfile.GFile(db_filename, 'rb') as f:
        db = pickle.load(f)
      images = [sfm120k.id2filename(db['cids'][i], ims_root) for i in
                range(len(db['cids']))]

      # Extract whitening vectors.
      global_features_utils.debug_and_log(
              '>> {}: Extracting...'.format(precompute_whitening))
      wvecs = global_model.extract_global_descriptors_from_list(net, images,
                                                                test_image_size)

      # Learning whitening.
      global_features_utils.debug_and_log(
              '>> {}: Learning...'.format(precompute_whitening))
      wvecs = wvecs.numpy()
      mean_vector, projection_matrix = whiten.whitenlearn(wvecs, db['qidxs'],
                                                          db['pidxs'])
      learned_whitening = {'m': mean_vector, 'P': projection_matrix}

      global_features_utils.debug_and_log(
              '>> {}: Elapsed time: {}'.format(precompute_whitening,
                                               global_features_utils.htime(
                                                       time.time() - start)))
      # Save learned_whitening parameters for a later use.
      with tf.io.gfile.GFile(filename, 'wb') as learned_whitening_file:
        pickle.dump(learned_whitening, learned_whitening_file)

      # Saving whitening as a layer.
      bias = -np.dot(mean_vector.T, projection_matrix.T)
      whitening_layer = tf.keras.layers.Dense(
              net.meta['outputdim'],
              activation=None,
              use_bias=True,
              kernel_initializer=tf.keras.initializers.Constant(
                      projection_matrix.T),
              bias_initializer=tf.keras.initializers.Constant(bias)
      )
      with tf.io.gfile.GFile(filename_layer, 'wb') as learned_whitening_file:
        pickle.dump(whitening_layer.get_config(), learned_whitening_file)
  else:
    learned_whitening = None

  # Evaluate on test datasets.
  for dataset in datasets:
    start = time.time()

    # Prepare config structure for the test dataset.
    cfg = test_dataset.CreateConfigForTestDataset(dataset,
                                                  os.path.join(data_root))
    images = [cfg['im_fname'](cfg, i) for i in range(cfg['n'])]
    qimages = [cfg['qim_fname'](cfg, i) for i in range(cfg['nq'])]
    bounding_boxes = [tuple(cfg['gnd'][i]['bbx']) for i in range(cfg['nq'])]

    # Extract database and query vectors.
    global_features_utils.debug_and_log(
            '>> {}: Extracting database images...'.format(dataset))
    vecs = global_model.extract_global_descriptors_from_list(
            net, images, test_image_size, scales=multiscale)
    global_features_utils.debug_and_log(
            '>> {}: Extracting query images...'.format(dataset))
    qvecs = global_model.extract_global_descriptors_from_list(
            net, qimages, test_image_size, bounding_boxes,
            scales=multiscale)

    global_features_utils.debug_and_log('>> {}: Evaluating...'.format(dataset))

    # Convert the obtained descriptors to numpy.
    vecs = vecs.numpy()
    qvecs = qvecs.numpy()

    # Search, rank and print test set metrics.
    _calculate_metrics_and_export_to_tensorboard(vecs, qvecs, dataset, cfg,
                                                 writer, epoch, whiten=False)

    if learned_whitening is not None:
      # Whiten the vectors.
      mean_vector = learned_whitening['m']
      projection_matrix = learned_whitening['P']
      vecs_lw = whiten.whitenapply(vecs, mean_vector, projection_matrix)
      qvecs_lw = whiten.whitenapply(qvecs, mean_vector, projection_matrix)

      # Search, rank, and print.
      _calculate_metrics_and_export_to_tensorboard(
              vecs_lw, qvecs_lw, dataset, cfg, writer, epoch, whiten=True)

    global_features_utils.debug_and_log(
            '>> {}: Elapsed time: {}'.format(
                    dataset, global_features_utils.htime(time.time() - start)))


def _calculate_metrics_and_export_to_tensorboard(vecs, qvecs, dataset, cfg,
                                                 writer, epoch, whiten=False):
  """
  Calculates metrics and exports them to tensorboard.

  Args:
    vecs: Numpy array dataset global descriptors.
    qvecs: Numpy array query global descriptors.
    dataset: String, one of `_TEST_DATASET_NAMES`.
    cfg: Dataset configuration.
    writer: Tensorboard writer.
    epoch: Integer, epoch number.
    whiten: Boolean, whether the metrics are with for whitening used as a
      post-processing step. Affects the name of the extracted TensorBoard
      metrics.
  """
  # Search, rank and print test set metrics.
  scores = np.dot(vecs.T, qvecs)
  ranks = np.transpose(np.argsort(-scores, axis=0))

  metrics = global_features_utils.compute_metrics_and_print(dataset, ranks,
                                                            cfg['gnd'])
  # Save calculated metrics in a tensorboard format.
  if writer:
    if whiten:
      metric_names = ['test_accuracy_whiten_{}_E'.format(dataset),
                      'test_accuracy_whiten_{}_M'.format(dataset),
                      'test_accuracy_whiten_{}_H'.format(dataset)]
    else:
      metric_names = ['test_accuracy_{}_E'.format(dataset),
                      'test_accuracy_{}_M'.format(dataset),
                      'test_accuracy_{}_H'.format(dataset)]
    tf.summary.scalar(metric_names[0], metrics[0][0], step=epoch)
    tf.summary.scalar(metric_names[1], metrics[1][0], step=epoch)
    tf.summary.scalar(metric_names[2], metrics[2][0], step=epoch)
    writer.flush()
  return None
