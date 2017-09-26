# Copyright 2016 Google Inc. All Rights Reserved.
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

"""Submatrix-wise Vector Embedding Learner.

Implementation of SwiVel algorithm described at:
http://arxiv.org/abs/1602.02215

This program expects an input directory that contains the following files.

  row_vocab.txt, col_vocab.txt

    The row an column vocabulary files.  Each file should contain one token per
    line; these will be used to generate a tab-separate file containing the
    trained embeddings.

  row_sums.txt, col_sum.txt

    The matrix row and column marginal sums.  Each file should contain one
    decimal floating point number per line which corresponds to the marginal
    count of the matrix for that row or column.

  shards.recs

    A file containing the sub-matrix shards, stored as TFRecords.  Each shard is
    expected to be a serialzed tf.Example protocol buffer with the following
    properties:

      global_row: the global row indicies contained in the shard
      global_col: the global column indicies contained in the shard
      sparse_local_row, sparse_local_col, sparse_value: three parallel arrays
      that are a sparse representation of the submatrix counts.

It will generate embeddings, training from the input directory for the specified
number of epochs.  When complete, it will output the trained vectors to a
tab-separated file that contains one line per embedding.  Row and column
embeddings are stored in separate files.

Swivel can be run "stand-alone" or "distributed".  The latter involves running
at least one parameter server process, along with one or more worker processes.
"""

from __future__ import division
from __future__ import print_function

import glob
import itertools
import os
import random

import numpy as np
import scipy.stats
import tensorflow as tf

flags = tf.app.flags

flags.DEFINE_string(
    'input_base_path', '/tmp/swivel_data',
    'Directory containing input shards, vocabularies, and marginals.')
flags.DEFINE_string(
    'output_base_path', '/tmp/swivel_data',
    'Path where to write the trained embeddings.')
flags.DEFINE_string('eval_base_path', '', 'Path to evaluation data')

# Control for training.
flags.DEFINE_float('num_epochs', 40, 'Number epochs to train')
flags.DEFINE_string('hparams', '', 'Model hyper-parameters')

# Model hyper-parameters. (Move these to tf.HParams once that gets integrated
# into TF from tf.contrib.)
flags.DEFINE_integer(
    'dim', 300, 'Embedding dimensionality')
flags.DEFINE_string(
    'optimizer', 'rmsprop', 'SGD optimizer; either "adagrad" or "rmsprop"')
flags.DEFINE_float(
    'learning_rate', 0.1, 'Optimizer learning rate')
flags.DEFINE_float(
    'momentum', 0.1, 'Optimizer momentum; used with RMSProp')
flags.DEFINE_float(
    'confidence_base', 0.0, 'Base for count weighting')
flags.DEFINE_float(
    'confidence_scale', 1.0, 'Scale for count weighting')
flags.DEFINE_float(
    'confidence_exponent', 0.5, 'Exponent for count weighting')
flags.DEFINE_integer(
    'submatrix_rows', 4096, 'Number of rows in each submatrix')
flags.DEFINE_integer(
    'submatrix_cols', 4096, 'Number of cols in each submatrix')

# For distributed training.
flags.DEFINE_string(
    'ps_hosts', '',
    'Comma-separated list of parameter server host:port; if empty, run local')
flags.DEFINE_string(
    'worker_hosts', '', 'Comma-separated list of worker host:port')
flags.DEFINE_string(
    'job_name', '', 'The job this process will run, either "ps" or "worker"')
flags.DEFINE_integer(
    'task_index', 0, 'The task index for this process')
flags.DEFINE_integer(
    'gpu_device', 0, 'The GPU device to use.')

FLAGS = flags.FLAGS


class Model(object):
  """A Swivel model."""

  def __init__(self, input_base_path, hparams):
    """Creates a new Swivel model."""
    # Read vocab
    self.row_ix_to_word, self.row_word_to_ix = self._read_vocab(
        os.path.join(input_base_path, 'row_vocab.txt'))
    self.col_ix_to_word, self.col_word_to_ix = self._read_vocab(
        os.path.join(input_base_path, 'col_vocab.txt'))

    # Read marginals.
    row_sums = self._read_marginals_file(
        os.path.join(input_base_path, 'row_sums.txt'))
    col_sums = self._read_marginals_file(
        os.path.join(input_base_path, 'col_sums.txt'))

    # Construct input tensors.
    count_matrix_files = glob.glob(
        os.path.join(input_base_path, 'shard-*.pb'))

    global_rows, global_cols, counts = self._count_matrix_input(
        count_matrix_files, hparams.submatrix_rows, hparams.submatrix_cols)

    # Create embedding variables.
    sigma = 1.0 / np.sqrt(hparams.dim)
    self.row_embedding = tf.get_variable(
        'row_embedding',
        shape=[len(row_sums), hparams.dim],
        initializer=tf.random_normal_initializer(0, sigma),
        dtype=tf.float32)
    self.col_embedding = tf.get_variable(
        'col_embedding',
        shape=[len(col_sums), hparams.dim],
        initializer=tf.random_normal_initializer(0, sigma),
        dtype=tf.float32)

    matrix_log_sum = np.log(np.sum(row_sums) + 1)
    row_bias = tf.constant(
        [np.log(x + 1) for x in row_sums], dtype=tf.float32)
    col_bias = tf.constant(
        [np.log(x + 1) for x in col_sums], dtype=tf.float32)

    # Fetch embeddings.
    selected_rows = tf.nn.embedding_lookup(self.row_embedding, global_rows)
    selected_cols = tf.nn.embedding_lookup(self.col_embedding, global_cols)

    selected_row_bias = tf.gather(row_bias, global_rows)
    selected_col_bias = tf.gather(col_bias, global_cols)

    predictions = tf.matmul(selected_rows, selected_cols, transpose_b=True)

    # These binary masks separate zero from non-zero values.
    count_is_nonzero = tf.to_float(tf.cast(counts, tf.bool))
    count_is_zero = 1 - count_is_nonzero

    objectives = count_is_nonzero * tf.log(counts + 1e-30)
    objectives -= tf.reshape(selected_row_bias, [-1, 1])
    objectives -= selected_col_bias
    objectives += matrix_log_sum

    err = predictions - objectives

    # The confidence function scales the L2 loss based on the raw
    # co-occurrence count.
    l2_confidence = (hparams.confidence_base +
                     hparams.confidence_scale * tf.pow(
                         counts, hparams.confidence_exponent))

    loss_multiplier = 1 / np.sqrt(
        hparams.submatrix_rows * hparams.submatrix_cols)

    l2_loss = loss_multiplier * tf.reduce_sum(
        0.5 * l2_confidence * tf.square(err))

    sigmoid_loss = loss_multiplier * tf.reduce_sum(
        tf.nn.softplus(err) * count_is_zero)

    self.loss_op = l2_loss + sigmoid_loss

    if hparams.optimizer == 'adagrad':
      opt = tf.train.AdagradOptimizer(hparams.learning_rate)
    elif hparams.optimizer == 'rmsprop':
      opt = tf.train.RMSPropOptimizer(hparams.learning_rate, hparams.momentum)
    else:
      raise ValueError('unknown optimizer "%s"' % hparams.optimizer)

    self.global_step = tf.get_variable(
        'global_step', initializer=0, trainable=False)

    self.train_op = opt.minimize(self.loss_op, global_step=self.global_step)

    # One epoch trains each submatrix once.
    self.steps_per_epoch = (
        (len(row_sums) / hparams.submatrix_rows) *
        (len(col_sums) / hparams.submatrix_cols))

  def _read_vocab(self, filename):
    """Reads the vocabulary file."""
    with open(filename) as lines:
      ix_to_word = [line.strip() for line in lines]
      word_to_ix = {word: ix for ix, word in enumerate(ix_to_word)}
      return ix_to_word, word_to_ix

  def _read_marginals_file(self, filename):
    """Reads text file with one number per line to an array."""
    with open(filename) as lines:
      return [float(line.strip()) for line in lines]

  def _count_matrix_input(self, filenames, submatrix_rows, submatrix_cols):
    """Creates ops that read submatrix shards from disk."""
    random.shuffle(filenames)
    filename_queue = tf.train.string_input_producer(filenames)
    reader = tf.WholeFileReader()
    _, serialized_example = reader.read(filename_queue)
    features = tf.parse_single_example(
        serialized_example,
        features={
            'global_row': tf.FixedLenFeature([submatrix_rows], dtype=tf.int64),
            'global_col': tf.FixedLenFeature([submatrix_cols], dtype=tf.int64),
            'sparse_local_row': tf.VarLenFeature(dtype=tf.int64),
            'sparse_local_col': tf.VarLenFeature(dtype=tf.int64),
            'sparse_value': tf.VarLenFeature(dtype=tf.float32)
        })

    global_row = features['global_row']
    global_col = features['global_col']

    sparse_local_row = features['sparse_local_row'].values
    sparse_local_col = features['sparse_local_col'].values
    sparse_count = features['sparse_value'].values

    sparse_indices = tf.concat(
        axis=1, values=[tf.expand_dims(sparse_local_row, 1),
                        tf.expand_dims(sparse_local_col, 1)])

    count = tf.sparse_to_dense(sparse_indices, [submatrix_rows, submatrix_cols],
                               sparse_count)

    return global_row, global_col, count

  def wordsim_eval_op(self, filename):
    """Returns an op that runs an eval on a word similarity dataset.

    The eval dataset is assumed to be tab-separated, one scored word pair per
    line.  The resulting value is Spearman's rho of the human judgements with
    the cosine similarity of the word embeddings.

    Args:
      filename: the filename containing the word similarity data.

    Returns:
      An operator that will compute Spearman's rho of the current row
      embeddings.
    """
    with open(filename, 'r') as fh:
      tuples = (line.strip().split('\t') for line in fh.read().splitlines())
      word1s, word2s, sims = zip(*tuples)
      actuals = map(float, sims)

    v1s_t = tf.nn.embedding_lookup(
        self.row_embedding,
        [self.row_word_to_ix.get(w, 0) for w in word1s])

    v2s_t = tf.nn.embedding_lookup(
        self.row_embedding,
        [self.row_word_to_ix.get(w, 0) for w in word2s])

    # Compute the predicted word similarity as the cosine similarity between the
    # embedding vectors.
    preds_t = tf.reduce_sum(
        tf.nn.l2_normalize(v1s_t, dim=1) * tf.nn.l2_normalize(v2s_t, dim=1),
        axis=1)

    def _op(preds):
      rho, _ = scipy.stats.spearmanr(preds, actuals)
      return rho

    return tf.py_func(_op, [preds_t], tf.float64)

  def analogy_eval_op(self, filename, max_vocab_size=20000):
    """Returns an op that runs an eval on an analogy dataset.

    The eval dataset is assumed to be tab-separated, with four tokens per
    line. The first three tokens are query terms, the last is the expected
    answer. For each line (e.g., "man king woman queen"), the vectors
    corresponding to the query terms are added ("king - man + woman") to produce
    a query vector.  If the expected answer's vector is the nearest neighbor to
    the query vector (not counting any of the query vectors themselves), then
    the line is scored as correct.  The reported accuracy is the number of
    correct rows divided by the total number of rows.  Missing terms are
    replaced with an arbitrary vector and will almost certainly result in
    incorrect answers.

    Note that the results are approximate: for efficiency's sake, only the first
    `max_vocab_size` terms are included in the nearest neighbor search.

    Args:
      filename: the filename containing the analogy data.
      max_vocab_size: the maximum number of tokens to include in the nearest
        neighbor search. By default, 20000.

    Returns:
      The accuracy on the analogy task.
    """
    analogy_ixs = []
    with open(filename, 'r') as lines:
      for line in lines:
        parts = line.strip().split('\t')
        if len(parts) == 4:
          analogy_ixs.append([self.row_word_to_ix.get(w, 0) for w in parts])

    # man:king :: woman:queen => king - man + woman == queen
    ix1s, ix2s, ix3s, _ = zip(*analogy_ixs)
    v1s_t, v2s_t, v3s_t = (
        tf.nn.l2_normalize(
            tf.nn.embedding_lookup(self.row_embedding, ixs),
            dim=1)
        for ixs in (ix1s, ix2s, ix3s))

    preds_t = v2s_t - v1s_t + v3s_t

    # Compute the nearest neighbors as the cosine similarity.  We only consider
    # up to max_vocab_size to avoid a matmul that swamps the machine.
    sims_t = tf.matmul(
        preds_t,
        tf.nn.l2_normalize(self.row_embedding[:max_vocab_size], dim=1),
        transpose_b=True)

    # Take the four nearest neighbors, since the eval explicitly discards the
    # query terms.
    _, preds_ixs_t = tf.nn.top_k(sims_t, 4)

    def _op(preds_ixs):
      correct, total = 0, 0
      for pred_ixs, actual_ixs in itertools.izip(preds_ixs, analogy_ixs):
        pred_ixs = [ix for ix in pred_ixs if ix not in actual_ixs[:3]]
        correct += pred_ixs[0] == actual_ixs[3]
        total += 1

      return correct / total

    return tf.py_func(_op, [preds_ixs_t], tf.float64)

  def _write_tensor(self, vocab_path, output_path, session, embedding):
    """Writes tensor to output_path as tsv."""
    embeddings = session.run(embedding)

    with open(output_path, 'w') as out_f:
      with open(vocab_path) as vocab_f:
        for index, word in enumerate(vocab_f):
          word = word.strip()
          embedding = embeddings[index]
          print('\t'.join([word.strip()] + [str(x) for x in embedding]),
                file=out_f)

  def write_embeddings(self, config, session):
    """Writes row and column embeddings disk."""
    self._write_tensor(
        os.path.join(config.input_base_path, 'row_vocab.txt'),
        os.path.join(config.output_base_path, 'row_embedding.tsv'),
        session, self.row_embedding)

    self._write_tensor(
        os.path.join(config.input_base_path, 'col_vocab.txt'),
        os.path.join(config.output_base_path, 'col_embedding.tsv'),
        session, self.col_embedding)


def main(_):
  tf.logging.set_verbosity(tf.logging.INFO)

  # If we have ps_hosts, then we'll assume that this is going to be a
  # distributed training run.  Configure the cluster appropriately.  Otherwise,
  # we just do everything in-process.
  if FLAGS.ps_hosts:
    cluster = tf.train.ClusterSpec({
        'ps': FLAGS.ps_hosts.split(','),
        'worker': FLAGS.worker_hosts.split(','),
    })

    if FLAGS.job_name == 'ps':
      # Ignore the GPU if we're the parameter server. This let's the PS run on
      # the same machine as a worker.
      config = tf.ConfigProto(device_count={'GPU': 0})
    elif FLAGS.job_name == 'worker':
      config = tf.ConfigProto(gpu_options=tf.GPUOptions(
          visible_device_list='%d' % FLAGS.gpu_device,
          allow_growth=True))
    else:
      raise ValueError('unknown job name "%s"' % FLAGS.job_name)

    server = tf.train.Server(
        cluster,
        job_name=FLAGS.job_name,
        task_index=FLAGS.task_index,
        config=config)

    if FLAGS.job_name == 'ps':
      return server.join()

    device_setter = tf.train.replica_device_setter(
        worker_device='/job:worker/task:%d' % FLAGS.task_index,
        cluster=cluster)

  else:
    server = None
    device_setter = tf.train.replica_device_setter(0)

  # Build the graph.
  with tf.Graph().as_default():
    with tf.device(device_setter):
      model = Model(FLAGS.input_base_path, FLAGS)

      # If an eval path is present, then create eval operators and set up scalar
      # summaries to report on the results.  Run the evals on the CPU since
      # the analogy eval requires a fairly enormous tensor to be allocated to
      # do the nearest neighbor search.
      if FLAGS.eval_base_path:
        wordsim_filenames = glob.glob(
            os.path.join(FLAGS.eval_base_path, '*.ws.tab'))

        for filename in wordsim_filenames:
          name = os.path.basename(filename).split('.')[0]
          with tf.device(tf.DeviceSpec(device_type='CPU')):
            op = model.wordsim_eval_op(filename)
            tf.summary.scalar(name, op)

        analogy_filenames = glob.glob(
            os.path.join(FLAGS.eval_base_path, '*.an.tab'))

        for filename in analogy_filenames:
          name = os.path.basename(filename).split('.')[0]
          with tf.device(tf.DeviceSpec(device_type='CPU')):
            op = model.analogy_eval_op(filename)
            tf.summary.scalar(name, op)

      tf.summary.scalar('loss', model.loss_op)

    # Train on, soldier.
    supervisor = tf.train.Supervisor(
        logdir=FLAGS.output_base_path,
        is_chief=(FLAGS.task_index == 0),
        save_summaries_secs=60,
        recovery_wait_secs=5)

    max_step = FLAGS.num_epochs * model.steps_per_epoch
    master = server.target if server else ''
    with supervisor.managed_session(master) as session:
      local_step = 0
      global_step = session.run(model.global_step)
      while not supervisor.should_stop() and global_step < max_step:
        global_step, loss, _ = session.run([
            model.global_step, model.loss_op, model.train_op])

        if not np.isfinite(loss):
          raise ValueError('non-finite cost at step %d' % global_step)

        local_step += 1
        if local_step % 10 == 0:
          tf.logging.info(
              'local_step=%d global_step=%d loss=%.1f, %.1f%% complete',
              local_step, global_step, loss, 100.0 * global_step / max_step)

      if FLAGS.task_index == 0:
        supervisor.saver.save(
            session, supervisor.save_path, global_step=global_step)

        model.write_embeddings(FLAGS, session)


if __name__ == '__main__':
  tf.app.run()
