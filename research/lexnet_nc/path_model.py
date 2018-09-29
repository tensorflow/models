# Copyright 2017, 2018 Google, Inc. All Rights Reserved.
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

"""LexNET Path-based Model."""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import collections
import itertools
import os

import lexnet_common
import numpy as np
import tensorflow as tf


class PathBasedModel(object):
  """The LexNET path-based model for classifying semantic relations."""

  @classmethod
  def default_hparams(cls):
    """Returns the default hyper-parameters."""
    return tf.contrib.training.HParams(
        max_path_len=8,
        num_classes=37,
        num_epochs=30,
        input_keep_prob=0.9,
        learning_rate=0.001,
        learn_lemmas=False,
        random_seed=133,  # zero means no random seed
        lemma_embeddings_file='glove/glove.6B.50d.bin',
        num_pos=len(lexnet_common.POSTAGS),
        num_dep=len(lexnet_common.DEPLABELS),
        num_directions=len(lexnet_common.DIRS),
        lemma_dim=50,
        pos_dim=4,
        dep_dim=5,
        dir_dim=1)

  def __init__(self, hparams, lemma_embeddings, instance):
    """Initialize the LexNET classifier.

    Args:
      hparams: the hyper-parameters.
      lemma_embeddings: word embeddings for the path-based component.
      instance: string tensor containing the input instance
    """
    self.hparams = hparams
    self.lemma_embeddings = lemma_embeddings
    self.instance = instance
    self.vocab_size, self.lemma_dim = self.lemma_embeddings.shape

    # Set the random seed
    if hparams.random_seed > 0:
      tf.set_random_seed(hparams.random_seed)

    # Create the network
    self.__create_computation_graph__()

  def __create_computation_graph__(self):
    """Initialize the model and define the graph."""
    self.lstm_input_dim = sum([self.hparams.lemma_dim, self.hparams.pos_dim,
                               self.hparams.dep_dim, self.hparams.dir_dim])
    self.lstm_output_dim = self.lstm_input_dim

    network_input = self.lstm_output_dim
    self.lemma_lookup = tf.get_variable(
        'lemma_lookup',
        initializer=self.lemma_embeddings,
        dtype=tf.float32,
        trainable=self.hparams.learn_lemmas)
    self.pos_lookup = tf.get_variable(
        'pos_lookup',
        shape=[self.hparams.num_pos, self.hparams.pos_dim],
        dtype=tf.float32)
    self.dep_lookup = tf.get_variable(
        'dep_lookup',
        shape=[self.hparams.num_dep, self.hparams.dep_dim],
        dtype=tf.float32)
    self.dir_lookup = tf.get_variable(
        'dir_lookup',
        shape=[self.hparams.num_directions, self.hparams.dir_dim],
        dtype=tf.float32)

    self.weights1 = tf.get_variable(
        'W1',
        shape=[network_input, self.hparams.num_classes],
        dtype=tf.float32)
    self.bias1 = tf.get_variable(
        'b1',
        shape=[self.hparams.num_classes],
        dtype=tf.float32)

    # Define the variables
    (self.batch_paths,
     self.path_counts,
     self.seq_lengths,
     self.path_strings,
     self.batch_labels) = _parse_tensorflow_example(
         self.instance, self.hparams.max_path_len, self.hparams.input_keep_prob)

    # Create the LSTM
    self.__lstm__()

    # Create the MLP
    self.__mlp__()

    self.instances_to_load = tf.placeholder(dtype=tf.string, shape=[None])
    self.labels_to_load = lexnet_common.load_all_labels(self.instances_to_load)

  def load_labels(self, session, batch_instances):
    """Loads the labels of the current instances.

    Args:
      session: the current TensorFlow session.
      batch_instances: the dataset instances.

    Returns:
      the labels.
    """
    return session.run(self.labels_to_load,
                       feed_dict={self.instances_to_load: batch_instances})

  def run_one_epoch(self, session, num_steps):
    """Train the model.

    Args:
      session: The current TensorFlow session.
      num_steps: The number of steps in each epoch.

    Returns:
      The mean loss for the epoch.

    Raises:
      ArithmeticError: if the loss becomes non-finite.
    """
    losses = []

    for step in range(num_steps):
      curr_loss, _ = session.run([self.cost, self.train_op])
      if not np.isfinite(curr_loss):
        raise ArithmeticError('nan loss at step %d' % step)

      losses.append(curr_loss)

    return np.mean(losses)

  def predict(self, session, inputs):
    """Predict the classification of the test set.

    Args:
      session: The current TensorFlow session.
      inputs: the train paths, x, y and/or nc vectors

    Returns:
      The test predictions.
    """
    predictions, _ = zip(*self.predict_with_score(session, inputs))
    return np.array(predictions)

  def predict_with_score(self, session, inputs):
    """Predict the classification of the test set.

    Args:
      session: The current TensorFlow session.
      inputs: the test paths, x, y and/or nc vectors

    Returns:
      The test predictions along with their scores.
    """
    test_pred = [0] * len(inputs)

    for index, instance in enumerate(inputs):

      prediction, scores = session.run(
          [self.predictions, self.scores],
          feed_dict={self.instance: instance})

      test_pred[index] = (prediction, scores[prediction])

    return test_pred

  def __mlp__(self):
    """Performs the MLP operations.

    Returns: the prediction object to be computed in a Session
    """
    # Feed the paths to the MLP: path_embeddings is
    # [num_batch_paths, output_dim], and when we multiply it by W
    # ([output_dim, num_classes]), we get a matrix of class distributions:
    # [num_batch_paths, num_classes].
    self.distributions = tf.matmul(self.path_embeddings, self.weights1)

    # Now, compute weighted average on the class distributions, using the path
    # frequency as weights.

    # First, reshape path_freq to the same shape of distributions
    self.path_freq = tf.tile(tf.expand_dims(self.path_counts, -1),
                             [1, self.hparams.num_classes])

    # Second, multiply the distributions and frequencies element-wise.
    self.weighted = tf.multiply(self.path_freq, self.distributions)

    # Finally, take the average to get a tensor of shape [1, num_classes].
    self.weighted_sum = tf.reduce_sum(self.weighted, 0)
    self.num_paths = tf.clip_by_value(tf.reduce_sum(self.path_counts),
                                      1, np.inf)
    self.num_paths = tf.tile(tf.expand_dims(self.num_paths, -1),
                             [self.hparams.num_classes])
    self.scores = tf.div(self.weighted_sum, self.num_paths)
    self.predictions = tf.argmax(self.scores)

    # Define the loss function and the optimization algorithm
    self.cross_entropies = tf.nn.sparse_softmax_cross_entropy_with_logits(
        logits=self.scores, labels=tf.reduce_mean(self.batch_labels))
    self.cost = tf.reduce_sum(self.cross_entropies, name='cost')
    self.global_step = tf.Variable(0, name='global_step', trainable=False)
    self.optimizer = tf.train.AdamOptimizer()
    self.train_op = self.optimizer.minimize(self.cost,
                                            global_step=self.global_step)

  def __lstm__(self):
    """Defines the LSTM operations.

    Returns:
      A matrix of path embeddings.
    """
    lookup_tables = [self.lemma_lookup, self.pos_lookup,
                     self.dep_lookup, self.dir_lookup]

    # Split the edges to components: list of 4 tensors
    # [num_batch_paths, max_path_len, 1]
    self.edge_components = tf.split(self.batch_paths, 4, axis=2)

    # Look up the components embeddings and concatenate them back together
    self.path_matrix = tf.concat([
        tf.squeeze(tf.nn.embedding_lookup(lookup_table, component), 2)
        for lookup_table, component in
        zip(lookup_tables, self.edge_components)
    ], axis=2)

    self.sequence_lengths = tf.reshape(self.seq_lengths, [-1])

    # Define the LSTM.
    # The input is [num_batch_paths, max_path_len, input_dim].
    lstm_cell = tf.contrib.rnn.BasicLSTMCell(self.lstm_output_dim)

    # The output is [num_batch_paths, max_path_len, output_dim].
    self.lstm_outputs, _ = tf.nn.dynamic_rnn(
        lstm_cell, self.path_matrix, dtype=tf.float32,
        sequence_length=self.sequence_lengths)

    # Slice the last *relevant* output for each instance ->
    # [num_batch_paths, output_dim]
    self.path_embeddings = _extract_last_relevant(self.lstm_outputs,
                                                  self.sequence_lengths)


def _parse_tensorflow_example(record, max_path_len, input_keep_prob):
  """Reads TensorFlow examples from a RecordReader.

  Args:
    record: a record with TensorFlow example.
    max_path_len: the maximum path length.
    input_keep_prob: 1 - the word dropout probability

  Returns:
    The paths and counts
  """
  features = tf.parse_single_example(record, {
      'lemmas':
          tf.FixedLenSequenceFeature(
              shape=(), dtype=tf.int64, allow_missing=True),
      'postags':
          tf.FixedLenSequenceFeature(
              shape=(), dtype=tf.int64, allow_missing=True),
      'deplabels':
          tf.FixedLenSequenceFeature(
              shape=(), dtype=tf.int64, allow_missing=True),
      'dirs':
          tf.FixedLenSequenceFeature(
              shape=(), dtype=tf.int64, allow_missing=True),
      'counts':
          tf.FixedLenSequenceFeature(
              shape=(), dtype=tf.int64, allow_missing=True),
      'pathlens':
          tf.FixedLenSequenceFeature(
              shape=(), dtype=tf.int64, allow_missing=True),
      'reprs':
          tf.FixedLenSequenceFeature(
              shape=(), dtype=tf.string, allow_missing=True),
      'rel_id':
          tf.FixedLenFeature([], dtype=tf.int64)
  })

  path_counts = tf.to_float(features['counts'])
  seq_lengths = features['pathlens']

  # Concatenate the edge components to create a path tensor:
  # [max_paths_per_ins, max_path_length, 4]
  lemmas = _word_dropout(
      tf.reshape(features['lemmas'], [-1, max_path_len]), input_keep_prob)

  paths = tf.stack(
      [lemmas] + [
          tf.reshape(features[f], [-1, max_path_len])
          for f in ('postags', 'deplabels', 'dirs')
      ],
      axis=-1)

  path_strings = features['reprs']

  # Add an empty path to pairs with no paths
  paths = tf.cond(
      tf.shape(paths)[0] > 0,
      lambda: paths,
      lambda: tf.zeros([1, max_path_len, 4], dtype=tf.int64))

  # Paths are left-padded. We reverse them to make them right-padded.
  #paths = tf.reverse(paths, axis=[1])

  path_counts = tf.cond(
      tf.shape(path_counts)[0] > 0,
      lambda: path_counts,
      lambda: tf.constant([1.0], dtype=tf.float32))

  seq_lengths = tf.cond(
      tf.shape(seq_lengths)[0] > 0,
      lambda: seq_lengths,
      lambda: tf.constant([1], dtype=tf.int64))

  # Duplicate the label for each path
  labels = tf.ones_like(path_counts, dtype=tf.int64) * features['rel_id']

  return paths, path_counts, seq_lengths, path_strings, labels


def _extract_last_relevant(output, seq_lengths):
  """Get the last relevant LSTM output cell for each batch instance.

  Args:
    output: the LSTM outputs - a tensor with shape
    [num_paths, output_dim, max_path_len]
    seq_lengths: the sequences length per instance

  Returns:
    The last relevant LSTM output cell for each batch instance.
  """
  max_length = int(output.get_shape()[1])
  path_lengths = tf.clip_by_value(seq_lengths - 1, 0, max_length)
  relevant = tf.reduce_sum(tf.multiply(output, tf.expand_dims(
      tf.one_hot(path_lengths, max_length), -1)), 1)
  return relevant


def _word_dropout(words, input_keep_prob):
  """Drops words with probability 1 - input_keep_prob.

  Args:
    words: a list of lemmas from the paths.
    input_keep_prob: the probability to keep the word.

  Returns:
    The revised list where some of the words are <UNK>ed.
  """
  # Create the mask: (-1) to drop, 1 to keep
  prob = tf.random_uniform(tf.shape(words), 0, 1)
  condition = tf.less(prob, (1 - input_keep_prob))
  mask = tf.where(condition,
                  tf.negative(tf.ones_like(words)), tf.ones_like(words))

  # We need to keep zeros (<PAD>), and change other numbers to 1 (<UNK>)
  # if their mask is -1. First, we multiply the mask and the words.
  # Zeros will stay zeros, and words to drop will become negative.
  # Then, we change negative values to 1.
  masked_words = tf.multiply(mask, words)
  condition = tf.less(masked_words, 0)
  dropped_words = tf.where(condition, tf.ones_like(words), words)
  return dropped_words


def compute_path_embeddings(model, session, instances):
  """Compute the path embeddings for all the distinct paths.

  Args:
    model: The trained path-based model.
    session: The current TensorFlow session.
    instances: All the train, test and validation instances.

  Returns:
    The path to ID index and the path embeddings.
  """
  # Get an index for each distinct path
  path_index = collections.defaultdict(itertools.count(0).next)
  path_vectors = {}

  for instance in instances:
    curr_path_embeddings, curr_path_strings = session.run(
        [model.path_embeddings, model.path_strings],
        feed_dict={model.instance: instance})

    for i, path in enumerate(curr_path_strings):
      if not path:
        continue

      # Set a new/existing index for the path
      index = path_index[path]

      # Save its vector
      path_vectors[index] = curr_path_embeddings[i, :]

  print('Number of distinct paths: %d' % len(path_index))
  return path_index, path_vectors


def save_path_embeddings(model, path_vectors, path_index, embeddings_base_path):
  """Saves the path embeddings.

  Args:
    model: The trained path-based model.
    path_vectors: The path embeddings.
    path_index: A map from path to ID.
    embeddings_base_path: The base directory where the embeddings are.
  """
  index_range = range(max(path_index.values()) + 1)
  path_matrix = [path_vectors[i] for i in index_range]
  path_matrix = np.vstack(path_matrix)

  # Save the path embeddings
  path_vector_filename = os.path.join(
      embeddings_base_path, '%d_path_vectors' % model.lstm_output_dim)
  with open(path_vector_filename, 'w') as f_out:
    np.save(f_out, path_matrix)

  index_to_path = {i: p for p, i in path_index.iteritems()}
  path_vocab = [index_to_path[i] for i in index_range]

  # Save the path vocabulary
  path_vocab_filename = os.path.join(
      embeddings_base_path, '%d_path_vocab' % model.lstm_output_dim)
  with open(path_vocab_filename, 'w') as f_out:
    f_out.write('\n'.join(path_vocab))
    f_out.write('\n')

  print('Saved path embeddings.')


def load_path_embeddings(path_embeddings_dir, path_dim):
  """Loads pretrained path embeddings from a binary file and returns the matrix.

  Args:
    path_embeddings_dir: The directory for the path embeddings.
    path_dim: The dimension of the path embeddings, used as prefix to the
    path_vocab and path_vectors files.

  Returns:
    The path embeddings matrix and the path_to_index dictionary.
  """
  prefix = path_embeddings_dir + '/%d' % path_dim + '_'
  with open(prefix + 'path_vocab') as f_in:
    vocab = f_in.read().splitlines()

  vocab_size = len(vocab)
  embedding_file = prefix + 'path_vectors'

  print('Embedding file "%s" has %d paths' % (embedding_file, vocab_size))

  with open(embedding_file) as f_in:
    embeddings = np.load(f_in)

  path_to_index = {p: i for i, p in enumerate(vocab)}
  return embeddings, path_to_index


def get_indicative_paths(model, session, path_index, path_vectors, classes,
                         save_dir, k=20, threshold=0.8):
  """Gets the most indicative paths for each class.

  Args:
    model: The trained path-based model.
    session: The current TensorFlow session.
    path_index: A map from path to ID.
    path_vectors: The path embeddings.
    classes: The class label names.
    save_dir: Where to save the paths.
    k: The k for top-k paths.
    threshold: The threshold above which to consider paths as indicative.
  """
  # Define graph variables for this operation
  p_path_embedding = tf.placeholder(dtype=tf.float32,
                                    shape=[1, model.lstm_output_dim])
  p_distributions = tf.nn.softmax(tf.matmul(p_path_embedding, model.weights1))

  # Treat each path as a pair instance with a single path, and get the
  # relation distribution for it. Then, take the top paths for each relation.

  # This dictionary contains a relation as a key, and the value is a list of
  # tuples of path index and score. A relation r will contain (p, s) if the
  # path p is classified to r with a confidence of s.
  prediction_per_relation = collections.defaultdict(list)

  index_to_path = {i: p for p, i in path_index.iteritems()}

  # Predict all the paths
  for index in range(len(path_index)):
    curr_path_vector = path_vectors[index]

    distribution = session.run(p_distributions,
                               feed_dict={
                                   p_path_embedding: np.reshape(
                                       curr_path_vector,
                                       [1, model.lstm_output_dim])})

    distribution = distribution[0, :]
    prediction = np.argmax(distribution)
    prediction_per_relation[prediction].append(
        (index, distribution[prediction]))

    if index % 10000 == 0:
      print('Classified %d/%d (%3.2f%%) of the paths' % (
          index, len(path_index), 100 * index / len(path_index)))

  # Retrieve k-best scoring paths for each relation
  for relation_index, relation in enumerate(classes):
    curr_paths = sorted(prediction_per_relation[relation_index],
                        key=lambda item: item[1], reverse=True)
    above_t = [(p, s) for (p, s) in curr_paths if s >= threshold]
    top_k = curr_paths[k+1]
    relation_paths = above_t if len(above_t) > len(top_k) else top_k

    paths_filename = os.path.join(save_dir, '%s.paths' % relation)
    with open(paths_filename, 'w') as f_out:
      for index, score in relation_paths:
        print('\t'.join([index_to_path[index], str(score)]), file=f_out)
