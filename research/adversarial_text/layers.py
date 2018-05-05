# Copyright 2017 Google Inc. All Rights Reserved.
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
"""Layers for VatxtModel."""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

# Dependency imports

from six.moves import xrange
import tensorflow as tf
K = tf.keras


def cl_logits_subgraph(layer_sizes, input_size, num_classes, keep_prob=1.):
  """Construct multiple ReLU layers with dropout and a linear layer."""
  subgraph = K.models.Sequential(name='cl_logits')
  for i, layer_size in enumerate(layer_sizes):
    if i == 0:
      subgraph.add(
          K.layers.Dense(layer_size, activation='relu', input_dim=input_size))
    else:
      subgraph.add(K.layers.Dense(layer_size, activation='relu'))

    if keep_prob < 1.:
      subgraph.add(K.layers.Dropout(1. - keep_prob))
  subgraph.add(K.layers.Dense(1 if num_classes == 2 else num_classes))
  return subgraph


class Embedding(K.layers.Layer):
  """Embedding layer with frequency-based normalization and dropout."""

  def __init__(self,
               vocab_size,
               embedding_dim,
               normalize=False,
               vocab_freqs=None,
               keep_prob=1.,
               **kwargs):
    self.vocab_size = vocab_size
    self.embedding_dim = embedding_dim
    self.normalized = normalize
    self.keep_prob = keep_prob

    if normalize:
      assert vocab_freqs is not None
      self.vocab_freqs = tf.constant(
          vocab_freqs, dtype=tf.float32, shape=(vocab_size, 1))

    super(Embedding, self).__init__(**kwargs)

  def build(self, input_shape):
    with tf.device('/cpu:0'):
      self.var = self.add_weight(
          shape=(self.vocab_size, self.embedding_dim),
          initializer=tf.random_uniform_initializer(-1., 1.),
          name='embedding')

    if self.normalized:
      self.var = self._normalize(self.var)

    super(Embedding, self).build(input_shape)

  def call(self, x):
    embedded = tf.nn.embedding_lookup(self.var, x)
    if self.keep_prob < 1.:
      shape = embedded.get_shape().as_list()

      # Use same dropout masks at each timestep with specifying noise_shape.
      # This slightly improves performance.
      # Please see https://arxiv.org/abs/1512.05287 for the theoretical
      # explanation.
      embedded = tf.nn.dropout(
          embedded, self.keep_prob, noise_shape=(shape[0], 1, shape[2]))
    return embedded

  def _normalize(self, emb):
    weights = self.vocab_freqs / tf.reduce_sum(self.vocab_freqs)
    mean = tf.reduce_sum(weights * emb, 0, keep_dims=True)
    var = tf.reduce_sum(weights * tf.pow(emb - mean, 2.), 0, keep_dims=True)
    stddev = tf.sqrt(1e-6 + var)
    return (emb - mean) / stddev


class LSTM(object):
  """LSTM layer using dynamic_rnn.

  Exposes variables in `trainable_weights` property.
  """

  def __init__(self, cell_size, num_layers=1, keep_prob=1., name='LSTM'):
    self.cell_size = cell_size
    self.num_layers = num_layers
    self.keep_prob = keep_prob
    self.reuse = None
    self.trainable_weights = None
    self.name = name

  def __call__(self, x, initial_state, seq_length):
    with tf.variable_scope(self.name, reuse=self.reuse) as vs:
      cell = tf.contrib.rnn.MultiRNNCell([
          tf.contrib.rnn.BasicLSTMCell(
              self.cell_size,
              forget_bias=0.0,
              reuse=tf.get_variable_scope().reuse)
          for _ in xrange(self.num_layers)
      ])

      # shape(x) = (batch_size, num_timesteps, embedding_dim)

      lstm_out, next_state = tf.nn.dynamic_rnn(
          cell, x, initial_state=initial_state, sequence_length=seq_length)

      # shape(lstm_out) = (batch_size, timesteps, cell_size)

      if self.keep_prob < 1.:
        lstm_out = tf.nn.dropout(lstm_out, self.keep_prob)

      if self.reuse is None:
        self.trainable_weights = vs.global_variables()

    self.reuse = True

    return lstm_out, next_state


class SoftmaxLoss(K.layers.Layer):
  """Softmax xentropy loss with candidate sampling."""

  def __init__(self,
               vocab_size,
               num_candidate_samples=-1,
               vocab_freqs=None,
               **kwargs):
    self.vocab_size = vocab_size
    self.num_candidate_samples = num_candidate_samples
    self.vocab_freqs = vocab_freqs
    super(SoftmaxLoss, self).__init__(**kwargs)
    self.multiclass_dense_layer = K.layers.Dense(self.vocab_size)

  def build(self, input_shape):
    input_shape = input_shape[0]
    with tf.device('/cpu:0'):
      self.lin_w = self.add_weight(
          shape=(input_shape[-1], self.vocab_size),
          name='lm_lin_w',
          initializer=K.initializers.glorot_uniform())
      self.lin_b = self.add_weight(
          shape=(self.vocab_size,),
          name='lm_lin_b',
          initializer=K.initializers.glorot_uniform())
      self.multiclass_dense_layer.build(input_shape)

    super(SoftmaxLoss, self).build(input_shape)

  def call(self, inputs):
    x, labels, weights = inputs
    if self.num_candidate_samples > -1:
      assert self.vocab_freqs is not None
      labels_reshaped = tf.reshape(labels, [-1])
      labels_reshaped = tf.expand_dims(labels_reshaped, -1)
      sampled = tf.nn.fixed_unigram_candidate_sampler(
          true_classes=labels_reshaped,
          num_true=1,
          num_sampled=self.num_candidate_samples,
          unique=True,
          range_max=self.vocab_size,
          unigrams=self.vocab_freqs)
      inputs_reshaped = tf.reshape(x, [-1, int(x.get_shape()[2])])

      lm_loss = tf.nn.sampled_softmax_loss(
          weights=tf.transpose(self.lin_w),
          biases=self.lin_b,
          labels=labels_reshaped,
          inputs=inputs_reshaped,
          num_sampled=self.num_candidate_samples,
          num_classes=self.vocab_size,
          sampled_values=sampled)
      lm_loss = tf.reshape(
          lm_loss,
          [int(x.get_shape()[0]), int(x.get_shape()[1])])
    else:
      logits = self.multiclass_dense_layer(x)
      lm_loss = tf.nn.sparse_softmax_cross_entropy_with_logits(
          logits=logits, labels=labels)

    lm_loss = tf.identity(
        tf.reduce_sum(lm_loss * weights) / _num_labels(weights),
        name='lm_xentropy_loss')
    return lm_loss


def classification_loss(logits, labels, weights):
  """Computes cross entropy loss between logits and labels.

  Args:
    logits: 2-D [timesteps*batch_size, m] float tensor, where m=1 if
      num_classes=2, otherwise m=num_classes.
    labels: 1-D [timesteps*batch_size] integer tensor.
    weights: 1-D [timesteps*batch_size] float tensor.

  Returns:
    Loss scalar of type float.
  """
  inner_dim = logits.get_shape().as_list()[-1]
  with tf.name_scope('classifier_loss'):
    # Logistic loss
    if inner_dim == 1:
      loss = tf.nn.sigmoid_cross_entropy_with_logits(
          logits=tf.squeeze(logits, -1), labels=tf.cast(labels, tf.float32))
    # Softmax loss
    else:
      loss = tf.nn.sparse_softmax_cross_entropy_with_logits(
          logits=logits, labels=labels)

    num_lab = _num_labels(weights)
    tf.summary.scalar('num_labels', num_lab)
    return tf.identity(
        tf.reduce_sum(weights * loss) / num_lab, name='classification_xentropy')


def accuracy(logits, targets, weights):
  """Computes prediction accuracy.

  Args:
    logits: 2-D classifier logits [timesteps*batch_size, num_classes]
    targets: 1-D [timesteps*batch_size] integer tensor.
    weights: 1-D [timesteps*batch_size] float tensor.

  Returns:
    Accuracy: float scalar.
  """
  with tf.name_scope('accuracy'):
    eq = tf.cast(tf.equal(predictions(logits), targets), tf.float32)
    return tf.identity(
        tf.reduce_sum(weights * eq) / _num_labels(weights), name='accuracy')


def predictions(logits):
  """Class prediction from logits."""
  inner_dim = logits.get_shape().as_list()[-1]
  with tf.name_scope('predictions'):
    # For binary classification
    if inner_dim == 1:
      pred = tf.cast(tf.greater(tf.squeeze(logits, -1), 0.), tf.int64)
    # For multi-class classification
    else:
      pred = tf.argmax(logits, 2)
    return pred


def _num_labels(weights):
  """Number of 1's in weights. Returns 1. if 0."""
  num_labels = tf.reduce_sum(weights)
  num_labels = tf.where(tf.equal(num_labels, 0.), 1., num_labels)
  return num_labels


def optimize(loss,
             global_step,
             max_grad_norm,
             lr,
             lr_decay,
             sync_replicas=False,
             replicas_to_aggregate=1,
             task_id=0):
  """Builds optimization graph.

  * Creates an optimizer, and optionally wraps with SyncReplicasOptimizer
  * Computes, clips, and applies gradients
  * Maintains moving averages for all trainable variables
  * Summarizes variables and gradients

  Args:
    loss: scalar loss to minimize.
    global_step: integer scalar Variable.
    max_grad_norm: float scalar. Grads will be clipped to this value.
    lr: float scalar, learning rate.
    lr_decay: float scalar, learning rate decay rate.
    sync_replicas: bool, whether to use SyncReplicasOptimizer.
    replicas_to_aggregate: int, number of replicas to aggregate when using
      SyncReplicasOptimizer.
    task_id: int, id of the current task; used to ensure proper initialization
      of SyncReplicasOptimizer.

  Returns:
    train_op
  """
  with tf.name_scope('optimization'):
    # Compute gradients.
    tvars = tf.trainable_variables()
    grads = tf.gradients(
        loss,
        tvars,
        aggregation_method=tf.AggregationMethod.EXPERIMENTAL_ACCUMULATE_N)

    # Clip non-embedding grads
    non_embedding_grads_and_vars = [(g, v) for (g, v) in zip(grads, tvars)
                                    if 'embedding' not in v.op.name]
    embedding_grads_and_vars = [(g, v) for (g, v) in zip(grads, tvars)
                                if 'embedding' in v.op.name]

    ne_grads, ne_vars = zip(*non_embedding_grads_and_vars)
    ne_grads, _ = tf.clip_by_global_norm(ne_grads, max_grad_norm)
    non_embedding_grads_and_vars = zip(ne_grads, ne_vars)

    grads_and_vars = embedding_grads_and_vars + non_embedding_grads_and_vars

    # Summarize
    _summarize_vars_and_grads(grads_and_vars)

    # Decaying learning rate
    lr = tf.train.exponential_decay(
        lr, global_step, 1, lr_decay, staircase=True)
    tf.summary.scalar('learning_rate', lr)
    opt = tf.train.AdamOptimizer(lr)

    # Track the moving averages of all trainable variables.
    variable_averages = tf.train.ExponentialMovingAverage(0.999, global_step)

    # Apply gradients
    if sync_replicas:
      opt = tf.train.SyncReplicasOptimizer(
          opt,
          replicas_to_aggregate,
          variable_averages=variable_averages,
          variables_to_average=tvars,
          total_num_replicas=replicas_to_aggregate)
      apply_gradient_op = opt.apply_gradients(
          grads_and_vars, global_step=global_step)
      with tf.control_dependencies([apply_gradient_op]):
        train_op = tf.no_op(name='train_op')

      # Initialization ops
      tf.add_to_collection(tf.GraphKeys.QUEUE_RUNNERS,
                           opt.get_chief_queue_runner())
      if task_id == 0:  # Chief task
        local_init_op = opt.chief_init_op
        tf.add_to_collection('chief_init_op', opt.get_init_tokens_op())
      else:
        local_init_op = opt.local_step_init_op
      tf.add_to_collection('local_init_op', local_init_op)
      tf.add_to_collection('ready_for_local_init_op',
                           opt.ready_for_local_init_op)
    else:
      # Non-sync optimizer
      apply_gradient_op = opt.apply_gradients(grads_and_vars, global_step)
      with tf.control_dependencies([apply_gradient_op]):
        train_op = variable_averages.apply(tvars)

    return train_op


def _summarize_vars_and_grads(grads_and_vars):
  tf.logging.info('Trainable variables:')
  tf.logging.info('-' * 60)
  for grad, var in grads_and_vars:
    tf.logging.info(var)

    def tag(name, v=var):
      return v.op.name + '_' + name

    # Variable summary
    mean = tf.reduce_mean(var)
    tf.summary.scalar(tag('mean'), mean)
    with tf.name_scope(tag('stddev')):
      stddev = tf.sqrt(tf.reduce_mean(tf.square(var - mean)))
    tf.summary.scalar(tag('stddev'), stddev)
    tf.summary.scalar(tag('max'), tf.reduce_max(var))
    tf.summary.scalar(tag('min'), tf.reduce_min(var))
    tf.summary.histogram(tag('histogram'), var)

    # Gradient summary
    if grad is not None:
      if isinstance(grad, tf.IndexedSlices):
        grad_values = grad.values
      else:
        grad_values = grad

      tf.summary.histogram(tag('gradient'), grad_values)
      tf.summary.scalar(tag('gradient_norm'), tf.global_norm([grad_values]))
    else:
      tf.logging.info('Var %s has no gradient', var.op.name)
