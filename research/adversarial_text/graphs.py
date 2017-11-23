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
"""Virtual adversarial text models."""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import csv
import os

# Dependency imports

import tensorflow as tf

import adversarial_losses as adv_lib
import inputs as inputs_lib
import layers as layers_lib

flags = tf.app.flags
FLAGS = flags.FLAGS

# Flags governing adversarial training are defined in adversarial_losses.py.

# Classifier
flags.DEFINE_integer('num_classes', 2, 'Number of classes for classification')

# Data path
flags.DEFINE_string('data_dir', '/tmp/IMDB',
                    'Directory path to preprocessed text dataset.')
flags.DEFINE_string('vocab_freq_path', None,
                    'Path to pre-calculated vocab frequency data. If '
                    'None, use FLAGS.data_dir/vocab_freq.txt.')
flags.DEFINE_integer('batch_size', 64, 'Size of the batch.')
flags.DEFINE_integer('num_timesteps', 100, 'Number of timesteps for BPTT')

# Model architechture
flags.DEFINE_bool('bidir_lstm', False, 'Whether to build a bidirectional LSTM.')
flags.DEFINE_integer('rnn_num_layers', 1, 'Number of LSTM layers.')
flags.DEFINE_integer('rnn_cell_size', 512,
                     'Number of hidden units in the LSTM.')
flags.DEFINE_integer('cl_num_layers', 1,
                     'Number of hidden layers of classification model.')
flags.DEFINE_integer('cl_hidden_size', 30,
                     'Number of hidden units in classification layer.')
flags.DEFINE_integer('num_candidate_samples', -1,
                     'Num samples used in the sampled output layer.')
flags.DEFINE_bool('use_seq2seq_autoencoder', False,
                  'If True, seq2seq auto-encoder is used to pretrain. '
                  'If False, standard language model is used.')

# Vocabulary and embeddings
flags.DEFINE_integer('embedding_dims', 256, 'Dimensions of embedded vector.')
flags.DEFINE_integer('vocab_size', 86934,
                     'The size of the vocaburary. This value '
                     'should be exactly same as the number of the '
                     'vocabulary used in dataset. Because the last '
                     'indexed vocabulary of the dataset preprocessed by '
                     'my preprocessed code, is always <eos> and here we '
                     'specify the <eos> with the the index.')
flags.DEFINE_bool('normalize_embeddings', True,
                  'Normalize word embeddings by vocab frequency')

# Optimization
flags.DEFINE_float('learning_rate', 0.001, 'Learning rate while fine-tuning.')
flags.DEFINE_float('learning_rate_decay_factor', 1.0,
                   'Learning rate decay factor')
flags.DEFINE_boolean('sync_replicas', False, 'sync_replica or not')
flags.DEFINE_integer('replicas_to_aggregate', 1,
                     'The number of replicas to aggregate')

# Regularization
flags.DEFINE_float('max_grad_norm', 1.0,
                   'Clip the global gradient norm to this value.')
flags.DEFINE_float('keep_prob_emb', 1.0, 'keep probability on embedding layer. '
                   '0.5 is optimal on IMDB with virtual adversarial training.')
flags.DEFINE_float('keep_prob_lstm_out', 1.0,
                   'keep probability on lstm output.')
flags.DEFINE_float('keep_prob_cl_hidden', 1.0,
                   'keep probability on classification hidden layer')


def get_model():
  if FLAGS.bidir_lstm:
    return VatxtBidirModel()
  else:
    return VatxtModel()


class VatxtModel(object):
  """Constructs training and evaluation graphs.

  Main methods: `classifier_training()`, `language_model_training()`,
  and `eval_graph()`.

  Variable reuse is a critical part of the model, both for sharing variables
  between the language model and the classifier, and for reusing variables for
  the adversarial loss calculation. To ensure correct variable reuse, all
  variables are created in Keras-style layers, wherein stateful layers (i.e.
  layers with variables) are represented as callable instances of the Layer
  class. Each time the Layer instance is called, it is using the same variables.

  All Layers are constructed in the __init__ method and reused in the various
  graph-building functions.
  """

  def __init__(self, cl_logits_input_dim=None):
    self.global_step = tf.train.get_or_create_global_step()
    self.vocab_freqs = _get_vocab_freqs()

    # Cache VatxtInput objects
    self.cl_inputs = None
    self.lm_inputs = None

    # Cache intermediate Tensors that are reused
    self.tensors = {}

    # Construct layers which are reused in constructing the LM and
    # Classification graphs. Instantiating them all once here ensures that
    # variable reuse works correctly.
    self.layers = {}
    self.layers['embedding'] = layers_lib.Embedding(
        FLAGS.vocab_size, FLAGS.embedding_dims, FLAGS.normalize_embeddings,
        self.vocab_freqs, FLAGS.keep_prob_emb)
    self.layers['lstm'] = layers_lib.LSTM(
        FLAGS.rnn_cell_size, FLAGS.rnn_num_layers, FLAGS.keep_prob_lstm_out)
    self.layers['lm_loss'] = layers_lib.SoftmaxLoss(
        FLAGS.vocab_size,
        FLAGS.num_candidate_samples,
        self.vocab_freqs,
        name='LM_loss')

    cl_logits_input_dim = cl_logits_input_dim or FLAGS.rnn_cell_size
    self.layers['cl_logits'] = layers_lib.cl_logits_subgraph(
        [FLAGS.cl_hidden_size] * FLAGS.cl_num_layers, cl_logits_input_dim,
        FLAGS.num_classes, FLAGS.keep_prob_cl_hidden)

  @property
  def pretrained_variables(self):
    return (self.layers['embedding'].trainable_weights +
            self.layers['lstm'].trainable_weights)

  def classifier_training(self):
    loss = self.classifier_graph()
    train_op = optimize(loss, self.global_step)
    return train_op, loss, self.global_step

  def language_model_training(self):
    loss = self.language_model_graph()
    train_op = optimize(loss, self.global_step)
    return train_op, loss, self.global_step

  def classifier_graph(self):
    """Constructs classifier graph from inputs to classifier loss.

    * Caches the VatxtInput object in `self.cl_inputs`
    * Caches tensors: `cl_embedded`, `cl_logits`, `cl_loss`

    Returns:
      loss: scalar float.
    """
    inputs = _inputs('train', pretrain=False)
    self.cl_inputs = inputs
    embedded = self.layers['embedding'](inputs.tokens)
    self.tensors['cl_embedded'] = embedded

    _, next_state, logits, loss = self.cl_loss_from_embedding(
        embedded, return_intermediates=True)
    tf.summary.scalar('classification_loss', loss)
    self.tensors['cl_logits'] = logits
    self.tensors['cl_loss'] = loss

    acc = layers_lib.accuracy(logits, inputs.labels, inputs.weights)
    tf.summary.scalar('accuracy', acc)

    adv_loss = (self.adversarial_loss() * tf.constant(
        FLAGS.adv_reg_coeff, name='adv_reg_coeff'))
    tf.summary.scalar('adversarial_loss', adv_loss)

    total_loss = loss + adv_loss
    tf.summary.scalar('total_classification_loss', total_loss)

    with tf.control_dependencies([inputs.save_state(next_state)]):
      total_loss = tf.identity(total_loss)

    return total_loss

  def language_model_graph(self, compute_loss=True):
    """Constructs LM graph from inputs to LM loss.

    * Caches the VatxtInput object in `self.lm_inputs`
    * Caches tensors: `lm_embedded`

    Args:
      compute_loss: bool, whether to compute and return the loss or stop after
        the LSTM computation.

    Returns:
      loss: scalar float.
    """
    inputs = _inputs('train', pretrain=True)
    self.lm_inputs = inputs
    return self._lm_loss(inputs, compute_loss=compute_loss)

  def _lm_loss(self,
               inputs,
               emb_key='lm_embedded',
               lstm_layer='lstm',
               lm_loss_layer='lm_loss',
               loss_name='lm_loss',
               compute_loss=True):
    embedded = self.layers['embedding'](inputs.tokens)
    self.tensors[emb_key] = embedded
    lstm_out, next_state = self.layers[lstm_layer](embedded, inputs.state,
                                                   inputs.length)
    if compute_loss:
      loss = self.layers[lm_loss_layer](
          [lstm_out, inputs.labels, inputs.weights])
      with tf.control_dependencies([inputs.save_state(next_state)]):
        loss = tf.identity(loss)
        tf.summary.scalar(loss_name, loss)

      return loss

  def eval_graph(self, dataset='test'):
    """Constructs classifier evaluation graph.

    Args:
      dataset: the labeled dataset to evaluate, {'train', 'test', 'valid'}.

    Returns:
      eval_ops: dict<metric name, tuple(value, update_op)>
      var_restore_dict: dict mapping variable restoration names to variables.
        Trainable variables will be mapped to their moving average names.
    """
    inputs = _inputs(dataset, pretrain=False)
    embedded = self.layers['embedding'](inputs.tokens)
    _, next_state, logits, _ = self.cl_loss_from_embedding(
        embedded, inputs=inputs, return_intermediates=True)

    eval_ops = {
        'accuracy':
            tf.contrib.metrics.streaming_accuracy(
                layers_lib.predictions(logits), inputs.labels, inputs.weights)
    }

    with tf.control_dependencies([inputs.save_state(next_state)]):
      acc, acc_update = eval_ops['accuracy']
      acc_update = tf.identity(acc_update)
      eval_ops['accuracy'] = (acc, acc_update)

    var_restore_dict = make_restore_average_vars_dict()
    return eval_ops, var_restore_dict

  def cl_loss_from_embedding(self,
                             embedded,
                             inputs=None,
                             return_intermediates=False):
    """Compute classification loss from embedding.

    Args:
      embedded: 3-D float Tensor [batch_size, num_timesteps, embedding_dim]
      inputs: VatxtInput, defaults to self.cl_inputs.
      return_intermediates: bool, whether to return intermediate tensors or only
        the final loss.

    Returns:
      If return_intermediates is True:
        lstm_out, next_state, logits, loss
      Else:
        loss
    """
    if inputs is None:
      inputs = self.cl_inputs

    lstm_out, next_state = self.layers['lstm'](embedded, inputs.state,
                                               inputs.length)
    logits = self.layers['cl_logits'](lstm_out)
    loss = layers_lib.classification_loss(logits, inputs.labels, inputs.weights)

    if return_intermediates:
      return lstm_out, next_state, logits, loss
    else:
      return loss

  def adversarial_loss(self):
    """Compute adversarial loss based on FLAGS.adv_training_method."""

    def random_perturbation_loss():
      return adv_lib.random_perturbation_loss(self.tensors['cl_embedded'],
                                              self.cl_inputs.length,
                                              self.cl_loss_from_embedding)

    def adversarial_loss():
      return adv_lib.adversarial_loss(self.tensors['cl_embedded'],
                                      self.tensors['cl_loss'],
                                      self.cl_loss_from_embedding)

    def virtual_adversarial_loss():
      """Computes virtual adversarial loss.

      Uses lm_inputs and constructs the language model graph if it hasn't yet
      been constructed.

      Also ensures that the LM input states are saved for LSTM state-saving
      BPTT.

      Returns:
        loss: float scalar.
      """
      if self.lm_inputs is None:
        self.language_model_graph(compute_loss=False)

      def logits_from_embedding(embedded, return_next_state=False):
        _, next_state, logits, _ = self.cl_loss_from_embedding(
            embedded, inputs=self.lm_inputs, return_intermediates=True)
        if return_next_state:
          return next_state, logits
        else:
          return logits

      next_state, lm_cl_logits = logits_from_embedding(
          self.tensors['lm_embedded'], return_next_state=True)

      va_loss = adv_lib.virtual_adversarial_loss(
          lm_cl_logits, self.tensors['lm_embedded'], self.lm_inputs,
          logits_from_embedding)

      with tf.control_dependencies([self.lm_inputs.save_state(next_state)]):
        va_loss = tf.identity(va_loss)

      return va_loss

    def combo_loss():
      return adversarial_loss() + virtual_adversarial_loss()

    adv_training_methods = {
        # Random perturbation
        'rp': random_perturbation_loss,
        # Adversarial training
        'at': adversarial_loss,
        # Virtual adversarial training
        'vat': virtual_adversarial_loss,
        # Both at and vat
        'atvat': combo_loss,
        '': lambda: tf.constant(0.),
        None: lambda: tf.constant(0.),
    }

    with tf.name_scope('adversarial_loss'):
      return adv_training_methods[FLAGS.adv_training_method]()


class VatxtBidirModel(VatxtModel):
  """Extension of VatxtModel that supports bidirectional input."""

  def __init__(self):
    super(VatxtBidirModel,
          self).__init__(cl_logits_input_dim=FLAGS.rnn_cell_size * 2)

    # Reverse LSTM and LM loss for bidirectional models
    self.layers['lstm_reverse'] = layers_lib.LSTM(
        FLAGS.rnn_cell_size,
        FLAGS.rnn_num_layers,
        FLAGS.keep_prob_lstm_out,
        name='LSTM_Reverse')
    self.layers['lm_loss_reverse'] = layers_lib.SoftmaxLoss(
        FLAGS.vocab_size,
        FLAGS.num_candidate_samples,
        self.vocab_freqs,
        name='LM_loss_reverse')

  @property
  def pretrained_variables(self):
    variables = super(VatxtBidirModel, self).pretrained_variables
    variables.extend(self.layers['lstm_reverse'].trainable_weights)
    return variables

  def classifier_graph(self):
    """Constructs classifier graph from inputs to classifier loss.

    * Caches the VatxtInput objects in `self.cl_inputs`
    * Caches tensors: `cl_embedded` (tuple of forward and reverse), `cl_logits`,
      `cl_loss`

    Returns:
      loss: scalar float.
    """
    inputs = _inputs('train', pretrain=False, bidir=True)
    self.cl_inputs = inputs
    f_inputs, _ = inputs

    # Embed both forward and reverse with a shared embedding
    embedded = [self.layers['embedding'](inp.tokens) for inp in inputs]
    self.tensors['cl_embedded'] = embedded

    _, next_states, logits, loss = self.cl_loss_from_embedding(
        embedded, return_intermediates=True)
    tf.summary.scalar('classification_loss', loss)
    self.tensors['cl_logits'] = logits
    self.tensors['cl_loss'] = loss

    acc = layers_lib.accuracy(logits, f_inputs.labels, f_inputs.weights)
    tf.summary.scalar('accuracy', acc)

    adv_loss = (self.adversarial_loss() * tf.constant(
        FLAGS.adv_reg_coeff, name='adv_reg_coeff'))
    tf.summary.scalar('adversarial_loss', adv_loss)

    total_loss = loss + adv_loss
    tf.summary.scalar('total_classification_loss', total_loss)

    saves = [inp.save_state(state) for (inp, state) in zip(inputs, next_states)]
    with tf.control_dependencies(saves):
      total_loss = tf.identity(total_loss)

    return total_loss

  def language_model_graph(self, compute_loss=True):
    """Constructs forward and reverse LM graphs from inputs to LM losses.

    * Caches the VatxtInput objects in `self.lm_inputs`
    * Caches tensors: `lm_embedded`, `lm_embedded_reverse`

    Args:
      compute_loss: bool, whether to compute and return the loss or stop after
        the LSTM computation.

    Returns:
      loss: scalar float, sum of forward and reverse losses.
    """
    inputs = _inputs('train', pretrain=True, bidir=True)
    self.lm_inputs = inputs
    f_inputs, r_inputs = inputs
    f_loss = self._lm_loss(f_inputs, compute_loss=compute_loss)
    r_loss = self._lm_loss(
        r_inputs,
        emb_key='lm_embedded_reverse',
        lstm_layer='lstm_reverse',
        lm_loss_layer='lm_loss_reverse',
        loss_name='lm_loss_reverse',
        compute_loss=compute_loss)
    if compute_loss:
      return f_loss + r_loss

  def eval_graph(self, dataset='test'):
    """Constructs classifier evaluation graph.

    Args:
      dataset: the labeled dataset to evaluate, {'train', 'test', 'valid'}.

    Returns:
      eval_ops: dict<metric name, tuple(value, update_op)>
      var_restore_dict: dict mapping variable restoration names to variables.
        Trainable variables will be mapped to their moving average names.
    """
    inputs = _inputs(dataset, pretrain=False, bidir=True)
    embedded = [self.layers['embedding'](inp.tokens) for inp in inputs]
    _, next_states, logits, _ = self.cl_loss_from_embedding(
        embedded, inputs=inputs, return_intermediates=True)
    f_inputs, _ = inputs

    eval_ops = {
        'accuracy':
            tf.contrib.metrics.streaming_accuracy(
                layers_lib.predictions(logits), f_inputs.labels,
                f_inputs.weights)
    }

    # Save states on accuracy update
    saves = [inp.save_state(state) for (inp, state) in zip(inputs, next_states)]
    with tf.control_dependencies(saves):
      acc, acc_update = eval_ops['accuracy']
      acc_update = tf.identity(acc_update)
      eval_ops['accuracy'] = (acc, acc_update)

    var_restore_dict = make_restore_average_vars_dict()
    return eval_ops, var_restore_dict

  def cl_loss_from_embedding(self,
                             embedded,
                             inputs=None,
                             return_intermediates=False):
    """Compute classification loss from embedding.

    Args:
      embedded: Length 2 tuple of 3-D float Tensor
        [batch_size, num_timesteps, embedding_dim].
      inputs: Length 2 tuple of VatxtInput, defaults to self.cl_inputs.
      return_intermediates: bool, whether to return intermediate tensors or only
        the final loss.

    Returns:
      If return_intermediates is True:
        lstm_out, next_states, logits, loss
      Else:
        loss
    """
    if inputs is None:
      inputs = self.cl_inputs

    out = []
    for (layer_name, emb, inp) in zip(['lstm', 'lstm_reverse'], embedded,
                                      inputs):
      out.append(self.layers[layer_name](emb, inp.state, inp.length))
    lstm_outs, next_states = zip(*out)

    # Concatenate output of forward and reverse LSTMs
    lstm_out = tf.concat(lstm_outs, 1)

    logits = self.layers['cl_logits'](lstm_out)
    f_inputs, _ = inputs  # pylint: disable=unpacking-non-sequence
    loss = layers_lib.classification_loss(logits, f_inputs.labels,
                                          f_inputs.weights)

    if return_intermediates:
      return lstm_out, next_states, logits, loss
    else:
      return loss

  def adversarial_loss(self):
    """Compute adversarial loss based on FLAGS.adv_training_method."""

    def random_perturbation_loss():
      return adv_lib.random_perturbation_loss_bidir(self.tensors['cl_embedded'],
                                                    self.cl_inputs[0].length,
                                                    self.cl_loss_from_embedding)

    def adversarial_loss():
      return adv_lib.adversarial_loss_bidir(self.tensors['cl_embedded'],
                                            self.tensors['cl_loss'],
                                            self.cl_loss_from_embedding)

    def virtual_adversarial_loss():
      """Computes virtual adversarial loss.

      Uses lm_inputs and constructs the language model graph if it hasn't yet
      been constructed.

      Also ensures that the LM input states are saved for LSTM state-saving
      BPTT.

      Returns:
        loss: float scalar.
      """
      if self.lm_inputs is None:
        self.language_model_graph(compute_loss=False)

      def logits_from_embedding(embedded, return_next_state=False):
        _, next_states, logits, _ = self.cl_loss_from_embedding(
            embedded, inputs=self.lm_inputs, return_intermediates=True)
        if return_next_state:
          return next_states, logits
        else:
          return logits

      lm_embedded = (self.tensors['lm_embedded'],
                     self.tensors['lm_embedded_reverse'])
      next_states, lm_cl_logits = logits_from_embedding(
          lm_embedded, return_next_state=True)

      va_loss = adv_lib.virtual_adversarial_loss_bidir(
          lm_cl_logits, lm_embedded, self.lm_inputs, logits_from_embedding)

      saves = [
          inp.save_state(state)
          for (inp, state) in zip(self.lm_inputs, next_states)
      ]
      with tf.control_dependencies(saves):
        va_loss = tf.identity(va_loss)

      return va_loss

    def combo_loss():
      return adversarial_loss() + virtual_adversarial_loss()

    adv_training_methods = {
        # Random perturbation
        'rp': random_perturbation_loss,
        # Adversarial training
        'at': adversarial_loss,
        # Virtual adversarial training
        'vat': virtual_adversarial_loss,
        # Both at and vat
        'atvat': combo_loss,
        '': lambda: tf.constant(0.),
        None: lambda: tf.constant(0.),
    }

    with tf.name_scope('adversarial_loss'):
      return adv_training_methods[FLAGS.adv_training_method]()


def _inputs(dataset='train', pretrain=False, bidir=False):
  return inputs_lib.inputs(
      data_dir=FLAGS.data_dir,
      phase=dataset,
      bidir=bidir,
      pretrain=pretrain,
      use_seq2seq=pretrain and FLAGS.use_seq2seq_autoencoder,
      state_size=FLAGS.rnn_cell_size,
      num_layers=FLAGS.rnn_num_layers,
      batch_size=FLAGS.batch_size,
      unroll_steps=FLAGS.num_timesteps,
      eos_id=FLAGS.vocab_size - 1)


def _get_vocab_freqs():
  """Returns vocab frequencies.

  Returns:
    List of integers, length=FLAGS.vocab_size.

  Raises:
    ValueError: if the length of the frequency file is not equal to the vocab
      size, or if the file is not found.
  """
  path = FLAGS.vocab_freq_path or os.path.join(FLAGS.data_dir, 'vocab_freq.txt')

  if tf.gfile.Exists(path):
    with tf.gfile.Open(path) as f:
      # Get pre-calculated frequencies of words.
      reader = csv.reader(f, quoting=csv.QUOTE_NONE)
      freqs = [int(row[-1]) for row in reader]
      if len(freqs) != FLAGS.vocab_size:
        raise ValueError('Frequency file length %d != vocab size %d' %
                         (len(freqs), FLAGS.vocab_size))
  else:
    if FLAGS.vocab_freq_path:
      raise ValueError('vocab_freq_path not found')
    freqs = [1] * FLAGS.vocab_size

  return freqs


def make_restore_average_vars_dict():
  """Returns dict mapping moving average names to variables."""
  var_restore_dict = {}
  variable_averages = tf.train.ExponentialMovingAverage(0.999)
  for v in tf.global_variables():
    if v in tf.trainable_variables():
      name = variable_averages.average_name(v)
    else:
      name = v.op.name
    var_restore_dict[name] = v
  return var_restore_dict


def optimize(loss, global_step):
  return layers_lib.optimize(
      loss, global_step, FLAGS.max_grad_norm, FLAGS.learning_rate,
      FLAGS.learning_rate_decay_factor, FLAGS.sync_replicas,
      FLAGS.replicas_to_aggregate, FLAGS.task)
