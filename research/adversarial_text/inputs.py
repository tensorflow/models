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
"""Input utils for virtual adversarial text classification."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os

# Dependency imports

import tensorflow as tf

from adversarial_text.data import data_utils


class VatxtInput(object):
  """Wrapper around NextQueuedSequenceBatch."""

  def __init__(self,
               batch,
               state_name=None,
               tokens=None,
               num_states=0,
               eos_id=None):
    """Construct VatxtInput.

    Args:
      batch: NextQueuedSequenceBatch.
      state_name: str, name of state to fetch and save.
      tokens: int Tensor, tokens. Defaults to batch's F_TOKEN_ID sequence.
      num_states: int The number of states to store.
      eos_id: int Id of end of Sequence.
    """
    self._batch = batch
    self._state_name = state_name
    self._tokens = (tokens if tokens is not None else
                    batch.sequences[data_utils.SequenceWrapper.F_TOKEN_ID])
    self._num_states = num_states

    # Once the tokens have passed through embedding and LSTM, the output Tensor
    # shapes will be time-major, i.e. shape = (time, batch, dim). Here we make
    # both weights and labels time-major with a transpose, and then merge the
    # time and batch dimensions such that they are both vectors of shape
    # (time*batch).
    w = batch.sequences[data_utils.SequenceWrapper.F_WEIGHT]
    w = tf.transpose(w, [1, 0])
    w = tf.reshape(w, [-1])
    self._weights = w

    l = batch.sequences[data_utils.SequenceWrapper.F_LABEL]
    l = tf.transpose(l, [1, 0])
    l = tf.reshape(l, [-1])
    self._labels = l

    # eos weights
    self._eos_weights = None
    if eos_id:
      ew = tf.cast(tf.equal(self._tokens, eos_id), tf.float32)
      ew = tf.transpose(ew, [1, 0])
      ew = tf.reshape(ew, [-1])
      self._eos_weights = ew

  @property
  def tokens(self):
    return self._tokens

  @property
  def weights(self):
    return self._weights

  @property
  def eos_weights(self):
    return self._eos_weights

  @property
  def labels(self):
    return self._labels

  @property
  def length(self):
    return self._batch.length

  @property
  def state_name(self):
    return self._state_name

  @property
  def state(self):
    # LSTM tuple states
    state_names = _get_tuple_state_names(self._num_states, self._state_name)
    return tuple([
        tf.contrib.rnn.LSTMStateTuple(
            self._batch.state(c_name), self._batch.state(h_name))
        for c_name, h_name in state_names
    ])

  def save_state(self, value):
    # LSTM tuple states
    state_names = _get_tuple_state_names(self._num_states, self._state_name)
    save_ops = []
    for (c_state, h_state), (c_name, h_name) in zip(value, state_names):
      save_ops.append(self._batch.save_state(c_name, c_state))
      save_ops.append(self._batch.save_state(h_name, h_state))
    return tf.group(*save_ops)


def _get_tuple_state_names(num_states, base_name):
  """Returns state names for use with LSTM tuple state."""
  state_names = [('{}_{}_c'.format(i, base_name), '{}_{}_h'.format(
      i, base_name)) for i in range(num_states)]
  return state_names


def _split_bidir_tokens(batch):
  tokens = batch.sequences[data_utils.SequenceWrapper.F_TOKEN_ID]
  # Tokens have shape [batch, time, 2]
  # forward and reverse have shape [batch, time].
  forward, reverse = [
      tf.squeeze(t, axis=[2]) for t in tf.split(tokens, 2, axis=2)
  ]
  return forward, reverse


def _filenames_for_data_spec(phase, bidir, pretrain, use_seq2seq):
  """Returns input filenames for configuration.

  Args:
    phase: str, 'train', 'test', or 'valid'.
    bidir: bool, bidirectional model.
    pretrain: bool, pretraining or classification.
    use_seq2seq: bool, seq2seq data, only valid if pretrain=True.

  Returns:
    Tuple of filenames.

  Raises:
    ValueError: if an invalid combination of arguments is provided that does not
      map to any data files (e.g. pretrain=False, use_seq2seq=True).
  """
  data_spec = (phase, bidir, pretrain, use_seq2seq)
  data_specs = {
      ('train', True, True, False): (data_utils.TRAIN_LM,
                                     data_utils.TRAIN_REV_LM),
      ('train', True, False, False): (data_utils.TRAIN_BD_CLASS,),
      ('train', False, True, False): (data_utils.TRAIN_LM,),
      ('train', False, True, True): (data_utils.TRAIN_SA,),
      ('train', False, False, False): (data_utils.TRAIN_CLASS,),
      ('test', True, True, False): (data_utils.TEST_LM,
                                    data_utils.TRAIN_REV_LM),
      ('test', True, False, False): (data_utils.TEST_BD_CLASS,),
      ('test', False, True, False): (data_utils.TEST_LM,),
      ('test', False, True, True): (data_utils.TEST_SA,),
      ('test', False, False, False): (data_utils.TEST_CLASS,),
      ('valid', True, False, False): (data_utils.VALID_BD_CLASS,),
      ('valid', False, False, False): (data_utils.VALID_CLASS,),
  }
  if data_spec not in data_specs:
    raise ValueError(
        'Data specification (phase, bidir, pretrain, use_seq2seq) %s not '
        'supported' % str(data_spec))

  return data_specs[data_spec]


def _read_single_sequence_example(file_list, tokens_shape=None):
  """Reads and parses SequenceExamples from TFRecord-encoded file_list."""
  tf.logging.info('Constructing TFRecordReader from files: %s', file_list)
  file_queue = tf.train.string_input_producer(file_list)
  reader = tf.TFRecordReader()
  seq_key, serialized_record = reader.read(file_queue)
  ctx, sequence = tf.parse_single_sequence_example(
      serialized_record,
      sequence_features={
          data_utils.SequenceWrapper.F_TOKEN_ID:
              tf.FixedLenSequenceFeature(tokens_shape or [], dtype=tf.int64),
          data_utils.SequenceWrapper.F_LABEL:
              tf.FixedLenSequenceFeature([], dtype=tf.int64),
          data_utils.SequenceWrapper.F_WEIGHT:
              tf.FixedLenSequenceFeature([], dtype=tf.float32),
      })
  return seq_key, ctx, sequence


def _read_and_batch(data_dir,
                    fname,
                    state_name,
                    state_size,
                    num_layers,
                    unroll_steps,
                    batch_size,
                    bidir_input=False):
  """Inputs for text model.

  Args:
    data_dir: str, directory containing TFRecord files of SequenceExample.
    fname: str, input file name.
    state_name: string, key for saved state of LSTM.
    state_size: int, size of LSTM state.
    num_layers: int, the number of layers in the LSTM.
    unroll_steps: int, number of timesteps to unroll for TBTT.
    batch_size: int, batch size.
    bidir_input: bool, whether the input is bidirectional. If True, creates 2
      states, state_name and state_name + '_reverse'.

  Returns:
    Instance of NextQueuedSequenceBatch

  Raises:
    ValueError: if file for input specification is not found.
  """
  data_path = os.path.join(data_dir, fname)
  if not tf.gfile.Exists(data_path):
    raise ValueError('Failed to find file: %s' % data_path)

  tokens_shape = [2] if bidir_input else []
  seq_key, ctx, sequence = _read_single_sequence_example(
      [data_path], tokens_shape=tokens_shape)
  # Set up stateful queue reader.
  state_names = _get_tuple_state_names(num_layers, state_name)
  initial_states = {}
  for c_state, h_state in state_names:
    initial_states[c_state] = tf.zeros(state_size)
    initial_states[h_state] = tf.zeros(state_size)
  if bidir_input:
    rev_state_names = _get_tuple_state_names(num_layers,
                                             '{}_reverse'.format(state_name))
    for rev_c_state, rev_h_state in rev_state_names:
      initial_states[rev_c_state] = tf.zeros(state_size)
      initial_states[rev_h_state] = tf.zeros(state_size)
  batch = tf.contrib.training.batch_sequences_with_states(
      input_key=seq_key,
      input_sequences=sequence,
      input_context=ctx,
      input_length=tf.shape(sequence['token_id'])[0],
      initial_states=initial_states,
      num_unroll=unroll_steps,
      batch_size=batch_size,
      allow_small_batch=False,
      num_threads=4,
      capacity=batch_size * 10,
      make_keys_unique=True,
      make_keys_unique_seed=29392)
  return batch


def inputs(data_dir=None,
           phase='train',
           bidir=False,
           pretrain=False,
           use_seq2seq=False,
           state_name='lstm',
           state_size=None,
           num_layers=0,
           batch_size=32,
           unroll_steps=100,
           eos_id=None):
  """Inputs for text model.

  Args:
    data_dir: str, directory containing TFRecord files of SequenceExample.
    phase: str, dataset for evaluation {'train', 'valid', 'test'}.
    bidir: bool, bidirectional LSTM.
    pretrain: bool, whether to read pretraining data or classification data.
    use_seq2seq: bool, whether to read seq2seq data or the language model data.
    state_name: string, key for saved state of LSTM.
    state_size: int, size of LSTM state.
    num_layers: int, the number of LSTM layers.
    batch_size: int, batch size.
    unroll_steps: int, number of timesteps to unroll for TBTT.
    eos_id: int, id of end of sequence. used for the kl weights on vat
  Returns:
    Instance of VatxtInput (x2 if bidir=True and pretrain=True, i.e. forward and
      reverse).
  """
  with tf.name_scope('inputs'):
    filenames = _filenames_for_data_spec(phase, bidir, pretrain, use_seq2seq)

    if bidir and pretrain:
      # Bidirectional pretraining
      # Requires separate forward and reverse language model data.
      forward_fname, reverse_fname = filenames
      forward_batch = _read_and_batch(data_dir, forward_fname, state_name,
                                      state_size, num_layers, unroll_steps,
                                      batch_size)
      state_name_rev = state_name + '_reverse'
      reverse_batch = _read_and_batch(data_dir, reverse_fname, state_name_rev,
                                      state_size, num_layers, unroll_steps,
                                      batch_size)
      forward_input = VatxtInput(
          forward_batch,
          state_name=state_name,
          num_states=num_layers,
          eos_id=eos_id)
      reverse_input = VatxtInput(
          reverse_batch,
          state_name=state_name_rev,
          num_states=num_layers,
          eos_id=eos_id)
      return forward_input, reverse_input

    elif bidir:
      # Classifier bidirectional LSTM
      # Shared data source, but separate token/state streams
      fname, = filenames
      batch = _read_and_batch(
          data_dir,
          fname,
          state_name,
          state_size,
          num_layers,
          unroll_steps,
          batch_size,
          bidir_input=True)
      forward_tokens, reverse_tokens = _split_bidir_tokens(batch)
      forward_input = VatxtInput(
          batch,
          state_name=state_name,
          tokens=forward_tokens,
          num_states=num_layers)
      reverse_input = VatxtInput(
          batch,
          state_name=state_name + '_reverse',
          tokens=reverse_tokens,
          num_states=num_layers)
      return forward_input, reverse_input
    else:
      # Unidirectional LM or classifier
      fname, = filenames
      batch = _read_and_batch(
          data_dir,
          fname,
          state_name,
          state_size,
          num_layers,
          unroll_steps,
          batch_size,
          bidir_input=False)
      return VatxtInput(
          batch, state_name=state_name, num_states=num_layers, eos_id=eos_id)
