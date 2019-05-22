# Copyright 2019 The TensorFlow Authors. All Rights Reserved.
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
"""BERT models that are compatible with TF 2.0."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import copy
import tensorflow as tf

from official.bert import modeling


def gather_indexes(sequence_tensor, positions):
  """Gathers the vectors at the specific positions.

  Args:
      sequence_tensor: Sequence output of `BertModel` layer of shape
        (`batch_size`, `seq_length`, num_hidden) where num_hidden is number of
        hidden units of `BertModel` layer.
      positions: Positions ids of tokens in sequence to mask for pretraining of
        with dimension (batch_size, max_predictions_per_seq) where
        `max_predictions_per_seq` is maximum number of tokens to mask out and
        predict per each sequence.

  Returns:
      Masked out sequence tensor of shape (batch_size * max_predictions_per_seq,
      num_hidden).
  """
  sequence_shape = modeling.get_shape_list(
      sequence_tensor, name='sequence_output_tensor')
  batch_size = sequence_shape[0]
  seq_length = sequence_shape[1]
  width = sequence_shape[2]

  flat_offsets = tf.keras.backend.reshape(
      tf.range(0, batch_size, dtype=tf.int32) * seq_length, [-1, 1])
  flat_positions = tf.keras.backend.reshape(positions + flat_offsets, [-1])
  flat_sequence_tensor = tf.keras.backend.reshape(
      sequence_tensor, [batch_size * seq_length, width])
  output_tensor = tf.gather(flat_sequence_tensor, flat_positions)

  return output_tensor


class BertPretrainLayer(tf.keras.layers.Layer):
  """Wrapper layer for pre-training a BERT model.

  This layer wraps an existing `bert_layer` which is a Keras Layer.
  It outputs `sequence_output` from TransformerBlock sub-layer and
  `sentence_output` which are suitable for feeding into a BertPretrainLoss
  layer. This layer can be used along with an unsupervised input to
  pre-train the embeddings for `bert_layer`.
  """

  def __init__(self,
               config,
               bert_layer,
               initializer=None,
               float_type=tf.float32,
               **kwargs):
    super(BertPretrainLayer, self).__init__(**kwargs)
    self.config = copy.deepcopy(config)
    self.float_type = float_type

    self.embedding_table = bert_layer.embedding_lookup.embeddings
    self.num_next_sentence_label = 2
    if initializer:
      self.initializer = initializer
    else:
      self.initializer = tf.keras.initializers.TruncatedNormal(
          stddev=self.config.initializer_range)

  def build(self, unused_input_shapes):
    """Implements build() for the layer."""
    self.lm_dense = tf.keras.layers.Dense(
        self.config.hidden_size,
        activation=modeling.get_activation(self.config.hidden_act),
        kernel_initializer=self.initializer)
    self.lm_bias = self.add_weight(
        shape=[self.config.vocab_size],
        name='lm_bias',
        initializer=tf.keras.initializers.Zeros())
    self.lm_layer_norm = tf.keras.layers.LayerNormalization(
        axis=-1, epsilon=1e-12)
    self.next_sentence_dense = tf.keras.layers.Dense(
        self.num_next_sentence_label, kernel_initializer=self.initializer)
    super(BertPretrainLayer, self).build(unused_input_shapes)

  def __call__(self,
               pooled_output,
               sequence_output=None,
               masked_lm_positions=None):
    inputs = modeling.pack_inputs(
        [pooled_output, sequence_output, masked_lm_positions])
    return super(BertPretrainLayer, self).__call__(inputs)

  def call(self, inputs):
    """Implements call() for the layer."""
    unpacked_inputs = modeling.unpack_inputs(inputs)
    pooled_output = unpacked_inputs[0]
    sequence_output = unpacked_inputs[1]
    masked_lm_positions = unpacked_inputs[2]

    mask_lm_input_tensor = gather_indexes(
        sequence_output, masked_lm_positions)
    lm_output = self.lm_dense(mask_lm_input_tensor)
    lm_output = self.lm_layer_norm(lm_output)
    lm_output = tf.keras.backend.dot(
        lm_output, tf.keras.backend.transpose(self.embedding_table))
    lm_output = tf.keras.backend.bias_add(lm_output, self.lm_bias)
    lm_output = tf.keras.backend.softmax(lm_output)
    lm_output = tf.keras.backend.log(lm_output)

    sentence_output = self.next_sentence_dense(pooled_output)
    sentence_output = tf.keras.backend.softmax(sentence_output)
    sentence_output = tf.keras.backend.log(sentence_output)
    return (lm_output, sentence_output)


class BertPretrainLossAndMetricLayer(tf.keras.layers.Layer):
  """Returns layer that computes custom loss and metrics for pretraining."""

  def __init__(self, bert_config, **kwargs):
    super(BertPretrainLossAndMetricLayer, self).__init__(**kwargs)
    self.config = copy.deepcopy(bert_config)

  def __call__(self,
               lm_output,
               sentence_output=None,
               lm_label_ids=None,
               lm_label_weights=None,
               sentence_labels=None):
    inputs = modeling.pack_inputs([
        lm_output, sentence_output, lm_label_ids, lm_label_weights,
        sentence_labels
    ])
    return super(BertPretrainLossAndMetricLayer, self).__call__(inputs)

  def _add_metrics(self, lm_output, lm_labels, lm_label_weights,
                   lm_per_example_loss, sentence_output, sentence_labels,
                   sentence_per_example_loss):
    """Adds metrics."""
    masked_lm_accuracy = tf.keras.metrics.sparse_categorical_accuracy(
        lm_labels, lm_output)
    masked_lm_accuracy = tf.reduce_mean(masked_lm_accuracy * lm_label_weights)
    self.add_metric(
        masked_lm_accuracy, name='masked_lm_accuracy', aggregation='mean')

    lm_example_loss = tf.reshape(lm_per_example_loss, [-1])
    lm_example_loss = tf.reduce_mean(lm_example_loss * lm_label_weights)
    self.add_metric(lm_example_loss, name='lm_example_loss', aggregation='mean')

    next_sentence_accuracy = tf.keras.metrics.sparse_categorical_accuracy(
        sentence_labels, sentence_output)
    self.add_metric(
        next_sentence_accuracy,
        name='next_sentence_accuracy',
        aggregation='mean')

    next_sentence_mean_loss = tf.reduce_mean(sentence_per_example_loss)
    self.add_metric(
        next_sentence_mean_loss, name='next_sentence_loss', aggregation='mean')

  def call(self, inputs):
    """Implements call() for the layer."""
    unpacked_inputs = modeling.unpack_inputs(inputs)
    lm_output = unpacked_inputs[0]
    sentence_output = unpacked_inputs[1]
    lm_label_ids = tf.keras.backend.cast(unpacked_inputs[2], tf.int32)
    lm_label_ids = tf.keras.backend.reshape(lm_label_ids, [-1])
    lm_label_ids_one_hot = tf.keras.backend.one_hot(lm_label_ids,
                                                    self.config.vocab_size)
    lm_label_weights = tf.keras.backend.cast(unpacked_inputs[3], tf.float32)
    lm_label_weights = tf.keras.backend.reshape(lm_label_weights, [-1])
    lm_per_example_loss = -tf.keras.backend.sum(
        lm_output * lm_label_ids_one_hot, axis=[-1])
    numerator = tf.keras.backend.sum(lm_label_weights * lm_per_example_loss)
    denominator = tf.keras.backend.sum(lm_label_weights) + 1e-5
    mask_label_loss = numerator / denominator

    sentence_labels = tf.keras.backend.cast(unpacked_inputs[4], dtype=tf.int32)
    sentence_labels = tf.keras.backend.reshape(sentence_labels, [-1])
    sentence_label_one_hot = tf.keras.backend.one_hot(sentence_labels, 2)
    per_example_loss_sentence = -tf.keras.backend.sum(
        sentence_label_one_hot * sentence_output, axis=-1)
    sentence_loss = tf.keras.backend.mean(per_example_loss_sentence)
    loss = mask_label_loss + sentence_loss
    final_loss = tf.fill(
        tf.keras.backend.shape(per_example_loss_sentence), loss)

    self._add_metrics(lm_output, lm_label_ids, lm_label_weights,
                      lm_per_example_loss, sentence_output, sentence_labels,
                      per_example_loss_sentence)
    return final_loss


def pretrain_model(bert_config,
                   seq_length,
                   max_predictions_per_seq,
                   initializer=None):
  """Returns model to be used for pre-training.

  Args:
      bert_config: Configuration that defines the core BERT model.
      seq_length: Maximum sequence length of the training data.
      max_predictions_per_seq: Maximum number of tokens in sequence to mask out
        and use for pretraining.
      initializer: Initializer for weights in BertPretrainLayer.

  Returns:
      Pretraining model as well as core BERT submodel from which to save
      weights after pretraining.
  """

  input_word_ids = tf.keras.layers.Input(
      shape=(seq_length,), name='input_word_ids', dtype=tf.int32)
  input_mask = tf.keras.layers.Input(
      shape=(seq_length,), name='input_mask', dtype=tf.int32)
  input_type_ids = tf.keras.layers.Input(
      shape=(seq_length,), name='input_type_ids', dtype=tf.int32)
  masked_lm_positions = tf.keras.layers.Input(
      shape=(max_predictions_per_seq,),
      name='masked_lm_positions',
      dtype=tf.int32)
  masked_lm_weights = tf.keras.layers.Input(
      shape=(max_predictions_per_seq,),
      name='masked_lm_weights',
      dtype=tf.int32)
  next_sentence_labels = tf.keras.layers.Input(
      shape=(1,), name='next_sentence_labels', dtype=tf.int32)
  masked_lm_ids = tf.keras.layers.Input(
      shape=(max_predictions_per_seq,), name='masked_lm_ids', dtype=tf.int32)

  bert_submodel_name = 'bert_core_layer'
  bert_submodel = modeling.get_bert_model(
      input_word_ids,
      input_mask,
      input_type_ids,
      name=bert_submodel_name,
      config=bert_config)
  pooled_output = bert_submodel.outputs[0]
  sequence_output = bert_submodel.outputs[1]

  pretrain_layer = BertPretrainLayer(
      bert_config,
      bert_submodel.get_layer(bert_submodel_name),
      initializer=initializer)
  lm_output, sentence_output = pretrain_layer(pooled_output, sequence_output,
                                              masked_lm_positions)

  pretrain_loss_layer = BertPretrainLossAndMetricLayer(bert_config)
  output_loss = pretrain_loss_layer(lm_output, sentence_output, masked_lm_ids,
                                    masked_lm_weights, next_sentence_labels)

  return tf.keras.Model(
      inputs={
          'input_word_ids': input_word_ids,
          'input_mask': input_mask,
          'input_type_ids': input_type_ids,
          'masked_lm_positions': masked_lm_positions,
          'masked_lm_ids': masked_lm_ids,
          'masked_lm_weights': masked_lm_weights,
          'next_sentence_labels': next_sentence_labels,
      },
      outputs=output_loss), bert_submodel


class BertSquadLogitsLayer(tf.keras.layers.Layer):
  """Returns a layer that computes custom logits for BERT squad model."""

  def __init__(self, initializer=None, float_type=tf.float32, **kwargs):
    super(BertSquadLogitsLayer, self).__init__(**kwargs)
    self.initializer = initializer
    self.float_type = float_type

  def build(self, unused_input_shapes):
    """Implements build() for the layer."""
    self.final_dense = tf.keras.layers.Dense(
        units=2, kernel_initializer=self.initializer, name='final_dense')
    super(BertSquadLogitsLayer, self).build(unused_input_shapes)

  def call(self, inputs):
    """Implements call() for the layer."""
    sequence_output = inputs

    input_shape = sequence_output.shape.as_list()
    sequence_length = input_shape[1]
    num_hidden_units = input_shape[2]

    final_hidden_input = tf.keras.backend.reshape(sequence_output,
                                                  [-1, num_hidden_units])
    logits = self.final_dense(final_hidden_input)
    logits = tf.keras.backend.reshape(logits, [-1, sequence_length, 2])
    logits = tf.transpose(logits, [2, 0, 1])
    unstacked_logits = tf.unstack(logits, axis=0)
    return unstacked_logits[0], unstacked_logits[1]


def squad_model(bert_config, max_seq_length, float_type, initializer=None):
  """Returns BERT Squad model along with core BERT model to import weights.

  Args:
    bert_config: BertConfig, the config defines the core Bert model.
    max_seq_length: integer, the maximum input sequence length.
    float_type: tf.dtype, tf.float32 or tf.bfloat16.
    initializer: Initializer for weights in BertSquadLogitsLayer.

  Returns:
    Two tensors, start logits and end logits, [batch x sequence length].
  """
  unique_ids = tf.keras.layers.Input(
      shape=(1,), dtype=tf.int32, name='unique_ids')
  input_word_ids = tf.keras.layers.Input(
      shape=(max_seq_length,), dtype=tf.int32, name='input_ids')
  input_mask = tf.keras.layers.Input(
      shape=(max_seq_length,), dtype=tf.int32, name='input_mask')
  input_type_ids = tf.keras.layers.Input(
      shape=(max_seq_length,), dtype=tf.int32, name='segment_ids')

  core_model = modeling.get_bert_model(
      input_word_ids,
      input_mask,
      input_type_ids,
      config=bert_config,
      name='bert_model',
      float_type=float_type)

  # `BertSquadModel` only uses the sequnce_output which
  # has dimensionality (batch_size, sequence_length, num_hidden).
  sequence_output = core_model.outputs[1]

  if initializer is None:
    initializer = tf.keras.initializers.TruncatedNormal(
        stddev=bert_config.initializer_range)
  squad_logits_layer = BertSquadLogitsLayer(
      initializer=initializer, float_type=float_type, name='squad_logits')
  start_logits, end_logits = squad_logits_layer(sequence_output)

  squad = tf.keras.Model(
      inputs={
          'unique_ids': unique_ids,
          'input_ids': input_word_ids,
          'input_mask': input_mask,
          'segment_ids': input_type_ids,
      },
      outputs=[unique_ids, start_logits, end_logits],
      name='squad_model')
  return squad, core_model


def classifier_model(bert_config,
                     float_type,
                     num_labels,
                     max_seq_length,
                     final_layer_initializer=None):
  """BERT classifier model in functional API style.

  Construct a Keras model for predicting `num_labels` outputs from an input with
  maximum sequence length `max_seq_length`.

  Args:
    bert_config: BertConfig, the config defines the core BERT model.
    float_type: dtype, tf.float32 or tf.bfloat16.
    num_labels: integer, the number of classes.
    max_seq_length: integer, the maximum input sequence length.
    final_layer_initializer: Initializer for final dense layer. Defaulted
      TruncatedNormal initializer.

  Returns:
    Combined prediction model (words, mask, type) -> (one-hot labels)
    BERT sub-model (words, mask, type) -> (bert_outputs)
  """
  input_word_ids = tf.keras.layers.Input(
      shape=(max_seq_length,), dtype=tf.int32, name='input_word_ids')
  input_mask = tf.keras.layers.Input(
      shape=(max_seq_length,), dtype=tf.int32, name='input_mask')
  input_type_ids = tf.keras.layers.Input(
      shape=(max_seq_length,), dtype=tf.int32, name='input_type_ids')
  bert_model = modeling.get_bert_model(
      input_word_ids,
      input_mask,
      input_type_ids,
      config=bert_config,
      float_type=float_type)
  pooled_output = bert_model.outputs[0]
  if final_layer_initializer is not None:
    initializer = final_layer_initializer
  else:
    initializer = tf.keras.initializers.TruncatedNormal(
        stddev=bert_config.initializer_range)

  output = tf.keras.layers.Dropout(rate=bert_config.hidden_dropout_prob)(
      pooled_output)
  output = tf.keras.layers.Dense(
      num_labels,
      kernel_initializer=initializer,
      name='output',
      dtype=float_type)(
          output)
  return tf.keras.Model(
      inputs={
          'input_word_ids': input_word_ids,
          'input_mask': input_mask,
          'input_type_ids': input_type_ids
      },
      outputs=output), bert_model
