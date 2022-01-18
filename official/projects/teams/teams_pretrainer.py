# Copyright 2022 The TensorFlow Authors. All Rights Reserved.
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

"""Trainer network for TEAMS models."""
# pylint: disable=g-classes-have-attributes

import tensorflow as tf

from official.modeling import tf_utils
from official.nlp.modeling import layers
from official.nlp.modeling import models

_LOGIT_PENALTY_MULTIPLIER = 10000


class ReplacedTokenDetectionHead(tf.keras.layers.Layer):
  """Replaced token detection discriminator head.

  Arguments:
    encoder_cfg: Encoder config, used to create hidden layers and head.
    num_task_agnostic_layers: Number of task agnostic layers in the
      discriminator.
    output: The output style for this network. Can be either 'logits' or
      'predictions'.
  """

  def __init__(self,
               encoder_cfg,
               num_task_agnostic_layers,
               output='logits',
               name='rtd',
               **kwargs):
    super(ReplacedTokenDetectionHead, self).__init__(name=name, **kwargs)
    self.num_task_agnostic_layers = num_task_agnostic_layers
    self.hidden_size = encoder_cfg['embedding_cfg']['hidden_size']
    self.num_hidden_instances = encoder_cfg['num_hidden_instances']
    self.hidden_cfg = encoder_cfg['hidden_cfg']
    self.activation = self.hidden_cfg['intermediate_activation']
    self.initializer = self.hidden_cfg['kernel_initializer']

    self.hidden_layers = []
    for i in range(self.num_task_agnostic_layers, self.num_hidden_instances):
      self.hidden_layers.append(
          layers.Transformer(
              num_attention_heads=self.hidden_cfg['num_attention_heads'],
              intermediate_size=self.hidden_cfg['intermediate_size'],
              intermediate_activation=self.activation,
              dropout_rate=self.hidden_cfg['dropout_rate'],
              attention_dropout_rate=self.hidden_cfg['attention_dropout_rate'],
              kernel_initializer=self.initializer,
              name='transformer/layer_%d_rtd' % i))
    self.dense = tf.keras.layers.Dense(
        self.hidden_size,
        activation=self.activation,
        kernel_initializer=self.initializer,
        name='transform/rtd_dense')
    self.rtd_head = tf.keras.layers.Dense(
        units=1, kernel_initializer=self.initializer,
        name='transform/rtd_head')

    if output not in ('predictions', 'logits'):
      raise ValueError(
          ('Unknown `output` value "%s". `output` can be either "logits" or '
           '"predictions"') % output)
    self._output_type = output

  def call(self, sequence_data, input_mask):
    """Compute inner-products of hidden vectors with sampled element embeddings.

    Args:
      sequence_data: A [batch_size, seq_length, num_hidden] tensor.
      input_mask: A [batch_size, seq_length] binary mask to separate the input
        from the padding.

    Returns:
      A [batch_size, seq_length] tensor.
    """
    attention_mask = layers.SelfAttentionMask()([sequence_data, input_mask])
    data = sequence_data
    for hidden_layer in self.hidden_layers:
      data = hidden_layer([sequence_data, attention_mask])
    rtd_logits = self.rtd_head(self.dense(data))
    return tf.squeeze(rtd_logits, axis=-1)


class MultiWordSelectionHead(tf.keras.layers.Layer):
  """Multi-word selection discriminator head.

  Arguments:
    embedding_table: The embedding table.
    activation: The activation, if any, for the dense layer.
    initializer: The intializer for the dense layer. Defaults to a Glorot
      uniform initializer.
    output: The output style for this network. Can be either 'logits' or
      'predictions'.
  """

  def __init__(self,
               embedding_table,
               activation=None,
               initializer='glorot_uniform',
               output='logits',
               name='mws',
               **kwargs):
    super(MultiWordSelectionHead, self).__init__(name=name, **kwargs)
    self.embedding_table = embedding_table
    self.activation = activation
    self.initializer = tf.keras.initializers.get(initializer)

    self._vocab_size, self.embed_size = self.embedding_table.shape
    self.dense = tf.keras.layers.Dense(
        self.embed_size,
        activation=self.activation,
        kernel_initializer=self.initializer,
        name='transform/mws_dense')
    self.layer_norm = tf.keras.layers.LayerNormalization(
        axis=-1, epsilon=1e-12, name='transform/mws_layernorm')

    if output not in ('predictions', 'logits'):
      raise ValueError(
          ('Unknown `output` value "%s". `output` can be either "logits" or '
           '"predictions"') % output)
    self._output_type = output

  def call(self, sequence_data, masked_positions, candidate_sets):
    """Compute inner-products of hidden vectors with sampled element embeddings.

    Args:
      sequence_data: A [batch_size, seq_length, num_hidden] tensor.
      masked_positions: A [batch_size, num_prediction] tensor.
      candidate_sets: A [batch_size, num_prediction, k] tensor.

    Returns:
      A [batch_size, num_prediction, k] tensor.
    """
    # Gets shapes for later usage
    candidate_set_shape = tf_utils.get_shape_list(candidate_sets)
    num_prediction = candidate_set_shape[1]

    # Gathers hidden vectors -> (batch_size, num_prediction, 1, embed_size)
    masked_lm_input = self._gather_indexes(sequence_data, masked_positions)
    lm_data = self.dense(masked_lm_input)
    lm_data = self.layer_norm(lm_data)
    lm_data = tf.expand_dims(
        tf.reshape(lm_data, [-1, num_prediction, self.embed_size]), 2)

    # Gathers embeddings -> (batch_size, num_prediction, embed_size, k)
    flat_candidate_sets = tf.reshape(candidate_sets, [-1])
    candidate_embeddings = tf.gather(self.embedding_table, flat_candidate_sets)
    candidate_embeddings = tf.reshape(
        candidate_embeddings,
        tf.concat([tf.shape(candidate_sets), [self.embed_size]], axis=0)
    )
    candidate_embeddings.set_shape(
        candidate_sets.shape.as_list() + [self.embed_size])
    candidate_embeddings = tf.transpose(candidate_embeddings, [0, 1, 3, 2])

    # matrix multiplication + squeeze -> (batch_size, num_prediction, k)
    logits = tf.matmul(lm_data, candidate_embeddings)
    logits = tf.squeeze(logits, 2)

    if self._output_type == 'logits':
      return logits
    return tf.nn.log_softmax(logits)

  def _gather_indexes(self, sequence_tensor, positions):
    """Gathers the vectors at the specific positions.

    Args:
        sequence_tensor: Sequence output of shape
          (`batch_size`, `seq_length`, `num_hidden`) where `num_hidden` is
          number of hidden units.
        positions: Positions ids of tokens in batched sequences.

    Returns:
        Sequence tensor of shape (batch_size * num_predictions,
        num_hidden).
    """
    sequence_shape = tf_utils.get_shape_list(
        sequence_tensor, name='sequence_output_tensor')
    batch_size, seq_length, width = sequence_shape

    flat_offsets = tf.reshape(
        tf.range(0, batch_size, dtype=tf.int32) * seq_length, [-1, 1])
    flat_positions = tf.reshape(positions + flat_offsets, [-1])
    flat_sequence_tensor = tf.reshape(sequence_tensor,
                                      [batch_size * seq_length, width])
    output_tensor = tf.gather(flat_sequence_tensor, flat_positions)

    return output_tensor


@tf.keras.utils.register_keras_serializable(package='Text')
class TeamsPretrainer(tf.keras.Model):
  """TEAMS network training model.

  This is an implementation of the network structure described in "Training
  ELECTRA Augmented with Multi-word Selection"
  (https://arxiv.org/abs/2106.00139).

  The TeamsPretrainer allows a user to pass in two transformer encoders, one
  for generator, the other for discriminator (multi-word selection). The
  pretrainer then instantiates the masked language model (at generator side) and
  classification networks (including both multi-word selection head and replaced
  token detection head) that are used to create the training objectives.

  *Note* that the model is constructed by Keras Subclass API, where layers are
  defined inside `__init__` and `call()` implements the computation.

  Args:
    generator_network: A transformer encoder for generator, this network should
      output a sequence output.
    discriminator_mws_network: A transformer encoder for multi-word selection
      discriminator, this network should output a sequence output.
    num_discriminator_task_agnostic_layers: Number of layers shared between
      multi-word selection and random token detection discriminators.
    vocab_size: Size of generator output vocabulary
    candidate_size: Candidate size for multi-word selection task,
      including the correct word.
    mlm_activation: The activation (if any) to use in the masked LM and
      classification networks. If None, no activation will be used.
    mlm_initializer: The initializer (if any) to use in the masked LM and
      classification networks. Defaults to a Glorot uniform initializer.
    output_type: The output style for this network. Can be either `logits` or
      `predictions`.
  """

  def __init__(self,
               generator_network,
               discriminator_mws_network,
               num_discriminator_task_agnostic_layers,
               vocab_size,
               candidate_size=5,
               mlm_activation=None,
               mlm_initializer='glorot_uniform',
               output_type='logits',
               **kwargs):
    super().__init__()
    self._config = {
        'generator_network':
            generator_network,
        'discriminator_mws_network':
            discriminator_mws_network,
        'num_discriminator_task_agnostic_layers':
            num_discriminator_task_agnostic_layers,
        'vocab_size':
            vocab_size,
        'candidate_size':
            candidate_size,
        'mlm_activation':
            mlm_activation,
        'mlm_initializer':
            mlm_initializer,
        'output_type':
            output_type,
    }
    for k, v in kwargs.items():
      self._config[k] = v

    self.generator_network = generator_network
    self.discriminator_mws_network = discriminator_mws_network
    self.vocab_size = vocab_size
    self.candidate_size = candidate_size
    self.mlm_activation = mlm_activation
    self.mlm_initializer = mlm_initializer
    self.output_type = output_type
    self.masked_lm = layers.MaskedLM(
        embedding_table=self.generator_network.embedding_network
        .get_embedding_table(),
        activation=mlm_activation,
        initializer=mlm_initializer,
        output=output_type,
        name='generator_masked_lm')
    discriminator_cfg = self.discriminator_mws_network.get_config()
    self.num_task_agnostic_layers = num_discriminator_task_agnostic_layers
    self.discriminator_rtd_head = ReplacedTokenDetectionHead(
        encoder_cfg=discriminator_cfg,
        num_task_agnostic_layers=self.num_task_agnostic_layers,
        output=output_type,
        name='discriminator_rtd')
    hidden_cfg = discriminator_cfg['hidden_cfg']
    self.discriminator_mws_head = MultiWordSelectionHead(
        embedding_table=self.discriminator_mws_network.embedding_network
        .get_embedding_table(),
        activation=hidden_cfg['intermediate_activation'],
        initializer=hidden_cfg['kernel_initializer'],
        output=output_type,
        name='discriminator_mws')

  def call(self, inputs):
    """TEAMS forward pass.

    Args:
      inputs: A dict of all inputs, same as the standard BERT model.

    Returns:
      outputs: A dict of pretrainer model outputs, including
        (1) lm_outputs: A `[batch_size, num_token_predictions, vocab_size]`
        tensor indicating logits on masked positions.
        (2) disc_rtd_logits: A `[batch_size, sequence_length]` tensor indicating
        logits for discriminator replaced token detection task.
        (3) disc_rtd_label: A `[batch_size, sequence_length]` tensor indicating
        target labels for discriminator replaced token detection task.
        (4) disc_mws_logits: A `[batch_size, num_token_predictions,
        candidate_size]` tensor indicating logits for discriminator multi-word
        selection task.
        (5) disc_mws_labels: A `[batch_size, num_token_predictions]` tensor
        indicating target labels for discriminator multi-word selection task.
    """
    input_word_ids = inputs['input_word_ids']
    input_mask = inputs['input_mask']
    input_type_ids = inputs['input_type_ids']
    masked_lm_positions = inputs['masked_lm_positions']

    # Runs generator.
    sequence_output = self.generator_network(
        [input_word_ids, input_mask, input_type_ids])['sequence_output']

    lm_outputs = self.masked_lm(sequence_output, masked_lm_positions)

    # Samples tokens from generator.
    fake_data = self._get_fake_data(inputs, lm_outputs)

    # Runs discriminator.
    disc_input = fake_data['inputs']
    disc_rtd_label = fake_data['is_fake_tokens']
    disc_mws_candidates = fake_data['candidate_set']
    mws_sequence_outputs = self.discriminator_mws_network([
        disc_input['input_word_ids'], disc_input['input_mask'],
        disc_input['input_type_ids']
    ])['encoder_outputs']

    # Applies replaced token detection with input selected based on
    # self.num_discriminator_task_agnostic_layers
    disc_rtd_logits = self.discriminator_rtd_head(
        mws_sequence_outputs[self.num_task_agnostic_layers - 1], input_mask)

    # Applies multi-word selection.
    disc_mws_logits = self.discriminator_mws_head(mws_sequence_outputs[-1],
                                                  masked_lm_positions,
                                                  disc_mws_candidates)
    disc_mws_label = tf.zeros_like(masked_lm_positions, dtype=tf.int32)

    outputs = {
        'lm_outputs': lm_outputs,
        'disc_rtd_logits': disc_rtd_logits,
        'disc_rtd_label': disc_rtd_label,
        'disc_mws_logits': disc_mws_logits,
        'disc_mws_label': disc_mws_label,
    }

    return outputs

  def _get_fake_data(self, inputs, mlm_logits):
    """Generate corrupted data for discriminator.

    Note it is poosible for sampled token to be the same as the correct one.
    Args:
      inputs: A dict of all inputs, same as the input of `call()` function
      mlm_logits: The generator's output logits

    Returns:
      A dict of generated fake data
    """
    inputs = models.electra_pretrainer.unmask(inputs, duplicate=True)

    # Samples replaced token.
    sampled_tokens = tf.stop_gradient(
        models.electra_pretrainer.sample_from_softmax(
            mlm_logits, disallow=None))
    sampled_tokids = tf.argmax(sampled_tokens, axis=-1, output_type=tf.int32)

    # Prepares input and label for replaced token detection task.
    updated_input_ids, masked = models.electra_pretrainer.scatter_update(
        inputs['input_word_ids'], sampled_tokids, inputs['masked_lm_positions'])
    rtd_labels = masked * (1 - tf.cast(
        tf.equal(updated_input_ids, inputs['input_word_ids']), tf.int32))
    updated_inputs = models.electra_pretrainer.get_updated_inputs(
        inputs, duplicate=True, input_word_ids=updated_input_ids)

    # Samples (candidate_size-1) negatives and concat with true tokens
    disallow = tf.one_hot(
        inputs['masked_lm_ids'], depth=self.vocab_size, dtype=tf.float32)
    sampled_candidates = tf.stop_gradient(
        sample_k_from_softmax(mlm_logits, k=self.candidate_size-1,
                              disallow=disallow))
    true_token_id = tf.expand_dims(inputs['masked_lm_ids'], -1)
    candidate_set = tf.concat([true_token_id, sampled_candidates], -1)

    return {
        'inputs': updated_inputs,
        'is_fake_tokens': rtd_labels,
        'sampled_tokens': sampled_tokens,
        'candidate_set': candidate_set
    }

  @property
  def checkpoint_items(self):
    """Returns a dictionary of items to be additionally checkpointed."""
    items = dict(encoder=self.discriminator_mws_network)
    return items

  def get_config(self):
    return self._config

  @classmethod
  def from_config(cls, config, custom_objects=None):
    return cls(**config)


def sample_k_from_softmax(logits, k, disallow=None, use_topk=False):
  """Implement softmax sampling using gumbel softmax trick to select k items.

  Args:
    logits: A [batch_size, num_token_predictions, vocab_size] tensor indicating
      the generator output logits for each masked position.
    k: Number of samples
    disallow: If `None`, we directly sample tokens from the logits. Otherwise,
      this is a tensor of size [batch_size, num_token_predictions, vocab_size]
      indicating the true word id in each masked position.
    use_topk: Whether to use tf.nn.top_k or using iterative approach where the
      latter is empirically faster.

  Returns:
    sampled_tokens: A [batch_size, num_token_predictions, k] tensor indicating
    the sampled word id in each masked position.
  """
  if use_topk:
    if disallow is not None:
      logits -= _LOGIT_PENALTY_MULTIPLIER * disallow
    uniform_noise = tf.random.uniform(
        tf_utils.get_shape_list(logits), minval=0, maxval=1)
    gumbel_noise = -tf.math.log(-tf.math.log(uniform_noise + 1e-9) + 1e-9)
    _, sampled_tokens = tf.nn.top_k(logits + gumbel_noise, k=k, sorted=False)
  else:
    sampled_tokens_list = []
    vocab_size = tf_utils.get_shape_list(logits)[-1]
    if disallow is not None:
      logits -= _LOGIT_PENALTY_MULTIPLIER * disallow

    uniform_noise = tf.random.uniform(
        tf_utils.get_shape_list(logits), minval=0, maxval=1)
    gumbel_noise = -tf.math.log(-tf.math.log(uniform_noise + 1e-9) + 1e-9)
    logits += gumbel_noise
    for _ in range(k):
      token_ids = tf.argmax(logits, -1, output_type=tf.int32)
      sampled_tokens_list.append(token_ids)
      logits -= _LOGIT_PENALTY_MULTIPLIER *  tf.one_hot(
          token_ids, depth=vocab_size, dtype=tf.float32)
    sampled_tokens = tf.stack(sampled_tokens_list, -1)
  return sampled_tokens
