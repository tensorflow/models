# Copyright 2020 The TensorFlow Authors. All Rights Reserved.
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
"""XLNet models."""
# pylint: disable=g-classes-have-attributes

from typing import Any, Mapping, Union

import tensorflow as tf

from official.nlp.modeling import layers
from official.nlp.modeling import networks


class XLNetMaskedLM(tf.keras.layers.Layer):
  """XLNet pretraining head."""

  def __init__(self,
               vocab_size: int,
               hidden_size: int,
               initializer: str = 'glorot_uniform',
               activation: str = 'gelu',
               name=None,
               **kwargs):
    super().__init__(name=name, **kwargs)
    self._vocab_size = vocab_size
    self._hidden_size = hidden_size
    self._initializer = initializer
    self._activation = activation

  def build(self, input_shape):
    self.dense = tf.keras.layers.Dense(
        units=self._hidden_size,
        activation=self._activation,
        kernel_initializer=self._initializer,
        name='transform/dense')
    self.layer_norm = tf.keras.layers.LayerNormalization(
        axis=-1, epsilon=1e-12, name='transform/LayerNorm')
    self.bias = self.add_weight(
        'output_bias/bias',
        shape=(self._vocab_size,),
        initializer='zeros',
        trainable=True)
    super().build(input_shape)

  def call(self,
           sequence_data: tf.Tensor,
           embedding_table: tf.Tensor):
    lm_data = self.dense(sequence_data)
    lm_data = self.layer_norm(lm_data)
    lm_data = tf.matmul(lm_data, embedding_table, transpose_b=True)
    logits = tf.nn.bias_add(lm_data, self.bias)
    return logits

  def get_config(self) -> Mapping[str, Any]:
    config = {
        'vocab_size':
            self._vocab_size,
        'hidden_size':
            self._hidden_size,
        'initializer':
            self._initializer
    }
    base_config = super(XLNetMaskedLM, self).get_config()
    return dict(list(base_config.items()) + list(config.items()))


@tf.keras.utils.register_keras_serializable(package='Text')
class XLNetPretrainer(tf.keras.Model):
  """XLNet-based pretrainer.

  This is an implementation of the network structure surrounding a
  Transformer-XL encoder as described in "XLNet: Generalized Autoregressive
  Pretraining for Language Understanding" (https://arxiv.org/abs/1906.08237).

  Arguments:
    network: An XLNet/Transformer-XL based network. This network should output a
      sequence output and list of `state` tensors.
    mlm_activation: The activation (if any) to use in the Masked LM network. If
      None, then no activation will be used.
    mlm_initializer: The initializer (if any) to use in the masked LM. Defaults
      to a Glorot uniform initializer.

  """

  def __init__(
      self,
      network: Union[tf.keras.layers.Layer, tf.keras.Model],
      mlm_activation=None,
      mlm_initializer='glorot_uniform',
      name: str = None,
      **kwargs):
    super().__init__(name=name, **kwargs)
    self._config = {
        'network': network,
        'mlm_activation': mlm_activation,
        'mlm_initializer': mlm_initializer,
    }
    self._network = network
    self._hidden_size = network.get_config()['hidden_size']
    self._vocab_size = network.get_config()['vocab_size']
    self._activation = mlm_activation
    self._initializer = mlm_initializer
    self._masked_lm = XLNetMaskedLM(
        vocab_size=self._vocab_size,
        hidden_size=self._hidden_size,
        initializer=self._initializer)

  def call(self, inputs: Mapping[str, Any]):
    input_word_ids = inputs['input_word_ids']
    input_type_ids = inputs['input_type_ids']
    masked_tokens = inputs['masked_tokens']
    permutation_mask = inputs['permutation_mask']
    target_mapping = inputs['target_mapping']
    state = inputs.get('state', None)

    attention_output, state = self._network(
        input_ids=input_word_ids,
        segment_ids=input_type_ids,
        input_mask=None,
        state=state,
        permutation_mask=permutation_mask,
        target_mapping=target_mapping,
        masked_tokens=masked_tokens)

    embedding_table = self._network.get_embedding_lookup_table()
    mlm_outputs = self._masked_lm(
        sequence_data=attention_output,
        embedding_table=embedding_table)
    return mlm_outputs, state

  def get_config(self) -> Mapping[str, Any]:
    return self._config

  @classmethod
  def from_config(cls, config, custom_objects=None):
    return cls(**config)

  @property
  def checkpoint_items(self):
    return dict(encoder=self._network)


@tf.keras.utils.register_keras_serializable(package='Text')
class XLNetClassifier(tf.keras.Model):
  """Classifier model based on XLNet.

  This is an implementation of the network structure surrounding a
  Transformer-XL encoder as described in "XLNet: Generalized Autoregressive
  Pretraining for Language Understanding" (https://arxiv.org/abs/1906.08237).

  Note: This model does not use utilize the memory mechanism used in the
  original XLNet Classifier.

  Arguments:
    network: An XLNet/Transformer-XL based network. This network should output a
      sequence output and list of `state` tensors.
    num_classes: Number of classes to predict from the classification network.
    initializer: The initializer (if any) to use in the classification networks.
      Defaults to a RandomNormal initializer.
    summary_type: Method used to summarize a sequence into a compact vector.
    dropout_rate: The dropout probability of the cls head.
  """

  def __init__(
      self,
      network: Union[tf.keras.layers.Layer, tf.keras.Model],
      num_classes: int,
      initializer: tf.keras.initializers.Initializer = 'random_normal',
      summary_type: str = 'last',
      dropout_rate: float = 0.1,
      **kwargs):
    super().__init__(**kwargs)
    self._network = network
    self._initializer = initializer
    self._summary_type = summary_type
    self._num_classes = num_classes
    self._config = {
        'network': network,
        'initializer': initializer,
        'num_classes': num_classes,
        'summary_type': summary_type,
        'dropout_rate': dropout_rate,
    }

    if summary_type == 'last':
      cls_token_idx = -1
    elif summary_type == 'first':
      cls_token_idx = 0
    else:
      raise ValueError('Invalid summary type provided: %s.' % summary_type)

    self.classifier = layers.ClassificationHead(
        inner_dim=network.get_config()['hidden_size'],
        num_classes=num_classes,
        initializer=initializer,
        dropout_rate=dropout_rate,
        cls_token_idx=cls_token_idx,
        name='sentence_prediction')

  def call(self, inputs: Mapping[str, Any]):
    input_ids = inputs['input_word_ids']
    segment_ids = inputs['input_type_ids']
    input_mask = tf.cast(inputs['input_mask'], tf.float32)
    state = inputs.get('mems', None)

    attention_output, _ = self._network(
        input_ids=input_ids,
        segment_ids=segment_ids,
        input_mask=input_mask,
        state=state)

    logits = self.classifier(attention_output)

    return logits

  def get_config(self):
    return self._config

  @classmethod
  def from_config(cls, config, custom_objects=None):
    return cls(**config)

  @property
  def checkpoint_items(self):
    items = dict(encoder=self._network)
    if hasattr(self.classifier, 'checkpoint_items'):
      for key, item in self.classifier.checkpoint_items.items():
        items['.'.join([self.classifier.name, key])] = item
    return items


@tf.keras.utils.register_keras_serializable(package='Text')
class XLNetSpanLabeler(tf.keras.Model):
  """Span labeler model based on XLNet.

  This is an implementation of the network structure surrounding a
  Transformer-XL encoder as described in "XLNet: Generalized Autoregressive
  Pretraining for Language Understanding" (https://arxiv.org/abs/1906.08237).

  Arguments:
    network: A transformer network. This network should output a sequence output
      and a classification output. Furthermore, it should expose its embedding
      table via a "get_embedding_table" method.
    start_n_top: Beam size for span start.
    end_n_top: Beam size for span end.
    dropout_rate: The dropout rate for the span labeling layer.
    span_labeling_activation: The activation for the span labeling head.
    initializer: The initializer (if any) to use in the span labeling network.
      Defaults to a Glorot uniform initializer.
  """

  def __init__(
      self,
      network: Union[tf.keras.layers.Layer, tf.keras.Model],
      start_n_top: int = 5,
      end_n_top: int = 5,
      dropout_rate: float = 0.1,
      span_labeling_activation: tf.keras.initializers.Initializer = 'tanh',
      initializer: tf.keras.initializers.Initializer = 'glorot_uniform',
      **kwargs):
    super().__init__(**kwargs)
    self._config = {
        'network': network,
        'start_n_top': start_n_top,
        'end_n_top': end_n_top,
        'dropout_rate': dropout_rate,
        'span_labeling_activation': span_labeling_activation,
        'initializer': initializer,
    }
    network_config = network.get_config()
    try:
      input_width = network_config['inner_size']
      self._xlnet_base = True
    except KeyError:
      # BertEncoder uses 'intermediate_size' due to legacy naming.
      input_width = network_config['intermediate_size']
      self._xlnet_base = False

    self._network = network
    self._initializer = initializer
    self._start_n_top = start_n_top
    self._end_n_top = end_n_top
    self._dropout_rate = dropout_rate
    self._activation = span_labeling_activation
    self.span_labeling = networks.XLNetSpanLabeling(
        input_width=input_width,
        start_n_top=self._start_n_top,
        end_n_top=self._end_n_top,
        activation=self._activation,
        dropout_rate=self._dropout_rate,
        initializer=self._initializer)

  def call(self, inputs: Mapping[str, Any]):
    input_word_ids = inputs['input_word_ids']
    input_type_ids = inputs['input_type_ids']
    input_mask = inputs['input_mask']
    class_index = inputs['class_index']
    paragraph_mask = inputs['paragraph_mask']
    start_positions = inputs.get('start_positions', None)

    if self._xlnet_base:
      attention_output, _ = self._network(
          input_ids=input_word_ids,
          segment_ids=input_type_ids,
          input_mask=input_mask)
    else:
      network_output_dict = self._network(dict(
          input_word_ids=input_word_ids,
          input_type_ids=input_type_ids,
          input_mask=input_mask))
      attention_output = network_output_dict['sequence_output']

    outputs = self.span_labeling(
        sequence_data=attention_output,
        class_index=class_index,
        paragraph_mask=paragraph_mask,
        start_positions=start_positions)
    return outputs

  @property
  def checkpoint_items(self):
    return dict(encoder=self._network)

  def get_config(self):
    return self._config

  @classmethod
  def from_config(cls, config, custom_objects=None):
    return cls(**config)

