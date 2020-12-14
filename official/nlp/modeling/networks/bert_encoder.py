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
"""Transformer-based text encoder network."""
# pylint: disable=g-classes-have-attributes
import collections
import tensorflow as tf

from official.modeling import activations
from official.nlp import keras_nlp


# This class is being replaced by keras_nlp.encoders.BertEncoder and merely
# acts as a wrapper if you need: 1) list outputs instead of dict outputs,
# 2) shared embedding layer.
@tf.keras.utils.register_keras_serializable(package='Text')
class BertEncoder(keras_nlp.encoders.BertEncoder):
  """Bi-directional Transformer-based encoder network.

  This network implements a bi-directional Transformer-based encoder as
  described in "BERT: Pre-training of Deep Bidirectional Transformers for
  Language Understanding" (https://arxiv.org/abs/1810.04805). It includes the
  embedding lookups and transformer layers, but not the masked language model
  or classification task networks.

  The default values for this object are taken from the BERT-Base implementation
  in "BERT: Pre-training of Deep Bidirectional Transformers for Language
  Understanding".

  *Note* that the network is constructed by
  [Keras Functional API](https://keras.io/guides/functional_api/).

  Arguments:
    vocab_size: The size of the token vocabulary.
    hidden_size: The size of the transformer hidden layers.
    num_layers: The number of transformer layers.
    num_attention_heads: The number of attention heads for each transformer. The
      hidden size must be divisible by the number of attention heads.
    sequence_length: [Deprecated]. TODO(hongkuny): remove this argument once no
      user is using it.
    max_sequence_length: The maximum sequence length that this encoder can
      consume. If None, max_sequence_length uses the value from sequence length.
      This determines the variable shape for positional embeddings.
    type_vocab_size: The number of types that the 'type_ids' input can take.
    intermediate_size: The intermediate size for the transformer layers.
    activation: The activation to use for the transformer layers.
    dropout_rate: The dropout rate to use for the transformer layers.
    attention_dropout_rate: The dropout rate to use for the attention layers
      within the transformer layers.
    initializer: The initialzer to use for all weights in this encoder.
    return_all_encoder_outputs: Whether to output sequence embedding outputs of
      all encoder transformer layers. Note: when the following `dict_outputs`
      argument is True, all encoder outputs are always returned in the dict,
      keyed by `encoder_outputs`.
    output_range: The sequence output range, [0, output_range), by slicing the
      target sequence of the last transformer layer. `None` means the entire
      target sequence will attend to the source sequence, which yeilds the full
      output.
    embedding_width: The width of the word embeddings. If the embedding width is
      not equal to hidden size, embedding parameters will be factorized into two
      matrices in the shape of ['vocab_size', 'embedding_width'] and
      ['embedding_width', 'hidden_size'] ('embedding_width' is usually much
      smaller than 'hidden_size').
    embedding_layer: The word embedding layer. `None` means we will create a new
      embedding layer. Otherwise, we will reuse the given embedding layer. This
      parameter is originally added for ELECTRA model which needs to tie the
      generator embeddings with the discriminator embeddings.
    dict_outputs: Whether to use a dictionary as the model outputs.
  """

  def __init__(self,
               vocab_size,
               hidden_size=768,
               num_layers=12,
               num_attention_heads=12,
               sequence_length=None,
               max_sequence_length=512,
               type_vocab_size=16,
               intermediate_size=3072,
               activation=activations.gelu,
               dropout_rate=0.1,
               attention_dropout_rate=0.1,
               initializer=tf.keras.initializers.TruncatedNormal(stddev=0.02),
               return_all_encoder_outputs=False,
               output_range=None,
               embedding_width=None,
               embedding_layer=None,
               dict_outputs=False,
               **kwargs):

    # b/164516224
    # Once we've created the network using the Functional API, we call
    # super().__init__ as though we were invoking the Functional API Model
    # constructor, resulting in this object having all the properties of a model
    # created using the Functional API. Once super().__init__ is called, we
    # can assign attributes to `self` - note that all `self` assignments are
    # below this line.
    super(BertEncoder, self).__init__(
        vocab_size=vocab_size,
        hidden_size=hidden_size,
        num_layers=num_layers,
        num_attention_heads=num_attention_heads,
        max_sequence_length=max_sequence_length,
        type_vocab_size=type_vocab_size,
        inner_dim=intermediate_size,
        inner_activation=activation,
        output_dropout=dropout_rate,
        attention_dropout=attention_dropout_rate,
        initializer=initializer,
        output_range=output_range,
        embedding_width=embedding_width,
        embedding_layer=embedding_layer)

    self._embedding_layer_instance = embedding_layer

    # Replace arguments from keras_nlp.encoders.BertEncoder.
    config_dict = self._config._asdict()
    config_dict['activation'] = config_dict.pop('inner_activation')
    config_dict['intermediate_size'] = config_dict.pop('inner_dim')
    config_dict['dropout_rate'] = config_dict.pop('output_dropout')
    config_dict['attention_dropout_rate'] = config_dict.pop('attention_dropout')
    config_dict['dict_outputs'] = dict_outputs
    config_dict['return_all_encoder_outputs'] = return_all_encoder_outputs
    config_cls = collections.namedtuple('Config', config_dict.keys())
    self._config = config_cls(**config_dict)

    if dict_outputs:
      return
    else:
      nested_output = self._nested_outputs
      cls_output = nested_output['pooled_output']
      if return_all_encoder_outputs:
        encoder_outputs = nested_output['encoder_outputs']
        outputs = [encoder_outputs, cls_output]
      else:
        sequence_output = nested_output['sequence_output']
        outputs = [sequence_output, cls_output]
    super(keras_nlp.encoders.BertEncoder, self).__init__(
        inputs=self.inputs, outputs=outputs, **kwargs)
