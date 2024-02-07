# Copyright 2024 The TensorFlow Authors. All Rights Reserved.
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

"""Perceiver networks."""

import copy

import tensorflow as tf, tf_keras

from official.nlp.modeling import layers


class Pretrainer(tf_keras.Model):
  """Perceiver Pretrainer.

  Adds the masked language model head upon the encoder output. Optionally
  incorporates decoder output.
  Forked from
  (https://github.com/tensorflow/models/blob/master/official/nlp/modeling/models/bert_pretrainer.py)

  Attributes:
    encoder:
      A perceiver encode and processor transformer network. This network
      should output a classification output. Furthermore, it should expose its
      embedding table via a "get_embedding_table" method.
    masked_lm:
      Masked language model network head for language modeling with encoder
      and optionally decoded output.
    inputs:
      A `Dict[str, tf_keras.Input]` with `input_word_ids`, `input_mask`, and
      `input_type_ids`. The shapes are all `(None)` with dtype `tf.int32`.
      If `masked_lm_positions` is included, it will run masked language
      modeling layer to return sequence of logits.
  """

  def __init__(self,
               encoder,
               decoder=None,
               mlm_activation=None,
               mlm_initializer='glorot_uniform',
               customized_masked_lm=None,
               name='pretrainer',
               **kwargs):
    """Init.

    Args:
      encoder:
        A perceiver encode and processor transformer network. It should expose
        its embedding table via a "get_embedding_table" method. Decoder won't
        be used if `sequence_output` is in the output of the encoder.
      decoder:
        A perceiver decoder network. This parameter is optional. This layer
        accepts the latent output of the encoder and emits logits. Decoder must
        accept a dictionary of `latent_output` and `input_mask` as inputs. This
        will not be used if `sequence_output` is an output from `encoder`.
      mlm_activation:
        The activation (if any) to use in the masked LM network. If `None`, no
        activation will be used.
      mlm_initializer:
        The initializer (if any) to use in the masked LM. Default
        to a Glorot uniform initializer.
      customized_masked_lm:
        A customized masked_lm layer. If None, will create
        a standard layer from `layers.MaskedLM`; if not None, will use the
        specified masked_lm layer. Above arguments `mlm_activation` and
        `mlm_initializer` will be ignored.
      name:
        Sets the `tf_keras.Model` name.
      **kwargs:
        Any keyword arguments to pass through to `tf_keras.Model`.
    """
    super().__init__(**kwargs, name=name)

    self._config = {
        'encoder': encoder,
        'decoder': decoder,
        'mlm_initializer': mlm_initializer,
        'mlm_activation': mlm_activation,
        'customized_masked_lm': customized_masked_lm,
        'name': name,
    }

    self._decoder = decoder
    self.encoder = encoder
    encoder_inputs = self.encoder.inputs

    # Makes sure the weights are built.
    encoder_outputs = self.encoder(encoder_inputs)

    if 'sequence_output' not in encoder_outputs:
      if 'latent_output' in encoder_outputs and self._decoder is not None:
        decoder_inputs = {
            'latent_output': encoder_outputs['latent_output'],
            'input_mask': encoder_inputs['input_mask'],
        }
        decoder_outputs = self._decoder(decoder_inputs)
        if 'sequence_output' not in decoder_outputs:
          raise ValueError('`sequence_output` must be in decoder output.')
      else:
        raise ValueError('if `sequence_output` is not in encoder output, '
                         '`latent_output` must be in encoder output and'
                         'decoder must exist.')

    encoder_inputs = copy.copy(self.encoder.inputs)
    inputs = dict(encoder_inputs)

    if self._decoder is not None:
      inputs.update(copy.copy(self._decoder.inputs))

    self.masked_lm = customized_masked_lm or layers.MaskedLM(
        embedding_table=self.encoder.get_embedding_table(),
        activation=mlm_activation,
        initializer=mlm_initializer,
        name='cls/predictions')
    masked_lm_positions = tf_keras.layers.Input(
        shape=(None,), name='masked_lm_positions', dtype=tf.int32)

    if isinstance(inputs, dict):
      inputs['masked_lm_positions'] = masked_lm_positions
    else:
      raise ValueError(f'Unexpected inputs type to {self.__class__}.')
    self.inputs = inputs

  def call(self, inputs):  # pytype: disable=signature-mismatch  # overriding-parameter-count-checks
    """Return perceiver pretrainer model output tensors in a dict.

    Accepts inputs as dictionary of tensors.
    Args:
      inputs:
        A `Dict[str, tf_keras.Input]` with `input_word_ids`, `input_mask`, and
        `input_type_ids`. The shapes are all `(None)` with dtype `tf.int32`.
        If `masked_lm_positions` is included, it will run masked language
        modeling layer to return sequence of logits.

    Returns:
      `Dict[str, tf.Tensor]` with `sequence_output` and optionally
      `mlm_logits`.
    """
    if not isinstance(inputs, dict):
      raise ValueError(f'Unexpected inputs type to {self.__class__}.')

    word_ids = inputs['input_word_ids']
    input_type_ids = inputs.get('input_type_ids')
    input_mask = inputs.get('input_mask')

    encoder_inputs = {
        'input_word_ids': word_ids,
        'input_mask': input_mask,
        'input_type_ids': input_type_ids,
    }
    encoder_outputs = self.encoder(encoder_inputs)

    if 'sequence_output' not in encoder_outputs:
      if 'latent_output' in encoder_outputs:
        z = encoder_outputs['latent_output']
        decoder_inputs = {'latent_output': z, 'input_mask': input_mask}
        decoder_output = self._decoder(decoder_inputs)

        outputs = dict()
        if isinstance(decoder_output, dict):
          outputs = decoder_output
        else:
          raise ValueError('decoder\'s output should be a dict,'
                           f'but got {decoder_output}')
      else:
        raise ValueError('If `sequence_output` is not in encoder output,'
                         '`latent_output` must be in encoder output.')
    else:
      outputs = encoder_outputs

    sequence_output = outputs['sequence_output']
    # Inference may not have masked_lm_positions and mlm_logits is not needed.
    if 'masked_lm_positions' in inputs:
      masked_lm_positions = inputs['masked_lm_positions']
      outputs['mlm_logits'] = self.masked_lm(
          sequence_output, masked_positions=masked_lm_positions)
    return outputs

  @property
  def checkpoint_items(self):
    """Returns a dictionary of items to be additionally checkpointed."""
    items = dict(
        encoder=self.encoder,
        masked_lm=self.masked_lm,
        decoder=self._decoder)
    return items

  def get_config(self):
    """Return the configuration to set up this object using `from_config`."""
    return self._config

  @classmethod
  def from_config(cls, config, custom_objects=None):
    """Initialize object using config from `get_config`.

    https://www.tensorflow.org/api_docs/python/tf/keras/models/model_from_config

    Args:
      config:
        Return the configuration to set up this object.
      custom_objects:
        Optional dictionary mapping names (strings) to custom classes or
        functions to be considered during deserialization.
    Returns:
      A Keras model instance (uncompiled).
    """
    return cls(**config)
