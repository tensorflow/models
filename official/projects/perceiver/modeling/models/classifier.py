# Copyright 2023 The TensorFlow Authors. All Rights Reserved.
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

"""Perceiver classifier."""

import numpy as np
import tensorflow as tf, tf_keras

from official.nlp.modeling import layers


class Classifier(tf_keras.Model):
  """Classifier model based on a shared encoder and optional decoder.

  This is an implementation of the network structure surrounding a transformer
  encoder as described in "Perceiver IO: A General Architecture for Structured
  Inputs & Outputs" (https://arxiv.org/abs/2107.14795).

  The Classifier allows a user to pass in an encoder stack and an optional
  decoder stack (e.g. perceiver decoder), and instantiates a classification
  network based on the passed `num_classes` argument. If `num_classes` is set
  to 1, a regression network is instantiated.

  This is forked from
  (https://github.com/tensorflow/models/blob/master/official/nlp/modeling/models/bert_classifier.py)

  Attributes:
    network:
      A perceiver encode and processor transformer network. This network
      should output a classification output. Furthermore, it should expose its
      embedding table via a "get_embedding_table" method.
    num_classes:
      Number of classes outputted by classification head.
    inputs:
      A `Dict[str, tf_keras.Input]` with `input_word_ids`, `input_mask`, and
      `input_type_ids`. The shapes are all `(None)` with dtype `tf.int32`.
    head_name:
      Name of the classification head.
    classifier:
      Classification head layer.
    initializer:
      `tf_keras.initializers.Initializer` used for classification head layer.
  """

  def __init__(self,
               network,
               num_classes,
               decoder=None,
               initializer=None,
               dropout_rate=0.0,
               head_name='glue',
               cls_head=None,
               name='classifier',
               **kwargs):
    """Init.

    Args:
      network:
        A perceiver encode and processor transformer network. This network
        should output a classification output. Furthermore, it should expose its
        embedding table via a "get_embedding_table" method.
      num_classes:
        Number of classes to predict from the classification network.
      decoder:
        A perceiver decoder network. This network should accept the
        latent output of the encoder and emits logits.
      initializer:
        The initializer (if any) to use in the classification networks.
        Defaults to a Glorot uniform initializer.
      dropout_rate:
        The dropout probability of the cls head.
      head_name:
        Name of the classification head.
      cls_head:
        (Optional) The layer instance to use for the classifier head.
        It should take in the output from network and produce the final logits.
        If set, the arguments ('num_classes', 'initializer', 'dropout_rate',
        'use_encoder_pooler', 'head_name') will be ignored.
      name:
        Sets the `tf_keras.Model` name.
      **kwargs:
        Any keyword arguments to pass through to `tf_keras.Model`.
    """
    super().__init__(name=name, **kwargs)

    self._config = {
        'network': network,
        'decoder': decoder,
        'num_classes': num_classes,
        'initializer': initializer,
        'dropout_rate': dropout_rate,
        'head_name': head_name,
        'cls_head': cls_head,
        'name': name,
    }

    self.num_classes = num_classes
    self.head_name = head_name
    self.initializer = initializer
    self._decoder = decoder
    self._network = network

    inputs = self._network.inputs
    outputs = self._network(inputs)

    if 'sequence_output' not in outputs:
      if 'latent_output' in outputs and self._decoder is not None:
        decoder_inputs = {
            'latent_output': outputs['latent_output'],
            'input_mask': inputs['input_mask'],
        }
        decoder_outputs = self._decoder(decoder_inputs)
        sequence_output = decoder_outputs['sequence_output']
      else:
        raise ValueError('if `sequence_output` is not in encoder output, '
                         '`latent_output` must be in encoder output and'
                         'decoder must exist.')
    else:
      sequence_output = outputs['sequence_output']

    cls_inputs = sequence_output

    if initializer is None:
      stddev = 1. / np.sqrt(cls_inputs.shape[-1])
      initializer = tf_keras.initializers.TruncatedNormal(stddev=stddev)

    if cls_head:
      classifier = cls_head
    else:
      classifier = layers.ClassificationHead(
          inner_dim=cls_inputs.shape[-1],
          num_classes=num_classes,
          initializer=initializer,
          dropout_rate=dropout_rate,
          name=head_name)

    _ = classifier(cls_inputs)
    self.inputs = inputs
    self._cls_head = cls_head
    self._name = name
    self.classifier = classifier

  def call(self, inputs):  # pytype: disable=signature-mismatch  # overriding-parameter-count-checks
    """Return perceiver classifier model output tensors in a dict.

    Accepts inputs as dictionary of tensors.
    Args:
      inputs:
        A `Dict[str, tf_keras.Input]` with `input_word_ids`, `input_mask`, and
        `input_type_ids`. The shapes are all `(None)` with dtype `tf.int32`.

    Returns:
      `tf.Tensor` classification output.
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
    encoder_outputs = self._network(encoder_inputs)

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

    return self.classifier(outputs['sequence_output'])

  @property
  def checkpoint_items(self):
    """Returns a dictionary of items to be additionally checkpointed."""
    items = dict(encoder=self._network, decoder=self._decoder)
    if hasattr(self.classifier, 'checkpoint_items'):
      for key, item in self.classifier.checkpoint_items.items():
        items['.'.join([self.classifier.name, key])] = item
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
