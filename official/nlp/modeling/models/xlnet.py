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
"""XLNet cls-token classifier."""
# pylint: disable=g-classes-have-attributes

from typing import Any, Mapping, Union

import tensorflow as tf

from official.nlp.modeling import layers


@tf.keras.utils.register_keras_serializable(package='Text')
class XLNetClassifier(tf.keras.Model):
  """Classifier model based on XLNet.

  This is an implementation of the network structure surrounding a
  Transformer-XL encoder as described in "XLNet: Generalized Autoregressive
  Pretraining for Language Understanding" (https://arxiv.org/abs/1906.08237).

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
        inner_dim=network.get_config()['inner_size'],
        num_classes=num_classes,
        initializer=initializer,
        dropout_rate=dropout_rate,
        cls_token_idx=cls_token_idx,
        name='sentence_prediction')

  def call(self, inputs: Mapping[str, Any]):
    input_ids = inputs['input_ids']
    segment_ids = inputs['segment_ids']
    input_mask = inputs['input_mask']
    state = inputs.get('mems', None)

    attention_output, new_states = self._network(
        input_ids=input_ids,
        segment_ids=segment_ids,
        input_mask=input_mask,
        state=state)

    logits = self.classifier(attention_output)

    return logits, new_states

  def get_config(self):
    return self._config

  @classmethod
  def from_config(cls, config, custom_objects=None):
    return cls(**config)
