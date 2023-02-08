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
"""Trainer network for dual encoder style models."""
# pylint: disable=g-classes-have-attributes
import collections
import tensorflow as tf

import tensorflow_models as tfm


@tf.keras.utils.register_keras_serializable(package='Text')
class LRADualEncoder(tf.keras.layers.Layer):
  """A dual encoder model based on a transformer-based encoder.

  This is an implementation of the dual encoder network structure based on the
  transfomer stack, as described in ["Language-agnostic BERT Sentence
  Embedding"](https://arxiv.org/abs/2007.01852)

  The DualEncoder allows a user to pass in a transformer stack, and build a dual
  encoder model based on the transformer stack.

  Args:
    network: A transformer network which should output an encoding output.
    max_seq_length: The maximum allowed sequence length for transformer.
    normalize: If set to True, normalize the encoding produced by transfomer.
    logit_scale: The scaling factor of dot products when doing training.
    logit_margin: The margin between positive and negative when doing training.
    output: The output style for this network. Can be either `logits` or
      `predictions`. If set to `predictions`, it will output the embedding
      producted by transformer network.
  """

  def __init__(
      self,
      network,
      num_classes,
      max_seq_length,
      dropout_rate=0.1,
      initializer='glorot_uniform',
      use_encoder_pooler=True,
      inner_dim=None,
      head_name='dual_encode',
      **kwargs
  ):
    super().__init__(**kwargs)

    config_dict = {
        'network': network,
        'num_classes': num_classes,
        'head_name': head_name,
        'max_seq_length': max_seq_length,
        'initializer': initializer,
        'use_encoder_pooler': use_encoder_pooler,
        'inner_dim': inner_dim,
    }
    # We are storing the config dict as a namedtuple here to ensure checkpoint
    # compatibility with an earlier version of this model which did not track
    # the config dict attribute. TF does not track immutable attrs which
    # do not contain Trackables, so by creating a config namedtuple instead of
    # a dict we avoid tracking it.
    config_cls = collections.namedtuple('Config', config_dict.keys())
    self._config = config_cls(**config_dict)
    self._use_encoder_pooler = use_encoder_pooler

    self.network = network
    self.classifier = tfm.nlp.layers.ClassificationHead(
        inner_dim=0 if use_encoder_pooler else inner_dim,
        num_classes=num_classes,
        initializer=initializer,
        dropout_rate=dropout_rate,
        name=head_name,
    )

  def call(self, inputs):
    if isinstance(inputs, dict):
      left_word_ids = inputs.get('left_word_ids')
      left_mask = inputs.get('left_mask')

      right_word_ids = inputs.get('right_word_ids')
      right_mask = inputs.get('right_mask')
    else:
      raise ValueError('Unexpected inputs type to %s.' % self.__class__)

    inputs = [left_word_ids, left_mask, right_word_ids, right_mask]

    left_inputs = [left_word_ids, left_mask]
    left_outputs = self.network(left_inputs)
    right_inputs = [right_word_ids, right_mask]
    right_outputs = self.network(right_inputs)

    if self._use_encoder_pooler:
      # Because we have a copy of inputs to create this Model object, we can
      # invoke the Network object with its own input tensors to start the Model.
      if isinstance(left_outputs, list):
        left_cls_inputs = left_outputs[1]
        right_cls_inputs = right_outputs[1]
      else:
        left_cls_inputs = left_outputs['pooled_output']
        right_cls_inputs = right_outputs['pooled_output']
    else:
      if isinstance(left_outputs, list):
        left_cls_inputs = left_outputs[0]
        right_cls_inputs = right_outputs[0]
      else:
        left_cls_inputs = left_outputs['sequence_output']
        right_cls_inputs = right_outputs['sequence_output']

    cls_inputs = tf.concat([left_cls_inputs, right_cls_inputs], -1)
    predictions = self.classifier(cls_inputs)
    return predictions

  def get_config(self):
    return dict(self._config._asdict())

  @classmethod
  def from_config(cls, config, custom_objects=None):
    return cls(**config)

  @property
  def checkpoint_items(self):
    """Returns a dictionary of items to be additionally checkpointed."""
    items = dict(encoder=self.network)
    return items
