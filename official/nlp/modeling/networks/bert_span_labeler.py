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
"""Trainer network for BERT-style models."""

from __future__ import absolute_import
from __future__ import division
# from __future__ import google_type_annotations
from __future__ import print_function

import tensorflow as tf

from official.nlp.modeling import networks


@tf.keras.utils.register_keras_serializable(package='Text')
class BertSpanLabeler(tf.keras.Model):
  """Span labeler model based on a BERT-style transformer-based encoder.

  This is an implementation of the network structure surrounding a transformer
  encoder as described in "BERT: Pre-training of Deep Bidirectional Transformers
  for Language Understanding" (https://arxiv.org/abs/1810.04805).

  The BertSpanLabeler allows a user to pass in a transformer stack, and
  instantiates a span labeling network based on a single dense layer.

  Attributes:
    network: A transformer network. This network should output a sequence output
      and a classification output. Furthermore, it should expose its embedding
      table via a "get_embedding_table" method.
    initializer: The initializer (if any) to use in the span labeling network.
      Defaults to a Glorot uniform initializer.
    output: The output style for this network. Can be either 'logits' or
      'predictions'.
  """

  def __init__(self,
               network,
               initializer='glorot_uniform',
               output='logits',
               **kwargs):
    self._self_setattr_tracking = False
    self._config = {
        'network': network,
        'initializer': initializer,
        'output': output,
    }
    # We want to use the inputs of the passed network as the inputs to this
    # Model. To do this, we need to keep a handle to the network inputs for use
    # when we construct the Model object at the end of init.
    inputs = network.inputs

    # Because we have a copy of inputs to create this Model object, we can
    # invoke the Network object with its own input tensors to start the Model.
    sequence_output, _ = network(inputs)

    # This is an instance variable for ease of access to the underlying task
    # network.
    self.span_labeling = networks.SpanLabeling(
        input_width=sequence_output.shape[-1],
        initializer=initializer,
        output=output,
        name='span_labeling')
    start_logits, end_logits = self.span_labeling(sequence_output)

    # Use identity layers wrapped in lambdas to explicitly name the output
    # tensors. This allows us to use string-keyed dicts in Keras fit/predict/
    # evaluate calls.
    start_logits = tf.keras.layers.Lambda(
        tf.identity, name='start_positions')(
            start_logits)
    end_logits = tf.keras.layers.Lambda(
        tf.identity, name='end_positions')(
            end_logits)

    logits = [start_logits, end_logits]

    super(BertSpanLabeler, self).__init__(
        inputs=inputs, outputs=logits, **kwargs)

  def get_config(self):
    return self._config

  @classmethod
  def from_config(cls, config, custom_objects=None):
    return cls(**config)
