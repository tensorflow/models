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
"""BERT token classifier."""
# pylint: disable=g-classes-have-attributes

import tensorflow as tf


@tf.keras.utils.register_keras_serializable(package='Text')
class BertTokenClassifier(tf.keras.Model):
  """Token classifier model based on a BERT-style transformer-based encoder.

  This is an implementation of the network structure surrounding a transformer
  encoder as described in "BERT: Pre-training of Deep Bidirectional Transformers
  for Language Understanding" (https://arxiv.org/abs/1810.04805).

  The BertTokenClassifier allows a user to pass in a transformer stack, and
  instantiates a token classification network based on the passed `num_classes`
  argument.

  *Note* that the model is constructed by
  [Keras Functional API](https://keras.io/guides/functional_api/).

  Arguments:
    network: A transformer network. This network should output a sequence output
      and a classification output. Furthermore, it should expose its embedding
      table via a "get_embedding_table" method.
    num_classes: Number of classes to predict from the classification network.
    initializer: The initializer (if any) to use in the classification networks.
      Defaults to a Glorot uniform initializer.
    output: The output style for this network. Can be either 'logits' or
      'predictions'.
  """

  def __init__(self,
               network,
               num_classes,
               initializer='glorot_uniform',
               output='logits',
               dropout_rate=0.1,
               **kwargs):
    self._self_setattr_tracking = False
    self._network = network
    self._config = {
        'network': network,
        'num_classes': num_classes,
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
    sequence_output = tf.keras.layers.Dropout(rate=dropout_rate)(
        sequence_output)

    self.classifier = tf.keras.layers.Dense(
        num_classes,
        activation=None,
        kernel_initializer=initializer,
        name='predictions/transform/logits')
    self.logits = self.classifier(sequence_output)
    if output == 'logits':
      output_tensors = self.logits
    elif output == 'predictions':
      output_tensors = tf.keras.layers.Activation(tf.nn.log_softmax)(
          self.logits)
    else:
      raise ValueError(
          ('Unknown `output` value "%s". `output` can be either "logits" or '
           '"predictions"') % output)
    super(BertTokenClassifier, self).__init__(
        inputs=inputs, outputs=output_tensors, **kwargs)

  @property
  def checkpoint_items(self):
    return dict(encoder=self._network)

  def get_config(self):
    return self._config

  @classmethod
  def from_config(cls, config, custom_objects=None):
    return cls(**config)
