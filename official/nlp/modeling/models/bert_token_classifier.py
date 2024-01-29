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

"""BERT token classifier."""
# pylint: disable=g-classes-have-attributes
import collections
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

  Args:
    network: A transformer network. This network should output a sequence output
      and a classification output. Furthermore, it should expose its embedding
      table via a `get_embedding_table` method.
    num_classes: Number of classes to predict from the classification network.
    initializer: The initializer (if any) to use in the classification networks.
      Defaults to a Glorot uniform initializer.
    output: The output style for this network. Can be either `logits` or
      `predictions`.
    dropout_rate: The dropout probability of the token classification head.
    output_encoder_outputs: Whether to include intermediate sequence output
      in the final output.
  """

  def __init__(self,
               network,
               num_classes,
               initializer='glorot_uniform',
               output='logits',
               dropout_rate=0.1,
               output_encoder_outputs=False,
               **kwargs):

    # We want to use the inputs of the passed network as the inputs to this
    # Model. To do this, we need to keep a handle to the network inputs for use
    # when we construct the Model object at the end of init.
    inputs = network.inputs

    # Because we have a copy of inputs to create this Model object, we can
    # invoke the Network object with its own input tensors to start the Model.
    outputs = network(inputs)
    if isinstance(outputs, list):
      sequence_output = outputs[0]
    else:
      sequence_output = outputs['sequence_output']
    sequence_output = tf.keras.layers.Dropout(rate=dropout_rate)(
        sequence_output)

    classifier = tf.keras.layers.Dense(
        num_classes,
        activation=None,
        kernel_initializer=initializer,
        name='predictions/transform/logits')
    logits = classifier(sequence_output)
    if output == 'logits':
      output_tensors = {'logits': logits}
    elif output == 'predictions':
      output_tensors = {
          'predictions': tf.keras.layers.Activation(tf.nn.log_softmax)(logits)
      }
    else:
      raise ValueError(
          ('Unknown `output` value "%s". `output` can be either "logits" or '
           '"predictions"') % output)

    if output_encoder_outputs:
      output_tensors['encoder_outputs'] = sequence_output

    # b/164516224
    # Once we've created the network using the Functional API, we call
    # super().__init__ as though we were invoking the Functional API Model
    # constructor, resulting in this object having all the properties of a model
    # created using the Functional API. Once super().__init__ is called, we
    # can assign attributes to `self` - note that all `self` assignments are
    # below this line.
    super(BertTokenClassifier, self).__init__(
        inputs=inputs, outputs=output_tensors, **kwargs)

    self._network = network
    config_dict = {
        'network': network,
        'num_classes': num_classes,
        'initializer': initializer,
        'output': output,
        'output_encoder_outputs': output_encoder_outputs
    }

    # We are storing the config dict as a namedtuple here to ensure checkpoint
    # compatibility with an earlier version of this model which did not track
    # the config dict attribute. TF does not track immutable attrs which
    # do not contain Trackables, so by creating a config namedtuple instead of
    # a dict we avoid tracking it.
    config_cls = collections.namedtuple('Config', config_dict.keys())
    self._config = config_cls(**config_dict)

    self.classifier = classifier
    self.logits = logits

  @property
  def checkpoint_items(self):
    return dict(encoder=self._network)

  def get_config(self):
    return dict(self._config._asdict())

  @classmethod
  def from_config(cls, config, custom_objects=None):
    return cls(**config)
