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

"""BERT Question Answering model."""
# pylint: disable=g-classes-have-attributes
import collections
import tensorflow as tf

from official.projects.qat.nlp.modeling.networks import span_labeling


@tf.keras.utils.register_keras_serializable(package='Text')
class BertSpanLabelerQuantized(tf.keras.Model):
  """Span labeler model based on a BERT-style transformer-based encoder.

  This is an implementation of the network structure surrounding a transformer
  encoder as described in "BERT: Pre-training of Deep Bidirectional Transformers
  for Language Understanding" (https://arxiv.org/abs/1810.04805).

  The BertSpanLabeler allows a user to pass in a transformer encoder, and
  instantiates a span labeling network based on a single dense layer.

  *Note* that the model is constructed by
  [Keras Functional API](https://keras.io/guides/functional_api/).

  Args:
    network: A transformer network. This network should output a sequence output
      and a classification output. Furthermore, it should expose its embedding
      table via a `get_embedding_table` method.
    initializer: The initializer (if any) to use in the span labeling network.
      Defaults to a Glorot uniform initializer.
    output: The output style for this network. Can be either `logit`' or
      `predictions`.
  """

  def __init__(self,
               network,
               initializer='glorot_uniform',
               output='logits',
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

    # The input network (typically a transformer model) may get outputs from all
    # layers. When this case happens, we retrieve the last layer output.
    if isinstance(sequence_output, list):
      sequence_output = sequence_output[-1]

    # This is an instance variable for ease of access to the underlying task
    # network.
    span_labeling_quantized = span_labeling.SpanLabelingQuantized(
        input_width=sequence_output.shape[-1],
        initializer=initializer,
        output=output,
        name='span_labeling')
    start_logits, end_logits = span_labeling_quantized(sequence_output)

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

    # b/164516224
    # Once we've created the network using the Functional API, we call
    # super().__init__ as though we were invoking the Functional API Model
    # constructor, resulting in this object having all the properties of a model
    # created using the Functional API. Once super().__init__ is called, we
    # can assign attributes to `self` - note that all `self` assignments are
    # below this line.
    super().__init__(
        inputs=inputs, outputs=logits, **kwargs)
    self._network = network
    config_dict = {
        'network': network,
        'initializer': initializer,
        'output': output,
    }
    # We are storing the config dict as a namedtuple here to ensure checkpoint
    # compatibility with an earlier version of this model which did not track
    # the config dict attribute. TF does not track immutable attrs which
    # do not contain Trackables, so by creating a config namedtuple instead of
    # a dict we avoid tracking it.
    config_cls = collections.namedtuple('Config', config_dict.keys())
    self._config = config_cls(**config_dict)
    self.span_labeling = span_labeling_quantized

  @property
  def checkpoint_items(self):
    return dict(encoder=self._network)

  def get_config(self):
    return dict(self._config._asdict())

  @classmethod
  def from_config(cls, config, custom_objects=None):
    return cls(**config)
