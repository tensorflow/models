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
# pylint: disable=g-classes-have-attributes
from __future__ import absolute_import
from __future__ import division
# from __future__ import google_type_annotations
from __future__ import print_function

import tensorflow as tf

from official.nlp.modeling import networks


@tf.keras.utils.register_keras_serializable(package='Text')
class BertClassifier(tf.keras.Model):
  """Classifier model based on a BERT-style transformer-based encoder.

  This is an implementation of the network structure surrounding a transformer
  encoder as described in "BERT: Pre-training of Deep Bidirectional Transformers
  for Language Understanding" (https://arxiv.org/abs/1810.04805).

  The BertClassifier allows a user to pass in a transformer stack, and
  instantiates a classification network based on the passed `num_classes`
  argument. If `num_classes` is set to 1, a regression network is instantiated.

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
    _, cls_output = network(inputs)
    cls_output = tf.keras.layers.Dropout(rate=dropout_rate)(cls_output)

    self.classifier = networks.Classification(
        input_width=cls_output.shape[-1],
        num_classes=num_classes,
        initializer=initializer,
        output=output,
        name='classification')
    predictions = self.classifier(cls_output)

    super(BertClassifier, self).__init__(
        inputs=inputs, outputs=predictions, **kwargs)

  def get_config(self):
    return self._config

  @classmethod
  def from_config(cls, config, custom_objects=None):
    return cls(**config)
