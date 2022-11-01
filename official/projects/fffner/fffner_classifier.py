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

"""BERT cls-token classifier."""
# pylint: disable=g-classes-have-attributes
import collections
import tensorflow as tf

from official.nlp.modeling import layers


@tf.keras.utils.register_keras_serializable(package='Text')
class FFFNerClassifier(tf.keras.Model):
  """Classifier model based on a BERT-style transformer-based encoder.

  This is an implementation of the network structure surrounding a transformer
  encoder as described in "BERT: Pre-training of Deep Bidirectional Transformers
  for Language Understanding" (https://arxiv.org/abs/1810.04805).

  The BertClassifier allows a user to pass in a transformer stack, and
  instantiates a classification network based on the passed `num_classes`
  argument. If `num_classes` is set to 1, a regression network is instantiated.

  *Note* that the model is constructed by
  [Keras Functional API](https://keras.io/guides/functional_api/).

  Args:
    network: A transformer network. This network should output a sequence output
      and a classification output. Furthermore, it should expose its embedding
      table via a "get_embedding_table" method.
    num_classes: Number of classes to predict from the classification network.
    initializer: The initializer (if any) to use in the classification networks.
      Defaults to a Glorot uniform initializer.
    dropout_rate: The dropout probability of the cls head.
    use_encoder_pooler: Whether to use the pooler layer pre-defined inside the
      encoder.
    head_name: Name of the classification head.
    cls_head: (Optional) The layer instance to use for the classifier head.
      It should take in the output from network and produce the final logits.
      If set, the arguments ('num_classes', 'initializer', 'dropout_rate',
      'use_encoder_pooler', 'head_name') will be ignored.
  """

  def __init__(self,
               network,
               num_classes_1,
               num_classes_2,
               initializer='glorot_uniform',
               dropout_rate=0.1,
               use_encoder_pooler=True,
               head_name_1='fffner_prediction_1',
               head_name_2='fffner_prediction_2',
               cls_head=None,
               **kwargs):
    self.num_classes_1 = num_classes_1
    self.num_classes_2 = num_classes_2
    self.head_name_1 = head_name_1
    self.head_name_2 = head_name_2
    self.initializer = initializer
    self.use_encoder_pooler = use_encoder_pooler
    assert use_encoder_pooler and not cls_head, "Customized pooling & classification function is used"

    # We want to use the inputs of the passed network as the inputs to this
    # Model. To do this, we need to keep a handle to the network inputs for use
    # when we construct the Model object at the end of init.
    inputs = network.inputs

    outputs = network(inputs)
    if isinstance(outputs, list):
        cls_inputs = outputs[1]
    else:
        cls_inputs = outputs['pooled_output']
    cls_inputs = tf.keras.layers.Dropout(rate=dropout_rate)(cls_inputs)

    classifier_1 = layers.ClassificationHead(
        inner_dim=0 if use_encoder_pooler else cls_inputs.shape[-1],
        num_classes=num_classes_1,
        initializer=initializer,
        dropout_rate=dropout_rate,
        name=head_name_1)
    classifier_2 = layers.ClassificationHead(
        inner_dim=0 if use_encoder_pooler else cls_inputs.shape[-1],
        num_classes=num_classes_2,
        initializer=initializer,
        dropout_rate=dropout_rate,
        name=head_name_2)


    predictions_1 = classifier_1(cls_inputs[:, 0, :])
    predictions_2 = classifier_2(cls_inputs[:, 1, :])

    # b/164516224
    # Once we've created the network using the Functional API, we call
    # super().__init__ as though we were invoking the Functional API Model
    # constructor, resulting in this object having all the properties of a model
    # created using the Functional API. Once super().__init__ is called, we
    # can assign attributes to `self` - note that all `self` assignments are
    # below this line.
    super(FFFNerClassifier, self).__init__(
        inputs=inputs, outputs=[predictions_1, predictions_2], **kwargs)
    self._network = network
    self._cls_head = cls_head

    config_dict = self._make_config_dict()
    # We are storing the config dict as a namedtuple here to ensure checkpoint
    # compatibility with an earlier version of this model which did not track
    # the config dict attribute. TF does not track immutable attrs which
    # do not contain Trackables, so by creating a config namedtuple instead of
    # a dict we avoid tracking it.
    config_cls = collections.namedtuple('Config', config_dict.keys())
    self._config = config_cls(**config_dict)
    self.classifier_1 = classifier_1
    self.classifier_2 = classifier_2

  @property
  def checkpoint_items(self):
    items = dict(encoder=self._network)
    if hasattr(self.classifier_1, 'checkpoint_items'):
      for key, item in self.classifier_1.checkpoint_items.items():
        items['.'.join([self.classifier_1.name, key])] = item
    if hasattr(self.classifier_2, 'checkpoint_items'):
      for key, item in self.classifier_2.checkpoint_items.items():
        items['.'.join([self.classifier_2.name, key])] = item
    return items

  def get_config(self):
    return dict(self._config._asdict())

  @classmethod
  def from_config(cls, config, custom_objects=None):
    return cls(**config)

  def _make_config_dict(self):
    return {
        'network': self._network,
        'num_classes_1': self.num_classes_1,
        'num_classes_2': self.num_classes_2,
        'head_name_1': self.head_name_1,
        'head_name_2': self.head_name_2,
        'initializer': self.initializer,
        'use_encoder_pooler': self.use_encoder_pooler,
        'cls_head': self._cls_head,
    }
