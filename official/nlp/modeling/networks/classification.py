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

"""Classification and regression network."""
# pylint: disable=g-classes-have-attributes
import collections
import tensorflow as tf
from tensorflow.python.util import deprecation


@tf.keras.utils.register_keras_serializable(package='Text')
class Classification(tf.keras.Model):
  """Classification network head for BERT modeling.

  This network implements a simple classifier head based on a dense layer. If
  num_classes is one, it can be considered as a regression problem.

  *Note* that the network is constructed by
  [Keras Functional API](https://keras.io/guides/functional_api/).

  Args:
    input_width: The innermost dimension of the input tensor to this network.
    num_classes: The number of classes that this network should classify to. If
      equal to 1, a regression problem is assumed.
    activation: The activation, if any, for the dense layer in this network.
    initializer: The initializer for the dense layer in this network. Defaults
      to a Glorot uniform initializer.
    output: The output style for this network. Can be either `logits` or
      `predictions`.
  """

  @deprecation.deprecated(None, 'Classification as a network is deprecated. '
                          'Please use the layers.ClassificationHead instead.')
  def __init__(self,
               input_width,
               num_classes,
               initializer='glorot_uniform',
               output='logits',
               **kwargs):

    cls_output = tf.keras.layers.Input(
        shape=(input_width,), name='cls_output', dtype=tf.float32)

    logits = tf.keras.layers.Dense(
        num_classes,
        activation=None,
        kernel_initializer=initializer,
        name='predictions/transform/logits')(
            cls_output)

    if output == 'logits':
      output_tensors = logits
    elif output == 'predictions':
      policy = tf.keras.mixed_precision.global_policy()
      if policy.name == 'mixed_bfloat16':
        # b/158514794: bf16 is not stable with post-softmax cross-entropy.
        policy = tf.float32
      output_tensors = tf.keras.layers.Activation(
          tf.nn.log_softmax, dtype=policy)(
              logits)
    else:
      raise ValueError(
          ('Unknown `output` value "%s". `output` can be either "logits" or '
           '"predictions"') % output)

    super().__init__(
        inputs=[cls_output], outputs=output_tensors, **kwargs)

    # b/164516224
    # Once we've created the network using the Functional API, we call
    # super().__init__ as though we were invoking the Functional API Model
    # constructor, resulting in this object having all the properties of a model
    # created using the Functional API. Once super().__init__ is called, we
    # can assign attributes to `self` - note that all `self` assignments are
    # below this line.
    config_dict = {
        'input_width': input_width,
        'num_classes': num_classes,
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
    self.logits = logits

  def get_config(self):
    return dict(self._config._asdict())

  @classmethod
  def from_config(cls, config, custom_objects=None):
    return cls(**config)
