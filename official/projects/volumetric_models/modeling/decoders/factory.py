# Copyright 2024 The TensorFlow Authors. All Rights Reserved.
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

"""Decoder registers and factory method.

One can register a new decoder model by the following two steps:

1 Import the factory and register the build in the decoder file.
2 Import the decoder class and add a build in __init__.py.

```
# my_decoder.py

from modeling.decoders import factory

class MyDecoder():
  ...

@factory.register_decoder_builder('my_decoder')
def build_my_decoder():
  return MyDecoder()

# decoders/__init__.py adds import
from modeling.decoders.my_decoder import MyDecoder
```

If one wants the MyDecoder class to be used only by those binary
then don't imported the decoder module in decoders/__init__.py, but import it
in place that uses it.
"""
from typing import Union, Mapping, Optional

# Import libraries

import tensorflow as tf, tf_keras

from official.core import registry
from official.modeling import hyperparams

_REGISTERED_DECODER_CLS = {}


def register_decoder_builder(key: str):
  """Decorates a builder of decoder class.

  The builder should be a Callable (a class or a function).
  This decorator supports registration of decoder builder as follows:

  ```
  class MyDecoder(tf_keras.Model):
    pass

  @register_decoder_builder('mydecoder')
  def builder(input_specs, config, l2_reg):
    return MyDecoder(...)

  # Builds a MyDecoder object.
  my_decoder = build_decoder_3d(input_specs, config, l2_reg)
  ```

  Args:
    key: A `str` of key to look up the builder.

  Returns:
    A callable for using as class decorator that registers the decorated class
    for creation from an instance of task_config_cls.
  """
  return registry.register(_REGISTERED_DECODER_CLS, key)


@register_decoder_builder('identity')
def build_identity(
    input_specs: Optional[Mapping[str, tf.TensorShape]] = None,
    model_config: Optional[hyperparams.Config] = None,
    l2_regularizer: Optional[tf_keras.regularizers.Regularizer] = None) -> None:
  del input_specs, model_config, l2_regularizer  # Unused by identity decoder.
  return None


def build_decoder(
    input_specs: Mapping[str, tf.TensorShape],
    model_config: hyperparams.Config,
    l2_regularizer: tf_keras.regularizers.Regularizer = None,
    **kwargs) -> Union[None, tf_keras.Model, tf_keras.layers.Layer]:  # pytype: disable=annotation-type-mismatch  # typed-keras
  """Builds decoder from a config.

  Args:
    input_specs: A `dict` of input specifications. A dictionary consists of
      {level: TensorShape} from a backbone.
    model_config: A `OneOfConfig` of model config.
    l2_regularizer: A `tf_keras.regularizers.Regularizer` object. Default to
      None.
    **kwargs: Additional keyword args to be passed to decoder builder.

  Returns:
    An instance of the decoder.
  """
  decoder_builder = registry.lookup(_REGISTERED_DECODER_CLS,
                                    model_config.decoder.type)

  return decoder_builder(
      input_specs=input_specs,
      model_config=model_config,
      l2_regularizer=l2_regularizer,
      **kwargs)
