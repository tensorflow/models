# Copyright 2025 The TensorFlow Authors. All Rights Reserved.
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

"""Backbone registers and factory method.

One can regitered a new backbone model by the following two steps:

1 Import the factory and register the build in the backbone file.
2 Import the backbone class and add a build in __init__.py.

```
# my_backbone.py

from modeling.backbones import factory

class MyBackbone():
  ...

@factory.register_backbone_builder('my_backbone')
def build_my_backbone():
  return MyBackbone()

# backbones/__init__.py adds import
from modeling.backbones.my_backbone import MyBackbone
```

If one wants the MyBackbone class to be used only by those binary
then don't imported the backbone module in backbones/__init__.py, but import it
in place that uses it.


"""
from typing import Sequence, Union

import tensorflow as tf, tf_keras

from official.core import registry
from official.modeling import hyperparams


_REGISTERED_BACKBONE_CLS = {}


def register_backbone_builder(key: str):
  """Decorates a builder of backbone class.

  The builder should be a Callable (a class or a function).
  This decorator supports registration of backbone builder as follows:

  ```
  class MyBackbone(tf_keras.Model):
    pass

  @register_backbone_builder('mybackbone')
  def builder(input_specs, config, l2_reg):
    return MyBackbone(...)

  # Builds a MyBackbone object.
  my_backbone = build_backbone_3d(input_specs, config, l2_reg)
  ```

  Args:
    key: A `str` of key to look up the builder.

  Returns:
    A callable for using as class decorator that registers the decorated class
    for creation from an instance of task_config_cls.
  """
  return registry.register(_REGISTERED_BACKBONE_CLS, key)


def build_backbone(input_specs: Union[tf_keras.layers.InputSpec,
                                      Sequence[tf_keras.layers.InputSpec]],
                   backbone_config: hyperparams.Config,
                   norm_activation_config: hyperparams.Config,
                   l2_regularizer: tf_keras.regularizers.Regularizer = None,
                   **kwargs) -> tf_keras.Model:  # pytype: disable=annotation-type-mismatch  # typed-keras
  """Builds backbone from a config.

  Args:
    input_specs: A (sequence of) `tf_keras.layers.InputSpec` of input.
    backbone_config: A `OneOfConfig` of backbone config.
    norm_activation_config: A config for normalization/activation layer.
    l2_regularizer: A `tf_keras.regularizers.Regularizer` object. Default to
      None.
    **kwargs: Additional keyword args to be passed to backbone builder.

  Returns:
    A `tf_keras.Model` instance of the backbone.
  """
  backbone_builder = registry.lookup(_REGISTERED_BACKBONE_CLS,
                                     backbone_config.type)

  return backbone_builder(
      input_specs=input_specs,
      backbone_config=backbone_config,
      norm_activation_config=norm_activation_config,
      l2_regularizer=l2_regularizer,
      **kwargs)
