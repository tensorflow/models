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

r"""Common library for API docs builder."""

import tensorflow as tf
from tensorflow_docs.api_generator import doc_controls


def hide_module_model_and_layer_methods():
  """Hide methods and properties defined in the base classes of Keras layers.

  We hide all methods and properties of the base classes, except:
  - `__init__` is always documented.
  - `call` is always documented, as it can carry important information for
    complex layers.
  """
  module_contents = list(tf.Module.__dict__.items())
  model_contents = list(tf.keras.Model.__dict__.items())
  layer_contents = list(tf.keras.layers.Layer.__dict__.items())

  for name, obj in module_contents + layer_contents + model_contents:
    if name == '__init__':
      # Always document __init__.
      continue

    if name == 'call':
      # Always document `call`.
      if hasattr(obj, doc_controls._FOR_SUBCLASS_IMPLEMENTERS):  # pylint: disable=protected-access
        delattr(obj, doc_controls._FOR_SUBCLASS_IMPLEMENTERS)  # pylint: disable=protected-access
      continue

    # Otherwise, exclude from documentation.
    if isinstance(obj, property):
      obj = obj.fget

    if isinstance(obj, (staticmethod, classmethod)):
      obj = obj.__func__

    try:
      doc_controls.do_not_doc_in_subclasses(obj)
    except AttributeError:
      pass
