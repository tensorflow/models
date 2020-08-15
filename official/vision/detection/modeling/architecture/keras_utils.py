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
"""Util functions to integrate with Keras internals."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function


from tensorflow.python.keras import backend

try:
  from tensorflow.python.keras.engine import keras_tensor  # pylint: disable=g-import-not-at-top,unused-import
  keras_tensor.disable_keras_tensors()
except ImportError:
  keras_tensor = None


class NoOpContextManager(object):

  def __enter__(self):
    pass

  def __exit__(self, *args):
    pass


def maybe_enter_backend_graph():
  if (keras_tensor is not None) and keras_tensor.keras_tensors_enabled():
    return NoOpContextManager()
  else:
    return backend.get_graph().as_default()
