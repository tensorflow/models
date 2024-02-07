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

"""Quantization helpers."""

import tensorflow_model_optimization as tfmot


class LayerQuantizerHelper(object):
  """Helper class that handles quantizers."""

  def __init__(self, *args, **kwargs):
    self._quantizers = {}
    self._quantizer_vars = {}
    super().__init__(*args, **kwargs)

  def _all_value_quantizer(self):
    return tfmot.quantization.keras.quantizers.AllValuesQuantizer(
        num_bits=8, per_axis=False, symmetric=False, narrow_range=False)

  def _moving_average_quantizer(self):
    return tfmot.quantization.keras.quantizers.MovingAverageQuantizer(
        num_bits=8, per_axis=False, symmetric=False, narrow_range=False)

  def _add_quantizer(self, name, all_value_quantizer=False):
    if all_value_quantizer:
      self._quantizers[name] = self._all_value_quantizer()
    else:
      self._quantizers[name] = self._moving_average_quantizer()

  def _apply_quantizer(self, name, inputs, training, **kwargs):
    return self._quantizers[name](
        inputs, training, self._quantizer_vars[name], **kwargs)

  def _build_quantizer_vars(self):
    for name in self._quantizers:
      self._quantizer_vars[name] = self._quantizers[name].build(
          tensor_shape=None, name=name, layer=self)
