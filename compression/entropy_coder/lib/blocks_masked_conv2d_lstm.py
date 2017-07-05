# Copyright 2017 The TensorFlow Authors All Rights Reserved.
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

"""Masked conv2d LSTM."""

import block_base
import block_util
import blocks_masked_conv2d
import blocks_lstm
import blocks_std

# pylint: disable=not-callable


class RasterScanConv2DLSTM(blocks_lstm.LSTMBase):
  """Convolutional LSTM implementation with optimizations inspired by [1].

  Note that when using the batch normalization feature, the bias initializer
  will not be used, since BN effectively cancels its effect out.

  [1] Zaremba, Sutskever, Vinyals. Recurrent Neural Network Regularization,
  2015. arxiv:1409.2329.
  """

  def __init__(self,
               depth,
               filter_size,
               hidden_filter_size,
               strides,
               padding,
               bias=blocks_lstm.LSTMBiasInit,
               initializer=block_util.RsqrtInitializer(dims=(0, 1, 2)),
               name=None):
    super(RasterScanConv2DLSTM, self).__init__([None, None, depth], name)

    with self._BlockScope():
      self._input_conv = blocks_masked_conv2d.RasterScanConv2D(
          4 * depth,
          filter_size,
          strides,
          padding,
          strict_order=False,
          bias=None,
          act=None,
          initializer=initializer,
          name='input_conv2d')

      self._hidden_conv = blocks_std.Conv2D(
          4 * depth,
          hidden_filter_size,
          [1, 1],
          'SAME',
          bias=None,
          act=None,
          initializer=initializer,
          name='hidden_conv2d')

      if bias is not None:
        self._bias = blocks_std.BiasAdd(bias, name='biases')
      else:
        self._bias = blocks_std.PassThrough()

  def _TransformInputs(self, x):
    return self._bias(self._input_conv(x))

  def _TransformHidden(self, h):
    return self._hidden_conv(h)
