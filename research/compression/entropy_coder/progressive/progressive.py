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

"""Code probability model used for entropy coding."""

import json

from six.moves import xrange
import tensorflow as tf

from entropy_coder.lib import blocks
from entropy_coder.model import entropy_coder_model
from entropy_coder.model import model_factory

# pylint: disable=not-callable


class BrnnPredictor(blocks.BlockBase):
  """BRNN prediction applied on one layer."""

  def __init__(self, code_depth, name=None):
    super(BrnnPredictor, self).__init__(name)

    with self._BlockScope():
      hidden_depth = 2 * code_depth

      # What is coming from the previous layer/iteration
      # is going through a regular Conv2D layer as opposed to the binary codes
      # of the current layer/iteration which are going through a masked
      # convolution.
      self._adaptation0 = blocks.RasterScanConv2D(
          hidden_depth, [7, 7], [1, 1], 'SAME',
          strict_order=True,
          bias=blocks.Bias(0), act=tf.tanh)
      self._adaptation1 = blocks.Conv2D(
          hidden_depth, [3, 3], [1, 1], 'SAME',
          bias=blocks.Bias(0), act=tf.tanh)
      self._predictor = blocks.CompositionOperator([
          blocks.LineOperator(
              blocks.RasterScanConv2DLSTM(
                  depth=hidden_depth,
                  filter_size=[1, 3],
                  hidden_filter_size=[1, 3],
                  strides=[1, 1],
                  padding='SAME')),
          blocks.Conv2D(hidden_depth, [1, 1], [1, 1], 'SAME',
                        bias=blocks.Bias(0), act=tf.tanh),
          blocks.Conv2D(code_depth, [1, 1], [1, 1], 'SAME',
                        bias=blocks.Bias(0), act=tf.tanh)
      ])

  def _Apply(self, x, s):
    # Code estimation using both:
    # - the state from the previous iteration/layer,
    # - the binary codes that are before in raster scan order.
    h = tf.concat(values=[self._adaptation0(x), self._adaptation1(s)], axis=3)

    estimated_codes = self._predictor(h)

    return estimated_codes


class LayerPrediction(blocks.BlockBase):
  """Binary code prediction for one layer."""

  def __init__(self, layer_count, code_depth, name=None):
    super(LayerPrediction, self).__init__(name)

    self._layer_count = layer_count

    # No previous layer.
    self._layer_state = None
    self._current_layer = 0

    with self._BlockScope():
      # Layers used to do the conditional code prediction.
      self._brnn_predictors = []
      for _ in xrange(layer_count):
        self._brnn_predictors.append(BrnnPredictor(code_depth))

      # Layers used to generate the input of the LSTM operating on the
      # iteration/depth domain.
      hidden_depth = 2 * code_depth
      self._state_blocks = []
      for _ in xrange(layer_count):
        self._state_blocks.append(blocks.CompositionOperator([
            blocks.Conv2D(
                hidden_depth, [3, 3], [1, 1], 'SAME',
                bias=blocks.Bias(0), act=tf.tanh),
            blocks.Conv2D(
                code_depth, [3, 3], [1, 1], 'SAME',
                bias=blocks.Bias(0), act=tf.tanh)
        ]))

      # Memory of the RNN is equivalent to the size of 2 layers of binary
      # codes.
      hidden_depth = 2 * code_depth
      self._layer_rnn = blocks.CompositionOperator([
          blocks.Conv2DLSTM(
              depth=hidden_depth,
              filter_size=[1, 1],
              hidden_filter_size=[1, 1],
              strides=[1, 1],
              padding='SAME'),
          blocks.Conv2D(hidden_depth, [1, 1], [1, 1], 'SAME',
                        bias=blocks.Bias(0), act=tf.tanh),
          blocks.Conv2D(code_depth, [1, 1], [1, 1], 'SAME',
                        bias=blocks.Bias(0), act=tf.tanh)
      ])

  def _Apply(self, x):
    assert self._current_layer < self._layer_count

    # Layer state is set to 0 when there is no previous iteration.
    if self._layer_state is None:
      self._layer_state = tf.zeros_like(x, dtype=tf.float32)

    # Code estimation using both:
    # - the state from the previous iteration/layer,
    # - the binary codes that are before in raster scan order.
    estimated_codes = self._brnn_predictors[self._current_layer](
        x, self._layer_state)

    # Compute the updated layer state.
    h = self._state_blocks[self._current_layer](x)
    self._layer_state = self._layer_rnn(h)
    self._current_layer += 1

    return estimated_codes


class ProgressiveModel(entropy_coder_model.EntropyCoderModel):
  """Progressive BRNN entropy coder model."""

  def __init__(self):
    super(ProgressiveModel, self).__init__()

  def Initialize(self, global_step, optimizer, config_string):
    if config_string is None:
      raise ValueError('The progressive model requires a configuration.')
    config = json.loads(config_string)
    if 'coded_layer_count' not in config:
      config['coded_layer_count'] = 0

    self._config = config
    self._optimizer = optimizer
    self._global_step = global_step

  def BuildGraph(self, input_codes):
    """Build the graph corresponding to the progressive BRNN model."""
    layer_depth = self._config['layer_depth']
    layer_count = self._config['layer_count']

    code_shape = input_codes.get_shape()
    code_depth = code_shape[-1].value
    if self._config['coded_layer_count'] > 0:
      prefix_depth = self._config['coded_layer_count'] * layer_depth
      if code_depth < prefix_depth:
        raise ValueError('Invalid prefix depth: {} VS {}'.format(
            prefix_depth, code_depth))
      input_codes = input_codes[:, :, :, :prefix_depth]

    code_shape = input_codes.get_shape()
    code_depth = code_shape[-1].value
    if code_depth % layer_depth != 0:
      raise ValueError(
          'Code depth must be a multiple of the layer depth: {} vs {}'.format(
              code_depth, layer_depth))
    code_layer_count = code_depth // layer_depth
    if code_layer_count > layer_count:
      raise ValueError('Input codes have too many layers: {}, max={}'.format(
          code_layer_count, layer_count))

    # Block used to estimate binary codes.
    layer_prediction = LayerPrediction(layer_count, layer_depth)

    # Block used to compute code lengths.
    code_length_block = blocks.CodeLength()

    # Loop over all the layers.
    code_length = []
    code_layers = tf.split(
        value=input_codes, num_or_size_splits=code_layer_count, axis=3)
    for k in xrange(code_layer_count):
      x = code_layers[k]
      predicted_x = layer_prediction(x)
      # Saturate the prediction to avoid infinite code length.
      epsilon = 0.001
      predicted_x = tf.clip_by_value(
          predicted_x, -1 + epsilon, +1 - epsilon)
      code_length.append(code_length_block(
          blocks.ConvertSignCodeToZeroOneCode(x),
          blocks.ConvertSignCodeToZeroOneCode(predicted_x)))
      tf.summary.scalar('code_length_layer_{:02d}'.format(k), code_length[-1])
    code_length = tf.stack(code_length)
    self.loss = tf.reduce_mean(code_length)
    tf.summary.scalar('loss', self.loss)

    # Loop over all the remaining layers just to make sure they are
    # instantiated. Otherwise, loading model params could fail.
    dummy_x = tf.zeros_like(code_layers[0])
    for _ in xrange(layer_count - code_layer_count):
      dummy_predicted_x = layer_prediction(dummy_x)

    # Average bitrate over total_line_count.
    self.average_code_length = tf.reduce_mean(code_length)

    if self._optimizer:
      optim_op = self._optimizer.minimize(self.loss,
                                          global_step=self._global_step)
      block_updates = blocks.CreateBlockUpdates()
      if block_updates:
        with tf.get_default_graph().control_dependencies([optim_op]):
          self.train_op = tf.group(*block_updates)
      else:
        self.train_op = optim_op
    else:
      self.train_op = None

  def GetConfigStringForUnitTest(self):
    s = '{\n'
    s += '"layer_depth": 1,\n'
    s += '"layer_count": 8\n'
    s += '}\n'
    return s


@model_factory.RegisterEntropyCoderModel('progressive')
def CreateProgressiveModel():
  return ProgressiveModel()
