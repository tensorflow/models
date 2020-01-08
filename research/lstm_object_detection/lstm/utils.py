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

"""Quantization related ops for LSTM."""

from __future__ import absolute_import
from __future__ import division

import tensorflow as tf
from tensorflow.contrib import framework as contrib_framework
from tensorflow.contrib import layers as contrib_layers
from tensorflow.python.training import moving_averages


def _quant_var(
    name,
    initializer_val,
    vars_collection=tf.GraphKeys.MOVING_AVERAGE_VARIABLES,
):
  """Create an var for storing the min/max quantization range."""
  return contrib_framework.model_variable(
      name,
      shape=[],
      initializer=tf.constant_initializer(initializer_val),
      collections=[vars_collection],
      trainable=False)


def quantizable_concat(inputs,
                       axis,
                       is_training,
                       is_quantized=True,
                       default_min=0,
                       default_max=6,
                       ema_decay=0.999,
                       scope='quantized_concat'):
  """Concat replacement with quantization option.

  Allows concat inputs to share the same min max ranges,
  from experimental/gazelle/synthetic/model/tpu/utils.py.

  Args:
    inputs: list of tensors to concatenate.
    axis: dimension along which to concatenate.
    is_training: true if the graph is a training graph.
    is_quantized: flag to enable/disable quantization.
    default_min: default min value for fake quant op.
    default_max: default max value for fake quant op.
    ema_decay: the moving average decay for the quantization variables.
    scope: Optional scope for variable_scope.

  Returns:
    Tensor resulting from concatenation of input tensors
  """
  if is_quantized:
    with tf.variable_scope(scope):
      tf.logging.info('inputs: {}'.format(inputs))
      for t in inputs:
        tf.logging.info(t)

      min_var = _quant_var('min', default_min)
      max_var = _quant_var('max', default_max)
      if not is_training:
        # If we are building an eval graph just use the values in the variables.
        quant_inputs = [
            tf.fake_quant_with_min_max_vars(t, min_var, max_var) for t in inputs
        ]
        tf.logging.info('min_val: {}'.format(min_var))
        tf.logging.info('max_val: {}'.format(max_var))
      else:
        concat_tensors = tf.concat(inputs, axis=axis)
        tf.logging.info('concat_tensors: {}'.format(concat_tensors))
        # TFLite requires that 0.0 is always in the [min; max] range.
        range_min = tf.minimum(
            tf.reduce_min(concat_tensors), 0.0, name='SafeQuantRangeMin')
        range_max = tf.maximum(
            tf.reduce_max(concat_tensors), 0.0, name='SafeQuantRangeMax')
        # Otherwise we need to keep track of the moving averages of the min and
        # of the elements of the input tensor max.
        min_val = moving_averages.assign_moving_average(
            min_var,
            range_min,
            ema_decay,
            name='AssignMinEma')
        max_val = moving_averages.assign_moving_average(
            max_var,
            range_max,
            ema_decay,
            name='AssignMaxEma')
        tf.logging.info('min_val: {}'.format(min_val))
        tf.logging.info('max_val: {}'.format(max_val))
        quant_inputs = [
            tf.fake_quant_with_min_max_vars(t, min_val, max_val) for t in inputs
        ]
      tf.logging.info('quant_inputs: {}'.format(quant_inputs))
      outputs = tf.concat(quant_inputs, axis=axis)
      tf.logging.info('outputs: {}'.format(outputs))
  else:
    outputs = tf.concat(inputs, axis=axis)
  return outputs


def quantizable_separable_conv2d(inputs,
                                 num_outputs,
                                 kernel_size,
                                 is_quantized=True,
                                 depth_multiplier=1,
                                 stride=1,
                                 activation_fn=tf.nn.relu6,
                                 normalizer_fn=None,
                                 scope=None):
  """Quantization friendly backward compatible separable conv2d.

  This op has the same API is separable_conv2d. The main difference is that an
  additional BiasAdd is manually inserted after the depthwise conv, such that
  the depthwise bias will not have name conflict with pointwise bias. The
  motivation of this op is that quantization script need BiasAdd in order to
  recognize the op, in which a native call to separable_conv2d do not create
  for the depthwise conv.

  Args:
    inputs: A tensor of size [batch_size, height, width, channels].
    num_outputs: The number of pointwise convolution output filters. If is
      None, then we skip the pointwise convolution stage.
    kernel_size: A list of length 2: [kernel_height, kernel_width] of the
      filters. Can be an int if both values are the same.
    is_quantized: flag to enable/disable quantization.
    depth_multiplier: The number of depthwise convolution output channels for
      each input channel. The total number of depthwise convolution output
      channels will be equal to num_filters_in * depth_multiplier.
    stride: A list of length 2: [stride_height, stride_width], specifying the
      depthwise convolution stride. Can be an int if both strides are the same.
    activation_fn: Activation function. The default value is a ReLU function.
      Explicitly set it to None to skip it and maintain a linear activation.
    normalizer_fn: Normalization function to use instead of biases.
    scope: Optional scope for variable_scope.

  Returns:
    Tensor resulting from concatenation of input tensors
  """
  if is_quantized:
    outputs = contrib_layers.separable_conv2d(
        inputs,
        None,
        kernel_size,
        depth_multiplier=depth_multiplier,
        stride=1,
        activation_fn=None,
        normalizer_fn=None,
        biases_initializer=None,
        scope=scope)
    outputs = contrib_layers.bias_add(
        outputs, trainable=True, scope='%s_bias' % scope)
    outputs = contrib_layers.conv2d(
        outputs,
        num_outputs, [1, 1],
        activation_fn=activation_fn,
        stride=stride,
        normalizer_fn=normalizer_fn,
        scope=scope)
  else:
    outputs = contrib_layers.separable_conv2d(
        inputs,
        num_outputs,
        kernel_size,
        depth_multiplier=depth_multiplier,
        stride=stride,
        activation_fn=activation_fn,
        normalizer_fn=normalizer_fn,
        scope=scope)
  return outputs


def quantize_op(inputs,
                is_training=True,
                is_quantized=True,
                default_min=0,
                default_max=6,
                ema_decay=0.999,
                scope='quant'):
  """Inserts a fake quantization op after inputs.

  Args:
    inputs: A tensor of size [batch_size, height, width, channels].
    is_training: true if the graph is a training graph.
    is_quantized: flag to enable/disable quantization.
    default_min: default min value for fake quant op.
    default_max: default max value for fake quant op.
    ema_decay: the moving average decay for the quantization variables.
    scope: Optional scope for variable_scope.

  Returns:
    Tensor resulting from quantizing the input tensors.
  """
  if is_quantized:
    with tf.variable_scope(scope):
      min_var = _quant_var('min', default_min)
      max_var = _quant_var('max', default_max)
      if is_training:
        # TFLite requires that 0.0 is always in the [min; max] range.
        range_min = tf.minimum(tf.reduce_min(inputs), 0.0, 'SafeQuantRangeMin')
        range_max = tf.maximum(tf.reduce_max(inputs), 0.0, 'SafeQuantRangeMax')
        min_val = moving_averages.assign_moving_average(
            min_var, range_min, ema_decay, name='AssignMinEma')
        max_val = moving_averages.assign_moving_average(
            max_var, range_max, ema_decay, name='AssignMaxEma')
        inputs = tf.fake_quant_with_min_max_vars(inputs, min_val, max_val)
      else:
        inputs = tf.fake_quant_with_min_max_vars(inputs, min_var, max_var)
  return inputs
