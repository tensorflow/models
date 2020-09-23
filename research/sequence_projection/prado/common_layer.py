# Copyright 2020 The TensorFlow Authors All Rights Reserved.
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
# python3
"""Common layer creator."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import tensorflow.compat.v1 as tf

from tensorflow.python.training import moving_averages  # pylint: disable=g-direct-tensorflow-import


class CommonLayers(object):
  """A base class that defines TfLite compatible NN layers."""

  def __init__(self,
               mode,
               regularizer_scale=0.0,
               weights_initializer=tf.keras.initializers.glorot_uniform(),
               quantization_enabled=True):
    """PoDLayers constructor.

    Args:
      mode: Graph creation mode.
      regularizer_scale: Optional regularizer for the weights.
      weights_initializer: Optional initializer for the weights.
      quantization_enabled: Enables quantization of weights and activation in
        the DNN.
    """
    self._mode = mode
    self._regularizer_scale = regularizer_scale
    self._weights_initializer = weights_initializer
    self._quantization_enabled = quantization_enabled
    # Batch normalization is the default normalization scheme.
    self._normalizer = self.batch_normalization
    self._moment_fn = None

  def qrange_sigmoid(self, tensor):
    """Quantize the tensor in sigmoid range (0.0, 1.0)."""
    return tf.fake_quant_with_min_max_args(
        tensor, 0.0, 1.0) if self._quantization_enabled else tensor

  def qrange_tanh(self, tensor):
    """Quantize the tensor in tanh range (-1.0, 1.0)."""
    return tf.fake_quant_with_min_max_args(
        tensor, -1.0, 1.0) if self._quantization_enabled else tensor

  def _quantized_tanh(self, tensor):
    """Apply tanh op and quantize in the range (-1.0, 1.0)."""
    return self.qrange_tanh(tf.tanh(tensor))

  def _quantized_sigmoid(self, tensor):
    """Apply sigmoid op and quantize in the range (0.0, 1.0)."""
    return self.qrange_sigmoid(tf.sigmoid(tensor))

  def set_moment_fn(self, moment_fn):
    """Set a moment function that will be used by batch norm."""
    self._moment_fn = moment_fn

  def set_regularizer_scale(self, regularizer_scale):
    """Override / set a new weights regularizer scale."""
    self._regularizer_scale = regularizer_scale

  def set_variable_length_moment_fn(self, sequence_length, max_sequence_length):
    """Set variable length moment function for use in batch norm.

    Args:
      sequence_length: An vector of sequence lengths.
      max_sequence_length: Padding length for the batch.

    Returns:
      Returns sequence mask.
    """
    mask = tf.sequence_mask(
        sequence_length, maxlen=max_sequence_length, dtype=tf.float32)
    mask = tf.expand_dims(mask, 2)

    mask_r4 = tf.expand_dims(mask, 3)
    mask_r2 = tf.reshape(mask, [-1, 1])
    inverse_numsteps = tf.math.reciprocal(tf.reduce_sum(mask))

    def _varlen_moment_fn(input_tensor, axes):
      """Moment function to use with batch normalization."""
      input_tensor_shape = input_tensor.get_shape().as_list()
      input_tensor_rank = len(input_tensor_shape)
      if input_tensor_rank == 2:
        input_tensor = mask_r2 * input_tensor
      elif input_tensor_rank == 4:
        assert input_tensor_shape[2] == 1
        input_tensor = mask_r4 * input_tensor
      else:
        assert False, "Supports rank2 and rank4 tensors."
      ex = tf.reduce_sum(input_tensor, axis=axes) * inverse_numsteps
      exx = tf.reduce_sum(
          input_tensor * input_tensor, axis=axes) * inverse_numsteps
      return ex, (exx - ex * ex)

    self._moment_fn = _varlen_moment_fn
    return mask

  def batch_normalization(self, input_tensor, decay=0.999):
    """Add batch normalization network structure after input_tensor.

    It performs batch normalization of the input tensor. This routine is
    verified to works for rank 4 or 2 tensors.

    Args:
      input_tensor: Input tensor that needs to be normalized.
      decay: Moving average decay

    Returns:
      A tensor that is normalized.
    """
    input_tensor_shape = input_tensor.get_shape().as_list()
    nstat = input_tensor_shape[-1]
    reduce_dims = list(range(len(input_tensor_shape) - 1))

    with tf.variable_scope(name_or_scope=None, default_name="batch_norm"):
      offset = tf.get_variable(
          "offset",
          shape=[nstat],
          initializer=tf.zeros_initializer,
          trainable=True)
      scale = tf.get_variable(
          "scale",
          shape=[nstat],
          initializer=tf.ones_initializer,
          trainable=True)
      moving_mean = tf.get_variable(
          "moving_mean",
          shape=[nstat],
          initializer=tf.zeros_initializer,
          trainable=False)
      moving_var = tf.get_variable(
          "moving_variance",
          shape=[nstat],
          initializer=tf.ones_initializer,
          trainable=False)

      if self._mode == tf.estimator.ModeKeys.TRAIN:
        # During training compute summay stats, update them to moving average
        # variables and use the summary stas for batch normalization.
        moment_fn = self._moment_fn or tf.nn.moments
        mean_mom, var_mom = moment_fn(input_tensor, reduce_dims)
        with tf.control_dependencies([
            moving_averages.assign_moving_average(
                moving_mean, mean_mom, decay, name="mean_op"),
            moving_averages.assign_moving_average(
                moving_var, var_mom, decay, name="variance_op")
        ]):
          tensor = tf.nn.batch_normalization(
              input_tensor,
              mean_mom,
              var_mom,
              offset,
              scale,
              1e-9,
              name="batch_norm_core")
      else:
        # During eval/inference use the moving average variable for batch
        # normalization. The variables would be frozen to constants before
        # saving graph.
        tensor = tf.nn.batch_normalization(
            input_tensor,
            moving_mean,
            moving_var,
            offset,
            scale,
            1e-9,
            name="batch_norm_core")
    return tensor

  def get_quantization_ranges(self, tensor, ema_decay=0.99):
    """Perform fake quantization of the tensor.

    The method computes ranges for quantization by first computing the
    batch min/max and then computing a moving average of the min/max across
    batches. The moving average of min/max is used for quantization during
    inference. During training the batch min/maxs are used directly.

    Args:
      tensor: Input tensor that needs to be quantized.
      ema_decay: Moving average decay

    Returns:
      Min/Max for fake quantization.
    """
    # If neither quantization is enabled, nor are we calculating ranges for
    # floating point models, this method is a no-op.
    if not self._quantization_enabled:
      return None, None

    # Calculate min/max for the tensor.
    min_var = tf.get_variable("min", initializer=0.0, trainable=False)
    max_var = tf.get_variable("max", initializer=1.0, trainable=False)

    if self._mode == tf.estimator.ModeKeys.TRAIN:
      # During training estimate moving average for min/max. Use the min/max
      # values directly for quantization.
      ops = []
      batch_min = tf.reduce_min(tensor, name="BatchMin")
      # Toco expects 0.0 to be part of the quantization range.
      batch_min = tf.minimum(batch_min, 0.0)
      ops.append(
          moving_averages.assign_moving_average(min_var, batch_min, ema_decay))

      batch_max = tf.reduce_max(tensor, name="BatchMax")
      # Toco expects 0.0 to be part of the quantization range.
      batch_max = tf.maximum(batch_max, 0.0)
      ops.append(
          moving_averages.assign_moving_average(max_var, batch_max, ema_decay))

      with tf.control_dependencies(ops):
        return tf.identity(batch_min), tf.identity(batch_max)
    else:
      # During inference/eval use the moving average min/maxs for
      # quantization.
      return min_var, max_var

  def quantization(self, tensor, ema_decay=0.99, num_bits=8):
    """Perform fake quantization of the tensor.

    The method performs fake quantization of the tensor by first computing the
    batch min/max and then computing a moving average of the min/max across
    batches. The moving average of min/max is used for quantization during
    inference. During training the batch min/maxs are used directly.

    Args:
      tensor: Input tensor that needs to be quantized.
      ema_decay: Moving average decay
      num_bits: Number of bits used for quantization

    Returns:
      Quantized tensor.
    """
    with tf.variable_scope(
        name_or_scope=None, default_name="MovingAvgQuantize"):
      min_tensor, max_tensor = self.get_quantization_ranges(tensor, ema_decay)
      if min_tensor is None or max_tensor is None:
        return tensor
      else:
        return tf.fake_quant_with_min_max_vars(
            tensor, min_tensor, max_tensor, num_bits=num_bits)

  def _weight_quantization(self, tensor, num_bits=8):
    """Quantize weights when enabled."""
    if not self._quantization_enabled:
      return tensor

    # For infer mode, toco computes the min/max from the weights offline to
    # quantize it. During train/eval this is computed from the current value
    # in the session by the graph itself.
    modes = set([tf.estimator.ModeKeys.TRAIN, tf.estimator.ModeKeys.EVAL])
    if self._mode in modes:
      batch_min = tf.reduce_min(tensor, name="BatchMin")
      # Toco expects 0.0 to be part of the quantization range.
      batch_min = tf.minimum(batch_min, 0.0)

      batch_max = tf.reduce_max(tensor, name="BatchMax")
      # Toco expects 0.0 to be part of the quantization range.
      batch_max = tf.maximum(batch_max, 0.0)

      return tf.fake_quant_with_min_max_vars(
          tensor, batch_min, batch_max, num_bits=num_bits)
    else:
      return tensor

  def _get_weight(self, shape, num_bits=8):
    """Return a weight variable for the given shape.

    The disable_pruning flag overrides the global pruning_obj object. When set
    to True, the returned weight tensor is not pruned.
    Args:
      shape: Shape of the weight tensor
      num_bits: Number of bits to use for the variable.

    Returns:
      Quantized tensor with the mask and threshold variables needed for pruning.

    """
    weight = tf.get_variable(
        "weight", shape, initializer=self._weights_initializer)
    if self._regularizer_scale > 0.0:
      reg_loss = tf.nn.l2_loss(weight) * tf.convert_to_tensor(
          self._regularizer_scale)
      tf.losses.add_loss(
          reg_loss, loss_collection=tf.GraphKeys.REGULARIZATION_LOSSES)
    return self._weight_quantization(weight, num_bits=num_bits)

  def _get_bias(self, shape):
    weight = tf.get_variable("bias", shape, initializer=tf.zeros_initializer())
    if self._regularizer_scale > 0.0:
      reg_loss = tf.nn.l2_loss(weight) * tf.convert_to_tensor(
          self._regularizer_scale)
      tf.losses.add_loss(
          reg_loss, loss_collection=tf.GraphKeys.REGULARIZATION_LOSSES)
    return weight

  def zero_beyond_sequence_length(self, sequence_length, gate):
    """Generate a binary mask for the sequence based on the timestep's validity.

    Args:
      sequence_length: The sequence length tensor of [batch size] elements.
      gate: A gate tensor used by the QuasiRNN cell to infer shape from it.

    Returns:
      Mask tensor with one for valid time and zero for invalid timestep.
    """
    mask = tf.sequence_mask(
        sequence_length, maxlen=tf.shape(gate)[1], dtype=tf.float32)
    return tf.expand_dims(mask, 2)

  def _convolution2d(self,
                     inputs,
                     kernel_size,
                     filters,
                     stride,
                     padding,
                     dilations=None,
                     weight_mask=None,
                     scope="convolution2d"):
    """Linear part of the convolution layer."""
    if isinstance(stride, int):
      strides = [1, stride, stride, 1]
    else:
      if not isinstance(stride, list) or len(stride) != 2:
        raise ValueError("`Stride` should be an integer or a list of length 2")
      strides = [1, stride[0], stride[1], 1]
    if dilations is not None:
      if not isinstance(dilations, list) or len(dilations) != 2:
        raise ValueError("`Dilations` should be an integer list of length 2")
      dilations = [1, dilations[0], dilations[1], 1]
    else:
      dilations = [1, 1, 1, 1]

    with tf.variable_scope(name_or_scope=None, default_name=scope):
      input_channels = inputs.get_shape().as_list()[-1]
      kernel_shape = kernel_size + [input_channels, filters]
      weight = self._get_weight(kernel_shape)
      if weight_mask is not None:
        # Tensor multiply for disabling backprop
        weight = weight * weight_mask
      bias = self._get_bias([filters])

      features = tf.nn.conv2d(
          inputs, weight, strides, padding, dilations=dilations)
      return tf.nn.bias_add(features, bias)

  def convolution2d(self,
                    inputs,
                    kernel_size,
                    filters,
                    scope="convolution2d",
                    stride=1,
                    padding="SAME",
                    dilations=None,
                    weight_mask=None,
                    activation=tf.nn.relu,
                    normalization=True):
    """Creates a 2d convolution layer.

    Performs batch normalization to the tensor pre activation and fake
    quantization post activation.

    Args:
      inputs: Input tensor, that is expected to be a rank 4 tensor.
      kernel_size: 2D convolution kernel size (2 tuple).
      filters: Number of output channels (integer).
      scope: A string that would be used as variable scope for the layer.
      stride: Convolution stride, can be a constant or a 2 tuple.
      padding: Padding to use for the convolution.
      dilations: tuple of size 2 specifying the dilation rates for input height
        and width respectively. Refer to tf.nn.conv2d API for more details.
      weight_mask: A floating point numpy array or constant tensor mask to turn
        off weights in the convolution kernel.
      activation: Activation function to be used, Relu is used by default.
      normalization: A boolean flag indicating if batchnorm should be performed.

    Returns:
      Tensor result of the convolution layer.

    Raises:
      ValueError: If inputs is not a rank 4 tensor
      ValueError: If kernel_size is not a list or tuple of length 2
    """
    if len(inputs.get_shape().as_list()) != 4:
      raise ValueError("`inputs` should be a rank 4 tensor. "
                       "Was: {}.".format(len(inputs.get_shape().as_list())))

    kernel_size = list(kernel_size)
    if len(kernel_size) != 2:
      raise ValueError("`kernel_size` should be a tuple or list of length 2. "
                       "Was: {}.".format(kernel_size))

    features_rank4 = self._convolution2d(
        inputs,
        kernel_size,
        filters,
        stride,
        padding,
        dilations,
        weight_mask=weight_mask,
        scope=scope)

    if normalization and self._normalizer:
      features_rank4 = self._normalizer(features_rank4)
    if activation is not None:
      features_rank4 = activation(features_rank4)

    return self.quantization(features_rank4)

  def _fully_connected(self,
                       features,
                       output_size,
                       scope="fully_connected",
                       use_bias=True):
    """Performs fully connected operation."""
    with tf.variable_scope(name_or_scope=None, default_name=scope):
      weight = self._get_weight(
          [features.get_shape().as_list()[-1], output_size])
      bias = self._get_bias([output_size])
      features = tf.matmul(features, weight)
      return tf.nn.bias_add(features, bias) if use_bias else features

  def fully_connected(self,
                      features,
                      output_size,
                      scope="fully_connected",
                      activation=tf.nn.relu,
                      normalization=True,
                      use_bias=True):
    """Creates a fully connected layer.

    Performs batch normalization to the tensor pre activation and fake
    quantization post activation.

    Args:
      features: Input features to the fully connected layer.
      output_size: Number of output features.
      scope: A variable scope for the connected layer.
      activation: activation function to be used, Relu is used by default.
      normalization: A flag indicating if batchnorm should be performed.
      use_bias: If True, bias is added to the result

    Returns:
      Tensor result of the fully connected layer.

    Raises:
      ValueError: If last dimension of features is dynamic (shape = None).
    """
    input_shape = features.get_shape().as_list()
    if not input_shape[-1]:
      raise ValueError("Last dimension of features should be static")

    need_reshape = len(input_shape) > 2
    input_tensor = features
    if need_reshape:
      features = tf.reshape(features, [-1, input_shape[-1]])

    features = self._fully_connected(
        features, output_size, scope=scope, use_bias=use_bias)

    if normalization and self._normalizer:
      features = self._normalizer(features)

    if activation:
      # Batch normalization is done pre activation as suggested in the original
      # paper. Quantization is done post activation because the range will
      # change after applying the squashing function.
      features = activation(features)
    features = self.quantization(features)
    if not need_reshape:
      return features
    else:
      # The fully connected layer changes the last dimension to output_size.
      # If a reshape was done before applying the fully connected layer, change
      # it back to the right rank. If the input dimensions are known use the
      # static shape otherwise use the shape tensor.
      if sum([val is None for val in input_shape]) <= 1:
        # Just one dynamic shape, we can reshape with -1
        output_shape = [-1 if val is None else val for val in input_shape]
      else:
        input_shape_tensor = tf.shape(input_tensor)
        output_shape = [
            shape or input_shape_tensor[index]
            for index, shape in enumerate(input_shape)
        ]
      output_shape[-1] = output_size
      return tf.reshape(features, output_shape)
