# Copyright 2016 The TensorFlow Authors. All Rights Reserved.
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

"""String network description language mapping to TF-Slim calls where possible.

See vglspecs.md for detailed description.
"""

import re
from string import maketrans

import nn_ops
import shapes
import tensorflow as tf
import tensorflow.contrib.slim as slim


# Class that builds a set of ops to manipulate variable-sized images.
class VGSLSpecs(object):
  """Layers that can be built from a string definition."""

  def __init__(self, widths, heights, is_training):
    """Constructs a VGSLSpecs.

    Args:
      widths:  Tensor of size batch_size of the widths of the inputs.
      heights: Tensor of size batch_size of the heights of the inputs.
      is_training: True if the graph should be build for training.
    """
    # The string that was used to build this model.
    self.model_str = None
    # True if we are training
    self.is_training = is_training
    # Tensor for the size of the images, of size batch_size.
    self.widths = widths
    self.heights = heights
    # Overall reduction factors of this model so far for each dimension.
    # TODO(rays) consider building a graph from widths and heights instead of
    # computing a scale factor.
    self.reduction_factors = [1.0, 1.0, 1.0, 1.0]
    # List of Op parsers.
    # TODO(rays) add more Op types as needed.
    self.valid_ops = [self.AddSeries, self.AddParallel, self.AddConvLayer,
                      self.AddMaxPool, self.AddDropout, self.AddReShape,
                      self.AddFCLayer, self.AddLSTMLayer]
    # Translation table to convert unacceptable characters that may occur
    # in op strings that cannot be used as names.
    self.transtab = maketrans('(,)', '___')

  def Build(self, prev_layer, model_str):
    """Builds a network with input prev_layer from a VGSLSpecs description.

    Args:
      prev_layer: The input tensor.
      model_str:  Model definition similar to Tesseract as follows:
        ============ FUNCTIONAL OPS ============
        C(s|t|r|l|m)[{name}]<y>,<x>,<d> Convolves using a y,x window, with no
          shrinkage, SAME infill, d outputs, with s|t|r|l|m non-linear layer.
          (s|t|r|l|m) specifies the type of non-linearity:
          s = sigmoid
          t = tanh
          r = relu
          l = linear (i.e., None)
          m = softmax
        F(s|t|r|l|m)[{name}]<d> Fully-connected with s|t|r|l|m non-linearity and
          d outputs. Reduces height, width to 1. Input height and width must be
          constant.
        L(f|r|b)(x|y)[s][{name}]<n> LSTM cell with n outputs.
          f runs the LSTM forward only.
          r runs the LSTM reversed only.
          b runs the LSTM bidirectionally.
          x runs the LSTM in the x-dimension (on data with or without the
             y-dimension).
          y runs the LSTM in the y-dimension (data must have a y dimension).
          s (optional) summarizes the output in the requested dimension,
             outputting only the final step, collapsing the dimension to a
             single element.
          Examples:
          Lfx128 runs a forward-only LSTM in the x-dimension with 128
                 outputs, treating any y dimension independently.
          Lfys64 runs a forward-only LSTM in the y-dimension with 64 outputs
                 and collapses the y-dimension to 1 element.
          NOTE that Lbxsn is implemented as (LfxsnLrxsn) since the summaries
          need to be taken from opposite ends of the output
        Do[{name}] Insert a dropout layer.
        ============ PLUMBING OPS ============
        [...] Execute ... networks in series (layers).
        (...) Execute ... networks in parallel, with their output concatenated
          in depth.
        S[{name}]<d>(<a>x<b>)<e>,<f> Splits one dimension, moves one part to
          another dimension.
          Splits input dimension d into a x b, sending the high part (a) to the
          high side of dimension e, and the low part (b) to the high side of
          dimension f. Exception: if d=e=f, then then dimension d is internally
          transposed to bxa.
          Either a or b can be zero, meaning whatever is left after taking out
          the other, allowing dimensions to be of variable size.
          Eg. S3(3x50)2,3 will split the 150-element depth into 3x50, with the 3
          going to the most significant part of the width, and the 50 part
          staying in depth.
          This will rearrange a 3x50 output parallel operation to spread the 3
          output sets over width.
        Mp[{name}]<y>,<x> Maxpool the input, reducing the (y,x) rectangle to a
          single vector value.

    Returns:
      Output tensor
    """
    self.model_str = model_str
    final_layer, _ = self.BuildFromString(prev_layer, 0)
    return final_layer

  def GetLengths(self, dim=2, factor=1):
    """Returns the lengths of the batch of elements in the given dimension.

    WARNING: The returned sizes may not exactly match TF's calculation.
    Args:
      dim: dimension to get the sizes of, in [1,2]. batch, depth not allowed.
      factor: A scalar value to multiply by.

    Returns:
      The original heights/widths scaled by the current scaling of the model and
      the given factor.

    Raises:
      ValueError: If the args are invalid.
    """
    if dim == 1:
      lengths = self.heights
    elif dim == 2:
      lengths = self.widths
    else:
      raise ValueError('Invalid dimension given to GetLengths')
    lengths = tf.cast(lengths, tf.float32)
    if self.reduction_factors[dim] is not None:
      lengths = tf.div(lengths, self.reduction_factors[dim])
    else:
      lengths = tf.ones_like(lengths)
    if factor != 1:
      lengths = tf.multiply(lengths, tf.cast(factor, tf.float32))
    return tf.cast(lengths, tf.int32)

  def BuildFromString(self, prev_layer, index):
    """Adds the layers defined by model_str[index:] to the model.

    Args:
      prev_layer: Input tensor.
      index:      Position in model_str to start parsing

    Returns:
      Output tensor, next model_str index.

    Raises:
      ValueError: If the model string is unrecognized.
    """
    index = self._SkipWhitespace(index)
    for op in self.valid_ops:
      output_layer, next_index = op(prev_layer, index)
      if output_layer is not None:
        return output_layer, next_index
    if output_layer is not None:
      return output_layer, next_index
    raise ValueError('Unrecognized model string:' + self.model_str[index:])

  def AddSeries(self, prev_layer, index):
    """Builds a sequence of layers for a VGSLSpecs model.

    Args:
      prev_layer: Input tensor.
      index:      Position in model_str to start parsing

    Returns:
      Output tensor of the series, end index in model_str.

    Raises:
      ValueError: If [] are unbalanced.
    """
    if self.model_str[index] != '[':
      return None, None
    index += 1
    while index < len(self.model_str) and self.model_str[index] != ']':
      prev_layer, index = self.BuildFromString(prev_layer, index)
    if index == len(self.model_str):
      raise ValueError('Missing ] at end of series!' + self.model_str)
    return prev_layer, index + 1

  def AddParallel(self, prev_layer, index):
    """tf.concats outputs of layers that run on the same inputs.

    Args:
      prev_layer: Input tensor.
      index:      Position in model_str to start parsing

    Returns:
      Output tensor of the parallel,  end index in model_str.

    Raises:
      ValueError: If () are unbalanced or the elements don't match.
    """
    if self.model_str[index] != '(':
      return None, None
    index += 1
    layers = []
    num_dims = 0
    # Each parallel must output the same, including any reduction factor, in
    # all dimensions except depth.
    # We have to save the starting factors, so they don't get reduced by all
    # the elements of the parallel, only once.
    original_factors = self.reduction_factors
    final_factors = None
    while index < len(self.model_str) and self.model_str[index] != ')':
      self.reduction_factors = original_factors
      layer, index = self.BuildFromString(prev_layer, index)
      if num_dims == 0:
        num_dims = len(layer.get_shape())
      elif num_dims != len(layer.get_shape()):
        raise ValueError('All elements of parallel must return same num dims')
      layers.append(layer)
      if final_factors:
        if final_factors != self.reduction_factors:
          raise ValueError('All elements of parallel must scale the same')
      else:
        final_factors = self.reduction_factors
    if index == len(self.model_str):
      raise ValueError('Missing ) at end of parallel!' + self.model_str)
    return tf.concat(axis=num_dims - 1, values=layers), index + 1

  def AddConvLayer(self, prev_layer, index):
    """Add a single standard convolutional layer.

    Args:
      prev_layer: Input tensor.
      index:      Position in model_str to start parsing

    Returns:
      Output tensor, end index in model_str.
    """
    pattern = re.compile(R'(C)(s|t|r|l|m)({\w+})?(\d+),(\d+),(\d+)')
    m = pattern.match(self.model_str, index)
    if m is None:
      return None, None
    name = self._GetLayerName(m.group(0), index, m.group(3))
    width = int(m.group(4))
    height = int(m.group(5))
    depth = int(m.group(6))
    fn = self._NonLinearity(m.group(2))
    return slim.conv2d(
        prev_layer, depth, [height, width], activation_fn=fn,
        scope=name), m.end()

  def AddMaxPool(self, prev_layer, index):
    """Add a maxpool layer.

    Args:
      prev_layer: Input tensor.
      index:      Position in model_str to start parsing

    Returns:
      Output tensor, end index in model_str.
    """
    pattern = re.compile(R'(Mp)({\w+})?(\d+),(\d+)(?:,(\d+),(\d+))?')
    m = pattern.match(self.model_str, index)
    if m is None:
      return None, None
    name = self._GetLayerName(m.group(0), index, m.group(2))
    height = int(m.group(3))
    width = int(m.group(4))
    y_stride = height if m.group(5) is None else m.group(5)
    x_stride = width if m.group(6) is None else m.group(6)
    self.reduction_factors[1] *= y_stride
    self.reduction_factors[2] *= x_stride
    return slim.max_pool2d(
        prev_layer, [height, width], [y_stride, x_stride],
        padding='SAME',
        scope=name), m.end()

  def AddDropout(self, prev_layer, index):
    """Adds a dropout layer.

    Args:
      prev_layer: Input tensor.
      index:      Position in model_str to start parsing

    Returns:
      Output tensor, end index in model_str.
    """
    pattern = re.compile(R'(Do)({\w+})?')
    m = pattern.match(self.model_str, index)
    if m is None:
      return None, None
    name = self._GetLayerName(m.group(0), index, m.group(2))
    layer = slim.dropout(
        prev_layer, 0.5, is_training=self.is_training, scope=name)
    return layer, m.end()

  def AddReShape(self, prev_layer, index):
    """Reshapes the input tensor by moving each (x_scale,y_scale) rectangle to.

       the depth dimension. NOTE that the TF convention is that inputs are
       [batch, y, x, depth].

    Args:
      prev_layer: Input tensor.
      index:      Position in model_str to start parsing

    Returns:
      Output tensor, end index in model_str.
    """
    pattern = re.compile(R'(S)(?:{(\w)})?(\d+)\((\d+)x(\d+)\)(\d+),(\d+)')
    m = pattern.match(self.model_str, index)
    if m is None:
      return None, None
    name = self._GetLayerName(m.group(0), index, m.group(2))
    src_dim = int(m.group(3))
    part_a = int(m.group(4))
    part_b = int(m.group(5))
    dest_dim_a = int(m.group(6))
    dest_dim_b = int(m.group(7))
    if part_a == 0:
      part_a = -1
    if part_b == 0:
      part_b = -1
    prev_shape = tf.shape(prev_layer)
    layer = shapes.transposing_reshape(
        prev_layer, src_dim, part_a, part_b, dest_dim_a, dest_dim_b, name=name)
    # Compute scale factors.
    result_shape = tf.shape(layer)
    for i in xrange(len(self.reduction_factors)):
      if self.reduction_factors[i] is not None:
        factor1 = tf.cast(self.reduction_factors[i], tf.float32)
        factor2 = tf.cast(prev_shape[i], tf.float32)
        divisor = tf.cast(result_shape[i], tf.float32)
        self.reduction_factors[i] = tf.div(tf.multiply(factor1, factor2), divisor)
    return layer, m.end()

  def AddFCLayer(self, prev_layer, index):
    """Parse expression and add Fully Connected Layer.

    Args:
      prev_layer: Input tensor.
      index:      Position in model_str to start parsing

    Returns:
      Output tensor, end index in model_str.
    """
    pattern = re.compile(R'(F)(s|t|r|l|m)({\w+})?(\d+)')
    m = pattern.match(self.model_str, index)
    if m is None:
      return None, None
    fn = self._NonLinearity(m.group(2))
    name = self._GetLayerName(m.group(0), index, m.group(3))
    depth = int(m.group(4))
    input_depth = shapes.tensor_dim(prev_layer, 1) * shapes.tensor_dim(
        prev_layer, 2) * shapes.tensor_dim(prev_layer, 3)
    # The slim fully connected is actually a 1x1 conv, so we have to crush the
    # dimensions on input.
    # Everything except batch goes to depth, and therefore has to be known.
    shaped = tf.reshape(
        prev_layer, [-1, input_depth], name=name + '_reshape_in')
    output = slim.fully_connected(shaped, depth, activation_fn=fn, scope=name)
    # Width and height are collapsed to 1.
    self.reduction_factors[1] = None
    self.reduction_factors[2] = None
    return tf.reshape(
        output, [shapes.tensor_dim(prev_layer, 0), 1, 1, depth],
        name=name + '_reshape_out'), m.end()

  def AddLSTMLayer(self, prev_layer, index):
    """Parse expression and add LSTM Layer.

    Args:
      prev_layer: Input tensor.
      index:      Position in model_str to start parsing

    Returns:
      Output tensor, end index in model_str.
    """
    pattern = re.compile(R'(L)(f|r|b)(x|y)(s)?({\w+})?(\d+)')
    m = pattern.match(self.model_str, index)
    if m is None:
      return None, None
    direction = m.group(2)
    dim = m.group(3)
    summarize = m.group(4) == 's'
    name = self._GetLayerName(m.group(0), index, m.group(5))
    depth = int(m.group(6))
    if direction == 'b' and summarize:
      fwd = self._LSTMLayer(prev_layer, 'forward', dim, True, depth,
                            name + '_forward')
      back = self._LSTMLayer(prev_layer, 'backward', dim, True, depth,
                             name + '_reverse')
      return tf.concat(axis=3, values=[fwd, back], name=name + '_concat'), m.end()
    if direction == 'f':
      direction = 'forward'
    elif direction == 'r':
      direction = 'backward'
    else:
      direction = 'bidirectional'
    outputs = self._LSTMLayer(prev_layer, direction, dim, summarize, depth,
                              name)
    if summarize:
      # The x or y dimension is getting collapsed.
      if dim == 'x':
        self.reduction_factors[2] = None
      else:
        self.reduction_factors[1] = None
    return outputs, m.end()

  def _LSTMLayer(self, prev_layer, direction, dim, summarize, depth, name):
    """Adds an LSTM layer with the given pre-parsed attributes.

    Always maps 4-D to 4-D regardless of summarize.
    Args:
      prev_layer: Input tensor.
      direction:  'forward' 'backward' or 'bidirectional'
      dim:        'x' or 'y', dimension to consider as time.
      summarize:  True if we are to return only the last timestep.
      depth:      Output depth.
      name:       Some string naming the op.

    Returns:
      Output tensor.
    """
    # If the target dimension is y, we need to transpose.
    if dim == 'x':
      lengths = self.GetLengths(2, 1)
      inputs = prev_layer
    else:
      lengths = self.GetLengths(1, 1)
      inputs = tf.transpose(prev_layer, [0, 2, 1, 3], name=name + '_ytrans_in')
    input_batch = shapes.tensor_dim(inputs, 0)
    num_slices = shapes.tensor_dim(inputs, 1)
    num_steps = shapes.tensor_dim(inputs, 2)
    input_depth = shapes.tensor_dim(inputs, 3)
    # Reshape away the other dimension.
    inputs = tf.reshape(
        inputs, [-1, num_steps, input_depth], name=name + '_reshape_in')
    # We need to replicate the lengths by the size of the other dimension, and
    # any changes that have been made to the batch dimension.
    tile_factor = tf.to_float(input_batch *
                              num_slices) / tf.to_float(tf.shape(lengths)[0])
    lengths = tf.tile(lengths, [tf.cast(tile_factor, tf.int32)])
    lengths = tf.cast(lengths, tf.int64)
    outputs = nn_ops.rnn_helper(
        inputs,
        lengths,
        cell_type='lstm',
        num_nodes=depth,
        direction=direction,
        name=name,
        stddev=0.1)
    # Output depth is doubled if bi-directional.
    if direction == 'bidirectional':
      output_depth = depth * 2
    else:
      output_depth = depth
    # Restore the other dimension.
    if summarize:
      outputs = tf.slice(
          outputs, [0, num_steps - 1, 0], [-1, 1, -1], name=name + '_sum_slice')
      outputs = tf.reshape(
          outputs, [input_batch, num_slices, 1, output_depth],
          name=name + '_reshape_out')
    else:
      outputs = tf.reshape(
          outputs, [input_batch, num_slices, num_steps, output_depth],
          name=name + '_reshape_out')
    if dim == 'y':
      outputs = tf.transpose(outputs, [0, 2, 1, 3], name=name + '_ytrans_out')
    return outputs

  def _NonLinearity(self, code):
    """Returns the non-linearity function pointer for the given string code.

    For forwards compatibility, allows the full names for stand-alone
    non-linearities, as well as the single-letter names used in ops like C,F.
    Args:
      code: String code representing a non-linearity function.
    Returns:
      non-linearity function represented by the code.
    """
    if code in ['s', 'Sig']:
      return tf.sigmoid
    elif code in ['t', 'Tanh']:
      return tf.tanh
    elif code in ['r', 'Relu']:
      return tf.nn.relu
    elif code in ['m', 'Smax']:
      return tf.nn.softmax
    return None

  def _GetLayerName(self, op_str, index, name_str):
    """Generates a name for the op, using a user-supplied name if possible.

    Args:
      op_str:     String representing the parsed op.
      index:      Position in model_str of the start of the op.
      name_str:   User-supplied {name} with {} that need removing or None.

    Returns:
      Selected name.
    """
    if name_str:
      return name_str[1:-1]
    else:
      return op_str.translate(self.transtab) + '_' + str(index)

  def _SkipWhitespace(self, index):
    """Skips any leading whitespace in the model description.

    Args:
      index:      Position in model_str to start parsing

    Returns:
      end index in model_str of whitespace.
    """
    pattern = re.compile(R'([ \t\n]+)')
    m = pattern.match(self.model_str, index)
    if m is None:
      return index
    return m.end()
