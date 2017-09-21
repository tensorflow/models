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

"""Define some typical masked 2D convolutions."""

import numpy as np
import tensorflow as tf

import block_util
import blocks_std

# pylint does not recognize block_base.BlockBase.__call__().
# pylint: disable=not-callable


class RasterScanConv2D(blocks_std.Conv2DBase):
  """Conv2D with no dependency on future pixels (in raster scan order).

  For example, assuming a 5 x 5 kernel, the kernel is applied a spatial mask:
    T T T T T
    T T T T T
    T T x F F
    F F F F F
    F F F F F
  where 'T' are pixels which are available when computing the convolution
  for pixel 'x'. All the pixels marked with 'F' are not available.
  'x' itself is not available if strict_order is True, otherwise, it is
  available.
  """

  def __init__(self, depth, filter_size, strides, padding,
               strict_order=True,
               bias=None, act=None, initializer=None, name=None):
    super(RasterScanConv2D, self).__init__(
        depth, filter_size, strides, padding, bias, act, name=name)

    if (filter_size[0] % 2) != 1 or (filter_size[1] % 2) != 1:
      raise ValueError('Kernel size should be odd.')

    with self._BlockScope():
      if initializer is None:
        initializer = block_util.RsqrtInitializer(dims=(0, 1, 2))
      self._initializer = initializer
      self._strict_order = strict_order

  def _CreateKernel(self, shape, dtype):
    init = self._initializer(shape, dtype)
    kernel = self.NewVar(init)

    mask = np.ones(shape[:2], dtype=dtype.as_numpy_dtype)
    center = shape[:2] // 2
    mask[center[0] + 1:, :] = 0
    if not self._strict_order:
      mask[center[0], center[1] + 1:] = 0
    else:
      mask[center[0], center[1]:] = 0
    mask = mask.reshape(mask.shape + (1, 1))

    return tf.convert_to_tensor(mask, dtype) * kernel


class DepthOrderConv2D(blocks_std.Conv2DBase):
  """Conv2D with no dependency on higher depth dimensions.

  More precisely, the output depth #n has only dependencies on input depths #k
  for k < n (if strict_order is True) or for k <= n (if strict_order is False).
  """

  def __init__(self, depth, filter_size, strides, padding,
               strict_order=True,
               bias=None, act=None, initializer=None, name=None):
    super(DepthOrderConv2D, self).__init__(
        depth, filter_size, strides, padding, bias, act, name=name)

    with self._BlockScope():
      if initializer is None:
        initializer = block_util.RsqrtInitializer(dims=(0, 1, 2))
      self._initializer = initializer
      self._strict_order = strict_order

  def _CreateKernel(self, shape, dtype):
    init = self._initializer(shape, dtype)
    kernel = self.NewVar(init)

    mask = np.ones(shape[2:], dtype=dtype.as_numpy_dtype)
    depth_output = shape[3]
    for d in xrange(depth_output):
      if self._strict_order:
        mask[d:, d] = 0
      else:
        mask[d + 1:, d] = 0
    mask = mask.reshape((1, 1) + mask.shape)

    return tf.convert_to_tensor(mask, dtype) * kernel


class GroupRasterScanConv2D(blocks_std.Conv2DBase):
  """Conv2D with no dependency on future pixels (in raster scan order).

  This version only introduces dependencies on previous pixels in raster scan
  order. It can also introduce some dependencies on previous depth positions
  of the current pixel (current pixel = center pixel of the kernel) in the
  following way:
  the depth dimension of the input is split into Ki groups of size
  |input_group_size|, the output dimension is split into Ko groups of size
  |output_group_size| (usually Ki == Ko). Each output group ko of the current
  pixel position can only depend on previous input groups ki
  (i.e. ki < ko if strict_order is True or ki <= ko if strict_order is False).

  Notes:
  - Block RasterScanConv2D is a special case of GroupRasterScanConv2D
    where Ki == Ko == 1 (i.e. input_group_size == input_depth and
    output_group_size == output_depth).
  - For 1x1 convolution, block DepthOrderConv2D is a special case of
    GroupRasterScanConv2D where input_group_size == 1 and
    output_group_size == 1.
  """

  def __init__(self, depth, filter_size, strides, padding,
               strict_order=True,
               input_group_size=1,
               output_group_size=1,
               bias=None, act=None, initializer=None, name=None):
    super(GroupRasterScanConv2D, self).__init__(
        depth, filter_size, strides, padding, bias, act, name=name)

    if (filter_size[0] % 2) != 1 or (filter_size[1] % 2) != 1:
      raise ValueError('Kernel size should be odd.')

    with self._BlockScope():
      if initializer is None:
        initializer = block_util.RsqrtInitializer(dims=(0, 1, 2))
      self._initializer = initializer
      self._input_group_size = input_group_size
      self._output_group_size = output_group_size
      self._strict_order = strict_order

      if depth % self._output_group_size != 0:
        raise ValueError(
            'Invalid depth group size: {} for depth {}'.format(
                self._output_group_size, depth))
      self._output_group_count = depth // self._output_group_size

  def _CreateKernel(self, shape, dtype):
    init = self._initializer(shape, dtype)
    kernel = self.NewVar(init)

    depth_input = shape[2]
    if depth_input % self._input_group_size != 0:
      raise ValueError(
          'Invalid depth group size: {} for depth {}'.format(
              self._input_group_size, depth_input))
    input_group_count = depth_input // self._input_group_size
    output_group_count = self._output_group_count

    # Set the mask to 0 for future pixels in raster scan order.
    center = shape[:2] // 2
    mask = np.ones([shape[0], shape[1],
                    input_group_count, self._input_group_size,
                    output_group_count, self._output_group_size],
                   dtype=dtype.as_numpy_dtype)
    mask[center[0] + 1:, :, :, :, :, :] = 0
    mask[center[0], center[1] + 1:, :, :, :, :] = 0

    # Adjust the mask for the current position (the center position).
    depth_output = shape[3]
    for d in xrange(output_group_count):
      mask[center[0], center[1], d + 1:, :, d:d + 1, :] = 0
      if self._strict_order:
        mask[center[0], center[1], d, :, d:d + 1, :] = 0

    mask = mask.reshape([shape[0], shape[1], depth_input, depth_output])
    return tf.convert_to_tensor(mask, dtype) * kernel


class InFillingConv2D(blocks_std.Conv2DBase):
  """Conv2D with kernel having no dependency on the current pixel.

  For example, assuming a 5 x 5 kernel, the kernel is applied a spatial mask:
    T T T T T
    T T T T T
    T T x T T
    T T T T T
    T T T T T
  where 'T' marks a pixel which is available when computing the convolution
  for pixel 'x'. 'x' itself is not available.
  """

  def __init__(self, depth, filter_size, strides, padding,
               bias=None, act=None, initializer=None, name=None):
    super(InFillingConv2D, self).__init__(
        depth, filter_size, strides, padding, bias, act, name=name)

    if (filter_size[0] % 2) != 1 or (filter_size[1] % 2) != 1:
      raise ValueError('Kernel size should be odd.')
    if filter_size[0] == 1 and filter_size[1] == 1:
      raise ValueError('Kernel size should be larger than 1x1.')

    with self._BlockScope():
      if initializer is None:
        initializer = block_util.RsqrtInitializer(dims=(0, 1, 2))
      self._initializer = initializer

  def _CreateKernel(self, shape, dtype):
    init = self._initializer(shape, dtype)
    kernel = self.NewVar(init)

    mask = np.ones(shape[:2], dtype=dtype.as_numpy_dtype)
    center = shape[:2] // 2
    mask[center[0], center[1]] = 0
    mask = mask.reshape(mask.shape + (1, 1))

    return tf.convert_to_tensor(mask, dtype) * kernel
