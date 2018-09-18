# Copyright 2018 The TensorFlow Authors All Rights Reserved.
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
"""OpRegularizers that concatenate and slice other OpRegularizers.

When we have a concatenation op in the network, which concatenates several
tensors, the regularizers of the concatenated ops (that is, the
regularization_vector-s and the alive_vector-s) should be concatenated as well.

Slicing is the complementary op - if regularizers Ra and Rb were concatenated
into a regularizer Rc, Ra and Rb can be obtained form Rc by slicing.
"""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import tensorflow as tf

from morph_net.framework import generic_regularizers


class ConcatRegularizer(generic_regularizers.OpRegularizer):
  """An OpRegularizer that concatenates others, to reflect a Concat op."""

  def __init__(self, regularizers_to_concatenate):
    for r in regularizers_to_concatenate:
      if not generic_regularizers.dimensions_are_compatible(r):
        raise ValueError('Bad regularizer: dimensions are not compatible')

    self._alive_vector = tf.concat(
        [r.alive_vector for r in regularizers_to_concatenate], 0)
    self._regularization_vector = tf.concat(
        [r.regularization_vector for r in regularizers_to_concatenate], 0)

  @property
  def regularization_vector(self):
    return self._regularization_vector

  @property
  def alive_vector(self):
    return self._alive_vector


class SlicingReferenceRegularizer(generic_regularizers.OpRegularizer):
  """An OpRegularizer that slices a segment of another regularizer.

  This is useful to complement the ConcatRegularizer. For example, suppose that
  we have two ops, one with 3 outputs (Op1) and the other with 4 outputs (Op2).
  Each has own regularizer, Reg1 and Reg2.

  Now suppose that a concat op concatenated Op1 and Op2 into OpC. Reg1 and Reg2
  should be concatenated to RegC. To make the situation more complicated, RegC
  was grouped in a group lasso with another op in the graph, resulting in RegG.

  Whan happens next? All references to RegC should obviously be replaced by
  RegG. But what about Reg1? The latter could be the first 3 outputs of RegG,
  and Reg2 would be the 4 last outputs of RegG.

  SlicingReferenceRegularizer is a regularizer that picks a segment of outputs
  form an existing OpRegularizer. When OpRegularizers are concatenated, they
  are replaced by SlicingReferenceRegularizer-s.
  """

  def __init__(self, get_regularizer_to_slice, begin, size):
    """Creates an instance.

    Args:
      get_regularizer_to_slice: A callable, such that get_regularizer_to_slice()
        returns an OpRegularizer that has to be sliced.
      begin: An integer, where to begin the slice.
      size: An integer, the length of the slice (so the slice ends at
        begin + size
    """
    self._get_regularizer_to_slice = get_regularizer_to_slice
    self._begin = begin
    self._size = size
    self._alive_vector = None
    self._regularization_vector = None

  @property
  def regularization_vector(self):
    if self._regularization_vector is None:
      regularizer_to_slice = self._get_regularizer_to_slice()
      self._regularization_vector = tf.slice(
          regularizer_to_slice.regularization_vector, [self._begin],
          [self._size])
    return self._regularization_vector

  @property
  def alive_vector(self):
    if self._alive_vector is None:
      regularizer_to_slice = self._get_regularizer_to_slice()
      assert regularizer_to_slice is not self
      self._alive_vector = tf.slice(regularizer_to_slice.alive_vector,
                                    [self._begin], [self._size])
    return self._alive_vector
