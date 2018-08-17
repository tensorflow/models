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
"""A NetworkRegularizer that targets the number of weights in the model."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from morph_net.framework import op_regularizer_manager
from morph_net.network_regularizers import bilinear_cost_utils
from morph_net.op_regularizers import gamma_l1_regularizer


# TODO: Add unit tests. This class is very similar to
# GammaFlopsRegularizer in flop_regularizer, so we have some indirect testing,
# but more is needed.
class GammaModelSizeRegularizer(
    bilinear_cost_utils.BilinearNetworkRegularizer):
  """A NetworkRegularizer that targets model size using Gamma L1 OpReg."""

  def __init__(self, ops, gamma_threshold):
    gamma_l1_reg_factory = gamma_l1_regularizer.GammaL1RegularizerFactory(
        gamma_threshold)
    opreg_manager = op_regularizer_manager.OpRegularizerManager(
        ops, {'Conv2D': gamma_l1_reg_factory.create_regularizer,
              'DepthwiseConv2dNative': gamma_l1_reg_factory.create_regularizer})
    super(GammaModelSizeRegularizer, self).__init__(
        opreg_manager, bilinear_cost_utils.num_weights_coeff)
