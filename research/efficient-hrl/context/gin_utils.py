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

"""Gin configurable utility functions.
"""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import numpy as np
import gin.tf


@gin.configurable
def gin_sparse_array(size, values, indices, fill_value=0):
  arr = np.zeros(size)
  arr.fill(fill_value)
  arr[indices] = values
  return arr


@gin.configurable
def gin_sum(values):
  result = values[0]
  for value in values[1:]:
    result += value
  return result


@gin.configurable
def gin_range(n):
  return range(n)
