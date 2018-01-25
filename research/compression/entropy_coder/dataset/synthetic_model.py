# Copyright 2016 The TensorFlow Authors All Rights Reserved.
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

"""Binary code sample generator."""

import numpy as np
from six.moves import xrange


_CRC_LINE = [
    [0, 1, 0],
    [1, 1, 0],
    [1, 0, 0]
]

_CRC_DEPTH = [1, 1, 0, 1]


def ComputeLineCrc(code, width, y, x, d):
  crc = 0
  for dy in xrange(len(_CRC_LINE)):
    i = y - 1 - dy
    if i < 0:
      continue
    for dx in xrange(len(_CRC_LINE[dy])):
      j = x - 2 + dx
      if j < 0 or j >= width:
        continue
      crc += 1 if (code[i, j, d] != _CRC_LINE[dy][dx]) else 0
  return crc


def ComputeDepthCrc(code, y, x, d):
  crc = 0
  for delta in xrange(len(_CRC_DEPTH)):
    k = d - 1 - delta
    if k < 0:
      continue
    crc += 1 if (code[y, x, k] != _CRC_DEPTH[delta]) else 0
  return crc


def GenerateSingleCode(code_shape):
  code = np.zeros(code_shape, dtype=np.int)

  keep_value_proba = 0.8

  height = code_shape[0]
  width = code_shape[1]
  depth = code_shape[2]

  for d in xrange(depth):
    for y in xrange(height):
      for x in xrange(width):
        v1 = ComputeLineCrc(code, width, y, x, d)
        v2 = ComputeDepthCrc(code, y, x, d)
        v = 1 if (v1 + v2 >= 6) else 0
        if np.random.rand() < keep_value_proba:
          code[y, x, d] = v
        else:
          code[y, x, d] = 1 - v

  return code
