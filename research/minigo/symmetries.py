# Copyright 2018 The TensorFlow Authors. All Rights Reserved.
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
"""Define symmetries for feature transformation.

Allowable symmetries:
  identity [12][34]
  rot90 [24][13]
  rot180 [43][21]
  rot270 [31][42]
  flip [13][24]
  fliprot90 [34][12]
  fliprot180 [42][31]
  fliprot270 [21][43]
"""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import functools
import random

import numpy as np

INVERSES = {
    'identity': 'identity',
    'rot90': 'rot270',
    'rot180': 'rot180',
    'rot270': 'rot90',
    'flip': 'flip',
    'fliprot90': 'fliprot90',
    'fliprot180': 'fliprot180',
    'fliprot270': 'fliprot270',
}

IMPLS = {
    'identity': lambda x: x,
    'rot90': np.rot90,
    'rot180': functools.partial(np.rot90, k=2),
    'rot270': functools.partial(np.rot90, k=3),
    'flip': lambda x: np.rot90(np.fliplr(x)),
    'fliprot90': np.flipud,
    'fliprot180': lambda x: np.rot90(np.flipud(x)),
    'fliprot270': np.fliplr,
}

assert set(INVERSES.keys()) == set(IMPLS.keys())
SYMMETRIES = list(INVERSES.keys())


# A symmetry is just a string describing the transformation.
def invert_symmetry(s):
  return INVERSES[s]


def apply_symmetry_feat(s, features):
  return IMPLS[s](features)


def apply_symmetry_pi(board_size, s, pi):
  pi = np.copy(pi)
  # rotate all moves except for the pass move at end
  pi[:-1] = IMPLS[s](pi[:-1].reshape([board_size, board_size])).ravel()
  return pi


def randomize_symmetries_feat(features):
  symmetries_used = [random.choice(SYMMETRIES) for f in features]
  return symmetries_used, [apply_symmetry_feat(s, f)
                           for s, f in zip(symmetries_used, features)]


def invert_symmetries_pi(board_size, symmetries, pis):
  return [apply_symmetry_pi(board_size, invert_symmetry(s), pi)
          for s, pi in zip(symmetries, pis)]
