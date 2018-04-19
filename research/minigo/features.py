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
"""Features used by AlphaGo Zero, in approximate order of importance.

Feature                 # Notes
Stone History           16 The stones of each color during the last 8 moves.
Ones                    1  Constant plane of 1s
All features with 8 planes are 1-hot encoded, with plane i marked with 1
only if the feature was equal to i. Any features >= 8 would be marked as 8.

This file includes the features from from AlphaGo Zero (AGZ) as NEW_FEATURES.
"""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import go
import numpy as np


def planes(num_planes):
  # to specify the number of planes in the features. For example, for a 19x19
  # go board, the input stone feature will be in the shape of [19, 19, 16],
  # where the third dimension is the num_planes.
  def deco(f):
    f.planes = num_planes
    return f
  return deco


@planes(16)
def stone_features(board_size, position):
  """Create the 16 planes of features for a given position.

  Args:
    board_size: the go board size.
    position: a given go board status.

  Returns:
    The 16 plane features.
  """
  # a bit easier to calculate it with axis 0 being the 16 board states,
  # and then roll axis 0 to the end.
  features = np.zeros([16, board_size, board_size], dtype=np.uint8)

  num_deltas_avail = position.board_deltas.shape[0]
  cumulative_deltas = np.cumsum(position.board_deltas, axis=0)
  last_eight = np.tile(position.board, [8, 1, 1])
  # apply deltas to compute previous board states
  last_eight[1:num_deltas_avail + 1] -= cumulative_deltas
  # if no more deltas are available, just repeat oldest board.
  last_eight[num_deltas_avail + 1:] = last_eight[num_deltas_avail].reshape(
      1, board_size, board_size)

  features[::2] = last_eight == position.to_play
  features[1::2] = last_eight == -position.to_play
  return np.rollaxis(features, 0, 3)


@planes(1)
def color_to_play_feature(board_size, position):
  if position.to_play == go.BLACK:
    return np.ones([board_size, board_size, 1], dtype=np.uint8)
  else:
    return np.zeros([board_size, board_size, 1], dtype=np.uint8)

NEW_FEATURES = [
    stone_features,
    color_to_play_feature
]

NEW_FEATURES_PLANES = sum(f.planes for f in NEW_FEATURES)


def extract_features(board_size, position, features=None):
  if features is None:
    features = NEW_FEATURES
  return np.concatenate([feature(board_size, position) for feature in features],
                        axis=2)
