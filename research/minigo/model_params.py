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
"""Defines MiniGo parameters."""


class MiniGoParams(object):
  """Parameters for MiniGo."""

  # Go params
  board_size = 9

  # RL pipeline
  max_games_per_generation = 10  # Number of games per selfplay generation
  max_iters_per_pipeline = 2  # Number of RL iterations in one pipeline

  # The shuffle buffer size determines how far an example could end up from
  # where it started; this and the interleave parameters in preprocessing can
  # give us an approximation of a uniform sampling.  The default of 4M is used
  # in training, but smaller numbers can be used for aggregation or validation.
  shuffle_buffer_size = 2000000  # shuffle buffer size in preprocessing

  # dual_net
  # How many positions to look at per generation.
  # Per AlphaGo Zero (AGZ), 2048 minibatch * 1k = 2M positions/generation
  examples_per_generation = 2000000

  # for learning rate
  l2_strength = 1e-4  # Regularization strength
  momentum = 0.9  # Momentum used in SGD

  kernel_size = [3, 3]  # kernel size of conv and res blocks is from AGZ paper

  # selfplay
  selfplay_readouts = 100  # How many simulations to run per move
  selfplay_verbose = 1  # >=2 will print debug info, >=3 will print boards
  # an absolute value of threshold to resign at
  selfplay_resign_threshold = 0.95

  # the number of simultaneous leaves in MCTS
  simultaneous_leaves = 8

  holdout_pct = 0.05  # How many games to hold out for validation
  holdout_generation = 50  # How many recent generations/models for holdout data

  # gather
  gather_generation = 50  # How many recent generations/models for gathered data

  # How many positions we should aggregate per 'chunk'.
  examples_per_chunk = 10000
  # How many positions to draw from for our training window.
  # AGZ used the most recent 500k games, which, assuming 250 moves/game = 125M
  train_window_size = 125000000

  # evaluation
  eval_games = 50  # The number of games to play in evaluation
  eval_readouts = 100  # How many readouts to make per move in evaluation
  eval_verbose = 1  # How verbose the players should be in evaluation
  eval_win_rate = 0.55  # Winner needs to win by a margin of 55%.


class DummyMiniGoParams(MiniGoParams):
  """Parameters for a dummy model."""
  num_filters = 8  # Number of filters in the convolution layer
  fc_width = 16  # Width of each fully connected layer
  num_shared_layers = 1  # Number of shared trunk layers
  batch_size = 16
  examples_per_generation = 64
  max_games_per_generation = 2
  max_iters_per_pipeline = 1
  selfplay_readouts = 10

  shuffle_buffer_size = 1000

    # evaluation
  eval_games = 10  # The number of games to play in evaluation
  eval_readouts = 10  # How many readouts to make per move in evaluation
  eval_verbose = 1  # How verbose the players should be in evaluation


class DummyValidationParams(DummyMiniGoParams, MiniGoParams):
  """Parameters for a dummy model."""
  holdout_pct = 1  # Set holdout percent as 1 for validation testing purpose
