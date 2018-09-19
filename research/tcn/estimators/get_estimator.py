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

"""Get a configured estimator."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from estimators import mvtcn_estimator as mvtcn_estimators
from estimators import svtcn_estimator


def get_mvtcn_estimator(loss_strategy, config, logdir):
  """Returns a configured MVTCN estimator."""
  loss_to_trainer = {
      'triplet_semihard': mvtcn_estimators.MVTCNTripletEstimator,
      'npairs': mvtcn_estimators.MVTCNNpairsEstimator,
  }
  if loss_strategy not in loss_to_trainer:
    raise ValueError('Unknown loss for MVTCN: %s' % loss_strategy)
  estimator = loss_to_trainer[loss_strategy](config, logdir)
  return estimator


def get_estimator(config, logdir):
  """Returns an unsupervised model trainer based on config.

  Args:
    config: A T object holding training configs.
    logdir: String, path to directory where model checkpoints and summaries
      are saved.
  Returns:
    estimator: A configured `TCNEstimator` object.
  Raises:
    ValueError: If unknown training strategy is specified.
  """
  # Get the training strategy.
  training_strategy = config.training_strategy
  if training_strategy == 'mvtcn':
    loss_strategy = config.loss_strategy
    estimator = get_mvtcn_estimator(
        loss_strategy, config, logdir)
  elif training_strategy == 'svtcn':
    estimator = svtcn_estimator.SVTCNTripletEstimator(config, logdir)
  else:
    raise ValueError('Unknown training strategy: %s' % training_strategy)
  return estimator
