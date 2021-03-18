# Copyright 2020 The TensorFlow Authors. All Rights Reserved.
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
"""The BASNet-style evaluator.

The following snippet demonstrates the use of interfaces:

  evaluator = BASNetEvaluator(...)
  for _ in range(num_evals):
    for _ in range(num_batches_per_eval):
      predictions, groundtruth = predictor.predict(...)  # pop a batch.
      evaluator.update_state(groundtruths, predictions)
    evaluator.result()  # finish one full eval and reset states.

This script is a revised version of Binary-Segmentation-Evaluation-Tool
(https://github.com/xuebinqin/Binary-Segmentation-Evaluation-Tool)
measures.py for evaluation in tensorflow model garden.

"""

import atexit
import tempfile
# Import libraries
from absl import logging
import numpy as np
import six
import tensorflow as tf

from official.vision.beta.projects.basnet.evaluation import f_score
from official.vision.beta.projects.basnet.evaluation import mae

class BASNetEvaluator(object):
  """BASNet evaluation metric class."""

  def __init__(self):
    """Constructs BASNet evaluation class.

    Args:

    """

    self._metric_names = ['MAE', 'maxF']
    self._mae = mae.MAE()
    self._f_score = f_score.Fscore()
    self.reset_states()

  @property
  def name(self):
    return 'basnet_metric'

  def reset_states(self):
    """Resets internal states for a fresh run."""

    self._mae.reset_states()
    self._f_score.reset_states()

  def result(self):
    """Evaluates segmentation results, and reset_states."""
    metric_dict = self.evaluate()    
    # Cleans up the internal variables in order for a fresh eval next time.
    self.reset_states()
    return metric_dict

  def evaluate(self):
    """Evaluates with masks from all images.

    Returns:
      basnet_metric: dictionary with float numpy.
    """
    average_mae = self._mae.evaluate()
    max_f = self._f_score.evaluate()

    metrics_dict = {}
    metrics_dict['MAE'] = average_mae
    metrics_dict['maxF'] = max_f

    return metrics_dict

  def update_state(self, groundtruths, predictions):
    """Update segmentation results and groundtruth data.

    Args:
      groundtruths : Tensor [batch, width, height, 1], groundtruth masks. range [0, 1]
      predictions  : Tensor [batch, width, height, 1], predicted masks. range [0, 1]
    
    """
    self._mae.update_state(groundtruths, predictions)
    self._f_score.update_state(groundtruths, predictions)
