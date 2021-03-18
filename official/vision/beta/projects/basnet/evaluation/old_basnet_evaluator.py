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


class BASNetEvaluator(object):
  """BASNet evaluation metric class."""

  def __init__(self):
    """Constructs BASNet evaluation class.

    Args:

    """

    self._metric_names = ['MAE', 'maxF']
    self.reset_states()

  @property
  def name(self):
    return 'basnet_metric'

  def reset_states(self):
    """Resets internal states for a fresh run."""
    self._predictions = []
    self._groundtruths = []

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

    mae_total = 0.0

    mybins = np.arange(0, 256)
    beta = 0.3
    precisions = np.zeros((len(self._groundtruths), len(mybins)-1))
    recalls = np.zeros((len(self._groundtruths), len(mybins)-1))

    for i, (true, pred) in enumerate(zip(self._groundtruths, self._predictions)):
      # Compute MAE
      mae = self._compute_mae(true, pred)
      mae_total += mae
      
      # Compute F-score
      true = self._mask_normalize(true)*255.0
      pred = self._mask_normalize(pred)*255.0
      pre, rec = self._compute_pre_rec(true, pred, mybins=np.arange(0,256))

      precisions[i,:] = pre
      recalls[i,:]    = rec

    average_mae = mae_total/len(self._groundtruths)

    precisions = np.sum(precisions,0)/(len(self._groundtruths)+1e-8)
    recalls    = np.sum(recalls,0)/(len(self._groundtruths)+1e-8)
    f          = (1+beta)*precisions*recalls/(beta*precisions+recalls+1e-8)
    f_max      = np.max(f)

    metrics_dict = {}
    metrics_dict['MAE'] = average_mae.astype(np.float32)
    metrics_dict['maxF'] = f_max.astype(np.float32)

    return metrics_dict

  def _mask_normalize(self, mask):
    return mask/(np.amax(mask)+1e-8)

  def _compute_mae(self, true, pred):
    h, w = true.shape[0], true.shape[1]
    mask1 = self._mask_normalize(true)
    mask2 = self._mask_normalize(pred)
    sum_error = np.sum(np.absolute((mask1.astype(float) - mask2.astype(float))))
    mae_error = sum_error/(float(h)*float(w)+1e-8)

    return mae_error

  def _compute_pre_rec(self, true, pred, mybins=np.arange(0,256)):
    gt_num = true[true>128].size # pixel number of ground truth foreground regions
    pp = pred[true>128] # mask predicted pixel values in the ground truth foreground region
    nn = pred[true<=128] # mask predicted pixel values in the ground truth bacground region

    pp_hist,pp_edges = np.histogram(pp,bins=mybins)
    nn_hist,nn_edges = np.histogram(nn,bins=mybins)

    pp_hist_flip = np.flipud(pp_hist)
    nn_hist_flip = np.flipud(nn_hist)

    pp_hist_flip_cum = np.cumsum(pp_hist_flip)
    nn_hist_flip_cum = np.cumsum(nn_hist_flip)

    precision = pp_hist_flip_cum/(pp_hist_flip_cum + nn_hist_flip_cum+1e-8) #TP/(TP+FP)
    recall = pp_hist_flip_cum/(gt_num+1e-8) #TP/(TP+FN)

    precision[np.isnan(precision)]= 0.0
    recall[np.isnan(recall)] = 0.0

    return np.reshape(precision,(len(precision))),np.reshape(recall,(len(recall)))

  def _convert_to_numpy(self, groundtruths, predictions):
    """Converts tesnors to numpy arrays."""
    numpy_groundtruths = groundtruths.numpy()
    numpy_predictions = predictions.numpy()
    
    return numpy_groundtruths, numpy_predictions

  def update_state(self, groundtruths, predictions):
    """Update segmentation results and groundtruth data.

    Args:
      groundtruths : Tensor [batch, width, height, 1], groundtruth masks. range [0, 1]
      predictions  : Tensor [batch, width, height, 1], predicted masks. range [0, 1]
    
    """
    groundtruths, predictions = self._convert_to_numpy(groundtruths[0],
                                                       predictions[0])
    for (true, pred) in zip(groundtruths, predictions):
      self._groundtruths.append(true)
      self._predictions.append(pred)

