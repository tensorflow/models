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
from scipy import signal
import six
import tensorflow as tf


class relaxedFscore(object):
  """BASNet evaluation metric class."""

  def __init__(self):
    """Constructs BASNet evaluation class.

    Args:

    """

    self.reset_states()

  @property
  def name(self):
    return 'relaxF'

  def reset_states(self):
    """Resets internal states for a fresh run."""
    self._predictions = []
    self._groundtruths = []

  def result(self):
    """Evaluates segmentation results, and reset_states."""
    metric_result = self.evaluate()
    # Cleans up the internal variables in order for a fresh eval next time.
    self.reset_states()
    return metric_result

  def evaluate(self):
    """Evaluates with masks from all images.

    Returns:
      relax_f: relaxed F-score value.
    """

    beta = 0.3
    rho = 3
    precisions = np.zeros(len(self._groundtruths))
    recalls = np.zeros(len(self._groundtruths))

    erode_kernel = np.ones((3,3))

    for i, (true, pred) in enumerate(zip(self._groundtruths, self._predictions)):
      true = self._mask_normalize(true)
      pred = self._mask_normalize(pred)

      true = np.squeeze(true, axis=-1)
      pred = np.squeeze(pred, axis=-1)
      # binary saliency mask (S_bw), threshold 0.5
      pred[pred>=0.5]= 1
      pred[pred<0.5]= 0
      # compute eroded binary mask (S_erd) of S_bw
      pred_erd = self._compute_erosion(pred, erode_kernel)
      
      pred_xor = np.logical_xor(pred_erd, pred)
      # convert True/False to 1/0
      pred_xor = pred_xor * 1
      
      # same method for ground truth
      true[true>=0.5]= 1
      true[true<0.5]= 0
      true_erd = self._compute_erosion(true, erode_kernel)
      true_xor = np.logical_xor(true_erd, true)
      true_xor = true_xor * 1
 
      pre, rec = self._compute_relax_pre_rec(true_xor, pred_xor, rho)
      precisions[i] = pre
      recalls[i] = rec

    precisions = np.sum(precisions,0)/(len(self._groundtruths)+1e-8)
    recalls    = np.sum(recalls,0)/(len(self._groundtruths)+1e-8)
    relax_f    = (1+beta)*precisions*recalls/(beta*precisions+recalls+1e-8)

    relax_f = relax_f.astype(np.float32)

    return relax_f

  def _mask_normalize(self, mask):
    return mask/(np.amax(mask)+1e-8)

  def _compute_erosion(self, mask, kernel):
    mask_erd = signal.convolve2d(mask, kernel, mode='same')
    mask_erd[mask_erd<9] = 0
    mask_erd[mask_erd==9] = 1
    return mask_erd

  def _compute_relax_pre_rec(self, true, pred, rho):
    kernel = np.ones((2*rho-1,2*rho-1))
    map_zeros = np.zeros_like(pred)
    map_ones = np.ones_like(pred)

    pred_filtered = signal.convolve2d(pred, kernel, mode='same')
    # True positive for relaxed precision
    relax_pre_tp = np.where((true==1) & (pred_filtered>0), map_ones, map_zeros)
    
    true_filtered = signal.convolve2d(true, kernel, mode='same')
    # True positive for relaxed recall
    relax_rec_tp = np.where((pred==1) & (true_filtered>0), map_ones, map_zeros)
    return np.sum(relax_pre_tp)/np.sum(pred), np.sum(relax_rec_tp)/np.sum(true) 
  
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
    groundtruths, predictions = self._convert_to_numpy(groundtruths[0], predictions[0])
    for (true, pred) in zip(groundtruths, predictions):
      self._groundtruths.append(true)
      self._predictions.append(pred)

