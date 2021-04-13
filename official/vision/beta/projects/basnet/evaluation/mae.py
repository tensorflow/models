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
"""

This source code is a modified version of
https://github.com/xuebinqin/Binary-Segmentation-Evaluation-Tool

"""

# Import libraries
import numpy as np

class MAE(object):
  """Mean Absolute Error(MAE) metric for basnet."""

  def __init__(self):
    """Constructs MAE metric class.
    
    Args:

    """
    self.reset_states()

  @property
  def name(self):
    return 'MAE'

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
      average_mae: average MAE with float numpy.
    """

    mae_total = 0.0

    for i, (true, pred) in enumerate(zip(self._groundtruths,
                                         self._predictions)):
      # Compute MAE
      mae = self._compute_mae(true, pred)
      mae_total += mae

    average_mae = mae_total/len(self._groundtruths)
    average_mae = average_mae.astype(np.float32)

    return average_mae

  def _mask_normalize(self, mask):
    return mask/(np.amax(mask)+1e-8)


  def _compute_mae(self, true, pred):
    h, w = true.shape[0], true.shape[1]
    mask1 = self._mask_normalize(true)
    mask2 = self._mask_normalize(pred)
    sum_error = np.sum(np.absolute((mask1.astype(float) - mask2.astype(float))))
    mae_error = sum_error/(float(h)*float(w)+1e-8)

    return mae_error

  def _convert_to_numpy(self, groundtruths, predictions):
    """Converts tesnors to numpy arrays."""
    numpy_groundtruths = groundtruths.numpy()
    numpy_predictions = predictions.numpy()
    
    return numpy_groundtruths, numpy_predictions

  def update_state(self, groundtruths, predictions):
    """Update segmentation results and groundtruth data.

    Args:
      groundtruths : Tuple of single Tensor [batch, width, height, 1],
                     groundtruth masks. range [0, 1]
      predictions  : Tuple of single Tensor [batch, width, height, 1],
                     predicted masks. range [0, 1]
    
    """
    groundtruths, predictions = self._convert_to_numpy(groundtruths[0],
                                                       predictions[0])
    for (true, pred) in zip(groundtruths, predictions):
      self._groundtruths.append(true)
      self._predictions.append(pred)

