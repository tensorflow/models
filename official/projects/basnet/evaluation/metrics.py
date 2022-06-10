# Copyright 2022 The TensorFlow Authors. All Rights Reserved.
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

"""Evaluation metrics for BASNet.

The MAE and maxFscore implementations are a modified version of
https://github.com/xuebinqin/Binary-Segmentation-Evaluation-Tool

"""
import numpy as np
import scipy.signal


class MAE:
  """Mean Absolute Error(MAE) metric for basnet."""

  def __init__(self):
    """Constructs MAE metric class."""
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

    for (true, pred) in zip(self._groundtruths, self._predictions):
      # Computes MAE
      mae = self._compute_mae(true, pred)
      mae_total += mae

    average_mae = mae_total / len(self._groundtruths)

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


class MaxFscore:
  """Maximum F-score metric for basnet."""

  def __init__(self):
    """Constructs BASNet evaluation class."""
    self.reset_states()

  @property
  def name(self):
    return 'MaxFScore'

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
      f_max: maximum F-score value.
    """

    mybins = np.arange(0, 256)
    beta = 0.3
    precisions = np.zeros((len(self._groundtruths), len(mybins)-1))
    recalls = np.zeros((len(self._groundtruths), len(mybins)-1))

    for i, (true, pred) in enumerate(zip(self._groundtruths,
                                         self._predictions)):
      # Compute F-score
      true = self._mask_normalize(true) * 255.0
      pred = self._mask_normalize(pred) * 255.0
      pre, rec = self._compute_pre_rec(true, pred, mybins=np.arange(0, 256))

      precisions[i, :] = pre
      recalls[i, :] = rec

    precisions = np.sum(precisions, 0) / (len(self._groundtruths) + 1e-8)
    recalls = np.sum(recalls, 0) / (len(self._groundtruths) + 1e-8)
    f = (1 + beta) * precisions * recalls / (beta * precisions + recalls + 1e-8)
    f_max = np.max(f)
    f_max = f_max.astype(np.float32)

    return f_max

  def _mask_normalize(self, mask):
    return mask / (np.amax(mask) + 1e-8)

  def _compute_pre_rec(self, true, pred, mybins=np.arange(0, 256)):
    """Computes relaxed precision and recall."""
    # pixel number of ground truth foreground regions
    gt_num = true[true > 128].size

    # mask predicted pixel values in the ground truth foreground region
    pp = pred[true > 128]
    # mask predicted pixel values in the ground truth bacground region
    nn = pred[true <= 128]

    pp_hist, _ = np.histogram(pp, bins=mybins)
    nn_hist, _ = np.histogram(nn, bins=mybins)

    pp_hist_flip = np.flipud(pp_hist)
    nn_hist_flip = np.flipud(nn_hist)

    pp_hist_flip_cum = np.cumsum(pp_hist_flip)
    nn_hist_flip_cum = np.cumsum(nn_hist_flip)

    precision = pp_hist_flip_cum / (pp_hist_flip_cum + nn_hist_flip_cum + 1e-8
                                   )  # TP/(TP+FP)
    recall = pp_hist_flip_cum / (gt_num + 1e-8)  # TP/(TP+FN)

    precision[np.isnan(precision)] = 0.0
    recall[np.isnan(recall)] = 0.0

    pre_len = len(precision)
    rec_len = len(recall)

    return np.reshape(precision, (pre_len)), np.reshape(recall, (rec_len))

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
      predictions  : Tuple of signle Tensor [batch, width, height, 1],
                     predicted masks. range [0, 1]
    """
    groundtruths, predictions = self._convert_to_numpy(groundtruths[0],
                                                       predictions[0])
    for (true, pred) in zip(groundtruths, predictions):
      self._groundtruths.append(true)
      self._predictions.append(pred)


class RelaxedFscore:
  """Relaxed F-score metric for basnet."""

  def __init__(self):
    """Constructs BASNet evaluation class."""
    self.reset_states()

  @property
  def name(self):
    return 'RelaxFScore'

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
    relax_fs = np.zeros(len(self._groundtruths))

    erode_kernel = np.ones((3, 3))

    for i, (true,
            pred) in enumerate(zip(self._groundtruths, self._predictions)):
      true = self._mask_normalize(true)
      pred = self._mask_normalize(pred)

      true = np.squeeze(true, axis=-1)
      pred = np.squeeze(pred, axis=-1)
      # binary saliency mask (S_bw), threshold 0.5
      pred[pred >= 0.5] = 1
      pred[pred < 0.5] = 0
      # compute eroded binary mask (S_erd) of S_bw
      pred_erd = self._compute_erosion(pred, erode_kernel)

      pred_xor = np.logical_xor(pred_erd, pred)
      # convert True/False to 1/0
      pred_xor = pred_xor * 1

      # same method for ground truth
      true[true >= 0.5] = 1
      true[true < 0.5] = 0
      true_erd = self._compute_erosion(true, erode_kernel)
      true_xor = np.logical_xor(true_erd, true)
      true_xor = true_xor * 1

      pre, rec = self._compute_relax_pre_rec(true_xor, pred_xor, rho)
      relax_fs[i] = (1 + beta) * pre * rec / (beta * pre + rec + 1e-8)

    relax_f = np.sum(relax_fs, 0) / (len(self._groundtruths) + 1e-8)
    relax_f = relax_f.astype(np.float32)

    return relax_f

  def _mask_normalize(self, mask):
    return mask/(np.amax(mask)+1e-8)

  def _compute_erosion(self, mask, kernel):
    kernel_full = np.sum(kernel)
    mask_erd = scipy.signal.convolve2d(mask, kernel, mode='same')
    mask_erd[mask_erd < kernel_full] = 0
    mask_erd[mask_erd == kernel_full] = 1
    return mask_erd

  def _compute_relax_pre_rec(self, true, pred, rho):
    """Computes relaxed precision and recall."""
    kernel = np.ones((2 * rho - 1, 2 * rho - 1))
    map_zeros = np.zeros_like(pred)
    map_ones = np.ones_like(pred)

    pred_filtered = scipy.signal.convolve2d(pred, kernel, mode='same')
    # True positive for relaxed precision
    relax_pre_tp = np.where((true == 1) & (pred_filtered > 0), map_ones,
                            map_zeros)

    true_filtered = scipy.signal.convolve2d(true, kernel, mode='same')
    # True positive for relaxed recall
    relax_rec_tp = np.where((pred == 1) & (true_filtered > 0), map_ones,
                            map_zeros)

    return np.sum(relax_pre_tp) / np.sum(pred), np.sum(relax_rec_tp) / np.sum(
        true)

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

