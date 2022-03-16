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

"""Instance center losses used for panoptic deeplab model."""

# Import libraries
import tensorflow as tf
from official.modeling import tf_utils


class CenterLoss:
  """Instance center loss."""
  
  _LOSS_FN = {
      'mse': tf.losses.mean_squared_error,
      'mae': tf.losses.mean_absolute_error
  }

  def __init__(self, use_groundtruth_dimension: bool, loss_type: str):
    if loss_type.lower() not in {'mse', 'mae'}:
      raise ValueError('Unsupported `loss_type` supported. Available loss '
                       'types: mse/mae')

    self._use_groundtruth_dimension = use_groundtruth_dimension
    self.loss_type = loss_type
    self._loss_fn = CenterLoss._LOSS_FN[self.loss_type]

  def __call__(self, logits, labels, sample_weight):
    _, height, width, _ = logits.get_shape().as_list()

    if self._use_groundtruth_dimension:
      logits = tf.image.resize(
          logits, tf.shape(labels)[1:3],
          method=tf.image.ResizeMethod.BILINEAR)
    else:
      labels = tf.image.resize(
          labels, (height, width),
          method=tf.image.ResizeMethod.BILINEAR)

    loss = self._loss_fn(y_true=labels, y_pred=logits)
    return tf_utils.safe_mean(loss * sample_weight)


class CenterHeatmapLoss(CenterLoss):
  def __init__(self, use_groundtruth_dimension):
    super(CenterHeatmapLoss, self).__init__(
        use_groundtruth_dimension=use_groundtruth_dimension,
        loss_type='mse')

class CenterOffsetLoss(CenterLoss):
  def __init__(self, use_groundtruth_dimension):
    super(CenterOffsetLoss, self).__init__(
        use_groundtruth_dimension=use_groundtruth_dimension,
        loss_type='mae')
