# Copyright 2023 The TensorFlow Authors. All Rights Reserved.
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

"""Tests for true_logits_loss."""

from absl.testing import parameterized
import tensorflow as tf, tf_keras

from official.recommendation.uplift import types
from official.recommendation.uplift.losses import true_logits_loss


class TrueLogitsLossTest(tf.test.TestCase, parameterized.TestCase):

  def _get_y_pred(self, **kwargs):
    # The shared embedding and control/treatment/uplift predictions are
    # distracting from the test logic.
    return types.TwoTowerTrainingOutputs(
        shared_embedding=tf.zeros((3, 1)),
        control_predictions=tf.zeros((3, 1)),
        treatment_predictions=tf.zeros((3, 1)),
        uplift=tf.zeros((3, 1)),
        **kwargs,
    )

  @parameterized.product(
      (
          dict(
              reduction_strategy=tf_keras.losses.Reduction.NONE,
              reduction_op=tf.identity,
          ),
          dict(
              reduction_strategy=tf_keras.losses.Reduction.SUM,
              reduction_op=tf.reduce_sum,
          ),
          dict(
              reduction_strategy=tf_keras.losses.Reduction.SUM_OVER_BATCH_SIZE,
              reduction_op=tf.reduce_mean,
          ),
      ),
      (
          dict(
              loss_fn=tf_keras.losses.mean_squared_error, loss_fn_kwargs=dict()
          ),
          dict(
              loss_fn=tf_keras.losses.mean_absolute_percentage_error,
              loss_fn_kwargs=dict(),
          ),
          dict(
              loss_fn=tf_keras.losses.huber,
              loss_fn_kwargs=dict(delta=0.2),
          ),
          dict(
              loss_fn=tf_keras.losses.categorical_crossentropy,
              loss_fn_kwargs=dict(from_logits=True),
          ),
      ),
  )
  def test_correctness(
      self, reduction_strategy, reduction_op, loss_fn, loss_fn_kwargs
  ):
    loss = true_logits_loss.TrueLogitsLoss(
        loss_fn=loss_fn,
        reduction=reduction_strategy,
        **loss_fn_kwargs,
    )
    y_true = tf.constant([[0.4], [1.0], [0.0]])
    y_pred = self._get_y_pred(
        control_logits=tf.constant([[0.6], [4.3], [-0.3]]),
        treatment_logits=tf.constant([[-2.0], [-0.1], [0.5]]),
        true_logits=tf.constant([[-2.0], [4.3], [0.5]]),
        is_treatment=tf.constant([[True], [False], [True]]),
    )
    expected_loss = reduction_op(
        loss_fn(y_true, y_pred.true_logits, **loss_fn_kwargs)
    )
    self.assertAllEqual(expected_loss, loss(y_true, y_pred))


if __name__ == "__main__":
  tf.test.main()
