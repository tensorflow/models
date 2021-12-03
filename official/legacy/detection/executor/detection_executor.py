# Copyright 2021 The TensorFlow Authors. All Rights Reserved.
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

"""An executor class for running model on TensorFlow 2.0."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from absl import logging

import tensorflow as tf
from official.legacy.detection.executor import distributed_executor as executor
from official.vision.utils.object_detection import visualization_utils


class DetectionDistributedExecutor(executor.DistributedExecutor):
  """Detection specific customer training loop executor.

  Subclasses the DistributedExecutor and adds support for numpy based metrics.
  """

  def __init__(self,
               predict_post_process_fn=None,
               trainable_variables_filter=None,
               **kwargs):
    super(DetectionDistributedExecutor, self).__init__(**kwargs)
    if predict_post_process_fn:
      assert callable(predict_post_process_fn)
    if trainable_variables_filter:
      assert callable(trainable_variables_filter)
    self._predict_post_process_fn = predict_post_process_fn
    self._trainable_variables_filter = trainable_variables_filter
    self.eval_steps = tf.Variable(
        0,
        trainable=False,
        dtype=tf.int32,
        synchronization=tf.VariableSynchronization.ON_READ,
        aggregation=tf.VariableAggregation.ONLY_FIRST_REPLICA,
        shape=[])

  def _create_replicated_step(self,
                              strategy,
                              model,
                              loss_fn,
                              optimizer,
                              metric=None):
    trainable_variables = model.trainable_variables
    if self._trainable_variables_filter:
      trainable_variables = self._trainable_variables_filter(
          trainable_variables)
    logging.info('Filter trainable variables from %d to %d',
                 len(model.trainable_variables), len(trainable_variables))
    update_state_fn = lambda labels, outputs: None
    if isinstance(metric, tf.keras.metrics.Metric):
      update_state_fn = metric.update_state
    else:
      logging.error('Detection: train metric is not an instance of '
                    'tf.keras.metrics.Metric.')

    def _replicated_step(inputs):
      """Replicated training step."""
      inputs, labels = inputs

      with tf.GradientTape() as tape:
        outputs = model(inputs, training=True)
        all_losses = loss_fn(labels, outputs)
        losses = {}
        for k, v in all_losses.items():
          losses[k] = tf.reduce_mean(v)
        per_replica_loss = losses['total_loss'] / strategy.num_replicas_in_sync
        update_state_fn(labels, outputs)

      grads = tape.gradient(per_replica_loss, trainable_variables)
      clipped_grads, _ = tf.clip_by_global_norm(grads, clip_norm=1.0)
      optimizer.apply_gradients(zip(clipped_grads, trainable_variables))
      return losses

    return _replicated_step

  def _create_test_step(self, strategy, model, metric):
    """Creates a distributed test step."""

    @tf.function
    def test_step(iterator, eval_steps):
      """Calculates evaluation metrics on distributed devices."""

      def _test_step_fn(inputs, eval_steps):
        """Replicated accuracy calculation."""
        inputs, labels = inputs
        model_outputs = model(inputs, training=False)
        if self._predict_post_process_fn:
          labels, prediction_outputs = self._predict_post_process_fn(
              labels, model_outputs)
          num_remaining_visualizations = (
              self._params.eval.num_images_to_visualize - eval_steps)
          # If there are remaining number of visualizations that needs to be
          # done, add next batch outputs for visualization.
          #
          # TODO(hongjunchoi): Once dynamic slicing is supported on TPU, only
          # write correct slice of outputs to summary file.
          if num_remaining_visualizations > 0:
            visualization_utils.visualize_images_with_bounding_boxes(
                inputs, prediction_outputs['detection_boxes'],
                self.global_train_step, self.eval_summary_writer)

        return labels, prediction_outputs

      labels, outputs = strategy.run(
          _test_step_fn, args=(
              next(iterator),
              eval_steps,
          ))
      outputs = tf.nest.map_structure(strategy.experimental_local_results,
                                      outputs)
      labels = tf.nest.map_structure(strategy.experimental_local_results,
                                     labels)

      eval_steps.assign_add(self._params.eval.batch_size)
      return labels, outputs

    return test_step

  def _run_evaluation(self, test_step, current_training_step, metric,
                      test_iterator):
    """Runs validation steps and aggregate metrics."""
    self.eval_steps.assign(0)
    if not test_iterator or not metric:
      logging.warning(
          'Both test_iterator (%s) and metrics (%s) must not be None.',
          test_iterator, metric)
      return None
    logging.info('Running evaluation after step: %s.', current_training_step)
    while True:
      try:
        labels, outputs = test_step(test_iterator, self.eval_steps)
        if metric:
          metric.update_state(labels, outputs)
      except (StopIteration, tf.errors.OutOfRangeError):
        break

    metric_result = metric.result()
    if isinstance(metric, tf.keras.metrics.Metric):
      metric_result = tf.nest.map_structure(lambda x: x.numpy().astype(float),
                                            metric_result)
    logging.info('Step: [%d] Validation metric = %s', current_training_step,
                 metric_result)
    return metric_result
