# Copyright 2022 The Orbit Authors. All Rights Reserved.
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

"""A trainer object that can train models with a single output."""

import orbit
import tensorflow as tf


class SingleTaskTrainer(orbit.StandardTrainer):
  """Trains a single-output model on a given dataset.

  This trainer will handle running a model with one output on a single
  dataset. It will apply the provided loss function to the model's output
  to calculate gradients and will apply them via the provided optimizer. It will
  also supply the output of that model to one or more `tf.keras.metrics.Metric`
  objects.
  """

  def __init__(self,
               train_dataset,
               label_key,
               model,
               loss_fn,
               optimizer,
               metrics=None,
               trainer_options=None):
    """Initializes a `SingleTaskTrainer` instance.

    If the `SingleTaskTrainer` should run its model under a distribution
    strategy, it should be created within that strategy's scope.

    This trainer will also calculate metrics during training. The loss metric
    is calculated by default, but other metrics can be passed to the `metrics`
    arg.

    Arguments:
      train_dataset: A `tf.data.Dataset` or `DistributedDataset` that contains a
        string-keyed dict of `Tensor`s.
      label_key: The key corresponding to the label value in feature
        dictionaries dequeued from `train_dataset`. This key will be removed
        from the dictionary before it is passed to the model.
      model: A `tf.Module` or Keras `Model` object to evaluate. It must accept a
        `training` kwarg.
      loss_fn: A per-element loss function of the form (target, output). The
        output of this loss function will be reduced via `tf.reduce_mean` to
        create the final loss. We recommend using the functions in the
        `tf.keras.losses` package or `tf.keras.losses.Loss` objects with
        `reduction=tf.keras.losses.reduction.NONE`.
      optimizer: A `tf.keras.optimizers.Optimizer` instance.
      metrics: A single `tf.keras.metrics.Metric` object, or a list of
        `tf.keras.metrics.Metric` objects.
      trainer_options: An optional `orbit.utils.StandardTrainerOptions` object.
    """
    self.label_key = label_key
    self.model = model
    self.loss_fn = loss_fn
    self.optimizer = optimizer

    # Capture the strategy from the containing scope.
    self.strategy = tf.distribute.get_strategy()

    # We always want to report training loss.
    self.train_loss = tf.keras.metrics.Mean('training_loss', dtype=tf.float32)

    # We need self.metrics to be an iterable later, so we handle that here.
    if metrics is None:
      self.metrics = []
    elif isinstance(metrics, list):
      self.metrics = metrics
    else:
      self.metrics = [metrics]

    super(SingleTaskTrainer, self).__init__(
        train_dataset=train_dataset, options=trainer_options)

  def train_loop_begin(self):
    """Actions to take once, at the beginning of each train loop."""
    self.train_loss.reset_states()
    for metric in self.metrics:
      metric.reset_states()

  def train_step(self, iterator):
    """A train step. Called multiple times per train loop by the superclass."""

    def train_fn(inputs):
      with tf.GradientTape() as tape:
        # Extract the target value and delete it from the input dict, so that
        # the model never sees it.
        target = inputs.pop(self.label_key)

        # Get the outputs of the model.
        output = self.model(inputs, training=True)

        # Get the average per-batch loss and scale it down by the number of
        # replicas. This ensures that we don't end up multiplying our loss by
        # the number of workers - gradients are summed, not averaged, across
        # replicas during the apply_gradients call.
        # Note, the reduction of loss is explicitly handled and scaled by
        # num_replicas_in_sync. Recommend to use a plain loss function.
        # If you're using tf.keras.losses.Loss object, you may need to set
        # reduction argument explicitly.
        loss = tf.reduce_mean(self.loss_fn(target, output))
        scaled_loss = loss / self.strategy.num_replicas_in_sync

        # Get the gradients by applying the loss to the model's trainable
        # variables.
        gradients = tape.gradient(scaled_loss, self.model.trainable_variables)

        # Apply the gradients via the optimizer.
        self.optimizer.apply_gradients(
            list(zip(gradients, self.model.trainable_variables)))

        # Update metrics.
        self.train_loss.update_state(loss)
        for metric in self.metrics:
          metric.update_state(target, output)

    # This is needed to handle distributed computation.
    self.strategy.run(train_fn, args=(next(iterator),))

  def train_loop_end(self):
    """Actions to take once after a training loop."""
    with self.strategy.scope():
      # Export the metrics.
      metrics = {metric.name: metric.result() for metric in self.metrics}
      metrics[self.train_loss.name] = self.train_loss.result()

    return metrics
