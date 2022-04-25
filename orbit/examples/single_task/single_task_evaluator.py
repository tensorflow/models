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

"""An evaluator object that can evaluate models with a single output."""
import orbit
import tensorflow as tf


class SingleTaskEvaluator(orbit.StandardEvaluator):
  """Evaluates a single-output model on a given dataset.

  This evaluator will handle running a model with one output on a single
  dataset, and will apply the output of that model to one or more
  `tf.keras.metrics.Metric` objects.
  """

  def __init__(self,
               eval_dataset,
               label_key,
               model,
               metrics,
               evaluator_options=None):
    """Initializes a `SingleTaskEvaluator` instance.

    If the `SingleTaskEvaluator` should run its model under a distribution
    strategy, it should be created within that strategy's scope.

    Arguments:
      eval_dataset: A `tf.data.Dataset` or `DistributedDataset` that contains a
        string-keyed dict of `Tensor`s.
      label_key: The key corresponding to the label value in feature
        dictionaries dequeued from `eval_dataset`. This key will be removed from
        the dictionary before it is passed to the model.
      model: A `tf.Module` or Keras `Model` object to evaluate.
      metrics: A single `tf.keras.metrics.Metric` object, or a list of
        `tf.keras.metrics.Metric` objects.
      evaluator_options: An optional `orbit.StandardEvaluatorOptions` object.
    """

    self.label_key = label_key
    self.model = model
    self.metrics = metrics if isinstance(metrics, list) else [metrics]

    # Capture the strategy from the containing scope.
    self.strategy = tf.distribute.get_strategy()

    super(SingleTaskEvaluator, self).__init__(
        eval_dataset=eval_dataset, options=evaluator_options)

  def eval_begin(self):
    """Actions to take once before every eval loop."""
    for metric in self.metrics:
      metric.reset_states()

  def eval_step(self, iterator):
    """One eval step. Called multiple times per eval loop by the superclass."""

    def step_fn(inputs):
      # Extract the target value and delete it from the input dict, so that
      # the model never sees it.
      target = inputs.pop(self.label_key)
      output = self.model(inputs)
      for metric in self.metrics:
        metric.update_state(target, output)

    # This is needed to handle distributed computation.
    self.strategy.run(step_fn, args=(next(iterator),))

  def eval_end(self):
    """Actions to take once after an eval loop."""
    with self.strategy.scope():
      # Export the metrics.
      metrics = {metric.name: metric.result() for metric in self.metrics}

    return metrics
