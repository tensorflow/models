# Copyright 2023 The Orbit Authors. All Rights Reserved.
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

"""Tests for the single_task_evaluator."""
import orbit
from orbit.examples.single_task import single_task_evaluator
from orbit.examples.single_task import single_task_trainer

import tensorflow as tf
import tensorflow_datasets as tfds


class SingleTaskEvaluatorTest(tf.test.TestCase):

  def test_single_task_evaluation(self):

    iris = tfds.load('iris')
    train_ds = iris['train'].batch(32)

    model = tf.keras.Sequential([
        tf.keras.Input(shape=(4,), name='features'),
        tf.keras.layers.Dense(10, activation=tf.nn.relu),
        tf.keras.layers.Dense(10, activation=tf.nn.relu),
        tf.keras.layers.Dense(3)
    ])

    trainer = single_task_trainer.SingleTaskTrainer(
        train_ds,
        label_key='label',
        model=model,
        loss_fn=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
        optimizer=tf.keras.optimizers.SGD(learning_rate=0.01))

    evaluator = single_task_evaluator.SingleTaskEvaluator(
        train_ds,
        label_key='label',
        model=model,
        metrics=[tf.keras.metrics.SparseCategoricalAccuracy()])

    controller = orbit.Controller(
        trainer=trainer,
        evaluator=evaluator,
        steps_per_loop=100,
        global_step=trainer.optimizer.iterations)

    controller.train(train_ds.cardinality().numpy())
    controller.evaluate()
    accuracy = evaluator.metrics[0].result().numpy()

    self.assertGreater(0.925, accuracy)


if __name__ == '__main__':
  tf.test.main()
