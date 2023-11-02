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

"""Tests for the single_task_trainer."""
import orbit
from orbit.examples.single_task import single_task_trainer

import tensorflow as tf, tf_keras
import tensorflow_datasets as tfds


class SingleTaskTrainerTest(tf.test.TestCase):

  def test_single_task_training(self):
    iris = tfds.load('iris')
    train_ds = iris['train'].batch(32).repeat()

    model = tf_keras.Sequential([
        tf_keras.Input(shape=(4,), name='features'),
        tf_keras.layers.Dense(10, activation=tf.nn.relu),
        tf_keras.layers.Dense(10, activation=tf.nn.relu),
        tf_keras.layers.Dense(3),
        tf_keras.layers.Softmax(),
    ])

    trainer = single_task_trainer.SingleTaskTrainer(
        train_ds,
        label_key='label',
        model=model,
        loss_fn=tf_keras.losses.sparse_categorical_crossentropy,
        optimizer=tf_keras.optimizers.SGD(learning_rate=0.01))

    controller = orbit.Controller(
        trainer=trainer,
        steps_per_loop=100,
        global_step=trainer.optimizer.iterations)

    controller.train(1)
    start_loss = trainer.train_loss.result().numpy()
    controller.train(500)
    end_loss = trainer.train_loss.result().numpy()

    # Assert that the model has trained 'significantly' - that the loss
    # has dropped by over 50%.
    self.assertLess(end_loss, start_loss / 2)


if __name__ == '__main__':
  tf.test.main()
