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

"""Tests for teams_task."""

from absl.testing import parameterized
import tensorflow as tf

from official.nlp.configs import encoders
from official.nlp.data import pretrain_dataloader
from official.nlp.projects.teams import teams
from official.nlp.projects.teams import teams_task


class TeamsPretrainTaskTest(tf.test.TestCase, parameterized.TestCase):

  @parameterized.parameters((1, 1), (0, 1), (0, 0), (1, 0))
  def test_task(self, num_shared_hidden_layers,
                num_task_agnostic_layers):
    config = teams_task.TeamsPretrainTaskConfig(
        model=teams.TeamsPretrainerConfig(
            generator=encoders.BertEncoderConfig(
                vocab_size=30522, num_layers=2),
            discriminator=encoders.BertEncoderConfig(
                vocab_size=30522, num_layers=2),
            num_shared_generator_hidden_layers=num_shared_hidden_layers,
            num_discriminator_task_agnostic_layers=num_task_agnostic_layers,
        ),
        train_data=pretrain_dataloader.BertPretrainDataConfig(
            input_path="dummy",
            max_predictions_per_seq=20,
            seq_length=128,
            global_batch_size=1))
    task = teams_task.TeamsPretrainTask(config)
    model = task.build_model()
    metrics = task.build_metrics()
    dataset = task.build_inputs(config.train_data)

    iterator = iter(dataset)
    optimizer = tf.keras.optimizers.SGD(lr=0.1)
    task.train_step(next(iterator), model, optimizer, metrics=metrics)
    task.validation_step(next(iterator), model, metrics=metrics)

if __name__ == "__main__":
  tf.test.main()
