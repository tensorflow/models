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

"""Unit tests for task."""

from absl.testing import parameterized
import tensorflow as tf

from official.core import exp_factory
from official.recommendation.ranking import data_pipeline
from official.recommendation.ranking import task


class TaskTest(parameterized.TestCase, tf.test.TestCase):

  @parameterized.parameters(('dlrm_criteo', True),
                            ('dlrm_criteo', False),
                            ('dcn_criteo', True),
                            ('dcn_criteo', False))
  def test_task(self, config_name, is_training):
    params = exp_factory.get_exp_config(config_name)

    params.task.train_data.global_batch_size = 16
    params.task.validation_data.global_batch_size = 16
    params.task.model.vocab_sizes = [40, 12, 11, 13, 2, 5]
    params.task.use_synthetic_data = True
    params.task.model.num_dense_features = 5

    ranking_task = task.RankingTask(params.task,
                                    params.trainer.optimizer_config)

    if is_training:
      dataset = data_pipeline.train_input_fn(params.task)
    else:
      dataset = data_pipeline.eval_input_fn(params.task)

    iterator = iter(dataset(ctx=None))
    model = ranking_task.build_model()

    if is_training:
      ranking_task.train_step(next(iterator), model, model.optimizer,
                              metrics=model.metrics)
    else:
      ranking_task.validation_step(next(iterator), model, metrics=model.metrics)


if __name__ == '__main__':
  tf.test.main()
