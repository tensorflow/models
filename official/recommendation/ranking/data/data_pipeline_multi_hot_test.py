# Copyright 2025 The TensorFlow Authors. All Rights Reserved.
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

"""Unit tests for data_pipeline."""

from absl.testing import parameterized
import tensorflow as tf, tf_keras

from official.recommendation.ranking.configs import config
from official.recommendation.ranking.data import data_pipeline_multi_hot


class DataPipelineTest(parameterized.TestCase, tf.test.TestCase):

  @parameterized.named_parameters(
      ('TrainCached', True, True),
      ('EvalNotCached', False, False),
      ('TrainNotCached', True, False),
      ('EvalCached', False, True),
  )
  def testSyntheticDataPipeline(self, is_training, use_cached_data):
    task = config.Task(
        model=config.ModelConfig(
            embedding_dim=4,
            num_dense_features=8,
            vocab_sizes=[40, 12, 11, 13, 2, 5],
            multi_hot_sizes=[2, 3, 1, 1, 3, 2],
            use_multi_hot=True,
            concat_dense=False,
            interaction='multi_layer_dcn',
            dcn_num_layers=3,
            dcn_low_rank_dim=64,
            bottom_mlp=[64, 32, 4],
            top_mlp=[64, 32, 1]),
        train_data=config.DataConfig(global_batch_size=16,
                                     use_cached_data=use_cached_data),
        validation_data=config.DataConfig(global_batch_size=16,
                                          use_cached_data=use_cached_data),
        use_synthetic_data=True)

    num_dense_features = task.model.num_dense_features
    num_sparse_features = len(task.model.vocab_sizes)
    batch_size = task.train_data.global_batch_size

    if is_training:
      dataset = data_pipeline_multi_hot.train_input_fn(task)
    else:
      dataset = data_pipeline_multi_hot.eval_input_fn(task)

    dataset_iter = iter(dataset(ctx=None))
    print('task model', task.model)
    # Consume full batches and validate shapes.
    for _ in range(10):
      features, label = next(dataset_iter)
      dense_features = features['dense_features']
      sparse_features = features['sparse_features']
      self.assertEqual(dense_features.shape, [batch_size, num_dense_features])
      self.assertLen(sparse_features, num_sparse_features)
      for idx, (_, val) in enumerate(sparse_features.items()):
        self.assertEqual(
            val.shape, [batch_size, task.model.multi_hot_sizes[idx]]
        )
      self.assertEqual(label.shape, [batch_size])


if __name__ == '__main__':
  tf.test.main()
