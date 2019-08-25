# Copyright 2018 The TensorFlow Authors. All Rights Reserved.
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
# ==============================================================================
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os


from absl import flags
from absl.testing import absltest
import tensorflow as tf

import train


FLAGS = flags.FLAGS


def provide_random_data(batch_size=2, patch_size=4, colors=1, **unused_kwargs):
  return tf.random_normal([batch_size, patch_size, patch_size, colors])


class TrainTest(absltest.TestCase):

  def setUp(self):
    self._config = {
        'start_height': 2,
        'start_width': 2,
        'scale_base': 2,
        'num_resolutions': 2,
        'batch_size_schedule': [2],
        'colors': 1,
        'to_rgb_use_tanh_activation': True,
        'kernel_size': 3,
        'stable_stage_num_images': 1,
        'transition_stage_num_images': 1,
        'total_num_images': 3,
        'save_summaries_num_images': 2,
        'latent_vector_size': 2,
        'fmap_base': 8,
        'fmap_decay': 1.0,
        'fmap_max': 8,
        'gradient_penalty_target': 1.0,
        'gradient_penalty_weight': 10.0,
        'real_score_penalty_weight': 0.001,
        'generator_learning_rate': 0.001,
        'discriminator_learning_rate': 0.001,
        'adam_beta1': 0.0,
        'adam_beta2': 0.99,
        'fake_grid_size': 2,
        'interp_grid_size': 2,
        'train_root_dir': os.path.join(FLAGS.test_tmpdir, 'progressive_gan'),
        'master': '',
        'task': 0
    }

  def test_train_success(self):
    train_root_dir = self._config['train_root_dir']
    if not tf.gfile.Exists(train_root_dir):
      tf.gfile.MakeDirs(train_root_dir)

    for stage_id in train.get_stage_ids(**self._config):
      batch_size = train.get_batch_size(stage_id, **self._config)
      tf.reset_default_graph()
      real_images = provide_random_data(batch_size=batch_size)
      model = train.build_model(stage_id, batch_size, real_images,
                                **self._config)
      train.add_model_summaries(model, **self._config)
      train.train(model, **self._config)

  def test_get_batch_size(self):
    config = {'num_resolutions': 5, 'batch_size_schedule': [8, 4, 2]}
    # batch_size_schedule is expanded to [8, 8, 8, 4, 2]
    # At stage level it is [8, 8, 8, 8, 8, 4, 4, 2, 2]
    for i, expected_batch_size in enumerate([8, 8, 8, 8, 8, 4, 4, 2, 2]):
      self.assertEqual(train.get_batch_size(i, **config), expected_batch_size)


if __name__ == '__main__':
  absltest.main()
