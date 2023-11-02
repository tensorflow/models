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

"""Tests for masked_ae."""

import numpy as np
import tensorflow as tf, tf_keras
import tensorflow_datasets as tfds

from official.modeling import optimization
from official.projects.mae.configs import mae as mae_cfg
from official.projects.mae.tasks import masked_ae
from official.vision.configs import image_classification


_NUM_EXAMPLES = 10


def _gen_fn():
  np.random.seed(0)  # Some seed may cause jpeg decoding to fail.
  h = np.random.randint(0, 300)
  w = np.random.randint(0, 300)
  return {
      'image': np.ones(shape=(h, w, 3), dtype=np.uint8),
      'label': np.random.randint(0, 100),
      'file_name': 'test',
  }


def _as_dataset(self, *args, **kwargs):
  del args
  del kwargs
  return tf.data.Dataset.from_generator(
      lambda: (_gen_fn() for i in range(_NUM_EXAMPLES)),
      output_types=self.info.features.dtype,
      output_shapes=self.info.features.shape,
  )


class MAETest(tf.test.TestCase):

  def test_train_step(self):
    config = mae_cfg.MAEConfig(
        train_data=image_classification.DataConfig(
            tfds_name='imagenet2012',
            tfds_split='validation',
            is_training=True,
            global_batch_size=2,
        ))
    with tfds.testing.mock_data(as_dataset_fn=_as_dataset):
      task = masked_ae.MaskedAETask(config)
      model = task.build_model()
      dataset = task.build_inputs(config.train_data)
      iterator = iter(dataset)
      opt_cfg = optimization.OptimizationConfig({
          'optimizer': {
              'type': 'adamw',
              'adamw': {
                  'weight_decay_rate': 0.05,
                  # Avoid AdamW legacy behavior.
                  'gradient_clip_norm': 0.0
              }
          },
          'learning_rate': {
              'type': 'cosine',
              'cosine': {
                  'initial_learning_rate': 1.5 * 1e-4,
                  'decay_steps': 5
              }
          },
          'warmup': {
              'type': 'linear',
              'linear': {
                  'warmup_steps': 1,
                  'warmup_learning_rate': 0
              }
          }
      })
      optimizer = masked_ae.MaskedAETask.create_optimizer(opt_cfg)
      task.train_step(next(iterator), model, optimizer)


if __name__ == '__main__':
  tf.test.main()
