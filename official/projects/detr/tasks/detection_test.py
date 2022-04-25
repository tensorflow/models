# Copyright 2022 The TensorFlow Authors. All Rights Reserved.
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

"""Tests for detection."""

import numpy as np
import tensorflow as tf
import tensorflow_datasets as tfds

from official.projects.detr import optimization
from official.projects.detr.configs import detr as detr_cfg
from official.projects.detr.dataloaders import coco
from official.projects.detr.tasks import detection


_NUM_EXAMPLES = 10


def _gen_fn():
  h = np.random.randint(0, 300)
  w = np.random.randint(0, 300)
  num_boxes = np.random.randint(0, 50)
  return {
      'image': np.ones(shape=(h, w, 3), dtype=np.uint8),
      'image/id': np.random.randint(0, 100),
      'image/filename': 'test',
      'objects': {
          'is_crowd': np.ones(shape=(num_boxes), dtype=np.bool),
          'bbox': np.ones(shape=(num_boxes, 4), dtype=np.float32),
          'label': np.ones(shape=(num_boxes), dtype=np.int64),
          'id': np.ones(shape=(num_boxes), dtype=np.int64),
          'area': np.ones(shape=(num_boxes), dtype=np.int64),
      }
  }


def _as_dataset(self, *args, **kwargs):
  del args
  del kwargs
  return tf.data.Dataset.from_generator(
      lambda: (_gen_fn() for i in range(_NUM_EXAMPLES)),
      output_types=self.info.features.dtype,
      output_shapes=self.info.features.shape,
  )


class DetectionTest(tf.test.TestCase):

  def test_train_step(self):
    config = detr_cfg.DetectionConfig(
        num_encoder_layers=1,
        num_decoder_layers=1,
        train_data=coco.COCODataConfig(
            tfds_name='coco/2017',
            tfds_split='validation',
            is_training=True,
            global_batch_size=2,
        ))
    with tfds.testing.mock_data(as_dataset_fn=_as_dataset):
      task = detection.DectectionTask(config)
      model = task.build_model()
      dataset = task.build_inputs(config.train_data)
      iterator = iter(dataset)
      opt_cfg = optimization.OptimizationConfig({
          'optimizer': {
              'type': 'detr_adamw',
              'detr_adamw': {
                  'weight_decay_rate': 1e-4,
                  'global_clipnorm': 0.1,
              }
          },
          'learning_rate': {
              'type': 'stepwise',
              'stepwise': {
                  'boundaries': [120000],
                  'values': [0.0001, 1.0e-05]
              }
          },
      })
      optimizer = detection.DectectionTask.create_optimizer(opt_cfg)
      task.train_step(next(iterator), model, optimizer)

  def test_validation_step(self):
    config = detr_cfg.DetectionConfig(
        num_encoder_layers=1,
        num_decoder_layers=1,
        validation_data=coco.COCODataConfig(
            tfds_name='coco/2017',
            tfds_split='validation',
            is_training=False,
            global_batch_size=2,
        ))

    with tfds.testing.mock_data(as_dataset_fn=_as_dataset):
      task = detection.DectectionTask(config)
      model = task.build_model()
      metrics = task.build_metrics(training=False)
      dataset = task.build_inputs(config.validation_data)
      iterator = iter(dataset)
      logs = task.validation_step(next(iterator), model, metrics)
      state = task.aggregate_logs(step_outputs=logs)
      task.reduce_aggregated_logs(state)

if __name__ == '__main__':
  tf.test.main()
