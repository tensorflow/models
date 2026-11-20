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

"""Tests for detection."""

import numpy as np
import tensorflow as tf
import tensorflow_datasets as tfds
import os

from official.modeling import optimization  
from official.projects.rngdet_plus.configs import rngdet as rngdet_cfg
from official.projects.rngdet_plus.tasks import rngdet_plus
from official.vision.configs import backbones
from official.vision.configs import decoders
import sys
import pdb;
_NUM_EXAMPLES = 10

datapath = os.getenv("DATAPATH", "/data2/cityscale/tfrecord/")
CITYSCALE_INPUT_PATH_BASE = datapath 

def _gen_fn():
  h = 128
  w = 128
  num_query = 10
  return {
      'sat_roi': np.ones(shape=(h, w, 3), dtype=np.uint8),
      'label_masks_roi': np.ones(shape=(h, w, 2), dtype=np.uint8),
      'historical_roi': np.ones(shape=(h, w, 1), dtype=np.uint8),
      'gt_probs': np.ones(shape=(num_query), dtype=np.int64),
      'gt_coords': np.ones(shape=(num_query, 2), dtype=np.float32),
      'list_len': 4,
      'gt_masks': np.ones(shape=(h, w, num_query), dtype=np.uint8),
  }

def _as_dataset(self, *args, **kwargs):
  del args
  del kwargs
  return tf.data.Dataset.from_generator(
      lambda: (_gen_fn() for i in range(_NUM_EXAMPLES)),
      output_types=self.info.features.dtype,
      output_shapes=self.info.features.shape,
  )

class RngdetTest(tf.test.TestCase):

  def test_train_step(self):
    config = rngdet_cfg.RngdetTask(
        model=rngdet_cfg.Rngdet(
            input_size=[128, 128, 3],
            roi_size=128,
            num_encoder_layers=6,
            num_decoder_layers=6,
            num_queries=10,
            hidden_size=256,
            num_classes=2,
            min_level=2,
            max_level=5,
            backbone_endpoint_name='5',
            backbone=backbones.Backbone(
                type='resnet',
                resnet=backbones.ResNet(model_id=50, bn_trainable=False)),
            decoder=decoders.Decoder(
                type='fpn',
                fpn=decoders.FPN())
        ),
        train_data=rngdet_cfg.DataConfig(
            input_path=os.path.join(CITYSCALE_INPUT_PATH_BASE, 'train-noise*'),
            is_training=True,
            dtype='float32',
            global_batch_size=3,
            shuffle_buffer_size=1000,
        ))
    with tfds.testing.mock_data(as_dataset_fn=_as_dataset, num_examples=5):
      task = rngdet_plus.RNGDetTask(config)  
      model = task.build_model()  
      opt_cfg = optimization.OptimizationConfig({
          'optimizer': {
              'type': 'adamw_experimental',
              'adamw_experimental': {
                  'epsilon': 1.0e-08,
                  'weight_decay': 1e-4,
                  'global_clipnorm': 0.1,
              }
          },
          'learning_rate': {
               'type': 'polynomial',
                  'polynomial': {
                  'initial_learning_rate': 0.0001,
                  'end_learning_rate': 0.000001,
                  'offset': 0,
                  'power': 1.0,
                  'decay_steps': 50 * 10,
              }
          },
          'warmup': {
                  'type': 'linear',
                  'linear': {
                      'warmup_steps': 2 * 10,
                      'warmup_learning_rate': 0,
                  },
          },
      })
      optimizer = rngdet_plus.RNGDetTask.create_optimizer(opt_cfg)
      dataset = task.build_inputs(config.train_data)  
      iterator = iter(dataset) 

      task.train_step(next(iterator), model, optimizer)
      dummy_images = tf.keras.Input([128, 128, 3]) 
      dummy_history = tf.keras.Input([128, 128, 1])
      _ = model(dummy_images, dummy_history, training=False)  

  def test_validation_step(self):
    config = rngdet_cfg.RngdetTask(
        model=rngdet_cfg.Rngdet(
            input_size=[128, 128, 3],
            roi_size=128,
            num_encoder_layers=6,
            num_decoder_layers=6,
            num_queries=10,
            hidden_size=256,
            num_classes=2,  
            min_level=2,
            max_level=5,
            backbone_endpoint_name='5',
            backbone=backbones.Backbone(
                type='resnet',
                resnet=backbones.ResNet(model_id=50, bn_trainable=False)),
            decoder=decoders.Decoder(
                type='fpn',
                fpn=decoders.FPN())

        ))

    with tfds.testing.mock_data(as_dataset_fn=_as_dataset):
      task = rngdet_plus.RNGDetTask(config)
      model = task.build_model()
      dummy_images = tf.keras.Input([128, 128, 3])
      dummy_history = tf.keras.Input([128, 128, 1])
      _ = model(dummy_images, dummy_history, training=False) 

if __name__ == '__main__':
  tf.test.main()
