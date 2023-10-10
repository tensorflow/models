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

from official.projects.rngdet import optimization
from official.projects.rngdet.configs import rngdet as rngdet_cfg
from official.projects.rngdet.tasks import rngdet
from official.vision.configs import backbones
from official.vision.configs import decoders

_NUM_EXAMPLES = 10


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

#CITYSCALE_INPUT_PATH_BASE = '/home/ghpark/cityscale'
CITYSCALE_INPUT_PATH_BASE = '/home/ghpark.epiclab/03_rngdet/models/official/projects/rngdet'

class RngdetTest(tf.test.TestCase):

  def test_train_step(self):
    config = rngdet_cfg.RngdetTask(
        init_checkpoint='gs://ghpark-ckpts/rngdet/test_02',
        init_checkpoint_modules='all',
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
            global_batch_size=1,
            shuffle_buffer_size=1000,
        ))
    with tfds.testing.mock_data(as_dataset_fn=_as_dataset):
      task = rngdet.RNGDetTask(config)
      model = task.build_model()
      
      task.initialize(model)
      """ckpt_dir_or_file = 'gs://ghpark-ckpts/rngdet/test_00'
      #ckpt_dir_or_file = '/home/ghpark.epiclab/03_rngdet/ckpt/test_00'
      ckpt = tf.train.Checkpoint(
          backbone=model.backbone,
          backbone_history=model.backbone_history,
          transformer=model.transformer,
          segment_fpn=model._segment_fpn,
          keypoint_fpn=model._keypoint_fpn,
          query_embeddings=model._query_embeddings,
          segment_head=model._segment_head,
          keypoint_head=model._keypoint_head,
          class_embed=model._class_embed,
          bbox_embed=model._bbox_embed,
          input_proj=model.input_proj)
      status = ckpt.restore(tf.train.latest_checkpoint(ckpt_dir_or_file))
      status.expect_partial().assert_existing_objects_matched()"""

      print("@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@")
      print("@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@")
      print("LOAD CHECKPOINT DONE")
      print("@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@")
      print("@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@")


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
      optimizer = rngdet.RNGDetTask.create_optimizer(opt_cfg)
      task.train_step(next(iterator), model, optimizer)
      print("***************************************")
      task.train_step(next(iterator), model, optimizer)

  """def test_validation_step(self):
    config = rngdet_cfg.DetrTask(
        model=rngdet_cfg.Detr(
            input_size=[1333, 1333, 3],
            num_encoder_layers=1,
            num_decoder_layers=1,
            num_classes=81,
            backbone=backbones.Backbone(
                type='resnet',
                resnet=backbones.ResNet(model_id=10, bn_trainable=False))
        ),
        validation_data=coco.COCODataConfig(
            tfds_name='coco/2017',
            tfds_split='validation',
            is_training=False,
            global_batch_size=2,
        ))

    with tfds.testing.mock_data(as_dataset_fn=_as_dataset):
      task = detection.DetectionTask(config)
      model = task.build_model()
      metrics = task.build_metrics(training=False)
      dataset = task.build_inputs(config.validation_data)
      iterator = iter(dataset)
      logs = task.validation_step(next(iterator), model, metrics)
      state = task.aggregate_logs(step_outputs=logs)
      task.reduce_aggregated_logs(state)"""
      

if __name__ == '__main__':
  tf.test.main()
