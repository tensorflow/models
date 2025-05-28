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

"""Tests for train."""
import json
import os

from absl import flags
from absl import logging
from absl.testing import flagsaver
import tensorflow as tf, tf_keras
from official.projects.volumetric_models import train as train_lib
from official.vision.dataloaders import tfexample_utils

FLAGS = flags.FLAGS


class TrainTest(tf.test.TestCase):

  def setUp(self):
    super().setUp()
    self._model_dir = os.path.join(self.get_temp_dir(), 'model_dir')
    tf.io.gfile.makedirs(self._model_dir)

    data_dir = os.path.join(self.get_temp_dir(), 'data')
    tf.io.gfile.makedirs(data_dir)
    self._data_path = os.path.join(data_dir, 'data.tfrecord')
    # pylint: disable=g-complex-comprehension
    examples = [
        tfexample_utils.create_3d_image_test_example(
            image_height=32, image_width=32, image_volume=32, image_channel=2)
        for _ in range(2)
    ]
    # pylint: enable=g-complex-comprehension
    tfexample_utils.dump_to_tfrecord(self._data_path, tf_examples=examples)

  def test_run(self):
    saved_flag_values = flagsaver.save_flag_values()
    train_lib.tfm_flags.define_flags()
    FLAGS.mode = 'train'
    FLAGS.model_dir = self._model_dir
    FLAGS.experiment = 'seg_unet3d_test'
    logging.info('Test pipeline correctness.')

    params_override = json.dumps({
        'runtime': {
            'mixed_precision_dtype': 'float32',
        },
        'trainer': {
            'train_steps': 1,
            'validation_steps': 1,
        },
        'task': {
            'model': {
                'backbone': {
                    'unet_3d': {
                        'model_id': 4,
                    },
                },
                'decoder': {
                    'unet_3d_decoder': {
                        'model_id': 4,
                    },
                },
            },
            'train_data': {
                'input_path': self._data_path,
                'file_type': 'tfrecord',
                'global_batch_size': 2,
            },
            'validation_data': {
                'input_path': self._data_path,
                'file_type': 'tfrecord',
                'global_batch_size': 2,
            }
        }
    })
    FLAGS.params_override = params_override

    train_lib.main('unused_args')

    FLAGS.mode = 'eval'

    with train_lib.gin.unlock_config():
      train_lib.main('unused_args')

    flagsaver.restore_flag_values(saved_flag_values)


if __name__ == '__main__':
  tf.test.main()
