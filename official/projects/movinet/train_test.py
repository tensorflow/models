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

"""Tests for train.py."""

import json
import os
import random

from absl import flags
from absl import logging
from absl.testing import flagsaver
import tensorflow as tf, tf_keras

from official.projects.movinet import train as train_lib
from official.vision.dataloaders import tfexample_utils

FLAGS = flags.FLAGS


class TrainTest(tf.test.TestCase):

  def setUp(self):
    super(TrainTest, self).setUp()
    self._model_dir = os.path.join(self.get_temp_dir(), 'model_dir')
    tf.io.gfile.makedirs(self._model_dir)

    data_dir = os.path.join(self.get_temp_dir(), 'data')
    tf.io.gfile.makedirs(data_dir)
    self._data_path = os.path.join(data_dir, 'data.tfrecord')
    # pylint: disable=g-complex-comprehension
    examples = [
        tfexample_utils.make_video_test_example(
            image_shape=(32, 32, 3),
            audio_shape=(20, 128),
            label=random.randint(0, 100)) for _ in range(2)
    ]
    # pylint: enable=g-complex-comprehension
    tfexample_utils.dump_to_tfrecord(self._data_path, tf_examples=examples)

  def test_train_and_evaluation_pipeline_runs(self):
    saved_flag_values = flagsaver.save_flag_values()
    train_lib.tfm_flags.define_flags()
    FLAGS.mode = 'train'
    FLAGS.model_dir = self._model_dir
    FLAGS.experiment = 'movinet_kinetics600'
    logging.info('Test pipeline correctness.')
    num_frames = 4

    # Test model training pipeline runs.
    params_override = json.dumps({
        'runtime': {
            'distribution_strategy': 'mirrored',
            'mixed_precision_dtype': 'float32',
        },
        'trainer': {
            'train_steps': 2,
            'validation_steps': 2,
        },
        'task': {
            'train_data': {
                'input_path': self._data_path,
                'file_type': 'tfrecord',
                'feature_shape': [num_frames, 32, 32, 3],
                'global_batch_size': 2,
            },
            'validation_data': {
                'input_path': self._data_path,
                'file_type': 'tfrecord',
                'global_batch_size': 2,
                'feature_shape': [num_frames * 2, 32, 32, 3],
            }
        }
    })
    FLAGS.params_override = params_override
    train_lib.main('unused_args')

    # Test model evaluation pipeline runs on newly produced checkpoint.
    FLAGS.mode = 'eval'
    with train_lib.gin.unlock_config():
      train_lib.main('unused_args')

    flagsaver.restore_flag_values(saved_flag_values)


if __name__ == '__main__':
  tf.test.main()
