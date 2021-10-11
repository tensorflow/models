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

# Lint as: python3

import json
import os

from absl import flags
from absl.testing import flagsaver
import numpy as np
import tensorflow as tf
from official.projects.yt8m import train as train_lib
from official.vision.beta.dataloaders import tfexample_utils

FLAGS = flags.FLAGS


def make_yt8m_example():
  rgb = np.random.randint(low=256, size=1024, dtype=np.uint8)
  audio = np.random.randint(low=256, size=128, dtype=np.uint8)

  seq_example = tf.train.SequenceExample()
  seq_example.context.feature['id'].bytes_list.value[:] = [b'id001']
  seq_example.context.feature['labels'].int64_list.value[:] = [1, 2, 3, 4]
  tfexample_utils.put_bytes_list_to_feature(
      seq_example, rgb.tobytes(), key='rgb', repeat_num=120)
  tfexample_utils.put_bytes_list_to_feature(
      seq_example, audio.tobytes(), key='audio', repeat_num=120)

  return seq_example


class TrainTest(tf.test.TestCase):

  def setUp(self):
    super(TrainTest, self).setUp()
    self._model_dir = os.path.join(self.get_temp_dir(), 'model_dir')
    tf.io.gfile.makedirs(self._model_dir)

    data_dir = os.path.join(self.get_temp_dir(), 'data')
    tf.io.gfile.makedirs(data_dir)
    self._data_path = os.path.join(data_dir, 'data.tfrecord')
    examples = [make_yt8m_example() for _ in range(8)]
    tfexample_utils.dump_to_tfrecord(self._data_path, tf_examples=examples)

  def test_run(self):
    saved_flag_values = flagsaver.save_flag_values()
    train_lib.tfm_flags.define_flags()
    FLAGS.mode = 'train'
    FLAGS.model_dir = self._model_dir
    FLAGS.experiment = 'yt8m_experiment'
    FLAGS.tpu = ''

    params_override = json.dumps({
        'runtime': {
            'distribution_strategy': 'mirrored',
            'mixed_precision_dtype': 'float32',
        },
        'trainer': {
            'train_steps': 1,
            'validation_steps': 1,
        },
        'task': {
            'model': {
                'cluster_size': 16,
                'hidden_size': 16,
                'use_context_gate_cluster_layer': True,
                'agg_model': {
                    'use_input_context_gate': True,
                    'use_output_context_gate': True,
                },
            },
            'train_data': {
                'input_path': self._data_path,
                'global_batch_size': 4,
            },
            'validation_data': {
                'input_path': self._data_path,
                'global_batch_size': 4,
            }
        }
    })
    FLAGS.params_override = params_override

    train_lib.train.main('unused_args')

    FLAGS.mode = 'eval'

    with train_lib.train.gin.unlock_config():
      train_lib.train.main('unused_args')

    flagsaver.restore_flag_values(saved_flag_values)


if __name__ == '__main__':
  tf.config.set_soft_device_placement(True)
  tf.test.main()
