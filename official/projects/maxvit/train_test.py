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

import json
import os

from absl import flags
from absl.testing import flagsaver
import gin
import tensorflow as tf, tf_keras

from official.projects.maxvit import train as train_lib
from official.vision.dataloaders import tfexample_utils


FLAGS = flags.FLAGS


class TrainTest(tf.test.TestCase):

  def setUp(self):
    super().setUp()

    self._model_dir = os.path.join(self.get_temp_dir(), 'model_dir')
    tf.io.gfile.makedirs(self._model_dir)
    self._test_tfrecord_file = os.path.join(
        self.get_temp_dir(), 'test.tfrecord'
    )
    num_samples = 3
    example = tf.train.Example.FromString(
        tfexample_utils.create_classification_example(
            image_height=224, image_width=224
        )
    )
    examples = [example] * num_samples
    tfexample_utils.dump_to_tfrecord(
        record_file=self._test_tfrecord_file, tf_examples=examples
    )

  def test_run(self):
    saved_flag_values = flagsaver.save_flag_values()
    train_lib.tfm_flags.define_flags()
    FLAGS.mode = 'train'
    FLAGS.model_dir = self._model_dir
    FLAGS.experiment = 'maxvit_imagenet'

    params_override = json.dumps({
        'runtime': {
            'mixed_precision_dtype': 'float32',
        },
        'trainer': {
            'train_steps': 1,
            'validation_steps': 1,
            'optimizer_config': {
                'ema': None,
            },
        },
        'task': {
            'init_checkpoint': '',
            'model': {
                'backbone': {
                    'maxvit': {
                        'model_name': 'maxvit-tiny-for-test',
                        'representation_size': 64,
                        'add_gap_layer_norm': True,
                    }
                },
                'input_size': [224, 224, 3],
                'num_classes': 3,
            },
            'train_data': {
                'global_batch_size': 2,
                'input_path': self._test_tfrecord_file,
            },
            'validation_data': {
                'global_batch_size': 2,
                'input_path': self._test_tfrecord_file,
            },
        },
    })
    FLAGS.params_override = params_override
    train_lib.train.main('unused_args')

    FLAGS.mode = 'eval'
    with gin.unlock_config():
      train_lib.train.main('unused_args')

    flagsaver.restore_flag_values(saved_flag_values)


if __name__ == '__main__':
  tf.test.main()
