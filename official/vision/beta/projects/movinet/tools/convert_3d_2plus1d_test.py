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

"""Tests for convert_3d_2plus1d."""

import os

from absl import flags
import tensorflow as tf

from official.vision.beta.projects.movinet.modeling import movinet
from official.vision.beta.projects.movinet.modeling import movinet_model
from official.vision.beta.projects.movinet.tools import convert_3d_2plus1d

FLAGS = flags.FLAGS


class Convert3d2plus1dTest(tf.test.TestCase):

  def test_convert_model(self):
    saved_model_path = self.get_temp_dir()
    input_checkpoint_path = os.path.join(saved_model_path, 'ckpt-input')
    output_checkpoint_path = os.path.join(saved_model_path, 'ckpt')

    model_3d_2plus1d = movinet_model.MovinetClassifier(
        backbone=movinet.Movinet(
            model_id='a0',
            conv_type='3d_2plus1d'),
        num_classes=600)
    model_3d_2plus1d.build([1, 1, 1, 1, 3])
    save_checkpoint = tf.train.Checkpoint(model=model_3d_2plus1d)
    save_checkpoint.save(input_checkpoint_path)

    FLAGS.input_checkpoint_path = f'{input_checkpoint_path}-1'
    FLAGS.output_checkpoint_path = output_checkpoint_path
    FLAGS.model_id = 'a0'
    FLAGS.use_positional_encoding = False
    FLAGS.num_classes = 600
    FLAGS.verify_output = True

    convert_3d_2plus1d.main('unused_args')

    print(os.listdir(saved_model_path))

    self.assertTrue(tf.io.gfile.exists(f'{output_checkpoint_path}-1.index'))


if __name__ == '__main__':
  tf.test.main()
