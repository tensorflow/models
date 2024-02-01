# Copyright 2024 The TensorFlow Authors. All Rights Reserved.
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

"""Tests for video_ssl_inputs."""

import io
import numpy as np
from PIL import Image

import tensorflow as tf, tf_keras

from official.projects.const_cl.configs import const_cl as exp_cfg
from official.projects.const_cl.datasets import video_ssl_inputs


AUDIO_KEY = 'features/audio'


def fake_seq_example():
  """Creates fake data."""
  random_image = np.random.randint(0, 256, size=(263, 320, 3), dtype=np.uint8)
  random_image = Image.fromarray(random_image)
  label = 42
  with io.BytesIO() as buffer:
    random_image.save(buffer, format='JPEG')
    raw_image_bytes = buffer.getvalue()

  seq_example = tf.train.SequenceExample()
  seq_example.feature_lists.feature_list.get_or_create(
      video_ssl_inputs.IMAGE_KEY).feature.add().bytes_list.value[:] = [
          raw_image_bytes
      ]
  seq_example.feature_lists.feature_list.get_or_create(
      video_ssl_inputs.IMAGE_KEY).feature.add().bytes_list.value[:] = [
          raw_image_bytes
      ]
  seq_example.context.feature[
      video_ssl_inputs.LABEL_KEY].int64_list.value[:] = [label]

  random_audio = np.random.normal(size=(10, 256)).tolist()
  for s in random_audio:
    seq_example.feature_lists.feature_list.get_or_create(
        AUDIO_KEY).feature.add().float_list.value[:] = s
  return seq_example, label


class VideoSslInputsTest(tf.test.TestCase):

  def test_video_ssl_input_pretrain(self):
    params = exp_cfg.const_cl_pretrain_kinetics400().task.train_data

    decoder = video_ssl_inputs.Decoder()
    parser = video_ssl_inputs.Parser(params).parse_fn(params.is_training)
    seq_example, _ = fake_seq_example()

    input_tensor = tf.constant(seq_example.SerializeToString())
    decoded_tensors = decoder.decode(input_tensor)
    output_tensor = parser(decoded_tensors)
    features, _ = output_tensor
    image = features['image']
    instances_position = features['instances_position']
    instances_mask = features['instances_mask']

    self.assertAllEqual(image.shape, (32, 224, 224, 3))
    self.assertAllEqual(instances_position.shape, (32, 8, 4))
    self.assertAllEqual(instances_mask.shape, (32, 8))


if __name__ == '__main__':
  tf.test.main()
