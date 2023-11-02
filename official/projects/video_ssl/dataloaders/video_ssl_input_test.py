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


import io

# Import libraries
import numpy as np
from PIL import Image
import tensorflow as tf, tf_keras

from official.projects.video_ssl.configs import video_ssl as exp_cfg
from official.projects.video_ssl.dataloaders import video_ssl_input

AUDIO_KEY = 'features/audio'


def fake_seq_example():
  # Create fake data.
  random_image = np.random.randint(0, 256, size=(263, 320, 3), dtype=np.uint8)
  random_image = Image.fromarray(random_image)
  label = 42
  with io.BytesIO() as buffer:
    random_image.save(buffer, format='JPEG')
    raw_image_bytes = buffer.getvalue()

  seq_example = tf.train.SequenceExample()
  seq_example.feature_lists.feature_list.get_or_create(
      video_ssl_input.IMAGE_KEY).feature.add().bytes_list.value[:] = [
          raw_image_bytes
      ]
  seq_example.feature_lists.feature_list.get_or_create(
      video_ssl_input.IMAGE_KEY).feature.add().bytes_list.value[:] = [
          raw_image_bytes
      ]
  seq_example.context.feature[video_ssl_input.LABEL_KEY].int64_list.value[:] = [
      label
  ]

  random_audio = np.random.normal(size=(10, 256)).tolist()
  for s in random_audio:
    seq_example.feature_lists.feature_list.get_or_create(
        AUDIO_KEY).feature.add().float_list.value[:] = s
  return seq_example, label


class VideoAndLabelParserTest(tf.test.TestCase):

  def test_video_ssl_input_pretrain(self):
    params = exp_cfg.video_ssl_pretrain_kinetics600().task.train_data

    decoder = video_ssl_input.Decoder()
    parser = video_ssl_input.Parser(params).parse_fn(params.is_training)
    seq_example, _ = fake_seq_example()

    input_tensor = tf.constant(seq_example.SerializeToString())
    decoded_tensors = decoder.decode(input_tensor)
    output_tensor = parser(decoded_tensors)
    image_features, _ = output_tensor
    image = image_features['image']

    self.assertAllEqual(image.shape, (32, 224, 224, 3))

  def test_video_ssl_input_linear_train(self):
    params = exp_cfg.video_ssl_linear_eval_kinetics600().task.train_data

    decoder = video_ssl_input.Decoder()
    parser = video_ssl_input.Parser(params).parse_fn(params.is_training)
    seq_example, label = fake_seq_example()

    input_tensor = tf.constant(seq_example.SerializeToString())
    decoded_tensors = decoder.decode(input_tensor)
    output_tensor = parser(decoded_tensors)
    image_features, label = output_tensor
    image = image_features['image']

    self.assertAllEqual(image.shape, (32, 224, 224, 3))
    self.assertAllEqual(label.shape, (600,))

  def test_video_ssl_input_linear_eval(self):
    params = exp_cfg.video_ssl_linear_eval_kinetics600().task.validation_data
    print('!!!', params)

    decoder = video_ssl_input.Decoder()
    parser = video_ssl_input.Parser(params).parse_fn(params.is_training)
    seq_example, label = fake_seq_example()

    input_tensor = tf.constant(seq_example.SerializeToString())
    decoded_tensors = decoder.decode(input_tensor)
    output_tensor = parser(decoded_tensors)
    image_features, label = output_tensor
    image = image_features['image']

    self.assertAllEqual(image.shape, (960, 256, 256, 3))
    self.assertAllEqual(label.shape, (600,))


if __name__ == '__main__':
  tf.test.main()
