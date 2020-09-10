# Lint as: python3
# Copyright 2020 The TensorFlow Authors. All Rights Reserved.
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
# ==============================================================================

import io

# Import libraries
import numpy as np
from PIL import Image
import tensorflow as tf

from official.vision.beta.configs import video_classification as exp_cfg
from official.vision.beta.dataloaders import video_input


class DecoderTest(tf.test.TestCase):
  """A tf.SequenceExample decoder for the video classification task."""

  def test_decoder(self):
    decoder = video_input.Decoder()

    # Create fake data.
    random_image = np.random.randint(0, 256, size=(263, 320, 3), dtype=np.uint8)
    random_image = Image.fromarray(random_image)
    label = 42
    with io.BytesIO() as buffer:
      random_image.save(buffer, format='JPEG')
      raw_image_bytes = buffer.getvalue()

    seq_example = tf.train.SequenceExample()
    seq_example.feature_lists.feature_list.get_or_create(
        video_input.IMAGE_KEY).feature.add().bytes_list.value[:] = [
            raw_image_bytes
        ]
    seq_example.feature_lists.feature_list.get_or_create(
        video_input.IMAGE_KEY).feature.add().bytes_list.value[:] = [
            raw_image_bytes
        ]
    seq_example.context.feature[video_input.LABEL_KEY].int64_list.value[:] = [
        label
    ]
    serialized_example = seq_example.SerializeToString()

    decoded_tensors = decoder.decode(tf.convert_to_tensor(serialized_example))
    results = tf.nest.map_structure(lambda x: x.numpy(), decoded_tensors)
    self.assertCountEqual([video_input.IMAGE_KEY, video_input.LABEL_KEY],
                          results.keys())
    self.assertEqual(label, results[video_input.LABEL_KEY])


class VideoAndLabelParserTest(tf.test.TestCase):

  def test_video_input(self):
    params = exp_cfg.kinetics600(is_training=True)
    params.feature_shape = (2, 224, 224, 3)
    params.min_image_size = 224
    decoder = video_input.Decoder()
    parser = video_input.Parser(params).parse_fn(params.is_training)

    # Create fake data.
    random_image = np.random.randint(0, 256, size=(263, 320, 3), dtype=np.uint8)
    random_image = Image.fromarray(random_image)
    with io.BytesIO() as buffer:
      random_image.save(buffer, format='JPEG')
      raw_image_bytes = buffer.getvalue()

    seq_example = tf.train.SequenceExample()
    seq_example.feature_lists.feature_list.get_or_create(
        video_input.IMAGE_KEY).feature.add().bytes_list.value[:] = [
            raw_image_bytes
        ]
    seq_example.feature_lists.feature_list.get_or_create(
        video_input.IMAGE_KEY).feature.add().bytes_list.value[:] = [
            raw_image_bytes
        ]
    seq_example.context.feature[video_input.LABEL_KEY].int64_list.value[:] = [
        42
    ]

    input_tensor = tf.constant(seq_example.SerializeToString())
    decoded_tensors = decoder.decode(input_tensor)
    output_tensor = parser(decoded_tensors)
    image_features, label = output_tensor
    image = image_features['image']

    self.assertAllEqual(image.shape, (2, 224, 224, 3))
    self.assertAllEqual(label.shape, (600,))


if __name__ == '__main__':
  tf.test.main()
