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


import io

# Import libraries
import numpy as np
from PIL import Image
import tensorflow as tf, tf_keras
import tensorflow_datasets as tfds

from official.vision.configs import common
from official.vision.configs import video_classification as exp_cfg
from official.vision.dataloaders import video_input


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

  random_audio = np.random.normal(size=(10, 256)).tolist()
  for s in random_audio:
    seq_example.feature_lists.feature_list.get_or_create(
        AUDIO_KEY).feature.add().float_list.value[:] = s
  return seq_example, label


class DecoderTest(tf.test.TestCase):
  """A tf.SequenceExample decoder for the video classification task."""

  def test_decoder(self):
    decoder = video_input.Decoder()

    seq_example, label = fake_seq_example()
    serialized_example = seq_example.SerializeToString()

    decoded_tensors = decoder.decode(tf.convert_to_tensor(serialized_example))
    results = tf.nest.map_structure(lambda x: x.numpy(), decoded_tensors)
    self.assertCountEqual([video_input.IMAGE_KEY, video_input.LABEL_KEY],
                          results.keys())
    self.assertEqual(label, results[video_input.LABEL_KEY])

  def test_decode_audio(self):
    decoder = video_input.Decoder()
    decoder.add_feature(AUDIO_KEY, tf.io.VarLenFeature(dtype=tf.float32))

    seq_example, label = fake_seq_example()
    serialized_example = seq_example.SerializeToString()

    decoded_tensors = decoder.decode(tf.convert_to_tensor(serialized_example))
    results = tf.nest.map_structure(lambda x: x.numpy(), decoded_tensors)
    self.assertCountEqual(
        [video_input.IMAGE_KEY, video_input.LABEL_KEY, AUDIO_KEY],
        results.keys())
    self.assertEqual(label, results[video_input.LABEL_KEY])
    self.assertEqual(results[AUDIO_KEY].shape, (10, 256))

  def test_tfds_decode(self):
    with tfds.testing.mock_data(num_examples=1):
      dataset = tfds.load('ucf101', split='train').take(1)
      data = next(iter(dataset))

    decoder = video_input.VideoTfdsDecoder()
    decoded_tensors = decoder.decode(data)
    self.assertContainsSubset([video_input.LABEL_KEY, video_input.IMAGE_KEY],
                              decoded_tensors.keys())


class VideoAndLabelParserTest(tf.test.TestCase):

  def test_video_input(self):
    params = exp_cfg.kinetics600(is_training=True)
    params.feature_shape = (2, 224, 224, 3)
    params.min_image_size = 224

    decoder = video_input.Decoder()
    parser = video_input.Parser(params).parse_fn(params.is_training)

    seq_example, label = fake_seq_example()

    input_tensor = tf.constant(seq_example.SerializeToString())
    decoded_tensors = decoder.decode(input_tensor)
    output_tensor = parser(decoded_tensors)
    image_features, label = output_tensor
    image = image_features['image']

    self.assertAllEqual(image.shape, (2, 224, 224, 3))
    self.assertAllEqual(label.shape, (600,))

  def test_video_audio_input(self):
    params = exp_cfg.kinetics600(is_training=True)
    params.feature_shape = (2, 224, 224, 3)
    params.min_image_size = 224
    params.output_audio = True
    params.audio_feature = AUDIO_KEY
    params.audio_feature_shape = (15, 256)

    decoder = video_input.Decoder()
    decoder.add_feature(params.audio_feature,
                        tf.io.VarLenFeature(dtype=tf.float32))
    parser = video_input.Parser(params).parse_fn(params.is_training)

    seq_example, label = fake_seq_example()

    input_tensor = tf.constant(seq_example.SerializeToString())
    decoded_tensors = decoder.decode(input_tensor)
    output_tensor = parser(decoded_tensors)
    features, label = output_tensor
    image = features['image']
    audio = features['audio']

    self.assertAllEqual(image.shape, (2, 224, 224, 3))
    self.assertAllEqual(label.shape, (600,))
    self.assertEqual(audio.shape, (15, 256))

  def test_video_input_random_stride(self):
    params = exp_cfg.kinetics600(is_training=True)
    params.feature_shape = (2, 224, 224, 3)
    params.min_image_size = 224

    params.temporal_stride = 2
    params.random_stride_range = 1

    decoder = video_input.Decoder()
    parser = video_input.Parser(params).parse_fn(params.is_training)

    seq_example, label = fake_seq_example()

    input_tensor = tf.constant(seq_example.SerializeToString())
    decoded_tensors = decoder.decode(input_tensor)
    output_tensor = parser(decoded_tensors)
    image_features, label = output_tensor
    image = image_features['image']

    self.assertAllEqual(image.shape, (2, 224, 224, 3))
    self.assertAllEqual(label.shape, (600,))

  def test_video_input_augmentation_returns_shape(self):
    params = exp_cfg.kinetics600(is_training=True)
    params.feature_shape = (2, 224, 224, 3)
    params.min_image_size = 224

    params.temporal_stride = 2
    params.aug_type = common.Augmentation(
        type='autoaug', autoaug=common.AutoAugment())

    decoder = video_input.Decoder()
    parser = video_input.Parser(params).parse_fn(params.is_training)

    seq_example, label = fake_seq_example()

    input_tensor = tf.constant(seq_example.SerializeToString())
    decoded_tensors = decoder.decode(input_tensor)
    output_tensor = parser(decoded_tensors)
    image_features, label = output_tensor
    image = image_features['image']

    self.assertAllEqual(image.shape, (2, 224, 224, 3))
    self.assertAllEqual(label.shape, (600,))

  def test_video_input_image_shape_label_type(self):
    params = exp_cfg.kinetics600(is_training=True)
    params.feature_shape = (2, 168, 224, 1)
    params.min_image_size = 168
    params.label_dtype = 'float32'
    params.one_hot = False

    decoder = video_input.Decoder()
    parser = video_input.Parser(params).parse_fn(params.is_training)

    seq_example, label = fake_seq_example()

    input_tensor = tf.constant(seq_example.SerializeToString())
    decoded_tensors = decoder.decode(input_tensor)
    output_tensor = parser(decoded_tensors)
    image_features, label = output_tensor
    image = image_features['image']

    self.assertAllEqual(image.shape, (2, 168, 224, 1))
    self.assertAllEqual(label.shape, (1,))
    self.assertDTypeEqual(label, tf.float32)


if __name__ == '__main__':
  tf.test.main()
