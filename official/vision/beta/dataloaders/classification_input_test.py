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
"""Tests classification_input.py."""

import io
# Import libraries
from absl.testing import parameterized
import numpy as np
from PIL import Image
import tensorflow as tf
from official.core import input_reader
from official.modeling.hyperparams import config_definitions as cfg
from official.vision.beta.dataloaders import classification_input


def _encode_image(image_array, fmt):
  image = Image.fromarray(image_array)
  with io.BytesIO() as output:
    image.save(output, format=fmt)
    return output.getvalue()


class DecoderTest(tf.test.TestCase, parameterized.TestCase):

  @parameterized.parameters(
      (100, 100, 0), (100, 100, 1), (100, 100, 2),
  )
  def test_decoder(self, image_height, image_width, num_instances):
    decoder = classification_input.Decoder()

    image = _encode_image(
        np.uint8(np.random.rand(image_height, image_width, 3) * 255),
        fmt='JPEG')
    label = 2
    serialized_example = tf.train.Example(
        features=tf.train.Features(
            feature={
                'image/encoded': (tf.train.Feature(
                    bytes_list=tf.train.BytesList(value=[image]))),
                'image/class/label': (
                    tf.train.Feature(
                        int64_list=tf.train.Int64List(value=[label]))),
            })).SerializeToString()
    decoded_tensors = decoder.decode(tf.convert_to_tensor(serialized_example))

    results = tf.nest.map_structure(lambda x: x.numpy(), decoded_tensors)
    self.assertCountEqual(
        ['image/encoded', 'image/class/label'], results.keys())
    self.assertEqual(label, results['image/class/label'])


class ParserTest(parameterized.TestCase, tf.test.TestCase):

  @parameterized.parameters(
      ([224, 224, 3], 'float32', True),
      ([224, 224, 3], 'float16', True),
      ([224, 224, 3], 'float32', False),
      ([224, 224, 3], 'float16', False),
      ([512, 640, 3], 'float32', True),
      ([512, 640, 3], 'float16', True),
      ([512, 640, 3], 'float32', False),
      ([512, 640, 3], 'float16', False),
      ([640, 640, 3], 'float32', True),
      ([640, 640, 3], 'bfloat16', True),
      ([640, 640, 3], 'float32', False),
      ([640, 640, 3], 'bfloat16', False),
  )
  def test_parser(self, output_size, dtype, is_training):

    params = cfg.DataConfig(
        input_path='imagenet-2012-tfrecord/train*',
        global_batch_size=2,
        is_training=True,
        examples_consume=4)

    decoder = classification_input.Decoder()
    parser = classification_input.Parser(
        output_size=output_size[:2],
        num_classes=1001,
        aug_rand_hflip=False,
        dtype=dtype)

    reader = input_reader.InputReader(
        params,
        dataset_fn=tf.data.TFRecordDataset,
        decoder_fn=decoder.decode,
        parser_fn=parser.parse_fn(params.is_training))

    dataset = reader.read()

    images, labels = next(iter(dataset))

    self.assertAllEqual(images.numpy().shape,
                        [params.global_batch_size] + output_size)
    self.assertAllEqual(labels.numpy().shape, [params.global_batch_size])

    if dtype == 'float32':
      self.assertAllEqual(images.dtype, tf.float32)
    elif dtype == 'float16':
      self.assertAllEqual(images.dtype, tf.float16)
    elif dtype == 'bfloat16':
      self.assertAllEqual(images.dtype, tf.bfloat16)


if __name__ == '__main__':
  tf.test.main()
