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

"""Tests classification_input.py."""

from absl.testing import parameterized
import tensorflow as tf
from official.projects.edgetpu.vision.dataloaders import classification_input
from official.vision.configs import common
from official.vision.dataloaders import tfexample_utils

IMAGE_FIELD_KEY = 'image/encoded'
LABEL_FIELD_KEY = 'image/class/label'


class DecoderTest(tf.test.TestCase, parameterized.TestCase):

  @parameterized.parameters(
      (100, 100, 0),
      (100, 100, 1),
      (100, 100, 2),
  )
  def test_decoder(self, image_height, image_width, num_instances):
    decoder = classification_input.Decoder(
        image_field_key=IMAGE_FIELD_KEY, label_field_key=LABEL_FIELD_KEY)

    serialized_example = tfexample_utils.create_classification_example(
        image_height, image_width)
    decoded_tensors = decoder.decode(tf.convert_to_tensor(serialized_example))

    results = tf.nest.map_structure(lambda x: x.numpy(), decoded_tensors)
    self.assertCountEqual([IMAGE_FIELD_KEY, LABEL_FIELD_KEY], results.keys())
    self.assertEqual(0, results[LABEL_FIELD_KEY])


class ParserTest(parameterized.TestCase, tf.test.TestCase):

  @parameterized.parameters(
      ([224, 224, 3], 'float32', True, 'autoaug', False, True, 'JPEG'),
      ([224, 224, 3], 'float16', True, 'randaug', False, False, 'PNG'),
      ([224, 224, 3], 'float32', False, None, False, True, 'JPEG'),
      ([224, 224, 3], 'float16', False, None, False, False, 'PNG'),
      ([512, 640, 3], 'float32', True, 'randaug', False, False, 'JPEG'),
      ([512, 640, 3], 'float16', True, 'autoaug', False, False, 'PNG'),
      ([512, 640, 3], 'float32', False, None, False, True, 'JPEG'),
      ([512, 640, 3], 'float16', False, None, False, False, 'PNG'),
      ([640, 640, 3], 'float32', True, None, False, False, 'JPEG'),
      ([640, 640, 3], 'bfloat16', True, None, False, False, 'PNG'),
      ([640, 640, 3], 'float32', False, None, False, False, 'JPEG'),
      ([640, 640, 3], 'bfloat16', False, None, False, False, 'PNG'),
      ([224, 224, 3], 'float32', True, 'autoaug', True, True, 'JPEG'),
      ([224, 224, 3], 'float16', True, 'randaug', True, False, 'PNG'),
  )
  def test_parser(self, output_size, dtype, is_training, aug_name,
                  is_multilabel, decode_jpeg_only, image_format):

    serialized_example = tfexample_utils.create_classification_example(
        output_size[0], output_size[1], image_format, is_multilabel)

    if aug_name == 'randaug':
      aug_type = common.Augmentation(
          type=aug_name, randaug=common.RandAugment(magnitude=10))
    elif aug_name == 'autoaug':
      aug_type = common.Augmentation(
          type=aug_name, autoaug=common.AutoAugment(augmentation_name='test'))
    else:
      aug_type = None

    decoder = classification_input.Decoder(
        image_field_key=IMAGE_FIELD_KEY, label_field_key=LABEL_FIELD_KEY,
        is_multilabel=is_multilabel)
    parser = classification_input.Parser(
        output_size=output_size[:2],
        num_classes=10,
        image_field_key=IMAGE_FIELD_KEY,
        label_field_key=LABEL_FIELD_KEY,
        is_multilabel=is_multilabel,
        decode_jpeg_only=decode_jpeg_only,
        aug_rand_hflip=False,
        aug_type=aug_type,
        dtype=dtype)

    decoded_tensors = decoder.decode(serialized_example)
    image, label = parser.parse_fn(is_training)(decoded_tensors)

    self.assertAllEqual(image.numpy().shape, output_size)

    if not is_multilabel:
      self.assertAllEqual(label, 0)
    else:
      self.assertAllEqual(label.numpy().shape, [10])

    if dtype == 'float32':
      self.assertAllEqual(image.dtype, tf.float32)
    elif dtype == 'float16':
      self.assertAllEqual(image.dtype, tf.float16)
    elif dtype == 'bfloat16':
      self.assertAllEqual(image.dtype, tf.bfloat16)


if __name__ == '__main__':
  tf.test.main()
