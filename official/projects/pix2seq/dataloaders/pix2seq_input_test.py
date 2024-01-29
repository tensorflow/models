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

"""Tests for Pix2Seq input."""
import io

# Import libraries
import numpy as np
from PIL import Image
import tensorflow as tf

from official.projects.pix2seq.dataloaders import pix2seq_input
from official.vision.dataloaders import tf_example_decoder


IMAGE_KEY = 'image/encoded'
LABEL_KEY = 'image/object/class/label'


def _bytes_feature(value):
  """Returns a bytes_list from a string / byte."""
  if isinstance(value, type(tf.constant(0))):
    value = (
        value.numpy()
    )  # BytesList won't unpack a string from an EagerTensor.
  return tf.train.Feature(bytes_list=tf.train.BytesList(value=[value]))


def _float_feature(value):
  """Returns a float_list from a float / double."""
  return tf.train.Feature(float_list=tf.train.FloatList(value=[value]))


def _int64_feature(value):
  """Returns an int64_list from a bool / enum / int / uint."""
  return tf.train.Feature(int64_list=tf.train.Int64List(value=[value]))


def fake_seq_example():
  # Create fake data.
  random_image = np.random.randint(0, 256, size=(480, 640, 3), dtype=np.uint8)
  random_image = Image.fromarray(random_image)
  labels = [42, 5]
  with io.BytesIO() as buffer:
    random_image.save(buffer, format='JPEG')
    raw_image_bytes = buffer.getvalue()

  xmins = [0.23, 0.15]
  xmaxs = [0.54, 0.60]
  ymins = [0.11, 0.5]
  ymaxs = [0.86, 0.72]

  feature = {
      'image/encoded': _bytes_feature(raw_image_bytes),
      'image/height': _int64_feature(480),
      'image/width': _int64_feature(640),
      'image/object/bbox/xmin': tf.train.Feature(
          float_list=tf.train.FloatList(value=xmins)
      ),
      'image/object/bbox/xmax': tf.train.Feature(
          float_list=tf.train.FloatList(value=xmaxs)
      ),
      'image/object/bbox/ymin': tf.train.Feature(
          float_list=tf.train.FloatList(value=ymins)
      ),
      'image/object/bbox/ymax': tf.train.Feature(
          float_list=tf.train.FloatList(value=ymaxs)
      ),
      'image/object/class/label': tf.train.Feature(
          int64_list=tf.train.Int64List(value=labels)
      ),
      'image/object/area': tf.train.Feature(
          float_list=tf.train.FloatList(value=[1., 2.])
      ),
      'image/object/is_crowd': tf.train.Feature(
          int64_list=tf.train.Int64List(value=[0, 0])
      ),
      'image/source_id': _bytes_feature(b'123'),
  }

  # Create a Features message using tf.train.Example.

  example_proto = tf.train.Example(features=tf.train.Features(feature=feature))

  return example_proto, labels


class Pix2SeqParserTest(tf.test.TestCase):

  def test_image_input_train(self):
    decoder = tf_example_decoder.TfExampleDecoder()
    parser = pix2seq_input.Parser(
        eos_token_weight=0.1,
        output_size=[640, 640],
        max_num_boxes=10,
    ).parse_fn(True)

    seq_example, _ = fake_seq_example()

    input_tensor = tf.constant(seq_example.SerializeToString())
    decoded_tensors = decoder.decode(input_tensor)
    output_tensor = parser(decoded_tensors)
    image, _ = output_tensor

    self.assertAllEqual(image.shape, (640, 640, 3))

  def test_image_input_eval(self):
    decoder = tf_example_decoder.TfExampleDecoder()
    parser = pix2seq_input.Parser(
        eos_token_weight=0.1,
        output_size=[640, 640],
        max_num_boxes=10,
    ).parse_fn(False)

    seq_example, _ = fake_seq_example()

    input_tensor = tf.constant(seq_example.SerializeToString())
    decoded_tensors = decoder.decode(input_tensor)
    output_tensor = parser(decoded_tensors)
    image, _ = output_tensor

    self.assertAllEqual(image.shape, (640, 640, 3))


if __name__ == '__main__':
  tf.test.main()
