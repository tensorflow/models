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

from official.projects.rngdet.dataloaders import rngdet_input
from official.vision.dataloaders import tf_example_decoder
from absl import app  # pylint:disable=unused-import


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
  random_image = np.random.randint(0, 256, size=(2048, 2048, 3), dtype=np.uint8)
  random_image = Image.fromarray(random_image)
  with io.BytesIO() as buffer:
    random_image.save(buffer, format='JPEG')
    raw_image_bytes = buffer.getvalue()

  segment_image = np.random.randint(0, 2, size=(2048, 2048, 1), dtype=np.uint8)
  segment_image = Image.fromarray(segment_image*255)
  with io.BytesIO() as buffer:
    segment_image.save(buffer, format='JPEG')
    raw_segment_bytes = buffer.getvalue()

  intersec_image = np.random.randint(0, 2, size=(2048, 2048, 1), dtype=np.uint8)
  intersec_image = Image.fromarray(intersec_image*255)
  with io.BytesIO() as buffer:
    intersec_image.save(buffer, format='JPEG')
    raw_intersec_bytes = buffer.getvalue()

  labels = [42, 5]

  xmins = [0.23, 0.15]
  xmaxs = [0.54, 0.60]
  ymins = [0.11, 0.5]
  ymaxs = [0.86, 0.72]

  feature = {
      'image/encoded': _bytes_feature(raw_image_bytes),
      'image/height': _int64_feature(2048),
      'image/width': _int64_feature(2048),
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

def fake_rngdet_example():
  # Create fake data.
  random_image = np.random.randint(0, 256, size=(480, 640, 3), dtype=np.uint8)
  random_image = Image.fromarray(random_image)
  labels = [42, 5]
  with io.BytesIO() as buffer:
    random_image.save(buffer, format='JPEG')
    raw_image_bytes = buffer.getvalue()
  {"edges":
    [{"id": 82, "src": 2379, "dst": 2380,
      "vertices": [[1990, 44], [1990, 45], [1989, 46], [1989, 47], [1989, 48], [1989, 49], [1988, 50], [1988, 51], [1988, 52], [1987, 53], [1987, 54]],
      "orientation": [46, 46, 46, 46, 46, 46, 46, 46, 46, 46, 46]}]
  }
  {"vertices":
    [{"id": 2379, "x": 1990, "y": 44, "neighbors": [82]},
     {"id": 2380, "x": 1987, "y": 54, "neighbors": [82, 3635, 3727, 4125]}]
  }
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
    parser = rngdet_input.Parser(
        eos_token_weight=0.1,
        output_size=[640, 640],
        max_num_boxes=10,
    ).parse_fn(True)

    seq_example, _ = fake_seq_example()

    input_tensor = tf.constant(seq_example.SerializeToString())
    decoded_tensors = decoder.decode(input_tensor)
    output_tensor = parser(decoded_tensors)
    image = output_tensor

    self.assertAllEqual(image.shape, (640, 640, 3))

  def test_image_input_eval(self):
    decoder = tf_example_decoder.TfExampleDecoder()
    parser = rngdet_input.Parser(
        roi_size=128,
        num_queries=10,
    ).parse_fn(False)

    seq_example, _ = fake_seq_example()

    input_tensor = tf.constant(seq_example.SerializeToString())
    decoded_tensors = decoder.decode(input_tensor)
    output_tensor = parser(decoded_tensors)
    image = output_tensor

    self.assertAllEqual(image.shape, (640, 640, 3))


def main(_):
  raw_dataset_train = tf.data.TFRecordDataset( '/home/mjyun/cityscale/tfrecord/train-noise-8-00000-of-00032.tfrecord')

  decoder = rngdet_input.Decoder()
  parser = rngdet_input.Parser(
        roi_size=128,
        num_queries=10,
    ).parse_fn(True)

  decoded_tensors = raw_dataset_train.map(decoder.decode)
  decoded_tensors = decoded_tensors.take(10)
  for i in decoded_tensors:
    images, labels = parser(i)


if __name__ == '__main__':
  #tf.test.main()
  app.run(main)
