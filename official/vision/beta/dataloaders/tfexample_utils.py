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

# Lint as: python3
"""Utility functions to create tf.Example and tf.SequnceExample for test.

Example:video classification end-to-end test
i.e. from reading input file to train and eval.

```python
class FooTrainTest(tf.test.TestCase):

  def setUp(self):
    super(TrainTest, self).setUp()

    # Write the fake tf.train.SequenceExample to file for test.
    data_dir = os.path.join(self.get_temp_dir(), 'data')
    tf.io.gfile.makedirs(data_dir)
    self._data_path = os.path.join(data_dir, 'data.tfrecord')
    examples = [
        tfexample_utils.make_video_test_example(
            image_shape=(36, 36, 3),
            audio_shape=(20, 128),
            label=random.randint(0, 100)) for _ in range(2)
    ]
    tfexample_utils.dump_to_tfrecord(self._data_path, tf_examples=examples)

  def test_foo(self):
    dataset = tf.data.TFRecordDataset(self._data_path)
    ...

```

"""
import io
from typing import Sequence, Union

# Import libraries
import numpy as np
from PIL import Image
import tensorflow as tf

IMAGE_KEY = 'image/encoded'
LABEL_KEY = 'clip/label/index'
AUDIO_KEY = 'features/audio'


def make_image_bytes(shape: Sequence[int]):
  """Generates image and return bytes in JPEG format."""
  random_image = np.random.randint(0, 256, size=shape, dtype=np.uint8)
  random_image = Image.fromarray(random_image)
  with io.BytesIO() as buffer:
    random_image.save(buffer, format='JPEG')
    raw_image_bytes = buffer.getvalue()
  return raw_image_bytes


def put_int64_to_context(seq_example: tf.train.SequenceExample,
                         label: int = 0,
                         key: str = LABEL_KEY):
  """Puts int64 to SequenceExample context with key."""
  seq_example.context.feature[key].int64_list.value[:] = [label]


def put_bytes_list_to_feature(seq_example: tf.train.SequenceExample,
                              raw_image_bytes: bytes,
                              key: str = IMAGE_KEY,
                              repeat_num: int = 2):
  """Puts bytes list to SequenceExample context with key."""
  for _ in range(repeat_num):
    seq_example.feature_lists.feature_list.get_or_create(
        key).feature.add().bytes_list.value[:] = [raw_image_bytes]


def put_float_list_to_feature(seq_example: tf.train.SequenceExample,
                              value: Sequence[Sequence[float]], key: str):
  """Puts float list to SequenceExample context with key."""
  for s in value:
    seq_example.feature_lists.feature_list.get_or_create(
        key).feature.add().float_list.value[:] = s


def make_video_test_example(image_shape: Sequence[int] = (263, 320, 3),
                            audio_shape: Sequence[int] = (10, 256),
                            label: int = 42):
  """Generates data for testing video models (inc. RGB, audio, & label)."""
  raw_image_bytes = make_image_bytes(shape=image_shape)
  random_audio = np.random.normal(size=audio_shape).tolist()

  seq_example = tf.train.SequenceExample()
  put_int64_to_context(seq_example, label=label, key=LABEL_KEY)
  put_bytes_list_to_feature(
      seq_example, raw_image_bytes, key=IMAGE_KEY, repeat_num=4)

  put_float_list_to_feature(seq_example, value=random_audio, key=AUDIO_KEY)
  return seq_example


def dump_to_tfrecord(record_file: str,
                     tf_examples: Sequence[Union[tf.train.Example,
                                                 tf.train.SequenceExample]]):
  """Writes serialized Example to TFRecord file with path."""
  with tf.io.TFRecordWriter(record_file) as writer:
    for tf_example in tf_examples:
      writer.write(tf_example.SerializeToString())
