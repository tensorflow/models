# Copyright 2022 The TensorFlow Authors. All Rights Reserved.
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
from typing import Mapping, Optional, Sequence, Union

import numpy as np
import tensorflow as tf

from official.core import file_writers
from official.vision.data import fake_feature_generator
from official.vision.data import image_utils
from official.vision.data import tf_example_builder

IMAGE_KEY = 'image/encoded'
CLASSIFICATION_LABEL_KEY = 'image/class/label'
DISTILLATION_LABEL_KEY = 'image/class/soft_labels'
LABEL_KEY = 'clip/label/index'
AUDIO_KEY = 'features/audio'
DUMP_SOURCE_ID = b'7435790'


def encode_image(image_array: np.ndarray, fmt: str) -> bytes:
  return image_utils.encode_image(image_array, fmt)


def make_image_bytes(shape: Sequence[int], fmt: str = 'JPEG') -> bytes:
  """Generates image and return bytes in specified format."""
  image = fake_feature_generator.generate_image_np(*shape)
  return encode_image(image, fmt=fmt)


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
  """Writes serialized Example to TFRecord file with path.

  Note that examples are expected to be not seriazlied.

  Args:
    record_file: The name of the output file.
    tf_examples: A list of examples to be stored.
  """
  file_writers.write_small_dataset(tf_examples, record_file, 'tfrecord')


def create_classification_example(
    image_height: int,
    image_width: int,
    image_format: str = 'JPEG',
    is_multilabel: bool = False,
    output_serialized_example: bool = True) -> tf.train.Example:
  """Creates image and labels for image classification input pipeline.

  Args:
    image_height: The height of test image.
    image_width: The width of test image.
    image_format: The format of test image.
    is_multilabel: A boolean flag represents whether the test image can have
      multiple labels.
    output_serialized_example: A boolean flag represents whether to return a
      serialized example.

  Returns:
    A tf.train.Example for testing.
  """
  image = fake_feature_generator.generate_image_np(image_height, image_width)
  labels = fake_feature_generator.generate_classes_np(2,
                                                      int(is_multilabel) +
                                                      1).tolist()
  builder = tf_example_builder.TfExampleBuilder()
  example = builder.add_image_matrix_feature(image, image_format,
                                             DUMP_SOURCE_ID).add_ints_feature(
                                                 CLASSIFICATION_LABEL_KEY,
                                                 labels).example
  if output_serialized_example:
    return example.SerializeToString()
  return example


def create_distillation_example(
    image_height: int,
    image_width: int,
    num_labels: int,
    image_format: str = 'JPEG',
    output_serialized_example: bool = True) -> tf.train.Example:
  """Creates image and labels for image classification with distillation.

  Args:
    image_height: The height of test image.
    image_width: The width of test image.
    num_labels: The number of labels used in test image.
    image_format: The format of test image.
    output_serialized_example: A boolean flag represents whether to return a
      serialized example.

  Returns:
    A tf.train.Example for testing.
  """
  image = fake_feature_generator.generate_image_np(image_height, image_width)
  labels = fake_feature_generator.generate_classes_np(2, 1).tolist()
  soft_labels = (fake_feature_generator.generate_classes_np(1, num_labels) +
                 0.6).tolist()
  builder = tf_example_builder.TfExampleBuilder()
  example = builder.add_image_matrix_feature(image, image_format,
                                             DUMP_SOURCE_ID).add_ints_feature(
                                                 CLASSIFICATION_LABEL_KEY,
                                                 labels).add_floats_feature(
                                                     DISTILLATION_LABEL_KEY,
                                                     soft_labels).example
  if output_serialized_example:
    return example.SerializeToString()
  return example


def create_3d_image_test_example(
    image_height: int,
    image_width: int,
    image_volume: int,
    image_channel: int,
    output_serialized_example: bool = False) -> tf.train.Example:
  """Creates 3D image and label.

  Args:
    image_height: The height of test 3D image.
    image_width: The width of test 3D image.
    image_volume: The volume of test 3D image.
    image_channel: The channel of test 3D image.
    output_serialized_example: A boolean flag represents whether to return a
      serialized example.

  Returns:
    A tf.train.Example for testing.
  """
  image = fake_feature_generator.generate_image_np(image_height, image_width,
                                                   image_channel)
  images = image[:, :, np.newaxis, :]
  images = np.tile(images, [1, 1, image_volume, 1]).astype(np.float32)

  shape = [image_height, image_width, image_volume, image_channel]
  labels = fake_feature_generator.generate_classes_np(
      2, np.prod(shape)).reshape(shape).astype(np.float32)

  builder = tf_example_builder.TfExampleBuilder()
  example = builder.add_bytes_feature(IMAGE_KEY,
                                      images.tobytes()).add_bytes_feature(
                                          CLASSIFICATION_LABEL_KEY,
                                          labels.tobytes()).example
  if output_serialized_example:
    return example.SerializeToString()
  return example


def create_detection_test_example(
    image_height: int,
    image_width: int,
    image_channel: int,
    num_instances: int,
    fill_image_size: bool = True,
    output_serialized_example: bool = False) -> tf.train.Example:
  """Creates and returns a test example containing box and mask annotations.

  Args:
    image_height: The height of test image.
    image_width: The width of test image.
    image_channel: The channel of test image.
    num_instances: The number of object instances per image.
    fill_image_size: If image height and width will be added to the example.
    output_serialized_example: A boolean flag represents whether to return a
      serialized example.

  Returns:
    A tf.train.Example for testing.
  """
  image = fake_feature_generator.generate_image_np(image_height, image_width,
                                                   image_channel)
  boxes = fake_feature_generator.generate_normalized_boxes_np(num_instances)
  ymins, xmins, ymaxs, xmaxs = boxes.T.tolist()
  is_crowds = [0] * num_instances
  labels = fake_feature_generator.generate_classes_np(
      2, size=num_instances).tolist()
  labels_text = [b'class_1'] * num_instances
  masks = fake_feature_generator.generate_instance_masks_np(
      image_height, image_width, boxes)

  builder = tf_example_builder.TfExampleBuilder()

  example = builder.add_image_matrix_feature(
      image, image_source_id=DUMP_SOURCE_ID).add_boxes_feature(
          xmins, xmaxs, ymins, ymaxs,
          labels).add_instance_mask_matrices_feature(masks).add_ints_feature(
              'image/object/is_crowd',
              is_crowds).add_bytes_feature('image/object/class/text',
                                           labels_text).example
  if not fill_image_size:
    del example.features.feature['image/height']
    del example.features.feature['image/width']

  if output_serialized_example:
    return example.SerializeToString()
  return example


def create_segmentation_test_example(
    image_height: int,
    image_width: int,
    image_channel: int,
    output_serialized_example: bool = False,
    dense_features: Optional[Mapping[str, int]] = None) -> tf.train.Example:
  """Creates and returns a test example containing mask annotations.

  Args:
    image_height: The height of test image.
    image_width: The width of test image.
    image_channel: The channel of test image.
    output_serialized_example: A boolean flag represents whether to return a
      serialized example.
    dense_features: An optional dictionary of additional dense features, where
      the key is the prefix of the feature key in tf.Example and the value is
      the number of the channels of this feature.
  Returns:
    A tf.train.Example for testing.
  """
  image = fake_feature_generator.generate_image_np(image_height, image_width,
                                                   image_channel)
  mask = fake_feature_generator.generate_semantic_mask_np(
      image_height, image_width, 3)
  builder = tf_example_builder.TfExampleBuilder()
  builder.add_image_matrix_feature(
      image,
      image_source_id=DUMP_SOURCE_ID).add_semantic_mask_matrix_feature(mask)

  if dense_features:
    for prefix, channel in dense_features.items():
      dense_feature = fake_feature_generator.generate_semantic_mask_np(
          image_height, image_width, channel)
      builder.add_semantic_mask_matrix_feature(
          dense_feature, feature_prefix=prefix)

  example = builder.example

  if output_serialized_example:
    return example.SerializeToString()
  return example
