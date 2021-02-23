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
import itertools
import numpy as np
from PIL import Image
import tensorflow as tf

from official.vision.beta.ops import preprocess_ops_3d


class ParserUtilsTest(tf.test.TestCase):

  def setUp(self):
    super().setUp()
    # [[0, 1, ..., 119], [1, 2, ..., 120], ..., [119, 120, ..., 218]].
    self._frames = tf.stack([tf.range(i, i + 120) for i in range(90)])
    self._frames = tf.cast(self._frames, tf.uint8)
    self._frames = self._frames[tf.newaxis, :, :, tf.newaxis]
    self._frames = tf.broadcast_to(self._frames, (6, 90, 120, 3))

    # Create an equivalent numpy array for assertions.
    self._np_frames = np.array([range(i, i + 120) for i in range(90)])
    self._np_frames = self._np_frames[np.newaxis, :, :, np.newaxis]
    self._np_frames = np.broadcast_to(self._np_frames, (6, 90, 120, 3))

  def test_sample_linspace_sequence(self):
    sequence = tf.range(100)
    sampled_seq_1 = preprocess_ops_3d.sample_linspace_sequence(
        sequence, 10, 10, 1)
    sampled_seq_2 = preprocess_ops_3d.sample_linspace_sequence(
        sequence, 7, 10, 1)
    sampled_seq_3 = preprocess_ops_3d.sample_linspace_sequence(
        sequence, 7, 5, 2)
    sampled_seq_4 = preprocess_ops_3d.sample_linspace_sequence(
        sequence, 101, 1, 1)

    self.assertAllEqual(sampled_seq_1, range(100))
    # [0, 1, 2, 3, 4, ..., 8, 9, 15, 16, ..., 97, 98, 99]
    self.assertAllEqual(
        sampled_seq_2,
        [15 * i + j for i, j in itertools.product(range(7), range(10))])
    # [0, 2, 4, 6, 8, 15, 17, 19, ..., 96, 98]
    self.assertAllEqual(
        sampled_seq_3,
        [15 * i + 2 * j for i, j in itertools.product(range(7), range(5))])
    self.assertAllEqual(sampled_seq_4, [0] + list(range(100)))

  def test_sample_sequence(self):
    sequence = tf.range(100)
    sampled_seq_1 = preprocess_ops_3d.sample_sequence(sequence, 10, False, 1)
    sampled_seq_2 = preprocess_ops_3d.sample_sequence(sequence, 10, False, 2)
    sampled_seq_3 = preprocess_ops_3d.sample_sequence(sequence, 10, True, 1)

    self.assertAllEqual(sampled_seq_1, range(45, 55))
    self.assertAllEqual(sampled_seq_2, range(40, 60, 2))

    offset_3 = sampled_seq_3[0]
    self.assertBetween(offset_3, 0, 99)
    self.assertAllEqual(sampled_seq_3, range(offset_3, offset_3 + 10))

  def test_decode_jpeg(self):
    # Create a random RGB JPEG image.
    random_image = np.random.randint(0, 256, size=(263, 320, 3), dtype=np.uint8)
    random_image = Image.fromarray(random_image)
    with io.BytesIO() as buffer:
      random_image.save(buffer, format='JPEG')
      raw_image_bytes = buffer.getvalue()

    raw_image = tf.constant([raw_image_bytes, raw_image_bytes])
    decoded_image = preprocess_ops_3d.decode_jpeg(raw_image, 3)

    self.assertEqual(decoded_image.shape.as_list()[3], 3)
    self.assertAllEqual(decoded_image.shape, (2, 263, 320, 3))

  def test_crop_image(self):
    cropped_image_1 = preprocess_ops_3d.crop_image(self._frames, 50, 70)
    cropped_image_2 = preprocess_ops_3d.crop_image(self._frames, 200, 200)
    cropped_image_3 = preprocess_ops_3d.crop_image(self._frames, 50, 70, True)
    cropped_image_4 = preprocess_ops_3d.crop_image(
        self._frames, 90, 90, False, 3)

    self.assertAllEqual(cropped_image_1.shape, (6, 50, 70, 3))
    self.assertAllEqual(cropped_image_1, self._np_frames[:, 20:70, 25:95, :])

    self.assertAllEqual(cropped_image_2.shape, (6, 200, 200, 3))
    expected = np.pad(
        self._np_frames, ((0, 0), (55, 55), (40, 40), (0, 0)), 'constant')
    self.assertAllEqual(cropped_image_2, expected)

    self.assertAllEqual(cropped_image_3.shape, (6, 50, 70, 3))
    offset = cropped_image_3[0, 0, 0, 0]
    expected = np.array([range(i, i + 70) for i in range(offset, offset + 50)])
    expected = expected[np.newaxis, :, :, np.newaxis]
    expected = np.broadcast_to(expected, (6, 50, 70, 3))
    self.assertAllEqual(cropped_image_3, expected)
    self.assertAllEqual(cropped_image_4.shape, (18, 90, 90, 3))

  def test_resize_smallest(self):
    resized_frames_1 = preprocess_ops_3d.resize_smallest(self._frames, 180)
    resized_frames_2 = preprocess_ops_3d.resize_smallest(self._frames, 45)
    resized_frames_3 = preprocess_ops_3d.resize_smallest(self._frames, 90)
    resized_frames_4 = preprocess_ops_3d.resize_smallest(
        tf.transpose(self._frames, (0, 2, 1, 3)), 45)

    self.assertAllEqual(resized_frames_1.shape, (6, 180, 240, 3))
    self.assertAllEqual(resized_frames_2.shape, (6, 45, 60, 3))
    self.assertAllEqual(resized_frames_3.shape, (6, 90, 120, 3))
    self.assertAllEqual(resized_frames_4.shape, (6, 60, 45, 3))

  def test_random_crop_resize(self):
    resized_frames_1 = preprocess_ops_3d.random_crop_resize(
        self._frames, 256, 256, 6, 3, (0.5, 2), (0.3, 1))
    resized_frames_2 = preprocess_ops_3d.random_crop_resize(
        self._frames, 224, 224, 6, 3, (0.5, 2), (0.3, 1))
    resized_frames_3 = preprocess_ops_3d.random_crop_resize(
        self._frames, 256, 256, 6, 3, (0.8, 1.2), (0.3, 1))
    resized_frames_4 = preprocess_ops_3d.random_crop_resize(
        self._frames, 256, 256, 6, 3, (0.5, 2), (0.1, 1))
    self.assertAllEqual(resized_frames_1.shape, (6, 256, 256, 3))
    self.assertAllEqual(resized_frames_2.shape, (6, 224, 224, 3))
    self.assertAllEqual(resized_frames_3.shape, (6, 256, 256, 3))
    self.assertAllEqual(resized_frames_4.shape, (6, 256, 256, 3))

  def test_random_flip_left_right(self):
    flipped_frames = preprocess_ops_3d.random_flip_left_right(self._frames)

    flipped = np.fliplr(self._np_frames[0, :, :, 0])
    flipped = flipped[np.newaxis, :, :, np.newaxis]
    flipped = np.broadcast_to(flipped, (6, 90, 120, 3))
    self.assertTrue((flipped_frames == self._np_frames).numpy().all() or (
        flipped_frames == flipped).numpy().all())

  def test_normalize_image(self):
    normalized_images_1 = preprocess_ops_3d.normalize_image(
        self._frames, False, tf.float32)
    normalized_images_2 = preprocess_ops_3d.normalize_image(
        self._frames, True, tf.float32)

    self.assertAllClose(normalized_images_1, self._np_frames / 255)
    self.assertAllClose(normalized_images_2, self._np_frames * 2 / 255 - 1.0)


if __name__ == '__main__':
  tf.test.main()
