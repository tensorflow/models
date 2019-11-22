# Lint as: python2, python3
# Copyright 2018 The TensorFlow Authors All Rights Reserved.
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

"""Tests for preprocess_utils."""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import numpy as np
from six.moves import range
import tensorflow as tf

from deeplab.core import preprocess_utils


class PreprocessUtilsTest(tf.test.TestCase):

  def testNoFlipWhenProbIsZero(self):
    numpy_image = np.dstack([[[5., 6.],
                              [9., 0.]],
                             [[4., 3.],
                              [3., 5.]]])
    image = tf.convert_to_tensor(numpy_image)

    with self.test_session():
      actual, is_flipped = preprocess_utils.flip_dim([image], prob=0, dim=0)
      self.assertAllEqual(numpy_image, actual.eval())
      self.assertAllEqual(False, is_flipped.eval())
      actual, is_flipped = preprocess_utils.flip_dim([image], prob=0, dim=1)
      self.assertAllEqual(numpy_image, actual.eval())
      self.assertAllEqual(False, is_flipped.eval())
      actual, is_flipped = preprocess_utils.flip_dim([image], prob=0, dim=2)
      self.assertAllEqual(numpy_image, actual.eval())
      self.assertAllEqual(False, is_flipped.eval())

  def testFlipWhenProbIsOne(self):
    numpy_image = np.dstack([[[5., 6.],
                              [9., 0.]],
                             [[4., 3.],
                              [3., 5.]]])
    dim0_flipped = np.dstack([[[9., 0.],
                               [5., 6.]],
                              [[3., 5.],
                               [4., 3.]]])
    dim1_flipped = np.dstack([[[6., 5.],
                               [0., 9.]],
                              [[3., 4.],
                               [5., 3.]]])
    dim2_flipped = np.dstack([[[4., 3.],
                               [3., 5.]],
                              [[5., 6.],
                               [9., 0.]]])
    image = tf.convert_to_tensor(numpy_image)

    with self.test_session():
      actual, is_flipped = preprocess_utils.flip_dim([image], prob=1, dim=0)
      self.assertAllEqual(dim0_flipped, actual.eval())
      self.assertAllEqual(True, is_flipped.eval())
      actual, is_flipped = preprocess_utils.flip_dim([image], prob=1, dim=1)
      self.assertAllEqual(dim1_flipped, actual.eval())
      self.assertAllEqual(True, is_flipped.eval())
      actual, is_flipped = preprocess_utils.flip_dim([image], prob=1, dim=2)
      self.assertAllEqual(dim2_flipped, actual.eval())
      self.assertAllEqual(True, is_flipped.eval())

  def testFlipMultipleImagesConsistentlyWhenProbIsOne(self):
    numpy_image = np.dstack([[[5., 6.],
                              [9., 0.]],
                             [[4., 3.],
                              [3., 5.]]])
    numpy_label = np.dstack([[[0., 1.],
                              [2., 3.]]])
    image_dim1_flipped = np.dstack([[[6., 5.],
                                     [0., 9.]],
                                    [[3., 4.],
                                     [5., 3.]]])
    label_dim1_flipped = np.dstack([[[1., 0.],
                                     [3., 2.]]])
    image = tf.convert_to_tensor(numpy_image)
    label = tf.convert_to_tensor(numpy_label)

    with self.test_session() as sess:
      image, label, is_flipped = preprocess_utils.flip_dim(
          [image, label], prob=1, dim=1)
      actual_image, actual_label = sess.run([image, label])
      self.assertAllEqual(image_dim1_flipped, actual_image)
      self.assertAllEqual(label_dim1_flipped, actual_label)
      self.assertEqual(True, is_flipped.eval())

  def testReturnRandomFlipsOnMultipleEvals(self):
    numpy_image = np.dstack([[[5., 6.],
                              [9., 0.]],
                             [[4., 3.],
                              [3., 5.]]])
    dim1_flipped = np.dstack([[[6., 5.],
                               [0., 9.]],
                              [[3., 4.],
                               [5., 3.]]])
    image = tf.convert_to_tensor(numpy_image)
    tf.compat.v1.set_random_seed(53)

    with self.test_session() as sess:
      actual, is_flipped = preprocess_utils.flip_dim(
          [image], prob=0.5, dim=1)
      actual_image, actual_is_flipped = sess.run([actual, is_flipped])
      self.assertAllEqual(numpy_image, actual_image)
      self.assertEqual(False, actual_is_flipped)
      actual_image, actual_is_flipped = sess.run([actual, is_flipped])
      self.assertAllEqual(dim1_flipped, actual_image)
      self.assertEqual(True, actual_is_flipped)

  def testReturnCorrectCropOfSingleImage(self):
    np.random.seed(0)

    height, width = 10, 20
    image = np.random.randint(0, 256, size=(height, width, 3))

    crop_height, crop_width = 2, 4

    image_placeholder = tf.placeholder(tf.int32, shape=(None, None, 3))
    [cropped] = preprocess_utils.random_crop([image_placeholder],
                                             crop_height,
                                             crop_width)

    with self.test_session():
      cropped_image = cropped.eval(feed_dict={image_placeholder: image})

    # Ensure we can find the cropped image in the original:
    is_found = False
    for x in range(0, width - crop_width + 1):
      for y in range(0, height - crop_height + 1):
        if np.isclose(image[y:y+crop_height, x:x+crop_width, :],
                      cropped_image).all():
          is_found = True
          break

    self.assertTrue(is_found)

  def testRandomCropMaintainsNumberOfChannels(self):
    np.random.seed(0)

    crop_height, crop_width = 10, 20
    image = np.random.randint(0, 256, size=(100, 200, 3))

    tf.compat.v1.set_random_seed(37)
    image_placeholder = tf.placeholder(tf.int32, shape=(None, None, 3))
    [cropped] = preprocess_utils.random_crop(
        [image_placeholder], crop_height, crop_width)

    with self.test_session():
      cropped_image = cropped.eval(feed_dict={image_placeholder: image})
      self.assertTupleEqual(cropped_image.shape, (crop_height, crop_width, 3))

  def testReturnDifferentCropAreasOnTwoEvals(self):
    tf.compat.v1.set_random_seed(0)

    crop_height, crop_width = 2, 3
    image = np.random.randint(0, 256, size=(100, 200, 3))
    image_placeholder = tf.placeholder(tf.int32, shape=(None, None, 3))
    [cropped] = preprocess_utils.random_crop(
        [image_placeholder], crop_height, crop_width)

    with self.test_session():
      crop0 = cropped.eval(feed_dict={image_placeholder: image})
      crop1 = cropped.eval(feed_dict={image_placeholder: image})
      self.assertFalse(np.isclose(crop0, crop1).all())

  def testReturnConsistenCropsOfImagesInTheList(self):
    tf.compat.v1.set_random_seed(0)

    height, width = 10, 20
    crop_height, crop_width = 2, 3
    labels = np.linspace(0, height * width-1, height * width)
    labels = labels.reshape((height, width, 1))
    image = np.tile(labels, (1, 1, 3))

    image_placeholder = tf.placeholder(tf.int32, shape=(None, None, 3))
    label_placeholder = tf.placeholder(tf.int32, shape=(None, None, 1))
    [cropped_image, cropped_label] = preprocess_utils.random_crop(
        [image_placeholder, label_placeholder], crop_height, crop_width)

    with self.test_session() as sess:
      cropped_image, cropped_labels = sess.run([cropped_image, cropped_label],
                                               feed_dict={
                                                   image_placeholder: image,
                                                   label_placeholder: labels})
      for i in range(3):
        self.assertAllEqual(cropped_image[:, :, i], cropped_labels.squeeze())

  def testDieOnRandomCropWhenImagesWithDifferentWidth(self):
    crop_height, crop_width = 2, 3
    image1 = tf.placeholder(tf.float32, name='image1', shape=(None, None, 3))
    image2 = tf.placeholder(tf.float32, name='image2', shape=(None, None, 1))
    cropped = preprocess_utils.random_crop(
        [image1, image2], crop_height, crop_width)

    with self.test_session() as sess:
      with self.assertRaises(tf.errors.InvalidArgumentError):
        sess.run(cropped, feed_dict={image1: np.random.rand(4, 5, 3),
                                     image2: np.random.rand(4, 6, 1)})

  def testDieOnRandomCropWhenImagesWithDifferentHeight(self):
    crop_height, crop_width = 2, 3
    image1 = tf.placeholder(tf.float32, name='image1', shape=(None, None, 3))
    image2 = tf.placeholder(tf.float32, name='image2', shape=(None, None, 1))
    cropped = preprocess_utils.random_crop(
        [image1, image2], crop_height, crop_width)

    with self.test_session() as sess:
      with self.assertRaisesWithPredicateMatch(
          tf.errors.InvalidArgumentError,
          'Wrong height for tensor'):
        sess.run(cropped, feed_dict={image1: np.random.rand(4, 5, 3),
                                     image2: np.random.rand(3, 5, 1)})

  def testDieOnRandomCropWhenCropSizeIsGreaterThanImage(self):
    crop_height, crop_width = 5, 9
    image1 = tf.placeholder(tf.float32, name='image1', shape=(None, None, 3))
    image2 = tf.placeholder(tf.float32, name='image2', shape=(None, None, 1))
    cropped = preprocess_utils.random_crop(
        [image1, image2], crop_height, crop_width)

    with self.test_session() as sess:
      with self.assertRaisesWithPredicateMatch(
          tf.errors.InvalidArgumentError,
          'Crop size greater than the image size.'):
        sess.run(cropped, feed_dict={image1: np.random.rand(4, 5, 3),
                                     image2: np.random.rand(4, 5, 1)})

  def testReturnPaddedImageWithNonZeroPadValue(self):
    for dtype in [np.int32, np.int64, np.float32, np.float64]:
      image = np.dstack([[[5, 6],
                          [9, 0]],
                         [[4, 3],
                          [3, 5]]]).astype(dtype)
      expected_image = np.dstack([[[255, 255, 255, 255, 255],
                                   [255, 255, 255, 255, 255],
                                   [255, 5, 6, 255, 255],
                                   [255, 9, 0, 255, 255],
                                   [255, 255, 255, 255, 255]],
                                  [[255, 255, 255, 255, 255],
                                   [255, 255, 255, 255, 255],
                                   [255, 4, 3, 255, 255],
                                   [255, 3, 5, 255, 255],
                                   [255, 255, 255, 255, 255]]]).astype(dtype)

      with self.session() as sess:
        padded_image = preprocess_utils.pad_to_bounding_box(
            image, 2, 1, 5, 5, 255)
        padded_image = sess.run(padded_image)
        self.assertAllClose(padded_image, expected_image)
        # Add batch size = 1 to image.
        padded_image = preprocess_utils.pad_to_bounding_box(
            np.expand_dims(image, 0), 2, 1, 5, 5, 255)
        padded_image = sess.run(padded_image)
        self.assertAllClose(padded_image, np.expand_dims(expected_image, 0))

  def testReturnOriginalImageWhenTargetSizeIsEqualToImageSize(self):
    image = np.dstack([[[5, 6],
                        [9, 0]],
                       [[4, 3],
                        [3, 5]]])
    with self.session() as sess:
      padded_image = preprocess_utils.pad_to_bounding_box(
          image, 0, 0, 2, 2, 255)
      padded_image = sess.run(padded_image)
      self.assertAllClose(padded_image, image)

  def testDieOnTargetSizeGreaterThanImageSize(self):
    image = np.dstack([[[5, 6],
                        [9, 0]],
                       [[4, 3],
                        [3, 5]]])
    with self.test_session():
      image_placeholder = tf.placeholder(tf.float32)
      padded_image = preprocess_utils.pad_to_bounding_box(
          image_placeholder, 0, 0, 2, 1, 255)
      with self.assertRaisesWithPredicateMatch(
          tf.errors.InvalidArgumentError,
          'target_width must be >= width'):
        padded_image.eval(feed_dict={image_placeholder: image})
      padded_image = preprocess_utils.pad_to_bounding_box(
          image_placeholder, 0, 0, 1, 2, 255)
      with self.assertRaisesWithPredicateMatch(
          tf.errors.InvalidArgumentError,
          'target_height must be >= height'):
        padded_image.eval(feed_dict={image_placeholder: image})

  def testDieIfTargetSizeNotPossibleWithGivenOffset(self):
    image = np.dstack([[[5, 6],
                        [9, 0]],
                       [[4, 3],
                        [3, 5]]])
    with self.test_session():
      image_placeholder = tf.placeholder(tf.float32)
      padded_image = preprocess_utils.pad_to_bounding_box(
          image_placeholder, 3, 0, 4, 4, 255)
      with self.assertRaisesWithPredicateMatch(
          tf.errors.InvalidArgumentError,
          'target size not possible with the given target offsets'):
        padded_image.eval(feed_dict={image_placeholder: image})

  def testDieIfImageTensorRankIsTwo(self):
    image = np.vstack([[5, 6],
                       [9, 0]])
    with self.test_session():
      image_placeholder = tf.placeholder(tf.float32)
      padded_image = preprocess_utils.pad_to_bounding_box(
          image_placeholder, 0, 0, 2, 2, 255)
      with self.assertRaisesWithPredicateMatch(
          tf.errors.InvalidArgumentError,
          'Wrong image tensor rank'):
        padded_image.eval(feed_dict={image_placeholder: image})

  def testResizeTensorsToRange(self):
    test_shapes = [[60, 40],
                   [15, 30],
                   [15, 50]]
    min_size = 50
    max_size = 100
    factor = None
    expected_shape_list = [(75, 50, 3),
                           (50, 100, 3),
                           (30, 100, 3)]
    for i, test_shape in enumerate(test_shapes):
      image = tf.random.normal([test_shape[0], test_shape[1], 3])
      new_tensor_list = preprocess_utils.resize_to_range(
          image=image,
          label=None,
          min_size=min_size,
          max_size=max_size,
          factor=factor,
          align_corners=True)
      with self.test_session() as session:
        resized_image = session.run(new_tensor_list[0])
        self.assertEqual(resized_image.shape, expected_shape_list[i])

  def testResizeTensorsToRangeWithFactor(self):
    test_shapes = [[60, 40],
                   [15, 30],
                   [15, 50]]
    min_size = 50
    max_size = 98
    factor = 8
    expected_image_shape_list = [(81, 57, 3),
                                 (49, 97, 3),
                                 (33, 97, 3)]
    expected_label_shape_list = [(81, 57, 1),
                                 (49, 97, 1),
                                 (33, 97, 1)]
    for i, test_shape in enumerate(test_shapes):
      image = tf.random.normal([test_shape[0], test_shape[1], 3])
      label = tf.random.normal([test_shape[0], test_shape[1], 1])
      new_tensor_list = preprocess_utils.resize_to_range(
          image=image,
          label=label,
          min_size=min_size,
          max_size=max_size,
          factor=factor,
          align_corners=True)
      with self.test_session() as session:
        new_tensor_list = session.run(new_tensor_list)
        self.assertEqual(new_tensor_list[0].shape, expected_image_shape_list[i])
        self.assertEqual(new_tensor_list[1].shape, expected_label_shape_list[i])

  def testResizeTensorsToRangeWithFactorAndLabelShapeCHW(self):
    test_shapes = [[60, 40],
                   [15, 30],
                   [15, 50]]
    min_size = 50
    max_size = 98
    factor = 8
    expected_image_shape_list = [(81, 57, 3),
                                 (49, 97, 3),
                                 (33, 97, 3)]
    expected_label_shape_list = [(5, 81, 57),
                                 (5, 49, 97),
                                 (5, 33, 97)]
    for i, test_shape in enumerate(test_shapes):
      image = tf.random.normal([test_shape[0], test_shape[1], 3])
      label = tf.random.normal([5, test_shape[0], test_shape[1]])
      new_tensor_list = preprocess_utils.resize_to_range(
          image=image,
          label=label,
          min_size=min_size,
          max_size=max_size,
          factor=factor,
          align_corners=True,
          label_layout_is_chw=True)
      with self.test_session() as session:
        new_tensor_list = session.run(new_tensor_list)
        self.assertEqual(new_tensor_list[0].shape, expected_image_shape_list[i])
        self.assertEqual(new_tensor_list[1].shape, expected_label_shape_list[i])

  def testResizeTensorsToRangeWithSimilarMinMaxSizes(self):
    test_shapes = [[60, 40],
                   [15, 30],
                   [15, 50]]
    # Values set so that one of the side = 97.
    min_size = 96
    max_size = 98
    factor = 8
    expected_image_shape_list = [(97, 65, 3),
                                 (49, 97, 3),
                                 (33, 97, 3)]
    expected_label_shape_list = [(97, 65, 1),
                                 (49, 97, 1),
                                 (33, 97, 1)]
    for i, test_shape in enumerate(test_shapes):
      image = tf.random.normal([test_shape[0], test_shape[1], 3])
      label = tf.random.normal([test_shape[0], test_shape[1], 1])
      new_tensor_list = preprocess_utils.resize_to_range(
          image=image,
          label=label,
          min_size=min_size,
          max_size=max_size,
          factor=factor,
          align_corners=True)
      with self.test_session() as session:
        new_tensor_list = session.run(new_tensor_list)
        self.assertEqual(new_tensor_list[0].shape, expected_image_shape_list[i])
        self.assertEqual(new_tensor_list[1].shape, expected_label_shape_list[i])

  def testResizeTensorsToRangeWithEqualMaxSize(self):
    test_shapes = [[97, 38],
                   [96, 97]]
    # Make max_size equal to the larger value of test_shapes.
    min_size = 97
    max_size = 97
    factor = 8
    expected_image_shape_list = [(97, 41, 3),
                                 (97, 97, 3)]
    expected_label_shape_list = [(97, 41, 1),
                                 (97, 97, 1)]
    for i, test_shape in enumerate(test_shapes):
      image = tf.random.normal([test_shape[0], test_shape[1], 3])
      label = tf.random.normal([test_shape[0], test_shape[1], 1])
      new_tensor_list = preprocess_utils.resize_to_range(
          image=image,
          label=label,
          min_size=min_size,
          max_size=max_size,
          factor=factor,
          align_corners=True)
      with self.test_session() as session:
        new_tensor_list = session.run(new_tensor_list)
        self.assertEqual(new_tensor_list[0].shape, expected_image_shape_list[i])
        self.assertEqual(new_tensor_list[1].shape, expected_label_shape_list[i])

  def testResizeTensorsToRangeWithPotentialErrorInTFCeil(self):
    test_shape = [3936, 5248]
    # Make max_size equal to the larger value of test_shapes.
    min_size = 1441
    max_size = 1441
    factor = 16
    expected_image_shape = (1089, 1441, 3)
    expected_label_shape = (1089, 1441, 1)
    image = tf.random.normal([test_shape[0], test_shape[1], 3])
    label = tf.random.normal([test_shape[0], test_shape[1], 1])
    new_tensor_list = preprocess_utils.resize_to_range(
        image=image,
        label=label,
        min_size=min_size,
        max_size=max_size,
        factor=factor,
        align_corners=True)
    with self.test_session() as session:
      new_tensor_list = session.run(new_tensor_list)
      self.assertEqual(new_tensor_list[0].shape, expected_image_shape)
      self.assertEqual(new_tensor_list[1].shape, expected_label_shape)

  def testResizeTensorsToRangeWithEqualMaxSizeWithoutAspectRatio(self):
    test_shapes = [[97, 38],
                   [96, 97]]
    # Make max_size equal to the larger value of test_shapes.
    min_size = 97
    max_size = 97
    factor = 8
    keep_aspect_ratio = False
    expected_image_shape_list = [(97, 97, 3),
                                 (97, 97, 3)]
    expected_label_shape_list = [(97, 97, 1),
                                 (97, 97, 1)]
    for i, test_shape in enumerate(test_shapes):
      image = tf.random.normal([test_shape[0], test_shape[1], 3])
      label = tf.random.normal([test_shape[0], test_shape[1], 1])
      new_tensor_list = preprocess_utils.resize_to_range(
          image=image,
          label=label,
          min_size=min_size,
          max_size=max_size,
          factor=factor,
          keep_aspect_ratio=keep_aspect_ratio,
          align_corners=True)
      with self.test_session() as session:
        new_tensor_list = session.run(new_tensor_list)
        self.assertEqual(new_tensor_list[0].shape, expected_image_shape_list[i])
        self.assertEqual(new_tensor_list[1].shape, expected_label_shape_list[i])


if __name__ == '__main__':
  tf.test.main()
