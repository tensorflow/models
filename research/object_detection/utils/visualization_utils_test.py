# Copyright 2017 The TensorFlow Authors. All Rights Reserved.
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

"""Tests for object_detection.utils.visualization_utils."""
import logging
import os

import numpy as np
import PIL.Image as Image
import tensorflow as tf

from object_detection.core import standard_fields as fields
from object_detection.utils import visualization_utils

_TESTDATA_PATH = 'object_detection/test_images'


class VisualizationUtilsTest(tf.test.TestCase):

  def test_get_prime_multiplier_for_color_randomness(self):
    # Show that default multipler is not 1 and does not divide the total number
    # of standard colors.
    multiplier = visualization_utils._get_multiplier_for_color_randomness()
    self.assertNotEqual(
        0, multiplier % len(visualization_utils.STANDARD_COLORS))
    self.assertNotEqual(1, multiplier)

    # Show that with 34 colors, the closest prime number to 34/10 that
    # satisfies the constraints is 5.
    visualization_utils.STANDARD_COLORS = [
        'color_{}'.format(str(i)) for i in range(34)
    ]
    multiplier = visualization_utils._get_multiplier_for_color_randomness()
    self.assertEqual(5, multiplier)

    # Show that with 110 colors, the closest prime number to 110/10 that
    # satisfies the constraints is 13 (since 11 equally divides 110).
    visualization_utils.STANDARD_COLORS = [
        'color_{}'.format(str(i)) for i in range(110)
    ]
    multiplier = visualization_utils._get_multiplier_for_color_randomness()
    self.assertEqual(13, multiplier)

  def create_colorful_test_image(self):
    """This function creates an image that can be used to test vis functions.

    It makes an image composed of four colored rectangles.

    Returns:
      colorful test numpy array image.
    """
    ch255 = np.full([100, 200, 1], 255, dtype=np.uint8)
    ch128 = np.full([100, 200, 1], 128, dtype=np.uint8)
    ch0 = np.full([100, 200, 1], 0, dtype=np.uint8)
    imr = np.concatenate((ch255, ch128, ch128), axis=2)
    img = np.concatenate((ch255, ch255, ch0), axis=2)
    imb = np.concatenate((ch255, ch0, ch255), axis=2)
    imw = np.concatenate((ch128, ch128, ch128), axis=2)
    imu = np.concatenate((imr, img), axis=1)
    imd = np.concatenate((imb, imw), axis=1)
    image = np.concatenate((imu, imd), axis=0)
    return image

  def create_test_image_with_five_channels(self):
    return np.full([100, 200, 5], 255, dtype=np.uint8)

  def create_test_grayscale_image(self):
    return np.full([100, 200, 1], 255, dtype=np.uint8)

  def test_draw_bounding_box_on_image(self):
    test_image = self.create_colorful_test_image()
    test_image = Image.fromarray(test_image)
    width_original, height_original = test_image.size
    ymin = 0.25
    ymax = 0.75
    xmin = 0.4
    xmax = 0.6

    visualization_utils.draw_bounding_box_on_image(test_image, ymin, xmin, ymax,
                                                   xmax)
    width_final, height_final = test_image.size

    self.assertEqual(width_original, width_final)
    self.assertEqual(height_original, height_final)

  def test_draw_bounding_box_on_image_array(self):
    test_image = self.create_colorful_test_image()
    width_original = test_image.shape[0]
    height_original = test_image.shape[1]
    ymin = 0.25
    ymax = 0.75
    xmin = 0.4
    xmax = 0.6

    visualization_utils.draw_bounding_box_on_image_array(
        test_image, ymin, xmin, ymax, xmax)
    width_final = test_image.shape[0]
    height_final = test_image.shape[1]

    self.assertEqual(width_original, width_final)
    self.assertEqual(height_original, height_final)

  def test_draw_bounding_boxes_on_image(self):
    test_image = self.create_colorful_test_image()
    test_image = Image.fromarray(test_image)
    width_original, height_original = test_image.size
    boxes = np.array([[0.25, 0.75, 0.4, 0.6],
                      [0.1, 0.1, 0.9, 0.9]])

    visualization_utils.draw_bounding_boxes_on_image(test_image, boxes)
    width_final, height_final = test_image.size

    self.assertEqual(width_original, width_final)
    self.assertEqual(height_original, height_final)

  def test_draw_bounding_boxes_on_image_array(self):
    test_image = self.create_colorful_test_image()
    width_original = test_image.shape[0]
    height_original = test_image.shape[1]
    boxes = np.array([[0.25, 0.75, 0.4, 0.6],
                      [0.1, 0.1, 0.9, 0.9]])

    visualization_utils.draw_bounding_boxes_on_image_array(test_image, boxes)
    width_final = test_image.shape[0]
    height_final = test_image.shape[1]

    self.assertEqual(width_original, width_final)
    self.assertEqual(height_original, height_final)

  def test_draw_bounding_boxes_on_image_tensors(self):
    """Tests that bounding box utility produces reasonable results."""
    category_index = {1: {'id': 1, 'name': 'dog'}, 2: {'id': 2, 'name': 'cat'}}

    fname = os.path.join(_TESTDATA_PATH, 'image1.jpg')
    image_np = np.array(Image.open(fname))
    images_np = np.stack((image_np, image_np), axis=0)
    original_image_shape = [[636, 512], [636, 512]]

    with tf.Graph().as_default():
      images_tensor = tf.constant(value=images_np, dtype=tf.uint8)
      image_shape = tf.constant(original_image_shape, dtype=tf.int32)
      boxes = tf.constant([[[0.4, 0.25, 0.75, 0.75], [0.5, 0.3, 0.6, 0.9]],
                           [[0.25, 0.25, 0.75, 0.75], [0.1, 0.3, 0.6, 1.0]]])
      classes = tf.constant([[1, 1], [1, 2]], dtype=tf.int64)
      scores = tf.constant([[0.8, 0.1], [0.6, 0.5]])
      images_with_boxes = (
          visualization_utils.draw_bounding_boxes_on_image_tensors(
              images_tensor,
              boxes,
              classes,
              scores,
              category_index,
              original_image_spatial_shape=image_shape,
              true_image_shape=image_shape,
              min_score_thresh=0.2))

      with self.test_session() as sess:
        sess.run(tf.global_variables_initializer())

        # Write output images for visualization.
        images_with_boxes_np = sess.run(images_with_boxes)
        self.assertEqual(images_np.shape[0], images_with_boxes_np.shape[0])
        self.assertEqual(images_np.shape[3], images_with_boxes_np.shape[3])
        self.assertEqual(
            tuple(original_image_shape[0]), images_with_boxes_np.shape[1:3])
        for i in range(images_with_boxes_np.shape[0]):
          img_name = 'image_' + str(i) + '.png'
          output_file = os.path.join(self.get_temp_dir(), img_name)
          logging.info('Writing output image %d to %s', i, output_file)
          image_pil = Image.fromarray(images_with_boxes_np[i, ...])
          image_pil.save(output_file)

  def test_draw_bounding_boxes_on_image_tensors_with_track_ids(self):
    """Tests that bounding box utility produces reasonable results."""
    category_index = {1: {'id': 1, 'name': 'dog'}, 2: {'id': 2, 'name': 'cat'}}

    fname = os.path.join(_TESTDATA_PATH, 'image1.jpg')
    image_np = np.array(Image.open(fname))
    images_np = np.stack((image_np, image_np), axis=0)
    original_image_shape = [[636, 512], [636, 512]]

    with tf.Graph().as_default():
      images_tensor = tf.constant(value=images_np, dtype=tf.uint8)
      image_shape = tf.constant(original_image_shape, dtype=tf.int32)
      boxes = tf.constant([[[0.4, 0.25, 0.75, 0.75],
                            [0.5, 0.3, 0.7, 0.9],
                            [0.7, 0.5, 0.8, 0.9]],
                           [[0.41, 0.25, 0.75, 0.75],
                            [0.51, 0.3, 0.7, 0.9],
                            [0.75, 0.5, 0.8, 0.9]]])
      classes = tf.constant([[1, 1, 2], [1, 1, 2]], dtype=tf.int64)
      scores = tf.constant([[0.8, 0.5, 0.7], [0.6, 0.5, 0.8]])
      track_ids = tf.constant([[3, 9, 7], [3, 9, 144]], dtype=tf.int32)
      images_with_boxes = (
          visualization_utils.draw_bounding_boxes_on_image_tensors(
              images_tensor,
              boxes,
              classes,
              scores,
              category_index,
              original_image_spatial_shape=image_shape,
              true_image_shape=image_shape,
              track_ids=track_ids,
              min_score_thresh=0.2))

      with self.test_session() as sess:
        sess.run(tf.global_variables_initializer())

        # Write output images for visualization.
        images_with_boxes_np = sess.run(images_with_boxes)
        self.assertEqual(images_np.shape[0], images_with_boxes_np.shape[0])
        self.assertEqual(images_np.shape[3], images_with_boxes_np.shape[3])
        self.assertEqual(
            tuple(original_image_shape[0]), images_with_boxes_np.shape[1:3])
        for i in range(images_with_boxes_np.shape[0]):
          img_name = 'image_with_track_ids_' + str(i) + '.png'
          output_file = os.path.join(self.get_temp_dir(), img_name)
          logging.info('Writing output image %d to %s', i, output_file)
          image_pil = Image.fromarray(images_with_boxes_np[i, ...])
          image_pil.save(output_file)

  def test_draw_bounding_boxes_on_image_tensors_with_additional_channels(self):
    """Tests the case where input image tensor has more than 3 channels."""
    category_index = {1: {'id': 1, 'name': 'dog'}}
    image_np = self.create_test_image_with_five_channels()
    images_np = np.stack((image_np, image_np), axis=0)

    with tf.Graph().as_default():
      images_tensor = tf.constant(value=images_np, dtype=tf.uint8)
      boxes = tf.constant(0, dtype=tf.float32, shape=[2, 0, 4])
      classes = tf.constant(0, dtype=tf.int64, shape=[2, 0])
      scores = tf.constant(0, dtype=tf.float32, shape=[2, 0])
      images_with_boxes = (
          visualization_utils.draw_bounding_boxes_on_image_tensors(
              images_tensor,
              boxes,
              classes,
              scores,
              category_index,
              min_score_thresh=0.2))

      with self.test_session() as sess:
        sess.run(tf.global_variables_initializer())

        final_images_np = sess.run(images_with_boxes)
        self.assertEqual((2, 100, 200, 3), final_images_np.shape)

  def test_draw_bounding_boxes_on_image_tensors_grayscale(self):
    """Tests the case where input image tensor has one channel."""
    category_index = {1: {'id': 1, 'name': 'dog'}}
    image_np = self.create_test_grayscale_image()
    images_np = np.stack((image_np, image_np), axis=0)

    with tf.Graph().as_default():
      images_tensor = tf.constant(value=images_np, dtype=tf.uint8)
      image_shape = tf.constant([[100, 200], [100, 200]], dtype=tf.int32)
      boxes = tf.constant(0, dtype=tf.float32, shape=[2, 0, 4])
      classes = tf.constant(0, dtype=tf.int64, shape=[2, 0])
      scores = tf.constant(0, dtype=tf.float32, shape=[2, 0])
      images_with_boxes = (
          visualization_utils.draw_bounding_boxes_on_image_tensors(
              images_tensor,
              boxes,
              classes,
              scores,
              category_index,
              original_image_spatial_shape=image_shape,
              true_image_shape=image_shape,
              min_score_thresh=0.2))

      with self.test_session() as sess:
        sess.run(tf.global_variables_initializer())

        final_images_np = sess.run(images_with_boxes)
        self.assertEqual((2, 100, 200, 3), final_images_np.shape)

  def test_draw_keypoints_on_image(self):
    test_image = self.create_colorful_test_image()
    test_image = Image.fromarray(test_image)
    width_original, height_original = test_image.size
    keypoints = [[0.25, 0.75], [0.4, 0.6], [0.1, 0.1], [0.9, 0.9]]

    visualization_utils.draw_keypoints_on_image(test_image, keypoints)
    width_final, height_final = test_image.size

    self.assertEqual(width_original, width_final)
    self.assertEqual(height_original, height_final)

  def test_draw_keypoints_on_image_array(self):
    test_image = self.create_colorful_test_image()
    width_original = test_image.shape[0]
    height_original = test_image.shape[1]
    keypoints = [[0.25, 0.75], [0.4, 0.6], [0.1, 0.1], [0.9, 0.9]]

    visualization_utils.draw_keypoints_on_image_array(test_image, keypoints)
    width_final = test_image.shape[0]
    height_final = test_image.shape[1]

    self.assertEqual(width_original, width_final)
    self.assertEqual(height_original, height_final)

  def test_draw_mask_on_image_array(self):
    test_image = np.asarray([[[0, 0, 0], [0, 0, 0]],
                             [[0, 0, 0], [0, 0, 0]]], dtype=np.uint8)
    mask = np.asarray([[0, 1],
                       [1, 1]], dtype=np.uint8)
    expected_result = np.asarray([[[0, 0, 0], [0, 0, 127]],
                                  [[0, 0, 127], [0, 0, 127]]], dtype=np.uint8)
    visualization_utils.draw_mask_on_image_array(test_image, mask,
                                                 color='Blue', alpha=.5)
    self.assertAllEqual(test_image, expected_result)

  def test_add_cdf_image_summary(self):
    values = [0.1, 0.2, 0.3, 0.4, 0.42, 0.44, 0.46, 0.48, 0.50]
    visualization_utils.add_cdf_image_summary(values, 'PositiveAnchorLoss')
    cdf_image_summary = tf.get_collection(key=tf.GraphKeys.SUMMARIES)[0]
    with self.test_session():
      cdf_image_summary.eval()

  def test_add_hist_image_summary(self):
    values = [0.1, 0.2, 0.3, 0.4, 0.42, 0.44, 0.46, 0.48, 0.50]
    bins = [0.01 * i for i in range(101)]
    visualization_utils.add_hist_image_summary(values, bins,
                                               'ScoresDistribution')
    hist_image_summary = tf.get_collection(key=tf.GraphKeys.SUMMARIES)[0]
    with self.test_session():
      hist_image_summary.eval()

  def test_eval_metric_ops(self):
    category_index = {1: {'id': 1, 'name': 'dog'}, 2: {'id': 2, 'name': 'cat'}}
    max_examples_to_draw = 4
    metric_op_base = 'Detections_Left_Groundtruth_Right'
    eval_metric_ops = visualization_utils.VisualizeSingleFrameDetections(
        category_index,
        max_examples_to_draw=max_examples_to_draw,
        summary_name_prefix=metric_op_base)
    original_image = tf.placeholder(tf.uint8, [4, None, None, 3])
    original_image_spatial_shape = tf.placeholder(tf.int32, [4, 2])
    true_image_shape = tf.placeholder(tf.int32, [4, 3])
    detection_boxes = tf.random_uniform([4, 20, 4],
                                        minval=0.0,
                                        maxval=1.0,
                                        dtype=tf.float32)
    detection_classes = tf.random_uniform([4, 20],
                                          minval=1,
                                          maxval=3,
                                          dtype=tf.int64)
    detection_scores = tf.random_uniform([4, 20],
                                         minval=0.,
                                         maxval=1.,
                                         dtype=tf.float32)
    groundtruth_boxes = tf.random_uniform([4, 8, 4],
                                          minval=0.0,
                                          maxval=1.0,
                                          dtype=tf.float32)
    groundtruth_classes = tf.random_uniform([4, 8],
                                            minval=1,
                                            maxval=3,
                                            dtype=tf.int64)
    eval_dict = {
        fields.DetectionResultFields.detection_boxes:
            detection_boxes,
        fields.DetectionResultFields.detection_classes:
            detection_classes,
        fields.DetectionResultFields.detection_scores:
            detection_scores,
        fields.InputDataFields.original_image:
            original_image,
        fields.InputDataFields.original_image_spatial_shape: (
            original_image_spatial_shape),
        fields.InputDataFields.true_image_shape: (true_image_shape),
        fields.InputDataFields.groundtruth_boxes:
            groundtruth_boxes,
        fields.InputDataFields.groundtruth_classes:
            groundtruth_classes
    }
    metric_ops = eval_metric_ops.get_estimator_eval_metric_ops(eval_dict)
    _, update_op = metric_ops[metric_ops.keys()[0]]

    with self.test_session() as sess:
      sess.run(tf.global_variables_initializer())
      value_ops = {}
      for key, (value_op, _) in metric_ops.iteritems():
        value_ops[key] = value_op

      # First run enough update steps to surpass `max_examples_to_draw`.
      for i in range(max_examples_to_draw):
        # Use a unique image shape on each eval image.
        sess.run(
            update_op,
            feed_dict={
                original_image:
                    np.random.randint(
                        low=0,
                        high=256,
                        size=(4, 6 + i, 7 + i, 3),
                        dtype=np.uint8),
                original_image_spatial_shape: [[6 + i, 7 + i], [6 + i, 7 + i],
                                               [6 + i, 7 + i], [6 + i, 7 + i]],
                true_image_shape: [[6 + i, 7 + i, 3], [6 + i, 7 + i, 3],
                                   [6 + i, 7 + i, 3], [6 + i, 7 + i, 3]]
            })
      value_ops_out = sess.run(value_ops)
      for key, value_op in value_ops_out.iteritems():
        self.assertNotEqual('', value_op)

      # Now run fewer update steps than `max_examples_to_draw`. A single value
      # op will be the empty string, since not enough image summaries can be
      # produced.
      for i in range(max_examples_to_draw - 1):
        # Use a unique image shape on each eval image.
        sess.run(
            update_op,
            feed_dict={
                original_image:
                    np.random.randint(
                        low=0,
                        high=256,
                        size=(4, 6 + i, 7 + i, 3),
                        dtype=np.uint8),
                original_image_spatial_shape: [[6 + i, 7 + i], [6 + i, 7 + i],
                                               [6 + i, 7 + i], [6 + i, 7 + i]],
                true_image_shape: [[6 + i, 7 + i, 3], [6 + i, 7 + i, 3],
                                   [6 + i, 7 + i, 3], [6 + i, 7 + i, 3]]
            })
      value_ops_out = sess.run(value_ops)
      self.assertEqual(
          '',
          value_ops_out[metric_op_base + '/' + str(max_examples_to_draw - 1)])


if __name__ == '__main__':
  tf.test.main()
