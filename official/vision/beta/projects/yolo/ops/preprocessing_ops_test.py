import numpy as np
import tensorflow as tf
from absl.testing import parameterized

from official.vision.beta.projects.yolo.ops import preprocessing_ops


class InputUtilsTest(parameterized.TestCase, tf.test.TestCase):

  @parameterized.parameters((416, 416, 5, 300, 300), (100, 200, 6, 50, 50))
  def testResizeCropFilter(self, default_width, default_height, num_boxes,
                           target_width, target_height):
    image = tf.convert_to_tensor(
        np.random.rand(default_width, default_height, 3))
    boxes = tf.convert_to_tensor(np.random.rand(num_boxes, 4))
    resized_image, resized_boxes = preprocessing_ops.resize_crop_filter(
        image, boxes, default_width, default_height, target_width,
        target_height)
    resized_image_shape = tf.shape(resized_image)
    resized_boxes_shape = tf.shape(resized_boxes)
    self.assertAllEqual([default_height, default_width, 3],
                        resized_image_shape.numpy())
    self.assertAllEqual([num_boxes, 4], resized_boxes_shape.numpy())

  @parameterized.parameters((7, 7., 5.), (25, 35., 45.))
  def testTranslateBoxes(self, num_boxes, translate_x, translate_y):
    boxes = tf.convert_to_tensor(np.random.rand(num_boxes, 4))
    translated_boxes = preprocessing_ops.translate_boxes(
        boxes, translate_x, translate_y)
    translated_boxes_shape = tf.shape(translated_boxes)
    self.assertAllEqual([num_boxes, 4], translated_boxes_shape.numpy())

  @parameterized.parameters((100, 200, 75., 25.), (400, 600, 25., 75.))
  def testTranslateImage(self, image_height, image_width, translate_x,
                         translate_y):
    image = tf.convert_to_tensor(np.random.rand(image_height, image_width, 4))
    translated_image = preprocessing_ops.translate_image(
        image, translate_x, translate_y)
    translated_image_shape = tf.shape(translated_image)
    self.assertAllEqual([image_height, image_width, 4],
                        translated_image_shape.numpy())

  @parameterized.parameters(([1, 2], 20, 0), ([13, 2, 4], 15, 0))
  def testPadMaxInstances(self, input_shape, instances, pad_axis):
    expected_output_shape = input_shape
    expected_output_shape[pad_axis] = instances
    output = preprocessing_ops.pad_max_instances(
        np.ones(input_shape), instances, pad_axis=pad_axis)
    self.assertAllEqual(expected_output_shape, tf.shape(output).numpy())


if __name__ == '__main__':
  tf.test.main()
