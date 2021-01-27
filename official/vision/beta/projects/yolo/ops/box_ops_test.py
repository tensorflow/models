import numpy as np
import tensorflow as tf
from absl.testing import parameterized

from official.vision.beta.projects.yolo.ops import box_ops


class InputUtilsTest(parameterized.TestCase, tf.test.TestCase):

  @parameterized.parameters((1), (4))
  def testBoxConversions(self, num_boxes):
    boxes = tf.convert_to_tensor(np.random.rand(num_boxes, 4))
    expected_shape = np.array([num_boxes, 4])
    xywh_box = box_ops.yxyx_to_xcycwh(boxes)
    yxyx_box = box_ops.xcycwh_to_yxyx(boxes)
    xyxy_box = box_ops.xcycwh_to_xyxy(boxes)
    self.assertAllEqual(tf.shape(xywh_box).numpy(), expected_shape)
    self.assertAllEqual(tf.shape(yxyx_box).numpy(), expected_shape)
    self.assertAllEqual(tf.shape(xyxy_box).numpy(), expected_shape)

  @parameterized.parameters((1), (5), (7))
  def testIOUs(self, num_boxes):
    boxes = tf.convert_to_tensor(np.random.rand(num_boxes, 4))
    expected_shape = np.array([
        num_boxes,
    ])
    expected_iou = np.ones([
        num_boxes,
    ])
    iou = box_ops.compute_iou(boxes, boxes)
    _, giou = box_ops.compute_giou(boxes, boxes)
    _, ciou = box_ops.compute_ciou(boxes, boxes)
    _, diou = box_ops.compute_diou(boxes, boxes)
    self.assertAllEqual(tf.shape(iou).numpy(), expected_shape)
    self.assertArrayNear(iou, expected_iou, 0.001)
    self.assertArrayNear(giou, expected_iou, 0.001)
    self.assertArrayNear(ciou, expected_iou, 0.001)
    self.assertArrayNear(diou, expected_iou, 0.001)


if __name__ == '__main__':
  tf.test.main()
