import tensorflow as tf
import tensorflow.keras as ks
import numpy as np
from absl.testing import parameterized

from official.vision.beta.projects.yolo.modeling.building_blocks import DarkTiny


class DarkTinyTest(tf.test.TestCase, parameterized.TestCase):

  @parameterized.named_parameters(("middle", 224, 224, 64, 2),
                                  ("last", 224, 224, 1024, 1))
  def test_pass_through(self, width, height, filters, strides):
    x = ks.Input(shape=(width, height, filters))
    test_layer = DarkTiny(filters=filters, strides=strides)
    outx = test_layer(x)
    self.assertEqual(width % strides, 0, msg="width % strides != 0")
    self.assertEqual(height % strides, 0, msg="height % strides != 0")
    self.assertAllEqual(outx.shape.as_list(),
                        [None, width // strides, height // strides, filters])

  @parameterized.named_parameters(("middle", 224, 224, 64, 2),
                                  ("last", 224, 224, 1024, 1))
  def test_gradient_pass_though(self, width, height, filters, strides):
    loss = ks.losses.MeanSquaredError()
    optimizer = ks.optimizers.SGD()
    test_layer = DarkTiny(filters=filters, strides=strides)

    init = tf.random_normal_initializer()
    x = tf.Variable(
        initial_value=init(shape=(1, width, height, filters), dtype=tf.float32))
    y = tf.Variable(initial_value=init(shape=(1, width // strides,
                                              height // strides, filters),
                                       dtype=tf.float32))

    with tf.GradientTape() as tape:
      x_hat = test_layer(x)
      grad_loss = loss(x_hat, y)
    grad = tape.gradient(grad_loss, test_layer.trainable_variables)
    optimizer.apply_gradients(zip(grad, test_layer.trainable_variables))

    self.assertNotIn(None, grad)


if __name__ == "__main__":
  tf.test.main()
