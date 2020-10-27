import tensorflow as tf
import tensorflow.keras as ks
import numpy as np
from absl.testing import parameterized

from official.vision.beta.projects.yolo.modeling import building_blocks as nn_blocks

class DarkResidualTest(tf.test.TestCase, parameterized.TestCase):

  @parameterized.named_parameters(("same", 224, 224, 64, False),
                                  ("downsample", 223, 223, 32, True),
                                  ("oddball", 223, 223, 32, False))
  def test_pass_through(self, width, height, filters, downsample):
    mod = 1
    if downsample:
      mod = 2
    x = ks.Input(shape=(width, height, filters))
    test_layer = nn_blocks.DarkResidual(filters=filters, downsample=downsample)
    outx = test_layer(x)
    print(outx)
    print(outx.shape.as_list())
    self.assertAllEqual(
        outx.shape.as_list(),
        [None, np.ceil(width / mod),
         np.ceil(height / mod), filters])

  @parameterized.named_parameters(("same", 64, 224, 224, False),
                                  ("downsample", 32, 223, 223, True),
                                  ("oddball", 32, 223, 223, False))
  def test_gradient_pass_though(self, filters, width, height, downsample):
    loss = ks.losses.MeanSquaredError()
    optimizer = ks.optimizers.SGD()
    test_layer = nn_blocks.DarkResidual(filters, downsample=downsample)

    if downsample:
      mod = 2
    else:
      mod = 1

    init = tf.random_normal_initializer()
    x = tf.Variable(
        initial_value=init(shape=(1, width, height, filters), dtype=tf.float32))
    y = tf.Variable(initial_value=init(shape=(1, int(np.ceil(width / mod)),
                                              int(np.ceil(height / mod)),
                                              filters),
                                       dtype=tf.float32))

    with tf.GradientTape() as tape:
      x_hat = test_layer(x)
      grad_loss = loss(x_hat, y)
    grad = tape.gradient(grad_loss, test_layer.trainable_variables)
    optimizer.apply_gradients(zip(grad, test_layer.trainable_variables))

    self.assertNotIn(None, grad)


if __name__ == "__main__":
  tf.test.main()
