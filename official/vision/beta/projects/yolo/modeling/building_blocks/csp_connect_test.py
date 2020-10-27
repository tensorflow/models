import tensorflow as tf
import tensorflow.keras as ks
import numpy as np
from absl.testing import parameterized

from official.vision.beta.projects.yolo.modeling.building_blocks import CSPDownSample as layer
from official.vision.beta.projects.yolo.modeling.building_blocks import CSPConnect as layer_companion


class CSPConnect(tf.test.TestCase, parameterized.TestCase):

  @parameterized.named_parameters(("same", 224, 224, 64, 1),
                                  ("downsample", 224, 224, 64, 2))
  def test_pass_through(self, width, height, filters, mod):
    x = ks.Input(shape=(width, height, filters))
    test_layer = layer(filters=filters, filter_reduce=mod)
    test_layer2 = layer_companion(filters=filters, filter_reduce=mod)
    outx, px = test_layer(x)
    outx = test_layer2([outx, px])
    print(outx)
    print(outx.shape.as_list())
    self.assertAllEqual(
        outx.shape.as_list(),
        [None, np.ceil(width // 2),
         np.ceil(height // 2), (filters)])

  @parameterized.named_parameters(("same", 224, 224, 64, 1),
                                  ("downsample", 224, 224, 128, 2))
  def test_gradient_pass_though(self, filters, width, height, mod):
    loss = ks.losses.MeanSquaredError()
    optimizer = ks.optimizers.SGD()
    test_layer = layer(filters, filter_reduce=mod)
    path_layer = layer_companion(filters, filter_reduce=mod)

    init = tf.random_normal_initializer()
    x = tf.Variable(
        initial_value=init(shape=(1, width, height, filters), dtype=tf.float32))
    y = tf.Variable(initial_value=init(shape=(1, int(np.ceil(width // 2)),
                                              int(np.ceil(height // 2)),
                                              filters),
                                       dtype=tf.float32))

    with tf.GradientTape() as tape:
      x_hat, x_prev = test_layer(x)
      x_hat = path_layer([x_hat, x_prev])
      grad_loss = loss(x_hat, y)
    grad = tape.gradient(grad_loss, test_layer.trainable_variables)
    optimizer.apply_gradients(zip(grad, test_layer.trainable_variables))

    self.assertNotIn(None, grad)


if __name__ == "__main__":
  tf.test.main()
