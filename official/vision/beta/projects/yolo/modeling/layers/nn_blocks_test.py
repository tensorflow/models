import tensorflow as tf
import tensorflow.keras as ks
import numpy as np
from absl.testing import parameterized

from official.vision.beta.projects.yolo.modeling.layers import nn_blocks



class CSPConnect(tf.test.TestCase, parameterized.TestCase):

  @parameterized.named_parameters(("same", 224, 224, 64, 1),
                                  ("downsample", 224, 224, 64, 2))
  def test_pass_through(self, width, height, filters, mod):
    x = ks.Input(shape=(width, height, filters))
    test_layer = nn_blocks.CSPDownSample(filters=filters, filter_reduce=mod)
    test_layer2 = nn_blocks.CSPConnect(filters=filters, filter_reduce=mod)
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
    test_layer = nn_blocks.CSPDownSample(filters, filter_reduce=mod)
    path_layer = nn_blocks.CSPConnect(filters, filter_reduce=mod)

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

class CSPDownSample(tf.test.TestCase, parameterized.TestCase):

  @parameterized.named_parameters(("same", 224, 224, 64, 1),
                                  ("downsample", 224, 224, 64, 2))
  def test_pass_through(self, width, height, filters, mod):
    x = ks.Input(shape=(width, height, filters))
    test_layer = nn_blocks.CSPDownSample(filters=filters, filter_reduce=mod)
    outx, px = test_layer(x)
    print(outx)
    print(outx.shape.as_list())
    self.assertAllEqual(
        outx.shape.as_list(),
        [None, np.ceil(width // 2),
         np.ceil(height // 2), (filters / mod)])

  @parameterized.named_parameters(("same", 224, 224, 64, 1),
                                  ("downsample", 224, 224, 128, 2))
  def test_gradient_pass_though(self, filters, width, height, mod):
    loss = ks.losses.MeanSquaredError()
    optimizer = ks.optimizers.SGD()
    test_layer = nn_blocks.CSPDownSample(filters, filter_reduce=mod)
    path_layer = nn_blocks.CSPConnect(filters, filter_reduce=mod)

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

class DarkConvTest(tf.test.TestCase, parameterized.TestCase):

  @parameterized.named_parameters(
      ("valid", (3, 3), "valid", (1, 1)), ("same", (3, 3), "same", (1, 1)),
      ("downsample", (3, 3), "same", (2, 2)), ("test", (1, 1), "valid", (1, 1)))
  def test_pass_through(self, kernel_size, padding, strides):
    if padding == "same":
      pad_const = 1
    else:
      pad_const = 0
    x = ks.Input(shape=(224, 224, 3))
    test_layer = nn_blocks.DarkConv(filters=64,
                          kernel_size=kernel_size,
                          padding=padding,
                          strides=strides,
                          trainable=False)
    outx = test_layer(x)
    print(outx.shape.as_list())
    test = [
        None,
        int((224 - kernel_size[0] + (2 * pad_const)) / strides[0] + 1),
        int((224 - kernel_size[1] + (2 * pad_const)) / strides[1] + 1), 64
    ]
    print(test)
    self.assertAllEqual(outx.shape.as_list(), test)

  @parameterized.named_parameters(("filters", 3))
  def test_gradient_pass_though(self, filters):
    loss = ks.losses.MeanSquaredError()
    optimizer = ks.optimizers.SGD()
    with tf.device("/CPU:0"):
      test_layer = nn_blocks.DarkConv(filters, kernel_size=(3, 3), padding="same")

    init = tf.random_normal_initializer()
    x = tf.Variable(initial_value=init(shape=(1, 224, 224,
                                              3), dtype=tf.float32))
    y = tf.Variable(
        initial_value=init(shape=(1, 224, 224, filters), dtype=tf.float32))

    with tf.GradientTape() as tape:
      x_hat = test_layer(x)
      grad_loss = loss(x_hat, y)
    grad = tape.gradient(grad_loss, test_layer.trainable_variables)
    optimizer.apply_gradients(zip(grad, test_layer.trainable_variables))
    self.assertNotIn(None, grad)

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

class DarkTinyTest(tf.test.TestCase, parameterized.TestCase):

  @parameterized.named_parameters(("middle", 224, 224, 64, 2),
                                  ("last", 224, 224, 1024, 1))
  def test_pass_through(self, width, height, filters, strides):
    x = ks.Input(shape=(width, height, filters))
    test_layer = nn_blocks.DarkTiny(filters=filters, strides=strides)
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
    test_layer = nn_blocks.DarkTiny(filters=filters, strides=strides)

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
