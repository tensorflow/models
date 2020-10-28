import tensorflow as tf
import tensorflow.keras as ks
import tensorflow_datasets as tfds
from absl.testing import parameterized
from official.vision.beta.projects.yolo.modeling import layers as nn_blocks


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

if __name__ == "__main__":
  tf.test.main()
