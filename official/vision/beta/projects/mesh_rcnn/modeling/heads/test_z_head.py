import tensorflow as tf
from absl.testing import parameterized
from official.vision.beta.projects.mesh_rcnn.modeling.heads import z_head

class ZHeadTest(parameterized.TestCase, tf.test.TestCase):
        
    def test_output_shape(self,
        num_fc: int,
        fc_dim: int,
        cls_agnostic: bool,
        num_classes: int,
        **kwargs):

        (batch_size, height, width, channels) = (64, 14, 14, 256)
        
        head = z_head.ZHead(num_fc, fc_dim, cls_agnostic, num_classes)
        # For example use batch size 64, h,w of 14, and 256 channels
        # NOTE: Don't actually know where the z-head attaches to architecture

        input = tf.zeros((batch_size, height, width, channels))
        output = head(input)
        expected_output = tf.zeros((batch_size, num_classes))
        self.assertAllEqual(output,expected_output)

    def test_serialize_deserialize(self,
        num_fc: int,
        fc_dim: int,
        cls_agnostic: bool,
        num_classes: int,
        **kwargs):

        (batch_size, height, width, channels) = (64, 14, 14, 256)
        head = z_head.ZHead(num_fc, fc_dim, cls_agnostic, num_classes)
        
        input = tf.zeros((batch_size, height, width, channels))
        _ = head(input)

        serialized = head.get_config()
        deserialized = z_head.ZHead.from_config(serialized)

        self.assertAllEqual(head.get_config(), deserialized.get_config())

    def test_gradient_pass_through(self,
        num_fc: int,
        fc_dim: int,
        cls_agnostic: bool,
        num_classes: int,
        **kwargs):

        (batch_size, height, width, channels) = (64, 14, 14, 256)
        head = z_head.ZHead(num_fc, fc_dim, cls_agnostic, num_classes)
        
        loss = tf.keras.losses.MeanSquaredError()
        optimizer = tf.keras.optimizers.SGD()

        init = tf.random_normal_initializer()
        input_shape = (batch_size, height, width, channels)
        x = tf.Variable(initial_value = init(shape=input_shape, dtype=tf.float32))

        output_shape = head(x).shape
        y = tf.Variable(initial_value = init(shape=output_shape, dtype=tf.float32))

        with tf.GradientTape() as tape:
            y_hat = head(x)
            grad_loss = loss(y_hat, y)
        grad = tape.gradient(grad_loss, head.trainable_variables)
        optimizer.apply_gradients(zip(grad, head.trainable_variables))

        self.assertNotIn(None, grad)

    def test_build_from_config(self,
        num_fc: int,
        fc_dim: int,
        cls_agnostic: bool,
        num_classes: int,
        **kwargs):
        pass

# zht = ZHeadTest()
# zht.test_output_shape(2, 1024, False, 100)
# zht.test_serialize_deserialize(2, 1024, False, 100)
# zht.test_gradient_pass_through(2, 1024, False, 100)
