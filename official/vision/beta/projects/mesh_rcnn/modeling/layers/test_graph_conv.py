"""Contains tests for Graph Convolution"""
import tensorflow as tf
from absl.testing import parameterized
import numpy as np
from nn_blocks import GraphConv, gather_scatter


class GraphConvTest(parameterized.TestCase, tf.test.TestCase):
    """Graph convolution tests"""
    def test_undirected(self):
        """Test with undirected graph"""
        verts = tf.constant([[1, 2, 3], [4, 5, 6], [7, 8, 9]], dtype=tf.float32)
        edges = tf.constant([[0, 1], [0, 2]])
        expected_y = tf.constant(
            [
                [1 + 2 + 3 - 4 - 5 - 6 - 7 - 8 - 9],
                [4 + 5 + 6 - 1 - 2 - 3],
                [7 + 8 + 9 - 1 - 2 - 3],
            ],
            dtype=tf.float32
        )

        conv = GraphConv(3, 1, directed=False)
        y = conv(verts, edges)

        w0 = np.array([[1.], [1.], [1.0]])
        w1 = np.array([[-1.], [-1.], [-1.0]])
        bias = np.array([0.])

        conv._w0.set_weights([w0, bias])
        conv._w1.set_weights([w1, bias])

        y = conv(verts, edges)

        self.assertAllEqual(y, expected_y)

    def test_no_edges(self):
        """Test with no edges"""
        dtype = tf.float32
        verts = tf.constant([[1, 2, 3], [4, 5, 6], [7, 8, 9]], dtype=tf.float32)
        edges = tf.zeros((0, 2), dtype=tf.int64)
        expected_y = tf.constant(
            [[1 - 2 - 2 * 3], [4 - 5 - 2 * 6], [7 - 8 - 2 * 9]], dtype=dtype
        )
        conv = GraphConv(3, 1)
        y = conv(verts, edges)

        w0 = np.array([[1.], [-1.], [-2.]])
        bias = np.array([0.])
        conv._w0.set_weights([w0, bias])

        y = conv(verts, edges)

        self.assertAllEqual(y, expected_y)

    def test_no_verts_and_edges(self):
        """Test with no vertices and no edges"""
        dtype = tf.float32

        verts = tf.constant([], dtype=dtype)
        edges = tf.constant([], dtype=dtype)

        conv = GraphConv(3, 1)
        conv.build((3,))
        y = conv(verts, edges)

        w0 = np.array([[1.], [-1.], [-2.]])
        bias = np.array([0.])
        conv._w0.set_weights([w0, bias])

        y = conv(verts, edges)
        self.assertAllEqual(y, tf.zeros((0, 1)))

        conv2 = GraphConv(3, 2)
        conv2.build((3,))
        y = conv(verts, edges)

        w0 = np.tile(w0, (1, 2))
        bias = np.array([0., 0.])

        conv2._w0.set_weights([w0, bias])
        y = conv2(verts, edges)

        self.assertAllEqual(y, tf.zeros((0, 2)))

    def test_directed(self):
        """Test with a directed graph"""
        dtype = tf.float32
        verts = tf.constant([[1, 2, 3], [4, 5, 6], [7, 8, 9]], dtype=dtype)
        edges = tf.constant([[0, 1], [0, 2]])
        expected_y = tf.constant(
            [
                [1 + 2 + 3 - 4 - 5 - 6 - 7 - 8 - 9],
                [4 + 5 + 6],
                [7 + 8 + 9],
            ],
            dtype=tf.float32
        )

        conv = GraphConv(3, 1, directed=True)
        y = conv(verts, edges)

        w0 = np.array([[1.], [1.], [1.0]])
        w1 = np.array([[-1.], [-1.], [-1.0]])
        bias = np.array([0.])

        conv._w0.set_weights([w0, bias])
        conv._w1.set_weights([w1, bias])

        y = conv(verts, edges)

        self.assertAllEqual(y, expected_y)

    def test_gather_scatter(self):
        """Test gather scatter"""
        dtype = tf.float32
        verts = tf.constant([[1, 2, 3], [4, 5, 6], [7, 8, 9]], dtype=dtype)
        edges = tf.constant([[0, 1], [0, 2]])
        expected_y = tf.constant(
            [
                [11., 13., 15.],
                [1., 2., 3.],
                [1., 2., 3.],
            ],
            dtype=tf.float32
        )
        y = gather_scatter(verts, edges)
        self.assertAllEqual(y, expected_y)

    def test_gradient_pass_through(self):
        """Test gradient pass through"""
        v, e = 100, 100
        verts = tf.random.uniform(shape=[v, 3], dtype=tf.dtypes.float32)
        edges = tf.random.uniform(shape=[e, 2], minval=0, maxval=v, dtype=tf.dtypes.int32)

        conv = GraphConv(3, 1, directed=False)

        with tf.GradientTape() as tape:
            y = conv(verts, edges)
        grads = tape.gradient(y, conv.trainable_variables)
        self.assertIsNotNone(grads)
