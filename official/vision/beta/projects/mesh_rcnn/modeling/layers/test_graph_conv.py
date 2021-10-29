import tensorflow as tf
from absl.testing import parameterized
import numpy as np
from nn_blocks import GraphConv, gather_scatter
class GraphConvTest(parameterized.TestCase, tf.test.TestCase):
    def test_undirected(self):
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
        b = np.array([0.])

        conv.w0.set_weights([w0, b])
        conv.w1.set_weights([w1, b])

        y = conv(verts, edges)

        self.assertAllEqual(y, expected_y)

    def test_no_edges(self):
        dtype = tf.float32
        verts = tf.constant([[1, 2, 3], [4, 5, 6], [7, 8, 9]], dtype=tf.float32)
        edges = tf.zeros((0, 2), dtype=tf.int64)
        expected_y = tf.constant(
            [[1 - 2 - 2 * 3], [4 - 5 - 2 * 6], [7 - 8 - 2 * 9]], dtype=dtype
        )
        conv = GraphConv(3, 1)
        y = conv(verts, edges)

        w0 = np.array([[1.], [-1.], [-2.]])
        b = np.array([0.])
        conv.w0.set_weights([w0, b])

        y = conv(verts, edges)

        self.assertAllEqual(y, expected_y)

    def test_no_verts_and_edges(self):
        dtype = tf.float32

        verts = tf.constant([], dtype=dtype)
        edges = tf.constant([], dtype=dtype)

        conv = GraphConv(3, 1)
        conv.build((3,))
        y = conv(verts, edges)

        w0 = np.array([[1.], [-1.], [-2.]])
        b = np.array([0.])
        conv.w0.set_weights([w0, b])

        y = conv(verts, edges)
        self.assertAllEqual(y, tf.zeros((0, 1)))

        conv2 = GraphConv(3, 2)
        conv2.build((3,))
        y = conv(verts, edges)

        w0 = np.tile(w0, (1, 2))
        b = np.array([0., 0.])

        conv2.w0.set_weights([w0, b])
        y = conv2(verts, edges)

        self.assertAllEqual(y, tf.zeros((0, 2)))

    def test_directed(self):
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
        b = np.array([0.])

        conv.w0.set_weights([w0, b])
        conv.w1.set_weights([w1, b])

        y = conv(verts, edges)

        self.assertAllEqual(y, expected_y)

    def test_gather_scatter(self):
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


gct = GraphConvTest()
gct.test_undirected()
gct.test_no_edges()
gct.test_no_verts_and_edges()
gct.test_directed()
gct.test_gather_scatter()