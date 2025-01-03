"""Tests for google3.third_party.tensorflow_models.object_detection.models.keras_models.nonlocal_block."""
import unittest
from absl.testing import parameterized
import tensorflow as tf

from object_detection.models.keras_models import nonlocal_block
from object_detection.utils import test_case
from object_detection.utils import tf_version


@unittest.skipIf(tf_version.is_tf1(), 'Skipping TF2.X only test.')
class NonlocalTest(test_case.TestCase, parameterized.TestCase):

  @parameterized.parameters([{'pool_size': None,
                              'add_coord_conv': False},
                             {'pool_size': None,
                              'add_coord_conv': True},
                             {'pool_size': 2,
                              'add_coord_conv': False},
                             {'pool_size': 2,
                              'add_coord_conv': True}])
  def test_run_nonlocal_block(self, pool_size, add_coord_conv):
    nonlocal_op = nonlocal_block.NonLocalBlock(
        8, pool_size=pool_size, add_coord_conv=add_coord_conv)
    def graph_fn():
      inputs = tf.zeros((4, 16, 16, 32), dtype=tf.float32)
      outputs = nonlocal_op(inputs)
      return outputs
    outputs = self.execute(graph_fn, [])
    self.assertAllEqual([4, 16, 16, 32], outputs.shape)


if __name__ == '__main__':
  tf.test.main()
