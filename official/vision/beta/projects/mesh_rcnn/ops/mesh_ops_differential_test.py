"""Differential Test for Mesh Ops"""

import numpy as np
import tensorflow as tf
import torch
from absl.testing import parameterized
from pytorch3d.ops import vert_align as torch_vert_align

from official.vision.beta.projects.mesh_rcnn.ops.mesh_ops import \
    vert_align as tf_vert_align


class MeshOpsDifferentialTest(parameterized.TestCase, tf.test.TestCase):

  def test_vert_align(self):
    verts = np.random.uniform(low=-1.0, high=1.0, size=(2, 5, 3))
    feature_map = np.random.uniform(low=-1.0, high=1.0, size=(2, 10, 10, 15))

    tf_verts = tf.convert_to_tensor(verts, tf.float32)
    torch_verts = torch.as_tensor(verts)

    tf_feature_map = tf.convert_to_tensor(feature_map, tf.float32)
    torch_feature_map = torch.as_tensor(feature_map)
    torch_feature_map = torch.permute(torch_feature_map, (0, 3, 1, 2))

    torch_outputs = torch_vert_align(
        torch_feature_map, torch_verts, padding_mode='border')
    tf_outputs = tf_vert_align(tf_feature_map, tf_verts)

    torch_outputs = torch_outputs.numpy()
    tf_outputs = tf_outputs.numpy()

    self.assertAllClose(torch_outputs, tf_outputs)

if __name__ == "__main__":
  tf.test.main()
