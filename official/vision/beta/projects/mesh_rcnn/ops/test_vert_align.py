"""Contains tests for Vertex Alignment Operation"""
import tensorflow as tf
import torch
import numpy as np
import pytorch3d
import pytorch3d.ops
from mesh_ops import vert_align

class VertAlignTest(tf.test.TestCase):
  def test_out_shape(self):
    feats = tf.constant(1.0, shape = (5,6,7,8))
    verts = tf.constant(1.0, shape = (5,10,3))
    sampled = vert_align(feats, verts, align_corners=True)
    self.assertEqual(sampled.shape, (5,10,6))

    sampled = vert_align(feats, verts, align_corners = False)
    self.assertEqual(sampled.shape, (5,10,6))

  def diff_torch_output(self, batch_size, channels, width, height, num_verts):

    np.random.seed(1)
    np_feats = np.random.uniform(-1000, 1000, (batch_size,channels,height,width))
    np_verts = np.random.uniform(-1000, 1000, (batch_size,num_verts,3))

    tf_feats = tf.convert_to_tensor(np_feats, dtype = tf.float32)
    tf_verts = tf.convert_to_tensor(np_verts, dtype = tf.float32)
    tf_sampled = vert_align(tf_feats,tf_verts, align_corners=True)

    torch_feats = torch.from_numpy(np_feats)
    torch_verts = torch.from_numpy(np_verts)
    torch_sampled = pytorch3d.ops.vert_align(torch_feats, torch_verts, padding_mode="border", align_corners=True)
    torch_sampled = torch_sampled.numpy()
    torch_sampled = tf.convert_to_tensor(torch_sampled, dtype = tf.float32)

    self.assertAllClose(tf_sampled, torch_sampled)

    
    tf_sampled = vert_align(tf_feats,tf_verts, align_corners=False)

    torch_sampled = pytorch3d.ops.vert_align(torch_feats, torch_verts, padding_mode="border", align_corners=False)
    torch_sampled = torch_sampled.numpy()
    torch_sampled = tf.convert_to_tensor(torch_sampled, dtype = tf.float32)

    self.assertAllClose(tf_sampled, torch_sampled)
    
# vat = VertAlignTest()
# vat.test_out_shape()
# vat.diff_torch_output(5,5,5,5,5)