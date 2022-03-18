from official.vision.beta.projects.mesh_rcnn.modeling.heads.mesh_head import \
    MeshHead
from official.vision.beta.projects.mesh_rcnn.ops.mesh_ops import \
    compute_mesh_shape
from official.vision.beta.projects.mesh_rcnn.ops.cubify import cubify
from official.vision.beta.projects.mesh_rcnn.utils.weight_utils.load_weights import (
    load_weights_mesh_head, pth_to_dict)
from official.vision.beta.projects.mesh_rcnn.ops.visualize_mesh import visualize_mesh

import tensorflow as tf
import numpy as np

PTH_PATH = r"C:\ML\Weights\meshrcnn_R50.pth"
BACKBONE_FEATURES = r"C:\ML\sofa_0134_mesh_features.npy"
VOXEL_HEAD_OUTPUT = r"C:\ML\sofa_0134_voxels.npy"

def print_layer_names(layers_dict, offset=0):
  if isinstance(layers_dict, dict):
    for k in layers_dict.keys():
      print(" " * offset + k)
      print_layer_names(layers_dict[k], offset+2)

def test_load_mesh_refinement_branch():
  weights_dict, n_read = pth_to_dict(PTH_PATH)

  grid_dims = 24
  mesh_shapes = compute_mesh_shape(1, grid_dims)
  verts_shape, verts_mask_shape, faces_shape, faces_mask_shape = mesh_shapes
  backbone_shape = [1, 14, 14, 256]
  input_specs = {
      'feature_map': backbone_shape,
      'verts': verts_shape,
      'verts_mask': verts_mask_shape,
      'faces': faces_shape,
      'faces_mask': faces_mask_shape
  }
  mesh_head = MeshHead(input_specs)

  n_weights = load_weights_mesh_head(
      mesh_head, weights_dict['roi_heads']['mesh_head'], 'pix3d')

  backbone_features = np.load(BACKBONE_FEATURES)
  voxels = np.load(VOXEL_HEAD_OUTPUT)

  backbone_features = tf.convert_to_tensor(backbone_features, tf.float32)
  voxels = tf.convert_to_tensor(voxels, tf.float32)

  backbone_features = tf.transpose(backbone_features, [0, 2, 3, 1])

  mesh = cubify(voxels, 0.2)
  verts = mesh['verts']
  faces = mesh['faces']
  verts_mask = mesh['verts_mask']
  faces_mask = mesh['faces_mask']

  inputs = {
      'feature_map': backbone_features,
      'verts': verts,
      'verts_mask': verts_mask,
      'faces': faces,
      'faces_mask': faces_mask
    }

  outputs = mesh_head(inputs)

  new_verts_0 = outputs['verts']['stage_0']
  new_verts_1 = outputs['verts']['stage_1']
  new_verts_2 = outputs['verts']['stage_2']

  visualize_mesh(verts[0, :], faces[0, :], verts_mask[0, :], faces_mask[0, :])

if __name__ == '__main__':
  test_load_mesh_refinement_branch()
