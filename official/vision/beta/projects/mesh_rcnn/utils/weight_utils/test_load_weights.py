import numpy as np
import tensorflow as tf
from matplotlib import pyplot as plt

from official.vision.beta.projects.mesh_rcnn.modeling.heads.mesh_head import \
    MeshHead
from official.vision.beta.projects.mesh_rcnn.ops.cubify import cubify
from official.vision.beta.projects.mesh_rcnn.ops.mesh_ops import \
    compute_mesh_shape
from official.vision.beta.projects.mesh_rcnn.ops.visualize_mesh import \
    visualize_mesh
from official.vision.beta.projects.mesh_rcnn.utils.weight_utils.load_weights import (
    load_weights_mesh_head, pth_to_dict)

PTH_PATH = r"C:\ML\Weights\meshrcnn_R50.pth"
BACKBONE_FEATURES = [
  r"C:\ML\sofa_0134_mesh_features.npy",
  r"C:\ML\bed_0003_mesh_features.npy",
  r"C:\ML\bookcase_0002_mesh_features.npy",
  r"C:\ML\chair_0093_mesh_features.npy",
  r"C:\ML\table_0169_mesh_features.npy",
]
VOXEL_HEAD_OUTPUTS = [
  r"C:\ML\sofa_0134_voxels.npy",
  r"C:\ML\bed_0003_voxels.npy",
  r"C:\ML\bookcase_0002_voxels.npy",
  r"C:\ML\chair_0093_voxels.npy",
  r"C:\ML\table_0169_voxels.npy",
]

def print_layer_names(layers_dict, offset=0):
  if isinstance(layers_dict, dict):
    for k in layers_dict.keys():
      print(" " * offset + k)
      print_layer_names(layers_dict[k], offset+2)

def test_load_mesh_refinement_branch():
  weights_dict, n_read = pth_to_dict(PTH_PATH)

  grid_dims = 24
  mesh_shapes = compute_mesh_shape(len(VOXEL_HEAD_OUTPUTS), grid_dims)
  verts_shape, verts_mask_shape, faces_shape, faces_mask_shape = mesh_shapes
  backbone_shape = [14, 14, 256]
  input_layer = {
      'feature_map': tf.keras.layers.Input(shape=backbone_shape),
      'verts': tf.keras.layers.Input(shape=verts_shape[1:]),
      'verts_mask': tf.keras.layers.Input(shape=verts_mask_shape[1:]),
      'faces': tf.keras.layers.Input(shape=faces_shape[1:]),
      'faces_mask': tf.keras.layers.Input(shape=faces_mask_shape[1:])
  }
  mesh_head = MeshHead()(input_layer)
  model = tf.keras.Model(inputs=[input_layer], outputs=[mesh_head])

  n_weights = load_weights_mesh_head(
      model, weights_dict['roi_heads']['mesh_head'], 'pix3d')

  batched_backbone_features = []
  print("backbone features shapes")
  for f in BACKBONE_FEATURES:
    backbone_features = np.load(f)
    print(backbone_features.shape)
    batched_backbone_features.append(backbone_features)
  
  batched_backbone_features = np.concatenate(batched_backbone_features, axis=0)

  batched_voxels = []
  print("voxels shapes")
  for f in VOXEL_HEAD_OUTPUTS:
    voxels = np.load(f)
    print(voxels.shape)
    batched_voxels.append(voxels)
  
  batched_voxels = np.concatenate(batched_voxels, axis=0)
  
  backbone_features = tf.convert_to_tensor(batched_backbone_features, tf.float32)
  backbone_features = tf.transpose(backbone_features, [0, 2, 3, 1])
  voxels = tf.convert_to_tensor(batched_voxels, tf.float32)

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

  outputs = model(inputs)[0]
  new_verts_0 = outputs['verts']['stage_0']
  new_verts_1 = outputs['verts']['stage_1']
  new_verts_2 = outputs['verts']['stage_2']

  batch_to_view = 1
  for batch_to_view in range(len(VOXEL_HEAD_OUTPUTS)):
    visualize_mesh(new_verts_2[batch_to_view, :],
                  faces[batch_to_view, :],
                  verts_mask[batch_to_view, :],
                  faces_mask[batch_to_view, :]
    )

  plt.show()

if __name__ == '__main__':
  test_load_mesh_refinement_branch()
