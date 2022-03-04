from official.vision.beta.projects.mesh_rcnn.modeling.heads.z_head import \
    ZHead
# from ...modeling.heads.z_head import \
#     ZHead
# from official.vision.beta.projects.mesh_rcnn.ops.mesh_ops import \
#     compute_mesh_shape
# from official.vision.beta.projects.mesh_rcnn.utils.weight_utils.load_weights import (
#     load_weights_zhead, pth_to_dict)

from load_weights import load_weights_zhead, pth_to_dict

PTH_PATH = r"C:\\Users\\johnm\\Downloads\\meshrcnn_R50.pth"

def print_layer_names(layers_dict, offset=0):
  if isinstance(layers_dict, dict):
    for k in layers_dict.keys():
      print(" " * offset + k)
      print_layer_names(layers_dict[k], offset+2)

def test_load_zhead():
  weights_dict, n_read = pth_to_dict(PTH_PATH)
  print(weights_dict.keys())
  print(weights_dict['roi_heads'].keys())
  print(weights_dict['roi_heads']['z_head'].keys())
  print(weights_dict['roi_heads']['z_head']['z_pred'].keys())
  print(weights_dict['roi_heads']['z_head']['z_pred']['weight'].shape)
  # grid_dims = 24
  # mesh_shapes = compute_mesh_shape(1, grid_dims)
  # verts_shape, verts_mask_shape, faces_shape, faces_mask_shape = mesh_shapes
  # backbone_shape = [1, 14, 14, 256]
  # input_specs = {
  #     'feature_map': backbone_shape,
  #     'verts': verts_shape,
  #     'verts_mask': verts_mask_shape,
  #     'faces': faces_shape,
  #     'faces_mask': faces_mask_shape
  # }
  # mesh_head = MeshHead(input_specs)
  # mesh_head.summary()

  input_specs = dict(
    num_fc = 2,
    fc_dim = 1024,
    cls_agnostic = False,
    num_classes = 9
  )

  zhead = ZHead.from_config(input_specs)

  n_weights = load_weights_zhead(
      zhead, weights_dict['roi_heads']['z_head'], 'pix3d')

if __name__ == '__main__':
  test_load_zhead()