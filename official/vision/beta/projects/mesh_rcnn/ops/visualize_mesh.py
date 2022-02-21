"""Mesh Visualization"""

import matplotlib.pyplot as plt
import tensorflow as tf
from mpl_toolkits.mplot3d import art3d

from official.vision.beta.projects.mesh_rcnn.ops.cubify import cubify


def create_voxels(grid_dims, batch_size, occupancy_locs):
  ones = tf.ones(shape=[len(occupancy_locs)])
  voxels = tf.scatter_nd(
      indices=tf.convert_to_tensor(occupancy_locs, tf.int32),
      updates=ones,
      shape=[batch_size, grid_dims, grid_dims, grid_dims])

  return voxels

def visualize_mesh(verts, faces, verts_mask, faces_mask):
  v = verts.numpy()
  f = faces.numpy()
  vm = verts_mask.numpy() == 1
  fm = faces_mask.numpy() == 1

  new_f = f[fm]

  fig = plt.figure()
  ax = fig.add_subplot(projection="3d")

  pc = art3d.Poly3DCollection(
      v[new_f], facecolors=(1, 0.5, 1, 1), edgecolor="black")

  ax.add_collection(pc)

  plt.axis('off')
  plt.show()


if __name__ == '__main__':
  _grid_dims = 2
  _batch_size = 5
  _occupancy_locs = [
      [0, 0, 0, 0], [0, 0, 0, 1], [0, 0, 1, 1],

      [1, 0, 0, 0], [1, 1, 1, 0], [1, 1, 0, 0], [1, 1, 0, 1],

      [3, 0, 0, 0], [3, 0, 0, 1], [3, 0, 1, 0], [3, 0, 1, 1],
      [3, 1, 0, 0], [3, 1, 0, 1], [3, 1, 1, 0], [3, 1, 1, 1],
  ]
  voxels = create_voxels(_grid_dims, _batch_size, _occupancy_locs)
  _verts, _faces, _verts_mask, _faces_mask = cubify(voxels, 0.5)

  batch_to_view = 0
  visualize_mesh(_verts[batch_to_view, :],
                 _faces[batch_to_view, :],
                 _verts_mask[batch_to_view, :],
                 _faces_mask[batch_to_view, :])
