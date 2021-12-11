import tensorflow as tf

from official.vision.beta.projects.mesh_rcnn.ops.cubify import cubify
from official.vision.beta.projects.mesh_rcnn.ops.mesh_sample import \
    sample_meshes


def create_voxels(grid_dims, batch_size, occupancy_locs):
  ones = tf.ones(shape=[len(occupancy_locs)])
  voxels = tf.scatter_nd(
      indices=tf.convert_to_tensor(occupancy_locs, tf.int32),
      updates=ones,
      shape=[batch_size, grid_dims, grid_dims, grid_dims])

  return voxels

if __name__ == "__main__":
  grid_dims = 2
  batch_size = 5
  occupancy_locs = [
      [3, 0, 0, 0], [3, 0, 0, 1], [3, 0, 1, 0], [3, 0, 1, 1],
      [3, 1, 0, 0], [3, 1, 0, 1], [3, 1, 1, 0], [3, 1, 1, 1],
  ]
  voxels = create_voxels(grid_dims, batch_size, occupancy_locs)
  verts, faces, verts_mask, faces_mask = cubify(voxels, 0.5)

  verts = tf.cast(verts, dtype=tf.float32)
  # print(verts)
  # print("\n\n")
  faces = tf.cast(faces, dtype=tf.int32)
  # print(faces)
  # print("\n\n")
  # print("\n\n")
  verts_mask = tf.cast(verts_mask, dtype=tf.int8)
  faces_mask = tf.cast(faces_mask, dtype=tf.int8)

  num_samples = 4
  sample_meshes_graph = tf.function(sample_meshes)
  samples, normals = sample_meshes_graph(verts, verts_mask, faces,
                                   faces_mask, num_samples)
