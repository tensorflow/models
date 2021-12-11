import matplotlib.pyplot as plt
import tensorflow as tf
from mpl_toolkits.mplot3d import art3d

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

def get_face_middle_pt(face, verts):
  v0, v1, v2 = verts[face[0]], verts[face[1]], verts[face[2]]
  return (v0 + v1 + v2) / 3

def visualize_mesh(verts, faces, verts_mask, faces_mask, normals):
  v = verts.numpy()
  f = faces.numpy()
  norms = normals.numpy()
  vm = verts_mask.numpy() == 1
  fm = faces_mask.numpy() == 1

  new_f = f[fm]

  fig = plt.figure()
  ax = fig.add_subplot(projection="3d")

  pc = art3d.Poly3DCollection(v[new_f], facecolors=(1, 0.5, 1, 0.3), edgecolor="black")

  ax.add_collection(pc)

  ################# Debugging - remove later  #################
  # cnt = 0
  # desired = 2
  #############################################################
  for i in range(fm.size):
    ################# Debugging - remove later  #################
    # if cnt > desired:
    #   break
    # elif cnt < desired:
    #   cnt += 1
    #   continue
    # else:
    #   cnt += 1
    #############################################################

    if fm[i] > 0:
      mid = get_face_middle_pt(f[i], v)
      ax.scatter(mid[0], mid[1], mid[2], c='blue', marker='.', s=20)
      norm = mid + norms[i]
      ax.scatter(norm[0], norm[1], norm[2], c='red', marker='.', s=20)

  for i in range(vm.size):
    if vm[i] > 0:
      ax.scatter(v[i][0], v[i][1], v[i][2], c='green', marker='*', s=50)

  for i in range(0,390,30):
    ax.view_init(elev=0, azim=i)
    plt.savefig(f"./official/vision/beta/projects/mesh_rcnn/ops/visualize_sampler_{i}.png")


if __name__ == '__main__':
  grid_dims = 2
  batch_size = 5
  occupancy_locs = [
      [0, 0, 0, 0], [0, 0, 0, 1], [0, 0, 1, 1],

      [1, 0, 0, 0], [1, 1, 1, 0], [1, 1, 0, 0], [1, 1, 0, 1],

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

  sample_meshes_graph = tf.function(sample_meshes)
  samples, normals = sample_meshes_graph(verts, verts_mask, faces,
                                   faces_mask, num_samples=4)

  batch_to_view = 1
  visualize_mesh(verts[batch_to_view, :],
                 faces[batch_to_view, :],
                 verts_mask[batch_to_view, :],
                 faces_mask[batch_to_view, :],
                 normals[batch_to_view, :]
                 )