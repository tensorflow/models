# Copyright 2021 The TensorFlow Authors. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Script to visualize sampled points and their normals for a given mesh."""

import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf
from mpl_toolkits.mplot3d import art3d

from official.vision.beta.projects.mesh_rcnn.ops.cubify import cubify
from official.vision.beta.projects.mesh_rcnn.ops.mesh_sample import MeshSampler
from official.vision.beta.projects.mesh_rcnn.ops.voxel_ops import create_voxels

matplotlib.use("TkAgg") # Needed for showing plot when running from WSL.


def visualize_mesh(
    verts: tf.Tensor,
    faces: tf.Tensor,
    faces_mask: tf.Tensor,
    samples: tf.Tensor,
    normals: tf.Tensor,
) -> None:
  """Plot the mesh, the sampled points, and their normals.

  Args:
    verts: A float `Tensor` of shape [B, Nv, 3], where the last dimension
      contains all (x,y,z) vertex coordinates in the initial mesh.
    faces: An int `Tensor` of shape [B, Nf, 3], where the last dimension
      contains the verts indices that make up the face. This may include
      duplicate faces.
    faces_mask: An int `Tensor` of shape [B, Nf], representing a mask for
      valid faces in the watertight mesh.
    samples: A float `Tensor` of shape [B, Ns, 3] holding the coordinates
      of sampled points from each mesh in the batch. A samples matrix for a
      mesh will be 0 (i.e. samples[i, :, :] = 0) if the mesh is empty
      (i.e. verts_mask[i,:] all 0).
    normals:  A float `Tensor` of shape [B, Ns, 3] holding the normal vector
      for each sampled point. Like `samples`, an empty mesh will correspond
      to a 0 normals matrix.
  """
  v = verts.numpy()
  f = faces.numpy()
  smpls = samples.numpy()
  norms = normals.numpy()
  fm = faces_mask.numpy() == 1

  fig = plt.figure()
  ax = fig.add_subplot(projection="3d")

  new_f = f[fm]
  pc = art3d.Poly3DCollection(
      v[new_f], facecolors=(1, 0.5, 1, 0.3), edgecolor="black"
  )

  ax.add_collection(pc)

  # visualize the sampled points and their normals
  assert np.shape(smpls) == np.shape(norms)
  for i in range(np.shape(smpls)[0]):
    ax.quiver3D(smpls[i][0], smpls[i][1], smpls[i][2],
                norms[i][0], norms[i][1], norms[i][2])

  plt.show()


def main():
  tf.random.set_seed(1)

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

  verts = tf.cast(verts, tf.float32)
  faces = tf.cast(faces, tf.int32)
  verts_mask = tf.cast(verts_mask, tf.int8)
  faces_mask = tf.cast(faces_mask, tf.int8)

  sampler = MeshSampler(num_samples=100)
  samples, normals, _ = sampler.sample_meshes(
      verts, verts_mask, faces, faces_mask
  )

  batch_to_view = 1
  visualize_mesh(
      verts[batch_to_view, :],
      faces[batch_to_view, :],
      faces_mask[batch_to_view, :],
      samples[batch_to_view, :],
      normals[batch_to_view, :],
  )


if __name__ == "__main__":
  main()
