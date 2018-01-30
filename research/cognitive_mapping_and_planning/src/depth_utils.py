# Copyright 2016 The TensorFlow Authors All Rights Reserved.
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
# ==============================================================================

"""Utilities for processing depth images.
"""
import numpy as np
import src.rotation_utils as ru
import src.utils as utils

def get_camera_matrix(width, height, fov):
  """Returns a camera matrix from image size and fov."""
  xc = (width-1.) / 2.
  zc = (height-1.) / 2.
  f = (width / 2.) / np.tan(np.deg2rad(fov / 2.))
  camera_matrix = utils.Foo(xc=xc, zc=zc, f=f)
  return camera_matrix

def get_point_cloud_from_z(Y, camera_matrix):
  """Projects the depth image Y into a 3D point cloud.
  Inputs:
    Y is ...xHxW
    camera_matrix
  Outputs:
    X is positive going right
    Y is positive into the image
    Z is positive up in the image
    XYZ is ...xHxWx3
  """
  x, z = np.meshgrid(np.arange(Y.shape[-1]),
                     np.arange(Y.shape[-2]-1, -1, -1))
  for i in range(Y.ndim-2):
    x = np.expand_dims(x, axis=0)
    z = np.expand_dims(z, axis=0)
  X = (x-camera_matrix.xc) * Y / camera_matrix.f
  Z = (z-camera_matrix.zc) * Y / camera_matrix.f
  XYZ = np.concatenate((X[...,np.newaxis], Y[...,np.newaxis],
                        Z[...,np.newaxis]), axis=X.ndim)
  return XYZ

def make_geocentric(XYZ, sensor_height, camera_elevation_degree):
  """Transforms the point cloud into geocentric coordinate frame.
  Input:
    XYZ                     : ...x3
    sensor_height           : height of the sensor
    camera_elevation_degree : camera elevation to rectify.
  Output:
    XYZ : ...x3
  """
  R = ru.get_r_matrix([1.,0.,0.], angle=np.deg2rad(camera_elevation_degree))
  XYZ = np.matmul(XYZ.reshape(-1,3), R.T).reshape(XYZ.shape)
  XYZ[...,2] = XYZ[...,2] + sensor_height
  return XYZ

def bin_points(XYZ_cms, map_size, z_bins, xy_resolution):
  """Bins points into xy-z bins
  XYZ_cms is ... x H x W x3
  Outputs is ... x map_size x map_size x (len(z_bins)+1)
  """
  sh = XYZ_cms.shape
  XYZ_cms = XYZ_cms.reshape([-1, sh[-3], sh[-2], sh[-1]])
  n_z_bins = len(z_bins)+1
  map_center = (map_size-1.)/2.
  counts = []
  isvalids = []
  for XYZ_cm in XYZ_cms:
    isnotnan = np.logical_not(np.isnan(XYZ_cm[:,:,0]))
    X_bin = np.round(XYZ_cm[:,:,0] / xy_resolution + map_center).astype(np.int32)
    Y_bin = np.round(XYZ_cm[:,:,1] / xy_resolution + map_center).astype(np.int32)
    Z_bin = np.digitize(XYZ_cm[:,:,2], bins=z_bins).astype(np.int32)

    isvalid = np.array([X_bin >= 0, X_bin < map_size, Y_bin >= 0, Y_bin < map_size,
                        Z_bin >= 0, Z_bin < n_z_bins, isnotnan])
    isvalid = np.all(isvalid, axis=0)

    ind = (Y_bin * map_size + X_bin) * n_z_bins + Z_bin
    ind[np.logical_not(isvalid)] = 0
    count = np.bincount(ind.ravel(), isvalid.ravel().astype(np.int32),
                         minlength=map_size*map_size*n_z_bins)
    count = np.reshape(count, [map_size, map_size, n_z_bins])
    counts.append(count)
    isvalids.append(isvalid)
  counts = np.array(counts).reshape(list(sh[:-3]) + [map_size, map_size, n_z_bins])
  isvalids = np.array(isvalids).reshape(list(sh[:-3]) + [sh[-3], sh[-2], 1])
  return counts, isvalids
