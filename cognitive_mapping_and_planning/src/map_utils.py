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

"""Various function to compute the ground truth map for training etc.
"""
import copy
import skimage.morphology
import numpy as np
import scipy.ndimage
import matplotlib.pyplot as plt
import PIL

import src.utils as utils
import cv2

def _get_xy_bounding_box(vertex, padding):
  """Returns the xy bounding box of the environment."""
  min_ = np.floor(np.min(vertex[:, :2], axis=0) - padding).astype(np.int)
  max_ = np.ceil(np.max(vertex[:, :2], axis=0) + padding).astype(np.int)
  return min_, max_

def _project_to_map(map, vertex, wt=None, ignore_points_outside_map=False):
  """Projects points to map, returns how many points are present at each
  location."""
  num_points = np.zeros((map.size[1], map.size[0]))
  vertex_ = vertex[:, :2] - map.origin
  vertex_ = np.round(vertex_ / map.resolution).astype(np.int)
  if ignore_points_outside_map:
    good_ind = np.all(np.array([vertex_[:,1] >= 0, vertex_[:,1] < map.size[1],
                                vertex_[:,0] >= 0, vertex_[:,0] < map.size[0]]),
                      axis=0)
    vertex_ = vertex_[good_ind, :]
    if wt is not None:
      wt = wt[good_ind, :]
  if wt is None:
    np.add.at(num_points, (vertex_[:, 1], vertex_[:, 0]), 1)
  else:
    assert(wt.shape[0] == vertex.shape[0]), \
      'number of weights should be same as vertices.'
    np.add.at(num_points, (vertex_[:, 1], vertex_[:, 0]), wt)
  return num_points

def make_map(padding, resolution, vertex=None, sc=1.):
  """Returns a map structure."""
  min_, max_ = _get_xy_bounding_box(vertex*sc, padding=padding)
  sz = np.ceil((max_ - min_ + 1) / resolution).astype(np.int32)
  max_ = min_ + sz * resolution - 1
  map = utils.Foo(origin=min_, size=sz, max=max_, resolution=resolution,
                  padding=padding)
  return map

def _fill_holes(img, thresh):
  """Fills holes less than thresh area (assumes 4 connectivity when computing
  hole area."""
  l, n = scipy.ndimage.label(np.logical_not(img))
  img_ = img == True
  cnts = np.bincount(l.reshape(-1))
  for i, cnt in enumerate(cnts):
    if cnt < thresh:
      l[l == i] = -1
  img_[l == -1] = True
  return img_

def compute_traversibility(map, robot_base, robot_height, robot_radius,
                           valid_min, valid_max, num_point_threshold, shapess,
                           sc=100., n_samples_per_face=200):
  """Returns a bit map with pixels that are traversible or not as long as the
  robot center is inside this volume we are good colisions can be detected by
  doing a line search on things, or walking from current location to final
  location in the bitmap, or doing bwlabel on the traversibility map."""

  tt = utils.Timer()
  tt.tic()
  num_obstcale_points = np.zeros((map.size[1], map.size[0]))
  num_points = np.zeros((map.size[1], map.size[0]))

  for i, shapes in enumerate(shapess):
    for j in range(shapes.get_number_of_meshes()):
      p, face_areas, face_idx = shapes.sample_points_on_face_of_shape(
          j, n_samples_per_face, sc)
      wt = face_areas[face_idx]/n_samples_per_face

      ind = np.all(np.concatenate(
        (p[:, [2]] > robot_base,
         p[:, [2]] < robot_base + robot_height), axis=1),axis=1)
      num_obstcale_points += _project_to_map(map, p[ind, :], wt[ind])

      ind = np.all(np.concatenate(
        (p[:, [2]] > valid_min,
         p[:, [2]] < valid_max), axis=1),axis=1)
      num_points += _project_to_map(map, p[ind, :], wt[ind])

  selem = skimage.morphology.disk(robot_radius / map.resolution)
  obstacle_free = skimage.morphology.binary_dilation(
      _fill_holes(num_obstcale_points > num_point_threshold, 20), selem) != True
  valid_space = _fill_holes(num_points > num_point_threshold, 20)
  traversible = np.all(np.concatenate((obstacle_free[...,np.newaxis],
                                       valid_space[...,np.newaxis]), axis=2),
                       axis=2)
  # plt.imshow(np.concatenate((obstacle_free, valid_space, traversible), axis=1))
  # plt.show()

  map_out = copy.deepcopy(map)
  map_out.num_obstcale_points = num_obstcale_points
  map_out.num_points = num_points
  map_out.traversible = traversible
  map_out.obstacle_free = obstacle_free
  map_out.valid_space = valid_space
  tt.toc(log_at=1, log_str='src.map_utils.compute_traversibility: ')
  return map_out


def resize_maps(map, map_scales, resize_method):
  scaled_maps = []
  for i, sc in enumerate(map_scales):
    if resize_method == 'antialiasing':
      # Resize using open cv so that we can compute the size.
      # Use PIL resize to use anti aliasing feature.
      map_ = cv2.resize(map*1, None, None, fx=sc, fy=sc, interpolation=cv2.INTER_LINEAR)
      w = map_.shape[1]; h = map_.shape[0]

      map_img = PIL.Image.fromarray((map*255).astype(np.uint8))
      map__img = map_img.resize((w,h), PIL.Image.ANTIALIAS)
      map_ = np.asarray(map__img).astype(np.float32)
      map_ = map_/255.
      map_ = np.minimum(map_, 1.0)
      map_ = np.maximum(map_, 0.0)
    elif resize_method == 'linear_noantialiasing':
      map_ = cv2.resize(map*1, None, None, fx=sc, fy=sc, interpolation=cv2.INTER_LINEAR)
    else:
      logging.error('Unknown resizing method')
    scaled_maps.append(map_)
  return scaled_maps


def pick_largest_cc(traversible):
  out = scipy.ndimage.label(traversible)[0]
  cnt = np.bincount(out.reshape(-1))[1:]
  return out == np.argmax(cnt) + 1

def get_graph_origin_loc(rng, traversible):
  """Erode the traversibility mask so that we get points in the bulk of the
  graph, and not end up with a situation where the graph is localized in the
  corner of a cramped room. Output Locs is in the coordinate frame of the
  map."""

  aa = pick_largest_cc(skimage.morphology.binary_erosion(traversible == True,
                                                         selem=np.ones((15,15))))
  y, x = np.where(aa > 0)
  ind = rng.choice(y.size)
  locs = np.array([x[ind], y[ind]])
  locs = locs + rng.rand(*(locs.shape)) - 0.5
  return locs


def generate_egocentric_maps(scaled_maps, map_scales, map_crop_sizes, loc,
                             x_axis, y_axis, theta):
  maps = []
  for i, (map_, sc, map_crop_size) in enumerate(zip(scaled_maps, map_scales, map_crop_sizes)):
    maps_i = np.array(get_map_to_predict(loc*sc, x_axis, y_axis, map_,
                                         map_crop_size,
                                         interpolation=cv2.INTER_LINEAR)[0])
    maps_i[np.isnan(maps_i)] = 0
    maps.append(maps_i)
  return maps

def generate_goal_images(map_scales, map_crop_sizes, n_ori, goal_dist,
                         goal_theta, rel_goal_orientation):
  goal_dist = goal_dist[:,0]
  goal_theta = goal_theta[:,0]
  rel_goal_orientation = rel_goal_orientation[:,0]

  goals = [];
  # Generate the map images.
  for i, (sc, map_crop_size) in enumerate(zip(map_scales, map_crop_sizes)):
    goal_i = np.zeros((goal_dist.shape[0], map_crop_size, map_crop_size, n_ori),
                      dtype=np.float32)
    x = goal_dist*np.cos(goal_theta)*sc + (map_crop_size-1.)/2.
    y = goal_dist*np.sin(goal_theta)*sc + (map_crop_size-1.)/2.

    for j in range(goal_dist.shape[0]):
      gc = rel_goal_orientation[j]
      x0 = np.floor(x[j]).astype(np.int32); x1 = x0 + 1;
      y0 = np.floor(y[j]).astype(np.int32); y1 = y0 + 1;
      if x0 >= 0 and x0 <= map_crop_size-1:
        if y0 >= 0 and y0 <= map_crop_size-1:
          goal_i[j, y0, x0, gc] = (x1-x[j])*(y1-y[j])
        if y1 >= 0 and y1 <= map_crop_size-1:
          goal_i[j, y1, x0, gc] = (x1-x[j])*(y[j]-y0)

      if x1 >= 0 and x1 <= map_crop_size-1:
        if y0 >= 0 and y0 <= map_crop_size-1:
          goal_i[j, y0, x1, gc] = (x[j]-x0)*(y1-y[j])
        if y1 >= 0 and y1 <= map_crop_size-1:
          goal_i[j, y1, x1, gc] = (x[j]-x0)*(y[j]-y0)

    goals.append(goal_i)
  return goals

def get_map_to_predict(src_locs, src_x_axiss, src_y_axiss, map, map_size,
                       interpolation=cv2.INTER_LINEAR):
  fss = []
  valids = []

  center = (map_size-1.0)/2.0
  dst_theta = np.pi/2.0
  dst_loc = np.array([center, center])
  dst_x_axis = np.array([np.cos(dst_theta), np.sin(dst_theta)])
  dst_y_axis = np.array([np.cos(dst_theta+np.pi/2), np.sin(dst_theta+np.pi/2)])

  def compute_points(center, x_axis, y_axis):
    points = np.zeros((3,2),dtype=np.float32)
    points[0,:] = center
    points[1,:] = center + x_axis
    points[2,:] = center + y_axis
    return points

  dst_points = compute_points(dst_loc, dst_x_axis, dst_y_axis)
  for i in range(src_locs.shape[0]):
    src_loc = src_locs[i,:]
    src_x_axis = src_x_axiss[i,:]
    src_y_axis = src_y_axiss[i,:]
    src_points = compute_points(src_loc, src_x_axis, src_y_axis)
    M = cv2.getAffineTransform(src_points, dst_points)

    fs = cv2.warpAffine(map, M, (map_size, map_size), None, flags=interpolation,
                        borderValue=np.NaN)
    valid = np.invert(np.isnan(fs))
    valids.append(valid)
    fss.append(fs)
  return fss, valids

