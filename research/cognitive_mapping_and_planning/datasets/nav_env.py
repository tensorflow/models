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

r"""Navidation Environment. Includes the following classes along with some
helper functions.
  Building: Loads buildings, computes traversibility, exposes functionality for
    rendering images.
  
  GridWorld: Base class which implements functionality for moving an agent on a
    grid world.
  
  NavigationEnv: Base class which generates navigation problems on a grid world.
  
  VisualNavigationEnv: Builds upon NavigationEnv and Building to provide
    interface that is used externally to train the agent. 
  
  MeshMapper: Class used for distilling the model, testing the mapper.
  
  BuildingMultiplexer: Wrapper class that instantiates a VisualNavigationEnv for
    each building and multiplexes between them as needed.
"""

import numpy as np
import os
import re
import matplotlib.pyplot as plt

import graph_tool as gt
import graph_tool.topology

from tensorflow.python.platform import gfile
import logging
import src.file_utils as fu
import src.utils as utils
import src.graph_utils as gu
import src.map_utils as mu
import src.depth_utils as du
import render.swiftshader_renderer as sru
from render.swiftshader_renderer import SwiftshaderRenderer
import cv2

label_nodes_with_class           = gu.label_nodes_with_class
label_nodes_with_class_geodesic  = gu.label_nodes_with_class_geodesic
get_distance_node_list           = gu.get_distance_node_list
convert_to_graph_tool            = gu.convert_to_graph_tool
generate_graph                   = gu.generate_graph
get_hardness_distribution        = gu.get_hardness_distribution
rng_next_goal_rejection_sampling = gu.rng_next_goal_rejection_sampling
rng_next_goal                    = gu.rng_next_goal
rng_room_to_room                 = gu.rng_room_to_room
rng_target_dist_field            = gu.rng_target_dist_field

compute_traversibility           = mu.compute_traversibility
make_map                         = mu.make_map
resize_maps                      = mu.resize_maps
pick_largest_cc                  = mu.pick_largest_cc
get_graph_origin_loc             = mu.get_graph_origin_loc
generate_egocentric_maps         = mu.generate_egocentric_maps
generate_goal_images             = mu.generate_goal_images
get_map_to_predict               = mu.get_map_to_predict

bin_points                       = du.bin_points
make_geocentric                  = du.make_geocentric
get_point_cloud_from_z           = du.get_point_cloud_from_z
get_camera_matrix                = du.get_camera_matrix

def _get_semantic_maps(folder_name, building_name, map, flip):
  # Load file from the cache.
  file_name = '{:s}_{:d}_{:d}_{:d}_{:d}_{:d}_{:d}.pkl'
  file_name = file_name.format(building_name, map.size[0], map.size[1],
                               map.origin[0], map.origin[1], map.resolution,
                               flip)
  file_name = os.path.join(folder_name, file_name)
  logging.info('Loading semantic maps from %s.', file_name)

  if fu.exists(file_name):
    a = utils.load_variables(file_name)
    maps = a['maps'] #HxWx#C
    cats = a['cats']
  else:
    logging.error('file_name: %s not found.', file_name)
    maps = None
    cats = None
  return maps, cats

def _select_classes(all_maps, all_cats, cats_to_use):
  inds = []
  for c in cats_to_use:
    ind = all_cats.index(c)
    inds.append(ind)
  out_maps = all_maps[:,:,inds]
  return out_maps

def _get_room_dimensions(file_name, resolution, origin, flip=False):
  if fu.exists(file_name):
    a = utils.load_variables(file_name)['room_dimension']
    names = a.keys()
    dims = np.concatenate(a.values(), axis=0).reshape((-1,6))
    ind = np.argsort(names)
    dims = dims[ind,:]
    names = [names[x] for x in ind]
    if flip:
      dims_new = dims*1
      dims_new[:,1] = -dims[:,4]
      dims_new[:,4] = -dims[:,1]
      dims = dims_new*1

    dims = dims*100.
    dims[:,0] = dims[:,0] - origin[0]
    dims[:,1] = dims[:,1] - origin[1]
    dims[:,3] = dims[:,3] - origin[0]
    dims[:,4] = dims[:,4] - origin[1]
    dims = dims / resolution
    out = {'names': names, 'dims': dims}
  else:
    out = None
  return out

def _filter_rooms(room_dims, room_regex):
  pattern = re.compile(room_regex)
  ind = []
  for i, name in enumerate(room_dims['names']):
    if pattern.match(name):
      ind.append(i)
  new_room_dims = {}
  new_room_dims['names'] = [room_dims['names'][i] for i in ind]
  new_room_dims['dims'] = room_dims['dims'][ind,:]*1
  return new_room_dims

def _label_nodes_with_room_id(xyt, room_dims):
  # Label the room with the ID into things.
  node_room_id = -1*np.ones((xyt.shape[0], 1))
  dims = room_dims['dims']
  for x, name in enumerate(room_dims['names']):
    all_ = np.concatenate((xyt[:,[0]] >= dims[x,0],
                           xyt[:,[0]] <= dims[x,3],
                           xyt[:,[1]] >= dims[x,1],
                           xyt[:,[1]] <= dims[x,4]), axis=1)
    node_room_id[np.all(all_, axis=1), 0] = x
  return node_room_id

def get_path_ids(start_node_id, end_node_id, pred_map):
  id = start_node_id
  path = [id]
  while id != end_node_id:
    id = pred_map[id]
    path.append(id)
  return path

def image_pre(images, modalities):
  # Assumes images are ...xHxWxC.
  # We always assume images are RGB followed by Depth.
  if 'depth' in modalities:
    d = images[...,-1][...,np.newaxis]*1.
    d[d < 0.01] = np.NaN; isnan = np.isnan(d);
    d = 100./d; d[isnan] = 0.;
    images = np.concatenate((images[...,:-1], d, isnan), axis=images.ndim-1)
  if 'rgb' in modalities:
    images[...,:3] = images[...,:3]*1. - 128
  return images

def _get_relative_goal_loc(goal_loc, loc, theta):
  r = np.sqrt(np.sum(np.square(goal_loc - loc), axis=1))
  t = np.arctan2(goal_loc[:,1] - loc[:,1], goal_loc[:,0] - loc[:,0])
  t = t-theta[:,0] + np.pi/2
  return np.expand_dims(r,axis=1), np.expand_dims(t, axis=1)

def _gen_perturbs(rng, batch_size, num_steps, lr_flip, delta_angle, delta_xy,
                  structured):
  perturbs = []
  for i in range(batch_size):
    # Doing things one by one for each episode in this batch. This way this
    # remains replicatable even when we change the batch size.
    p = np.zeros((num_steps+1, 4))
    if lr_flip:
      # Flip the whole trajectory.
      p[:,3] = rng.rand(1)-0.5
    if delta_angle > 0:
      if structured:
        p[:,2] = (rng.rand(1)-0.5)* delta_angle
      else:
        p[:,2] = (rng.rand(p.shape[0])-0.5)* delta_angle
    if delta_xy > 0:
      if structured:
        p[:,:2] = (rng.rand(1, 2)-0.5)*delta_xy
      else:
        p[:,:2] = (rng.rand(p.shape[0], 2)-0.5)*delta_xy
    perturbs.append(p)
  return perturbs

def get_multiplexer_class(args, task_number):
  assert(args.task_params.base_class == 'Building')
  logging.info('Returning BuildingMultiplexer')
  R = BuildingMultiplexer(args, task_number)
  return R

class GridWorld():
  def __init__(self):
    """Class members that will be assigned by any class that actually uses this
    class."""
    self.restrict_to_largest_cc = None
    self.robot = None
    self.env = None
    self.category_list = None
    self.traversible = None

  def get_loc_axis(self, node, delta_theta, perturb=None):
    """Based on the node orientation returns X, and Y axis. Used to sample the
    map in egocentric coordinate frame.
    """
    if type(node) == tuple:
      node = np.array([node])
    if perturb is None:
      perturb = np.zeros((node.shape[0], 4))
    xyt = self.to_actual_xyt_vec(node)
    x = xyt[:,[0]] + perturb[:,[0]]
    y = xyt[:,[1]] + perturb[:,[1]]
    t = xyt[:,[2]] + perturb[:,[2]]
    theta = t*delta_theta
    loc = np.concatenate((x,y), axis=1)
    x_axis = np.concatenate((np.cos(theta), np.sin(theta)), axis=1)
    y_axis = np.concatenate((np.cos(theta+np.pi/2.), np.sin(theta+np.pi/2.)),
                            axis=1)
    # Flip the sampled map where need be.
    y_axis[np.where(perturb[:,3] > 0)[0], :] *= -1.
    return loc, x_axis, y_axis, theta

  def to_actual_xyt(self, pqr):
    """Converts from node to location on the map."""
    (p, q, r) = pqr
    if self.task.n_ori == 6:
      out = (p - q * 0.5 + self.task.origin_loc[0],
             q * np.sqrt(3.) / 2. + self.task.origin_loc[1], r)
    elif self.task.n_ori == 4:
      out = (p + self.task.origin_loc[0],
             q + self.task.origin_loc[1], r)
    return out

  def to_actual_xyt_vec(self, pqr):
    """Converts from node array to location array on the map."""
    p = pqr[:,0][:, np.newaxis]
    q = pqr[:,1][:, np.newaxis]
    r = pqr[:,2][:, np.newaxis]
    if self.task.n_ori == 6:
      out = np.concatenate((p - q * 0.5 + self.task.origin_loc[0],
                            q * np.sqrt(3.) / 2. + self.task.origin_loc[1],
                            r), axis=1)
    elif self.task.n_ori == 4:
      out = np.concatenate((p + self.task.origin_loc[0],
                            q + self.task.origin_loc[1],
                            r), axis=1)
    return out

  def raw_valid_fn_vec(self, xyt):
    """Returns if the given set of nodes is valid or not."""
    height = self.traversible.shape[0]
    width = self.traversible.shape[1]
    x = np.round(xyt[:,[0]]).astype(np.int32)
    y = np.round(xyt[:,[1]]).astype(np.int32)
    is_inside = np.all(np.concatenate((x >= 0, y >= 0,
                                       x < width, y < height), axis=1), axis=1)
    x = np.minimum(np.maximum(x, 0), width-1)
    y = np.minimum(np.maximum(y, 0), height-1)
    ind = np.ravel_multi_index((y,x), self.traversible.shape)
    is_traversible = self.traversible.ravel()[ind]

    is_valid = np.all(np.concatenate((is_inside[:,np.newaxis], is_traversible),
                                     axis=1), axis=1)
    return is_valid


  def valid_fn_vec(self, pqr):
    """Returns if the given set of nodes is valid or not."""
    xyt = self.to_actual_xyt_vec(np.array(pqr))
    height = self.traversible.shape[0]
    width = self.traversible.shape[1]
    x = np.round(xyt[:,[0]]).astype(np.int32)
    y = np.round(xyt[:,[1]]).astype(np.int32)
    is_inside = np.all(np.concatenate((x >= 0, y >= 0,
                                       x < width, y < height), axis=1), axis=1)
    x = np.minimum(np.maximum(x, 0), width-1)
    y = np.minimum(np.maximum(y, 0), height-1)
    ind = np.ravel_multi_index((y,x), self.traversible.shape)
    is_traversible = self.traversible.ravel()[ind]

    is_valid = np.all(np.concatenate((is_inside[:,np.newaxis], is_traversible),
                                     axis=1), axis=1)
    return is_valid

  def get_feasible_actions(self, node_ids):
    """Returns the feasible set of actions from the current node."""
    a = np.zeros((len(node_ids), self.task_params.num_actions), dtype=np.int32)
    gtG = self.task.gtG
    next_node = []
    for i, c in enumerate(node_ids):
      neigh = gtG.vertex(c).out_neighbours()
      neigh_edge = gtG.vertex(c).out_edges()
      nn = {}
      for n, e in zip(neigh, neigh_edge):
        _ = gtG.ep['action'][e]
        a[i,_] = 1
        nn[_] = int(n)
      next_node.append(nn)
    return a, next_node

  def take_action(self, current_node_ids, action):
    """Returns the new node after taking the action action. Stays at the current
    node if the action is invalid."""
    actions, next_node_ids = self.get_feasible_actions(current_node_ids)
    new_node_ids = []
    for i, (c,a) in enumerate(zip(current_node_ids, action)):
      if actions[i,a] == 1:
        new_node_ids.append(next_node_ids[i][a])
      else:
        new_node_ids.append(c)
    return new_node_ids

  def set_r_obj(self, r_obj):
    """Sets the SwiftshaderRenderer object used for rendering."""
    self.r_obj = r_obj

class Building(GridWorld):
  def __init__(self, building_name, robot, env,
               category_list=None, small=False, flip=False, logdir=None,
               building_loader=None):

    self.restrict_to_largest_cc = True
    self.robot = robot
    self.env = env
    self.logdir = logdir

    # Load the building meta data.
    building = building_loader.load_building(building_name)
    if small:
      building['mesh_names'] = building['mesh_names'][:5]

    # New code.
    shapess = building_loader.load_building_meshes(building)
    if flip:
      for shapes in shapess:
        shapes.flip_shape()

    vs = []
    for shapes in shapess:
      vs.append(shapes.get_vertices()[0])
    vs = np.concatenate(vs, axis=0)
    map = make_map(env.padding, env.resolution, vertex=vs, sc=100.)
    map = compute_traversibility(
        map, robot.base, robot.height, robot.radius, env.valid_min,
        env.valid_max, env.num_point_threshold, shapess=shapess, sc=100.,
        n_samples_per_face=env.n_samples_per_face)

    room_dims = _get_room_dimensions(building['room_dimension_file'],
                                     env.resolution, map.origin, flip=flip)
    class_maps, class_map_names = _get_semantic_maps(
        building['class_map_folder'], building_name, map, flip)

    self.class_maps      = class_maps
    self.class_map_names = class_map_names
    self.building        = building
    self.shapess         = shapess
    self.map             = map
    self.traversible     = map.traversible*1
    self.building_name   = building_name
    self.room_dims       = room_dims
    self.flipped         = flip
    self.renderer_entitiy_ids = []

    if self.restrict_to_largest_cc:
      self.traversible = pick_largest_cc(self.traversible)

  def load_building_into_scene(self):
    # Loads the scene.
    self.renderer_entitiy_ids += self.r_obj.load_shapes(self.shapess)
    # Free up memory, we dont need the mesh or the materials anymore.
    self.shapess = None

  def add_entity_at_nodes(self, nodes, height, shape):
    xyt = self.to_actual_xyt_vec(nodes)
    nxy = xyt[:,:2]*1.
    nxy = nxy * self.map.resolution
    nxy = nxy + self.map.origin
    Ts = np.concatenate((nxy, nxy[:,:1]), axis=1)
    Ts[:,2] = height; Ts = Ts / 100.;

    # Merge all the shapes into a single shape and add that shape.
    shape.replicate_shape(Ts)
    entity_ids = self.r_obj.load_shapes([shape])
    self.renderer_entitiy_ids += entity_ids
    return entity_ids

  def add_shapes(self, shapes):
    scene = self.r_obj.viz.scene()
    for shape in shapes:
      scene.AddShape(shape)

  def add_materials(self, materials):
    scene = self.r_obj.viz.scene()
    for material in materials:
      scene.AddOrUpdateMaterial(material)

  def set_building_visibility(self, visibility):
    self.r_obj.set_entity_visible(self.renderer_entitiy_ids, visibility)

  def render_nodes(self, nodes, perturb=None, aux_delta_theta=0.):
    self.set_building_visibility(True)
    if perturb is None:
      perturb = np.zeros((len(nodes), 4))

    imgs = []
    r = 2
    elevation_z = r * np.tan(np.deg2rad(self.robot.camera_elevation_degree))

    for i in range(len(nodes)):
      xyt = self.to_actual_xyt(nodes[i])
      lookat_theta = 3.0 * np.pi / 2.0 - (xyt[2]+perturb[i,2]+aux_delta_theta) * (self.task.delta_theta)
      nxy = np.array([xyt[0]+perturb[i,0], xyt[1]+perturb[i,1]]).reshape(1, -1)
      nxy = nxy * self.map.resolution
      nxy = nxy + self.map.origin
      camera_xyz = np.zeros((1, 3))
      camera_xyz[...] = [nxy[0, 0], nxy[0, 1], self.robot.sensor_height]
      camera_xyz = camera_xyz / 100.
      lookat_xyz = np.array([-r * np.sin(lookat_theta),
                             -r * np.cos(lookat_theta), elevation_z])
      lookat_xyz = lookat_xyz + camera_xyz[0, :]
      self.r_obj.position_camera(camera_xyz[0, :].tolist(),
                                 lookat_xyz.tolist(), [0.0, 0.0, 1.0])
      img = self.r_obj.render(take_screenshot=True, output_type=0)
      img = [x for x in img if x is not None]
      img = np.concatenate(img, axis=2).astype(np.float32)
      if perturb[i,3]>0:
        img = img[:,::-1,:]
      imgs.append(img)

    self.set_building_visibility(False)
    return imgs


class MeshMapper(Building):
  def __init__(self, robot, env, task_params, building_name, category_list,
               flip, logdir=None, building_loader=None):
    Building.__init__(self, building_name, robot, env, category_list,
                      small=task_params.toy_problem, flip=flip, logdir=logdir,
                      building_loader=building_loader)
    self.task_params = task_params
    self.task = None
    self._preprocess_for_task(self.task_params.building_seed)

  def _preprocess_for_task(self, seed):
    if self.task is None or self.task.seed != seed:
      rng = np.random.RandomState(seed)
      origin_loc = get_graph_origin_loc(rng, self.traversible)
      self.task = utils.Foo(seed=seed, origin_loc=origin_loc,
                            n_ori=self.task_params.n_ori)
      G = generate_graph(self.valid_fn_vec,
                                  self.task_params.step_size, self.task.n_ori,
                                  (0, 0, 0))
      gtG, nodes, nodes_to_id = convert_to_graph_tool(G)
      self.task.gtG = gtG
      self.task.nodes = nodes
      self.task.delta_theta = 2.0*np.pi/(self.task.n_ori*1.)
      self.task.nodes_to_id = nodes_to_id
      logging.info('Building %s, #V=%d, #E=%d', self.building_name,
                   self.task.nodes.shape[0], self.task.gtG.num_edges())

      if self.logdir is not None:
        write_traversible = cv2.applyColorMap(self.traversible.astype(np.uint8)*255, cv2.COLORMAP_JET)
        img_path = os.path.join(self.logdir,
                                '{:s}_{:d}_graph.png'.format(self.building_name,
                                                             seed))
        node_xyt = self.to_actual_xyt_vec(self.task.nodes)
        plt.set_cmap('jet');
        fig, ax = utils.subplot(plt, (1,1), (12,12))
        ax.plot(node_xyt[:,0], node_xyt[:,1], 'm.')
        ax.imshow(self.traversible, origin='lower');
        ax.set_axis_off(); ax.axis('equal');
        ax.set_title('{:s}, {:d}, {:d}'.format(self.building_name,
                                               self.task.nodes.shape[0],
                                               self.task.gtG.num_edges()))
        if self.room_dims is not None:
          for i, r in enumerate(self.room_dims['dims']*1):
            min_ = r[:3]*1
            max_ = r[3:]*1
            xmin, ymin, zmin = min_
            xmax, ymax, zmax = max_

            ax.plot([xmin, xmax, xmax, xmin, xmin],
                    [ymin, ymin, ymax, ymax, ymin], 'g')
        with fu.fopen(img_path, 'w') as f:
          fig.savefig(f, bbox_inches='tight', transparent=True, pad_inches=0)
        plt.close(fig)


  def _gen_rng(self, rng):
    # instances is a list of list of node_ids.
    if self.task_params.move_type == 'circle':
      _, _, _, _, paths = rng_target_dist_field(self.task_params.batch_size,
                                                self.task.gtG, rng, 0, 1,
                                                compute_path=True)
      instances_ = paths

      instances = []
      for instance_ in instances_:
        instance = instance_
        for i in range(self.task_params.num_steps):
          instance.append(self.take_action([instance[-1]], [1])[0])
        instances.append(instance)

    elif self.task_params.move_type == 'shortest_path':
      _, _, _, _, paths = rng_target_dist_field(self.task_params.batch_size,
                                                self.task.gtG, rng,
                                                self.task_params.num_steps,
                                                self.task_params.num_steps+1,
                                                compute_path=True)
      instances = paths

    elif self.task_params.move_type == 'circle+forward':
      _, _, _, _, paths = rng_target_dist_field(self.task_params.batch_size,
                                                self.task.gtG, rng, 0, 1,
                                                compute_path=True)
      instances_ = paths
      instances = []
      for instance_ in instances_:
        instance = instance_
        for i in range(self.task_params.n_ori-1):
          instance.append(self.take_action([instance[-1]], [1])[0])
        while len(instance) <= self.task_params.num_steps:
          while self.take_action([instance[-1]], [3])[0] == instance[-1] and len(instance) <= self.task_params.num_steps:
            instance.append(self.take_action([instance[-1]], [2])[0])
          if len(instance) <= self.task_params.num_steps:
            instance.append(self.take_action([instance[-1]], [3])[0])
        instances.append(instance)

    # Do random perturbation if needed.
    perturbs = _gen_perturbs(rng, self.task_params.batch_size,
                             self.task_params.num_steps,
                             self.task_params.data_augment.lr_flip,
                             self.task_params.data_augment.delta_angle,
                             self.task_params.data_augment.delta_xy,
                             self.task_params.data_augment.structured)
    return instances, perturbs

  def worker(self, instances, perturbs):
    # Output the images and the free space.

    # Make the instances be all the same length.
    for i in range(len(instances)):
      for j in range(self.task_params.num_steps - len(instances[i]) + 1):
        instances[i].append(instances[i][-1])
      if perturbs[i].shape[0] < self.task_params.num_steps+1:
        p = np.zeros((self.task_params.num_steps+1, 4))
        p[:perturbs[i].shape[0], :] = perturbs[i]
        p[perturbs[i].shape[0]:, :] = perturbs[i][-1,:]
        perturbs[i] = p

    instances_ = []
    for instance in instances:
      instances_ = instances_ + instance
    perturbs_ = np.concatenate(perturbs, axis=0)

    instances_nodes = self.task.nodes[instances_,:]
    instances_nodes = [tuple(x) for x in instances_nodes]

    imgs_ = self.render_nodes(instances_nodes, perturbs_)
    imgs = []; next = 0;
    for instance in instances:
      img_i = []
      for _ in instance:
        img_i.append(imgs_[next])
        next = next+1
      imgs.append(img_i)
    imgs = np.array(imgs)

    # Render out the maps in the egocentric view for all nodes and not just the
    # last node.
    all_nodes = []
    for x in instances:
      all_nodes = all_nodes + x
    all_perturbs = np.concatenate(perturbs, axis=0)
    loc, x_axis, y_axis, theta = self.get_loc_axis(
        self.task.nodes[all_nodes, :]*1, delta_theta=self.task.delta_theta,
        perturb=all_perturbs)
    fss = None
    valids = None
    loc_on_map = None
    theta_on_map = None
    cum_fs = None
    cum_valid = None
    incremental_locs = None
    incremental_thetas = None

    if self.task_params.output_free_space:
      fss, valids = get_map_to_predict(loc, x_axis, y_axis,
                                       map=self.traversible*1.,
                                       map_size=self.task_params.map_size)
      fss = np.array(fss) > 0.5
      fss = np.reshape(fss, [self.task_params.batch_size,
                             self.task_params.num_steps+1,
                             self.task_params.map_size,
                             self.task_params.map_size])
      valids = np.reshape(np.array(valids), fss.shape)

    if self.task_params.output_transform_to_global_map:
      # Output the transform to the global map.
      loc_on_map = np.reshape(loc*1, [self.task_params.batch_size,
                                      self.task_params.num_steps+1, -1])
      # Converting to location wrt to first location so that warping happens
      # properly.
      theta_on_map = np.reshape(theta*1, [self.task_params.batch_size,
                                            self.task_params.num_steps+1, -1])

    if self.task_params.output_incremental_transform:
      # Output the transform to the global map.
      incremental_locs_ = np.reshape(loc*1, [self.task_params.batch_size,
                                             self.task_params.num_steps+1, -1])
      incremental_locs_[:,1:,:] -= incremental_locs_[:,:-1,:]
      t0 = -np.pi/2+np.reshape(theta*1, [self.task_params.batch_size,
                                        self.task_params.num_steps+1, -1])
      t = t0*1
      incremental_locs = incremental_locs_*1
      incremental_locs[:,:,0] = np.sum(incremental_locs_ * np.concatenate((np.cos(t), np.sin(t)), axis=-1), axis=-1)
      incremental_locs[:,:,1] = np.sum(incremental_locs_ * np.concatenate((np.cos(t+np.pi/2), np.sin(t+np.pi/2)), axis=-1), axis=-1)
      incremental_locs[:,0,:] = incremental_locs_[:,0,:]
      # print incremental_locs_[0,:,:], incremental_locs[0,:,:], t0[0,:,:]

      incremental_thetas = np.reshape(theta*1, [self.task_params.batch_size,
                                                self.task_params.num_steps+1,
                                                -1])
      incremental_thetas[:,1:,:] += -incremental_thetas[:,:-1,:]

    if self.task_params.output_canonical_map:
      loc_ = loc[0::(self.task_params.num_steps+1), :]
      x_axis = np.zeros_like(loc_); x_axis[:,1] = 1
      y_axis = np.zeros_like(loc_); y_axis[:,0] = -1
      cum_fs, cum_valid = get_map_to_predict(loc_, x_axis, y_axis,
                                             map=self.traversible*1.,
                                             map_size=self.task_params.map_size)
      cum_fs = np.array(cum_fs) > 0.5
      cum_fs = np.reshape(cum_fs, [self.task_params.batch_size, 1,
                                   self.task_params.map_size,
                                   self.task_params.map_size])
      cum_valid = np.reshape(np.array(cum_valid), cum_fs.shape)


    inputs = {'fs_maps': fss,
              'valid_maps': valids,
              'imgs': imgs,
              'loc_on_map': loc_on_map,
              'theta_on_map': theta_on_map,
              'cum_fs_maps': cum_fs,
              'cum_valid_maps': cum_valid,
              'incremental_thetas': incremental_thetas,
              'incremental_locs': incremental_locs}
    return inputs

  def pre(self, inputs):
    inputs['imgs'] = image_pre(inputs['imgs'], self.task_params.modalities)
    if inputs['loc_on_map'] is not None:
      inputs['loc_on_map'] = inputs['loc_on_map'] - inputs['loc_on_map'][:,[0],:]
    if inputs['theta_on_map'] is not None:
      inputs['theta_on_map'] = np.pi/2. - inputs['theta_on_map']
    return inputs

def _nav_env_reset_helper(type, rng, nodes, batch_size, gtG, max_dist,
                          num_steps, num_goals, data_augment, **kwargs):
  """Generates and returns a new episode."""
  max_compute = max_dist + 4*num_steps
  if type == 'general':
    start_node_ids, end_node_ids, dist, pred_map, paths = \
        rng_target_dist_field(batch_size, gtG, rng, max_dist, max_compute,
                              nodes=nodes, compute_path=False)
    target_class = None

  elif type == 'room_to_room_many':
    goal_node_ids = []; dists = [];
    node_room_ids = kwargs['node_room_ids']
    # Sample the first one
    start_node_ids_, end_node_ids_, dist_, _, _ = rng_room_to_room(
        batch_size, gtG, rng, max_dist, max_compute,
        node_room_ids=node_room_ids, nodes=nodes)
    start_node_ids = start_node_ids_
    goal_node_ids.append(end_node_ids_)
    dists.append(dist_)
    for n in range(num_goals-1):
      start_node_ids_, end_node_ids_, dist_, _, _ = rng_next_goal(
          goal_node_ids[n], batch_size, gtG, rng, max_dist,
          max_compute, node_room_ids=node_room_ids, nodes=nodes,
          dists_from_start_node=dists[n])
      goal_node_ids.append(end_node_ids_)
      dists.append(dist_)
    target_class = None

  elif type == 'rng_rejection_sampling_many':
    num_goals = num_goals
    goal_node_ids = []; dists = [];

    n_ori = kwargs['n_ori']
    step_size = kwargs['step_size']
    min_dist = kwargs['min_dist']
    sampling_distribution = kwargs['sampling_distribution']
    target_distribution = kwargs['target_distribution']
    rejection_sampling_M = kwargs['rejection_sampling_M']
    distribution_bins = kwargs['distribution_bins']

    for n in range(num_goals):
      if n == 0: input_nodes = None
      else: input_nodes = goal_node_ids[n-1]
      start_node_ids_, end_node_ids_, dist_, _, _, _, _ = rng_next_goal_rejection_sampling(
              input_nodes, batch_size, gtG, rng, max_dist, min_dist,
              max_compute, sampling_distribution, target_distribution, nodes,
              n_ori, step_size, distribution_bins, rejection_sampling_M)
      if n == 0: start_node_ids = start_node_ids_
      goal_node_ids.append(end_node_ids_)
      dists.append(dist_)
    target_class = None

  elif type == 'room_to_room_back':
    num_goals = num_goals
    assert(num_goals == 2), 'num_goals must be 2.'
    goal_node_ids = []; dists = [];
    node_room_ids = kwargs['node_room_ids']
    # Sample the first one.
    start_node_ids_, end_node_ids_, dist_, _, _ = rng_room_to_room(
        batch_size, gtG, rng, max_dist, max_compute,
        node_room_ids=node_room_ids, nodes=nodes)
    start_node_ids = start_node_ids_
    goal_node_ids.append(end_node_ids_)
    dists.append(dist_)

    # Set second goal to be starting position, and compute distance to the start node.
    goal_node_ids.append(start_node_ids)
    dist = []
    for i in range(batch_size):
      dist_ = gt.topology.shortest_distance(
          gt.GraphView(gtG, reversed=True),
          source=gtG.vertex(start_node_ids[i]), target=None)
      dist_ = np.array(dist_.get_array())
      dist.append(dist_)
    dists.append(dist)
    target_class = None

  elif type[:14] == 'to_nearest_obj':
    # Generate an episode by sampling one of the target classes (with
    # probability proportional to the number of nodes in the world).
    # With the sampled class sample a node that is within some distance from
    # the sampled class.
    class_nodes   = kwargs['class_nodes']
    sampling      = kwargs['sampling']
    dist_to_class = kwargs['dist_to_class']

    assert(num_goals == 1), 'Only supports a single goal.'
    ind = rng.choice(class_nodes.shape[0], size=batch_size)
    target_class = class_nodes[ind,1]
    start_node_ids = []; dists = []; goal_node_ids = [];

    for t in target_class:
      if sampling == 'uniform':
        max_dist = max_dist
        cnts = np.bincount(dist_to_class[t], minlength=max_dist+1)*1.
        cnts[max_dist+1:] = 0
        p_each = 1./ cnts / (max_dist+1.)
        p_each[cnts == 0] = 0
        p = p_each[dist_to_class[t]]*1.; p = p/np.sum(p)
        start_node_id = rng.choice(p.shape[0], size=1, p=p)[0]
      else:
        logging.fatal('Sampling not one of uniform.')
      start_node_ids.append(start_node_id)
      dists.append(dist_to_class[t])
      # Dummy goal node, same as the start node, so that vis is better.
      goal_node_ids.append(start_node_id)
    dists = [dists]
    goal_node_ids = [goal_node_ids]

  return start_node_ids, goal_node_ids, dists, target_class


class NavigationEnv(GridWorld, Building):
  """Wrapper around GridWorld which sets up navigation tasks.
  """
  def _debug_save_hardness(self, seed):
    out_path = os.path.join(self.logdir, '{:s}_{:d}_hardness.png'.format(self.building_name, seed))
    batch_size = 4000
    rng = np.random.RandomState(0)
    start_node_ids, end_node_ids, dists, pred_maps, paths, hardnesss, gt_dists = \
      rng_next_goal_rejection_sampling(
          None, batch_size, self.task.gtG, rng, self.task_params.max_dist,
          self.task_params.min_dist, self.task_params.max_dist,
          self.task.sampling_distribution, self.task.target_distribution,
          self.task.nodes, self.task_params.n_ori, self.task_params.step_size,
          self.task.distribution_bins, self.task.rejection_sampling_M)
    bins = self.task.distribution_bins 
    n_bins = self.task.n_bins
    with plt.style.context('ggplot'):
      fig, axes = utils.subplot(plt, (1,2), (10,10))
      ax = axes[0]
      _ = ax.hist(hardnesss, bins=bins, weights=np.ones_like(hardnesss)/len(hardnesss))
      ax.plot(bins[:-1]+0.5/n_bins, self.task.target_distribution, 'g')
      ax.plot(bins[:-1]+0.5/n_bins, self.task.sampling_distribution, 'b')
      ax.grid('on')
      
      ax = axes[1]
      _ = ax.hist(gt_dists, bins=np.arange(self.task_params.max_dist+1))
      ax.grid('on')
      ax.set_title('Mean: {:0.2f}, Median: {:0.2f}'.format(np.mean(gt_dists),
                                                           np.median(gt_dists)))
      with fu.fopen(out_path, 'w') as f:
        fig.savefig(f, bbox_inches='tight', transparent=True, pad_inches=0)

  def _debug_save_map_nodes(self, seed):
    """Saves traversible space along with nodes generated on the graph. Takes
    the seed as input."""
    img_path = os.path.join(self.logdir, '{:s}_{:d}_graph.png'.format(self.building_name, seed))
    node_xyt = self.to_actual_xyt_vec(self.task.nodes)
    plt.set_cmap('jet');
    fig, ax = utils.subplot(plt, (1,1), (12,12))
    ax.plot(node_xyt[:,0], node_xyt[:,1], 'm.')
    ax.set_axis_off(); ax.axis('equal');
    
    if self.room_dims is not None:
      for i, r in enumerate(self.room_dims['dims']*1):
        min_ = r[:3]*1
        max_ = r[3:]*1
        xmin, ymin, zmin = min_
        xmax, ymax, zmax = max_

        ax.plot([xmin, xmax, xmax, xmin, xmin],
                [ymin, ymin, ymax, ymax, ymin], 'g')
    ax.imshow(self.traversible, origin='lower');
    with fu.fopen(img_path, 'w') as f:
      fig.savefig(f, bbox_inches='tight', transparent=True, pad_inches=0)

  def _debug_semantic_maps(self, seed):
    """Saves traversible space along with nodes generated on the graph. Takes
    the seed as input."""
    for i, cls in enumerate(self.task_params.semantic_task.class_map_names):
      img_path = os.path.join(self.logdir, '{:s}_flip{:d}_{:s}_graph.png'.format(self.building_name, seed, cls))
      maps = self.traversible*1.
      maps += 0.5*(self.task.class_maps_dilated[:,:,i])
      write_traversible = (maps*1.+1.)/3.0
      write_traversible = (write_traversible*255.).astype(np.uint8)[:,:,np.newaxis]
      write_traversible = write_traversible + np.zeros((1,1,3), dtype=np.uint8)
      fu.write_image(img_path, write_traversible[::-1,:,:])

  def _preprocess_for_task(self, seed):
    """Sets up the task field for doing navigation on the grid world."""
    if self.task is None or self.task.seed != seed:
      rng = np.random.RandomState(seed)
      origin_loc = get_graph_origin_loc(rng, self.traversible)
      self.task = utils.Foo(seed=seed, origin_loc=origin_loc,
                            n_ori=self.task_params.n_ori)
      G = generate_graph(self.valid_fn_vec, self.task_params.step_size,
                         self.task.n_ori, (0, 0, 0))
      gtG, nodes, nodes_to_id = convert_to_graph_tool(G)
      self.task.gtG = gtG
      self.task.nodes = nodes
      self.task.delta_theta = 2.0*np.pi/(self.task.n_ori*1.)
      self.task.nodes_to_id = nodes_to_id

      logging.info('Building %s, #V=%d, #E=%d', self.building_name,
                   self.task.nodes.shape[0], self.task.gtG.num_edges())
      type = self.task_params.type
      if type == 'general':
        # Do nothing
        _ = None

      elif type == 'room_to_room_many' or type == 'room_to_room_back':
        if type == 'room_to_room_back':
          assert(self.task_params.num_goals == 2), 'num_goals must be 2.'

        self.room_dims = _filter_rooms(self.room_dims, self.task_params.room_regex)
        xyt = self.to_actual_xyt_vec(self.task.nodes)
        self.task.node_room_ids = _label_nodes_with_room_id(xyt, self.room_dims)
        self.task.reset_kwargs = {'node_room_ids': self.task.node_room_ids}

      elif type == 'rng_rejection_sampling_many':
        n_bins = 20
        rejection_sampling_M = self.task_params.rejection_sampling_M
        min_dist = self.task_params.min_dist
        bins = np.arange(n_bins+1)/(n_bins*1.)
        target_d = np.zeros(n_bins); target_d[...] = 1./n_bins;

        sampling_d = get_hardness_distribution(
            self.task.gtG, self.task_params.max_dist, self.task_params.min_dist,
            np.random.RandomState(0), 4000, bins, self.task.nodes,
            self.task_params.n_ori, self.task_params.step_size)

        self.task.reset_kwargs = {'distribution_bins': bins,
                                  'target_distribution': target_d,
                                  'sampling_distribution': sampling_d,
                                  'rejection_sampling_M': rejection_sampling_M,
                                  'n_bins': n_bins, 
                                  'n_ori': self.task_params.n_ori,
                                  'step_size': self.task_params.step_size,
                                  'min_dist': self.task_params.min_dist}
        self.task.n_bins = n_bins
        self.task.distribution_bins = bins
        self.task.target_distribution = target_d
        self.task.sampling_distribution = sampling_d
        self.task.rejection_sampling_M = rejection_sampling_M

        if self.logdir is not None:
          self._debug_save_hardness(seed)

      elif type[:14] == 'to_nearest_obj':
        self.room_dims = _filter_rooms(self.room_dims, self.task_params.room_regex)
        xyt = self.to_actual_xyt_vec(self.task.nodes)

        self.class_maps = _select_classes(self.class_maps,
                                          self.class_map_names,
                                          self.task_params.semantic_task.class_map_names)*1
        self.class_map_names = self.task_params.semantic_task.class_map_names
        nodes_xyt = self.to_actual_xyt_vec(np.array(self.task.nodes))

        tt = utils.Timer(); tt.tic();
        if self.task_params.type == 'to_nearest_obj_acc':
          self.task.class_maps_dilated, self.task.node_class_label = label_nodes_with_class_geodesic(
            nodes_xyt, self.class_maps,
            self.task_params.semantic_task.pix_distance+8, self.map.traversible,
            ff_cost=1., fo_cost=1., oo_cost=4., connectivity=8.)

        dists = []
        for i in range(len(self.class_map_names)):
          class_nodes_ = np.where(self.task.node_class_label[:,i])[0]
          dists.append(get_distance_node_list(gtG, source_nodes=class_nodes_, direction='to'))
        self.task.dist_to_class = dists
        a_, b_ = np.where(self.task.node_class_label)
        self.task.class_nodes = np.concatenate((a_[:,np.newaxis], b_[:,np.newaxis]), axis=1)
        
        if self.logdir is not None:
          self._debug_semantic_maps(seed)
        
        self.task.reset_kwargs = {'sampling': self.task_params.semantic_task.sampling,
                                  'class_nodes': self.task.class_nodes,
                                  'dist_to_class': self.task.dist_to_class}

      if self.logdir is not None:
        self._debug_save_map_nodes(seed)

  def reset(self, rngs):
    rng = rngs[0]; rng_perturb = rngs[1];
    nodes = self.task.nodes
    tp = self.task_params

    start_node_ids, goal_node_ids, dists, target_class = \
        _nav_env_reset_helper(tp.type, rng, self.task.nodes, tp.batch_size,
                              self.task.gtG, tp.max_dist, tp.num_steps,
                              tp.num_goals, tp.data_augment,
                              **(self.task.reset_kwargs))

    start_nodes = [tuple(nodes[_,:]) for _ in start_node_ids]
    goal_nodes = [[tuple(nodes[_,:]) for _ in __] for __ in goal_node_ids]
    data_augment = tp.data_augment
    perturbs = _gen_perturbs(rng_perturb, tp.batch_size,
                             (tp.num_steps+1)*tp.num_goals,
                             data_augment.lr_flip, data_augment.delta_angle,
                             data_augment.delta_xy, data_augment.structured)
    perturbs = np.array(perturbs) # batch x steps x 4
    end_perturbs = perturbs[:,-(tp.num_goals):,:]*1 # fixed perturb for the goal.
    perturbs = perturbs[:,:-(tp.num_goals),:]*1

    history = -np.ones((tp.batch_size, tp.num_steps*tp.num_goals), dtype=np.int32)
    self.episode = utils.Foo(
        start_nodes=start_nodes, start_node_ids=start_node_ids,
        goal_nodes=goal_nodes, goal_node_ids=goal_node_ids, dist_to_goal=dists,
        perturbs=perturbs, goal_perturbs=end_perturbs, history=history,
        target_class=target_class, history_frames=[])
    return start_node_ids

  def take_action(self, current_node_ids, action, step_number):
    """In addition to returning the action, also returns the reward that the
    agent receives."""
    goal_number = step_number / self.task_params.num_steps
    new_node_ids = GridWorld.take_action(self, current_node_ids, action)
    rewards = []
    for i, n in enumerate(new_node_ids):
      reward = 0
      if n == self.episode.goal_node_ids[goal_number][i]:
        reward = self.task_params.reward_at_goal
      reward = reward - self.task_params.reward_time_penalty
      rewards.append(reward)
    return new_node_ids, rewards


  def get_optimal_action(self, current_node_ids, step_number):
    """Returns the optimal action from the current node."""
    goal_number = step_number / self.task_params.num_steps
    gtG = self.task.gtG
    a = np.zeros((len(current_node_ids), self.task_params.num_actions), dtype=np.int32)
    d_dict = self.episode.dist_to_goal[goal_number]
    for i, c in enumerate(current_node_ids):
      neigh = gtG.vertex(c).out_neighbours()
      neigh_edge = gtG.vertex(c).out_edges()
      ds = np.array([d_dict[i][int(x)] for x in neigh])
      ds_min = np.min(ds)
      for i_, e in enumerate(neigh_edge):
        if ds[i_] == ds_min:
          _ = gtG.ep['action'][e]
          a[i, _] = 1
    return a

  def get_targets(self, current_node_ids, step_number):
    """Returns the target actions from the current node."""
    action = self.get_optimal_action(current_node_ids, step_number)
    action = np.expand_dims(action, axis=1)
    return vars(utils.Foo(action=action))

  def get_targets_name(self):
    """Returns the list of names of the targets."""
    return ['action']

  def cleanup(self):
    self.episode = None

class VisualNavigationEnv(NavigationEnv):
  """Class for doing visual navigation in environments. Functions for computing
  features on states, etc.
  """
  def __init__(self, robot, env, task_params, category_list=None,
               building_name=None, flip=False, logdir=None,
               building_loader=None, r_obj=None):
    tt = utils.Timer()
    tt.tic()
    Building.__init__(self, building_name, robot, env, category_list,
                      small=task_params.toy_problem, flip=flip, logdir=logdir,
                      building_loader=building_loader)

    self.set_r_obj(r_obj)
    self.task_params = task_params
    self.task = None
    self.episode = None
    self._preprocess_for_task(self.task_params.building_seed)
    if hasattr(self.task_params, 'map_scales'):
      self.task.scaled_maps = resize_maps(
          self.traversible.astype(np.float32)*1, self.task_params.map_scales,
          self.task_params.map_resize_method)
    else:
      logging.fatal('VisualNavigationEnv does not support scale_f anymore.')
    self.task.readout_maps_scaled = resize_maps(
      self.traversible.astype(np.float32)*1,
      self.task_params.readout_maps_scales,
      self.task_params.map_resize_method)
    tt.toc(log_at=1, log_str='VisualNavigationEnv __init__: ')

  def get_weight(self):
    return self.task.nodes.shape[0]

  def get_common_data(self):
    goal_nodes = self.episode.goal_nodes
    start_nodes = self.episode.start_nodes
    perturbs = self.episode.perturbs
    goal_perturbs = self.episode.goal_perturbs
    target_class = self.episode.target_class

    goal_locs = []; rel_goal_locs = [];
    for i in range(len(goal_nodes)):
      end_nodes = goal_nodes[i]
      goal_loc, _, _, goal_theta = self.get_loc_axis(
          np.array(end_nodes), delta_theta=self.task.delta_theta,
          perturb=goal_perturbs[:,i,:])

      # Compute the relative location to all goals from the starting location.
      loc, _, _, theta = self.get_loc_axis(np.array(start_nodes),
                                           delta_theta=self.task.delta_theta,
                                           perturb=perturbs[:,0,:])
      r_goal, t_goal = _get_relative_goal_loc(goal_loc*1., loc, theta)
      rel_goal_loc = np.concatenate((r_goal*np.cos(t_goal), r_goal*np.sin(t_goal),
                                     np.cos(goal_theta-theta),
                                     np.sin(goal_theta-theta)), axis=1)
      rel_goal_locs.append(np.expand_dims(rel_goal_loc, axis=1))
      goal_locs.append(np.expand_dims(goal_loc, axis=1))

    map = self.traversible*1.
    maps = np.repeat(np.expand_dims(np.expand_dims(map, axis=0), axis=0),
                     self.task_params.batch_size, axis=0)*1
    if self.task_params.type[:14] == 'to_nearest_obj':
      for i in range(self.task_params.batch_size):
        maps[i,0,:,:] += 0.5*(self.task.class_maps_dilated[:,:,target_class[i]])

    rel_goal_locs = np.concatenate(rel_goal_locs, axis=1)
    goal_locs = np.concatenate(goal_locs, axis=1)
    maps = np.expand_dims(maps, axis=-1)

    if self.task_params.type[:14] == 'to_nearest_obj':
      rel_goal_locs = np.zeros((self.task_params.batch_size, 1,
                                len(self.task_params.semantic_task.class_map_names)),
                               dtype=np.float32)
      goal_locs = np.zeros((self.task_params.batch_size, 1, 2),
                           dtype=np.float32)
      for i in range(self.task_params.batch_size):
          t = target_class[i]
          rel_goal_locs[i,0,t] = 1.
          goal_locs[i,0,0] = t
          goal_locs[i,0,1] = np.NaN

    return vars(utils.Foo(orig_maps=maps, goal_loc=goal_locs,
                          rel_goal_loc_at_start=rel_goal_locs))

  def pre_common_data(self, inputs):
    return inputs


  def get_features(self, current_node_ids, step_number):
    task_params = self.task_params
    goal_number = step_number / self.task_params.num_steps
    end_nodes = self.task.nodes[self.episode.goal_node_ids[goal_number],:]*1
    current_nodes = self.task.nodes[current_node_ids,:]*1
    end_perturbs = self.episode.goal_perturbs[:,goal_number,:][:,np.newaxis,:]
    perturbs = self.episode.perturbs
    target_class = self.episode.target_class

    # Append to history.
    self.episode.history[:,step_number] = np.array(current_node_ids)

    # Render out the images from current node.
    outs = {}

    if self.task_params.outputs.images:
      imgs_all = []
      imgs = self.render_nodes([tuple(x) for x in current_nodes],
                               perturb=perturbs[:,step_number,:])
      imgs_all.append(imgs)
      aux_delta_thetas = self.task_params.aux_delta_thetas
      for i in range(len(aux_delta_thetas)):
        imgs = self.render_nodes([tuple(x) for x in current_nodes],
                                 perturb=perturbs[:,step_number,:],
                                 aux_delta_theta=aux_delta_thetas[i])
        imgs_all.append(imgs)
      imgs_all = np.array(imgs_all) # A x B x H x W x C
      imgs_all = np.transpose(imgs_all, axes=[1,0,2,3,4])
      imgs_all = np.expand_dims(imgs_all, axis=1) # B x N x A x H x W x C
      if task_params.num_history_frames > 0:
        if step_number == 0:
          # Append the same frame 4 times
          for i in range(task_params.num_history_frames+1):
            self.episode.history_frames.insert(0, imgs_all*1.)
        self.episode.history_frames.insert(0, imgs_all)
        self.episode.history_frames.pop()
        imgs_all_with_history = np.concatenate(self.episode.history_frames, axis=2)
      else:
        imgs_all_with_history = imgs_all
      outs['imgs'] = imgs_all_with_history # B x N x A x H x W x C

    if self.task_params.outputs.node_ids:
      outs['node_ids'] = np.array(current_node_ids).reshape((-1,1,1))
      outs['perturbs'] = np.expand_dims(perturbs[:,step_number, :]*1., axis=1)

    if self.task_params.outputs.analytical_counts:
      assert(self.task_params.modalities == ['depth'])
      d = image_pre(outs['imgs']*1., self.task_params.modalities)
      cm = get_camera_matrix(self.task_params.img_width,
                             self.task_params.img_height,
                             self.task_params.img_fov)
      XYZ = get_point_cloud_from_z(100./d[...,0], cm)
      XYZ = make_geocentric(XYZ*100., self.robot.sensor_height,
                                      self.robot.camera_elevation_degree)
      for i in range(len(self.task_params.analytical_counts.map_sizes)):
        non_linearity = self.task_params.analytical_counts.non_linearity[i]
        count, isvalid = bin_points(XYZ*1.,
                                    map_size=self.task_params.analytical_counts.map_sizes[i],
                                    xy_resolution=self.task_params.analytical_counts.xy_resolution[i],
                                    z_bins=self.task_params.analytical_counts.z_bins[i])
        assert(count.shape[2] == 1), 'only works for n_views equal to 1.'
        count = count[:,:,0,:,:,:]
        isvalid = isvalid[:,:,0,:,:,:]
        if non_linearity == 'none':
          None
        elif non_linearity == 'min10':
          count = np.minimum(count, 10.)
        elif non_linearity == 'sqrt':
          count = np.sqrt(count)
        else:
          logging.fatal('Undefined non_linearity.')
        outs['analytical_counts_{:d}'.format(i)] = count

    # Compute the goal location in the cordinate frame of the robot.
    if self.task_params.outputs.rel_goal_loc:
      if self.task_params.type[:14] != 'to_nearest_obj':
        loc, _, _, theta = self.get_loc_axis(current_nodes,
                                             delta_theta=self.task.delta_theta,
                                             perturb=perturbs[:,step_number,:])
        goal_loc, _, _, goal_theta = self.get_loc_axis(end_nodes,
                                                       delta_theta=self.task.delta_theta,
                                                       perturb=end_perturbs[:,0,:])
        r_goal, t_goal = _get_relative_goal_loc(goal_loc, loc, theta)

        rel_goal_loc = np.concatenate((r_goal*np.cos(t_goal), r_goal*np.sin(t_goal),
                                       np.cos(goal_theta-theta),
                                       np.sin(goal_theta-theta)), axis=1)
        outs['rel_goal_loc'] = np.expand_dims(rel_goal_loc, axis=1)
      elif self.task_params.type[:14] == 'to_nearest_obj':
        rel_goal_loc = np.zeros((self.task_params.batch_size, 1,
                                 len(self.task_params.semantic_task.class_map_names)),
                                dtype=np.float32)
        for i in range(self.task_params.batch_size):
          t = target_class[i]
          rel_goal_loc[i,0,t] = 1.
        outs['rel_goal_loc'] = rel_goal_loc

    # Location on map to plot the trajectory during validation.
    if self.task_params.outputs.loc_on_map:
      loc, x_axis, y_axis, theta = self.get_loc_axis(current_nodes,
                                                     delta_theta=self.task.delta_theta,
                                                     perturb=perturbs[:,step_number,:])
      outs['loc_on_map'] = np.expand_dims(loc, axis=1)

    # Compute gt_dist to goal
    if self.task_params.outputs.gt_dist_to_goal:
      gt_dist_to_goal = np.zeros((len(current_node_ids), 1), dtype=np.float32)
      for i, n in enumerate(current_node_ids):
        gt_dist_to_goal[i,0] = self.episode.dist_to_goal[goal_number][i][n]
      outs['gt_dist_to_goal'] = np.expand_dims(gt_dist_to_goal, axis=1)

    # Free space in front of you, map and goal as images.
    if self.task_params.outputs.ego_maps:
      loc, x_axis, y_axis, theta = self.get_loc_axis(current_nodes,
                                                     delta_theta=self.task.delta_theta,
                                                     perturb=perturbs[:,step_number,:])
      maps = generate_egocentric_maps(self.task.scaled_maps,
                                      self.task_params.map_scales,
                                      self.task_params.map_crop_sizes, loc,
                                      x_axis, y_axis, theta)

      for i in range(len(self.task_params.map_scales)):
        outs['ego_maps_{:d}'.format(i)] = \
            np.expand_dims(np.expand_dims(maps[i], axis=1), axis=-1)

    if self.task_params.outputs.readout_maps:
      loc, x_axis, y_axis, theta = self.get_loc_axis(current_nodes,
                                                     delta_theta=self.task.delta_theta,
                                                     perturb=perturbs[:,step_number,:])
      maps = generate_egocentric_maps(self.task.readout_maps_scaled,
                                      self.task_params.readout_maps_scales,
                                      self.task_params.readout_maps_crop_sizes,
                                      loc, x_axis, y_axis, theta)
      for i in range(len(self.task_params.readout_maps_scales)):
        outs['readout_maps_{:d}'.format(i)] = \
            np.expand_dims(np.expand_dims(maps[i], axis=1), axis=-1)

    # Images for the goal.
    if self.task_params.outputs.ego_goal_imgs:
      if self.task_params.type[:14] != 'to_nearest_obj': 
        loc, x_axis, y_axis, theta = self.get_loc_axis(current_nodes,
                                                       delta_theta=self.task.delta_theta,
                                                       perturb=perturbs[:,step_number,:])
        goal_loc, _, _, _ = self.get_loc_axis(end_nodes,
                                              delta_theta=self.task.delta_theta,
                                              perturb=end_perturbs[:,0,:])
        rel_goal_orientation = np.mod(
            np.int32(current_nodes[:,2:] - end_nodes[:,2:]), self.task_params.n_ori)
        goal_dist, goal_theta = _get_relative_goal_loc(goal_loc, loc, theta)
        goals = generate_goal_images(self.task_params.map_scales,
                                     self.task_params.map_crop_sizes,
                                     self.task_params.n_ori, goal_dist,
                                     goal_theta, rel_goal_orientation)
        for i in range(len(self.task_params.map_scales)):
          outs['ego_goal_imgs_{:d}'.format(i)] = np.expand_dims(goals[i], axis=1)

      elif self.task_params.type[:14] == 'to_nearest_obj':
        for i in range(len(self.task_params.map_scales)):
          num_classes = len(self.task_params.semantic_task.class_map_names)
          outs['ego_goal_imgs_{:d}'.format(i)] = np.zeros((self.task_params.batch_size, 1,
                                                           self.task_params.map_crop_sizes[i],
                                                           self.task_params.map_crop_sizes[i],
                                                           self.task_params.goal_channels))
        for i in range(self.task_params.batch_size):
          t = target_class[i]
          for j in range(len(self.task_params.map_scales)):
            outs['ego_goal_imgs_{:d}'.format(j)][i,:,:,:,t] = 1.

    # Incremental locs and theta (for map warping), always in the original scale
    # of the map, the subequent steps in the tf code scale appropriately.
    # Scaling is done by just multiplying incremental_locs appropriately.
    if self.task_params.outputs.egomotion:
      if step_number == 0:
        # Zero Ego Motion
        incremental_locs = np.zeros((self.task_params.batch_size, 1, 2), dtype=np.float32)
        incremental_thetas = np.zeros((self.task_params.batch_size, 1, 1), dtype=np.float32)
      else:
        previous_nodes = self.task.nodes[self.episode.history[:,step_number-1], :]*1
        loc, _, _, theta = self.get_loc_axis(current_nodes,
                                             delta_theta=self.task.delta_theta,
                                             perturb=perturbs[:,step_number,:])
        previous_loc, _, _, previous_theta = self.get_loc_axis(
            previous_nodes, delta_theta=self.task.delta_theta,
            perturb=perturbs[:,step_number-1,:])

        incremental_locs_ = np.reshape(loc-previous_loc, [self.task_params.batch_size, 1, -1])

        t = -np.pi/2+np.reshape(theta*1, [self.task_params.batch_size, 1, -1])
        incremental_locs = incremental_locs_*1
        incremental_locs[:,:,0] = np.sum(incremental_locs_ *
                                         np.concatenate((np.cos(t), np.sin(t)),
                                                        axis=-1), axis=-1)
        incremental_locs[:,:,1] = np.sum(incremental_locs_ *
                                         np.concatenate((np.cos(t+np.pi/2),
                                                         np.sin(t+np.pi/2)),
                                                        axis=-1), axis=-1)
        incremental_thetas = np.reshape(theta-previous_theta,
                                        [self.task_params.batch_size, 1, -1])
      outs['incremental_locs'] = incremental_locs
      outs['incremental_thetas'] = incremental_thetas

    if self.task_params.outputs.visit_count:
      # Output the visit count for this state, how many times has the current
      # state been visited, and how far in the history was the last visit
      # (except this one)
      visit_count = np.zeros((self.task_params.batch_size, 1), dtype=np.int32)
      last_visit = -np.ones((self.task_params.batch_size, 1), dtype=np.int32)
      if step_number >= 1:
        h = self.episode.history[:,:(step_number)]
        visit_count[:,0] = np.sum(h == np.array(current_node_ids).reshape([-1,1]),
                                  axis=1)
        last_visit[:,0] = np.argmax(h[:,::-1] == np.array(current_node_ids).reshape([-1,1]),
                                    axis=1) + 1
        last_visit[visit_count == 0] = -1 # -1 if not visited.
      outs['visit_count'] = np.expand_dims(visit_count, axis=1)
      outs['last_visit'] = np.expand_dims(last_visit, axis=1)
    return outs

  def get_features_name(self):
    f = []
    if self.task_params.outputs.images:
      f.append('imgs')
    if self.task_params.outputs.rel_goal_loc:
      f.append('rel_goal_loc')
    if self.task_params.outputs.loc_on_map:
      f.append('loc_on_map')
    if self.task_params.outputs.gt_dist_to_goal:
      f.append('gt_dist_to_goal')
    if self.task_params.outputs.ego_maps:
      for i in range(len(self.task_params.map_scales)):
        f.append('ego_maps_{:d}'.format(i))
    if self.task_params.outputs.readout_maps:
      for i in range(len(self.task_params.readout_maps_scales)):
        f.append('readout_maps_{:d}'.format(i))
    if self.task_params.outputs.ego_goal_imgs:
      for i in range(len(self.task_params.map_scales)):
        f.append('ego_goal_imgs_{:d}'.format(i))
    if self.task_params.outputs.egomotion:
      f.append('incremental_locs')
      f.append('incremental_thetas')
    if self.task_params.outputs.visit_count:
      f.append('visit_count')
      f.append('last_visit')
    if self.task_params.outputs.analytical_counts:
      for i in range(len(self.task_params.analytical_counts.map_sizes)):
        f.append('analytical_counts_{:d}'.format(i))
    if self.task_params.outputs.node_ids:
      f.append('node_ids')
      f.append('perturbs')
    return f

  def pre_features(self, inputs):
    if self.task_params.outputs.images:
      inputs['imgs'] = image_pre(inputs['imgs'], self.task_params.modalities)
    return inputs

class BuildingMultiplexer():
  def __init__(self, args, task_number):
    params = vars(args)
    for k in params.keys():
      setattr(self, k, params[k])
    self.task_number = task_number
    self._pick_data(task_number)
    logging.info('Env Class: %s.', self.env_class)
    if self.task_params.task == 'planning':
      self._setup_planner()
    elif self.task_params.task == 'mapping':
      self._setup_mapper()
    elif self.task_params.task == 'map+plan':
      self._setup_mapper()
    else:
      logging.error('Undefined task: %s'.format(self.task_params.task))

  def _pick_data(self, task_number):
    logging.error('Input Building Names: %s', self.building_names)
    self.flip = [np.mod(task_number / len(self.building_names), 2) == 1]
    id = np.mod(task_number, len(self.building_names))
    self.building_names = [self.building_names[id]]
    self.task_params.building_seed = task_number
    logging.error('BuildingMultiplexer: Picked Building Name: %s', self.building_names)
    self.building_names = self.building_names[0].split('+')
    self.flip = [self.flip[0] for _ in self.building_names]
    logging.error('BuildingMultiplexer: Picked Building Name: %s', self.building_names)
    logging.error('BuildingMultiplexer: Flipping Buildings: %s', self.flip)
    logging.error('BuildingMultiplexer: Set building_seed: %d', self.task_params.building_seed)
    self.num_buildings = len(self.building_names)
    logging.error('BuildingMultiplexer: Num buildings: %d', self.num_buildings)

  def _setup_planner(self):
    # Load building env class.
    self.buildings = []
    for i, building_name in enumerate(self.building_names):
      b = self.env_class(robot=self.robot, env=self.env,
                         task_params=self.task_params,
                         building_name=building_name, flip=self.flip[i],
                         logdir=self.logdir, building_loader=self.dataset)
      self.buildings.append(b)

  def _setup_mapper(self):
    # Set up the renderer.
    cp = self.camera_param
    rgb_shader, d_shader = sru.get_shaders(cp.modalities)
    r_obj = SwiftshaderRenderer()
    r_obj.init_display(width=cp.width, height=cp.height, fov=cp.fov,
                       z_near=cp.z_near, z_far=cp.z_far, rgb_shader=rgb_shader,
                       d_shader=d_shader)
    self.r_obj = r_obj
    r_obj.clear_scene()

    # Load building env class.
    self.buildings = []
    wt = []
    for i, building_name in enumerate(self.building_names):
      b = self.env_class(robot=self.robot, env=self.env,
                         task_params=self.task_params,
                         building_name=building_name, flip=self.flip[i],
                         logdir=self.logdir, building_loader=self.dataset,
                         r_obj=r_obj)
      wt.append(b.get_weight())
      b.load_building_into_scene()
      b.set_building_visibility(False)
      self.buildings.append(b)
    wt = np.array(wt).astype(np.float32)
    wt = wt / np.sum(wt+0.0001)
    self.building_sampling_weights = wt

  def sample_building(self, rng):
    if self.num_buildings == 1:
      building_id = rng.choice(range(len(self.building_names)))
    else:
      building_id = rng.choice(self.num_buildings,
                               p=self.building_sampling_weights)
    b = self.buildings[building_id]
    instances = b._gen_rng(rng)
    self._building_id = building_id
    return self.buildings[building_id], instances

  def sample_env(self, rngs):
    rng = rngs[0];
    if self.num_buildings == 1:
      building_id = rng.choice(range(len(self.building_names)))
    else:
      building_id = rng.choice(self.num_buildings,
                               p=self.building_sampling_weights)
    return self.buildings[building_id]

  def pre(self, inputs):
    return self.buildings[self._building_id].pre(inputs)
  
  def __del__(self):
    self.r_obj.clear_scene()
    logging.error('Clearing scene.')
