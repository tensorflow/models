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

import os
import glob
import numpy as np
import logging
import cPickle
from datasets import nav_env
from datasets import factory
from src import utils 
from src import map_utils as mu

logging.basicConfig(level=logging.INFO)
DATA_DIR = 'data/stanford_building_parser_dataset_raw/'

mkdir_if_missing = utils.mkdir_if_missing
save_variables = utils.save_variables

def _get_semantic_maps(building_name, transform, map_, flip, cats):
  rooms = get_room_in_building(building_name)
  maps = []
  for cat in cats:
    maps.append(np.zeros((map_.size[1], map_.size[0])))
  
  for r in rooms:
    room = load_room(building_name, r, category_list=cats)
    classes = room['class_id']
    for i, cat in enumerate(cats):
      c_ind = cats.index(cat)
      ind = [_ for _, c in enumerate(classes) if c == c_ind]
      if len(ind) > 0:
        vs = [room['vertexs'][x]*1 for x in ind]
        vs = np.concatenate(vs, axis=0)
        if transform:
          vs = np.array([vs[:,1], vs[:,0], vs[:,2]]).T
          vs[:,0] = -vs[:,0]
          vs[:,1] += 4.20
          vs[:,0] += 6.20
        vs = vs*100.
        if flip:
          vs[:,1] = -vs[:,1]
        maps[i] = maps[i] + \
            mu._project_to_map(map_, vs, ignore_points_outside_map=True)
  return maps

def _map_building_name(building_name):
  b = int(building_name.split('_')[0][4])
  out_name = 'Area_{:d}'.format(b)
  if b == 5:
    if int(building_name.split('_')[0][5]) == 1:
      transform = True
    else:
      transform = False
  else:
    transform = False
  return out_name, transform

def get_categories():
  cats = ['beam', 'board', 'bookcase', 'ceiling', 'chair', 'clutter', 'column',
          'door', 'floor', 'sofa', 'table', 'wall', 'window']
  return cats

def _write_map_files(b_in, b_out, transform):
  cats = get_categories()

  env = utils.Foo(padding=10, resolution=5, num_point_threshold=2,
                  valid_min=-10, valid_max=200, n_samples_per_face=200)
  robot = utils.Foo(radius=15, base=10, height=140, sensor_height=120,
                    camera_elevation_degree=-15)
  
  building_loader = factory.get_dataset('sbpd')
  for flip in [False, True]:
    b = nav_env.Building(b_out, robot, env, flip=flip,
                         building_loader=building_loader)
    logging.info("building_in: %s, building_out: %s, transform: %d", b_in,
                 b_out, transform)
    maps = _get_semantic_maps(b_in, transform, b.map, flip, cats)
    maps = np.transpose(np.array(maps), axes=[1,2,0])

    #  Load file from the cache.
    file_name = '{:s}_{:d}_{:d}_{:d}_{:d}_{:d}_{:d}.pkl'
    file_name = file_name.format(b.building_name, b.map.size[0], b.map.size[1],
                                 b.map.origin[0], b.map.origin[1],
                                 b.map.resolution, flip)
    out_file = os.path.join(DATA_DIR, 'processing', 'class-maps', file_name)
    logging.info('Writing semantic maps to %s.', out_file)
    save_variables(out_file, [maps, cats], ['maps', 'cats'], overwrite=True)

def _transform_area5b(room_dimension):
  for a in room_dimension.keys():
    r = room_dimension[a]*1
    r[[0,1,3,4]] = r[[1,0,4,3]]
    r[[0,3]] = -r[[3,0]]
    r[[1,4]] += 4.20
    r[[0,3]] += 6.20
    room_dimension[a] = r
  return room_dimension

def collect_room(building_name, room_name):
  room_dir = os.path.join(DATA_DIR, 'Stanford3dDataset_v1.2', building_name,
                          room_name, 'Annotations')
  files = glob.glob1(room_dir, '*.txt')
  files = sorted(files, key=lambda s: s.lower())
  vertexs = []; colors = [];
  for f in files:
    file_name = os.path.join(room_dir, f)
    logging.info('  %s', file_name)
    a = np.loadtxt(file_name)
    vertex = a[:,:3]*1.
    color = a[:,3:]*1
    color = color.astype(np.uint8)
    vertexs.append(vertex)
    colors.append(color)
  files = [f.split('.')[0] for f in files]
  out = {'vertexs': vertexs, 'colors': colors, 'names': files}
  return out

def load_room(building_name, room_name, category_list=None):
  room = collect_room(building_name, room_name)
  room['building_name'] = building_name
  room['room_name']     = room_name
  instance_id = range(len(room['names']))
  room['instance_id'] = instance_id
  if category_list is not None:
    name = [r.split('_')[0] for r in room['names']]
    class_id = []
    for n in name:
      if n in category_list:
        class_id.append(category_list.index(n))
      else:
        class_id.append(len(category_list))
    room['class_id'] = class_id
    room['category_list'] = category_list
  return room

def get_room_in_building(building_name):
  building_dir = os.path.join(DATA_DIR, 'Stanford3dDataset_v1.2', building_name)
  rn = os.listdir(building_dir)
  rn = [x for x in rn if os.path.isdir(os.path.join(building_dir, x))]
  rn = sorted(rn, key=lambda s: s.lower())
  return rn

def write_room_dimensions(b_in, b_out, transform):
  rooms = get_room_in_building(b_in)
  room_dimension = {}
  for r in rooms:
    room = load_room(b_in, r, category_list=None)
    vertex = np.concatenate(room['vertexs'], axis=0)
    room_dimension[r] = np.concatenate((np.min(vertex, axis=0), np.max(vertex, axis=0)), axis=0)
  if transform == 1:
    room_dimension = _transform_area5b(room_dimension)
  
  out_file = os.path.join(DATA_DIR, 'processing', 'room-dimension', b_out+'.pkl')
  save_variables(out_file, [room_dimension], ['room_dimension'], overwrite=True)

def write_room_dimensions_all(I):
  mkdir_if_missing(os.path.join(DATA_DIR, 'processing', 'room-dimension'))
  bs_in = ['Area_1', 'Area_2', 'Area_3', 'Area_4', 'Area_5', 'Area_5', 'Area_6']
  bs_out = ['area1', 'area2', 'area3', 'area4', 'area5a', 'area5b', 'area6']
  transforms = [0, 0, 0, 0, 0, 1, 0]
  
  for i in I:
    b_in = bs_in[i]
    b_out = bs_out[i]
    t = transforms[i]
    write_room_dimensions(b_in, b_out, t)

def write_class_maps_all(I):
  mkdir_if_missing(os.path.join(DATA_DIR, 'processing', 'class-maps'))
  bs_in = ['Area_1', 'Area_2', 'Area_3', 'Area_4', 'Area_5', 'Area_5', 'Area_6']
  bs_out = ['area1', 'area2', 'area3', 'area4', 'area5a', 'area5b', 'area6']
  transforms = [0, 0, 0, 0, 0, 1, 0]
  
  for i in I:
    b_in = bs_in[i]
    b_out = bs_out[i]
    t = transforms[i]
    _write_map_files(b_in, b_out, t)


if __name__ == '__main__':
  write_room_dimensions_all([0, 2, 3, 4, 5, 6])
  write_class_maps_all([0, 2, 3, 4, 5, 6])

