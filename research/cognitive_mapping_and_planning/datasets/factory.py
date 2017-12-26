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

r"""Wrapper for selecting the navigation environment that we want to train and
test on.
"""
import numpy as np
import os, glob
import platform

import logging
from tensorflow.python.platform import app
from tensorflow.python.platform import flags

import render.swiftshader_renderer as renderer 
import src.file_utils as fu
import src.utils as utils

def get_dataset(dataset_name):
  if dataset_name == 'sbpd':
    dataset = StanfordBuildingParserDataset(dataset_name)
  else:
    logging.fatal('Not one of sbpd')
  return dataset

class Loader():
  def get_data_dir():
    pass

  def get_meta_data(self, file_name, data_dir=None):
    if data_dir is None:
      data_dir = self.get_data_dir()
    full_file_name = os.path.join(data_dir, 'meta', file_name)
    assert(fu.exists(full_file_name)), \
      '{:s} does not exist'.format(full_file_name)
    ext = os.path.splitext(full_file_name)[1]
    if ext == '.txt':
      ls = []
      with fu.fopen(full_file_name, 'r') as f:
        for l in f:
          ls.append(l.rstrip())
    elif ext == '.pkl':
      ls = utils.load_variables(full_file_name)
    return ls

  def load_building(self, name, data_dir=None):
    if data_dir is None:
      data_dir = self.get_data_dir()
    out = {}
    out['name'] = name
    out['data_dir'] = data_dir
    out['room_dimension_file'] = os.path.join(data_dir, 'room-dimension',
                                              name+'.pkl')
    out['class_map_folder'] = os.path.join(data_dir, 'class-maps')
    return out

  def load_building_meshes(self, building):
    dir_name = os.path.join(building['data_dir'], 'mesh', building['name'])
    mesh_file_name = glob.glob1(dir_name, '*.obj')[0]
    mesh_file_name_full = os.path.join(dir_name, mesh_file_name)
    logging.error('Loading building from obj file: %s', mesh_file_name_full)
    shape = renderer.Shape(mesh_file_name_full, load_materials=True, 
                           name_prefix=building['name']+'_')
    return [shape]

class StanfordBuildingParserDataset(Loader):
  def __init__(self, ver):
    self.ver = ver
    self.data_dir = None
  
  def get_data_dir(self):
    if self.data_dir is None:
      self.data_dir = 'data/stanford_building_parser_dataset/'
    return self.data_dir

  def get_benchmark_sets(self):
    return self._get_benchmark_sets()

  def get_split(self, split_name):
    if self.ver == 'sbpd':
      return self._get_split(split_name)
    else:
      logging.fatal('Unknown version.')

  def _get_benchmark_sets(self):
    sets = ['train1', 'val', 'test']
    return sets

  def _get_split(self, split_name):
    train = ['area1', 'area5a', 'area5b', 'area6']
    train1 = ['area1']
    val = ['area3']
    test = ['area4']

    sets = {}
    sets['train'] = train
    sets['train1'] = train1
    sets['val'] = val
    sets['test'] = test
    sets['all'] = sorted(list(set(train + val + test)))
    return sets[split_name]
