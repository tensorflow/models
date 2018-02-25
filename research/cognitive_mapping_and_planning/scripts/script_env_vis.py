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

"""A simple python function to walk in the enviornments that we have created.
PYTHONPATH='.' PYOPENGL_PLATFORM=egl python scripts/script_env_vis.py \
  --dataset_name sbpd --building_name area3
"""
import sys
import numpy as np
import matplotlib
matplotlib.use('TkAgg')
from PIL import ImageTk, Image
import Tkinter as tk
import logging
from tensorflow.python.platform import app
from tensorflow.python.platform import flags

import datasets.nav_env_config as nec
import datasets.nav_env as nav_env
import cv2
from datasets import factory
import render.swiftshader_renderer as renderer

SwiftshaderRenderer = renderer.SwiftshaderRenderer
VisualNavigationEnv = nav_env.VisualNavigationEnv

FLAGS = flags.FLAGS
flags.DEFINE_string('dataset_name', 'sbpd', 'Name of the dataset.')
flags.DEFINE_float('fov', 60., 'Field of view')
flags.DEFINE_integer('image_size', 512, 'Size of the image.')
flags.DEFINE_string('building_name', '', 'Name of the building.')

def get_args():
  navtask = nec.nav_env_base_config()
  navtask.task_params.type = 'rng_rejection_sampling_many'
  navtask.task_params.rejection_sampling_M = 2000
  navtask.task_params.min_dist = 10
  sz = FLAGS.image_size
  navtask.camera_param.fov = FLAGS.fov
  navtask.camera_param.height = sz
  navtask.camera_param.width = sz
  navtask.task_params.img_height = sz
  navtask.task_params.img_width = sz

  # navtask.task_params.semantic_task.class_map_names = ['chair', 'door', 'table']
  # navtask.task_params.type = 'to_nearest_obj_acc'

  logging.info('navtask: %s', navtask)
  return navtask

def load_building(dataset_name, building_name):
  dataset = factory.get_dataset(dataset_name)

  navtask = get_args()
  cp = navtask.camera_param
  rgb_shader, d_shader = renderer.get_shaders(cp.modalities)
  r_obj = SwiftshaderRenderer()
  r_obj.init_display(width=cp.width, height=cp.height,
                     fov=cp.fov, z_near=cp.z_near, z_far=cp.z_far,
                     rgb_shader=rgb_shader, d_shader=d_shader)
  r_obj.clear_scene()
  b = VisualNavigationEnv(robot=navtask.robot, env=navtask.env,
                          task_params=navtask.task_params,
                          building_name=building_name, flip=False,
                          logdir=None, building_loader=dataset,
                          r_obj=r_obj)
  b.load_building_into_scene()
  b.set_building_visibility(False)
  return b

def walk_through(b):
  # init agent at a random location in the environment.
  init_env_state = b.reset([np.random.RandomState(0), np.random.RandomState(0)])

  global current_node
  rng = np.random.RandomState(0)
  current_node = rng.choice(b.task.nodes.shape[0])

  root = tk.Tk()
  image = b.render_nodes(b.task.nodes[[current_node],:])[0]
  print(image.shape)
  image = image.astype(np.uint8)
  im = Image.fromarray(image)
  im = ImageTk.PhotoImage(im)
  panel = tk.Label(root, image=im)

  map_size = b.traversible.shape
  sc = np.max(map_size)/256.
  loc = np.array([[map_size[1]/2., map_size[0]/2.]])
  x_axis = np.zeros_like(loc); x_axis[:,1] = sc
  y_axis = np.zeros_like(loc); y_axis[:,0] = -sc
  cum_fs, cum_valid = nav_env.get_map_to_predict(loc, x_axis, y_axis,
                                                   map=b.traversible*1.,
                                                   map_size=256)
  cum_fs = cum_fs[0]
  cum_fs = cv2.applyColorMap((cum_fs*255).astype(np.uint8), cv2.COLORMAP_JET)
  im = Image.fromarray(cum_fs)
  im = ImageTk.PhotoImage(im)
  panel_overhead = tk.Label(root, image=im)

  def refresh():
    global current_node
    image = b.render_nodes(b.task.nodes[[current_node],:])[0]
    image = image.astype(np.uint8)
    im = Image.fromarray(image)
    im = ImageTk.PhotoImage(im)
    panel.configure(image=im)
    panel.image = im

  def left_key(event):
    global current_node
    current_node = b.take_action([current_node], [2], 1)[0][0]
    refresh()

  def up_key(event):
    global current_node
    current_node = b.take_action([current_node], [3], 1)[0][0]
    refresh()

  def right_key(event):
    global current_node
    current_node = b.take_action([current_node], [1], 1)[0][0]
    refresh()

  def quit(event):
    root.destroy()

  panel_overhead.grid(row=4, column=5, rowspan=1, columnspan=1,
                      sticky=tk.W+tk.E+tk.N+tk.S)
  panel.bind('<Left>', left_key)
  panel.bind('<Up>', up_key)
  panel.bind('<Right>', right_key)
  panel.bind('q', quit)
  panel.focus_set()
  panel.grid(row=0, column=0, rowspan=5, columnspan=5,
             sticky=tk.W+tk.E+tk.N+tk.S)
  root.mainloop()

def simple_window():
  root = tk.Tk()

  image = np.zeros((128, 128, 3), dtype=np.uint8)
  image[32:96, 32:96, 0] = 255
  im = Image.fromarray(image)
  im = ImageTk.PhotoImage(im)

  image = np.zeros((128, 128, 3), dtype=np.uint8)
  image[32:96, 32:96, 1] = 255
  im2 = Image.fromarray(image)
  im2 = ImageTk.PhotoImage(im2)

  panel = tk.Label(root, image=im)

  def left_key(event):
    panel.configure(image=im2)
    panel.image = im2

  def quit(event):
    sys.exit()

  panel.bind('<Left>', left_key)
  panel.bind('<Up>', left_key)
  panel.bind('<Down>', left_key)
  panel.bind('q', quit)
  panel.focus_set()
  panel.pack(side = "bottom", fill = "both", expand = "yes")
  root.mainloop()

def main(_):
  b = load_building(FLAGS.dataset_name, FLAGS.building_name)
  walk_through(b)

if __name__ == '__main__':
  app.run()
