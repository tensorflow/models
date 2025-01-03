# Copyright 2018 The TensorFlow Authors All Rights Reserved.
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

"""Initializes at random location and visualizes the optimal path.

Different modes of execution:
1) benchmark: It generates benchmark_iter sample trajectory to random goals
   and plots the histogram of path lengths. It can be also used to see how fast
   it runs.
2) vis: It visualizes the generated paths by image, semantic segmentation, and
   so on.
3) human: allows the user to navigate through environment from keyboard input.

python viz_active_vision_dataset_main -- \
  --mode=benchmark --benchmark_iter=1000 --gin_config=envs/configs/active_vision_config.gin

python viz_active_vision_dataset_main -- \
  --mode=vis \
  --gin_config=envs/configs/active_vision_config.gin

python viz_active_vision_dataset_main -- \
  --mode=human \
  --gin_config=envs/configs/active_vision_config.gin

python viz_active_vision_dataset_main.py --mode=eval --eval_folder=/usr/local/google/home/$USER/checkin_log_det/evals/ --output_folder=/usr/local/google/home/$USER/test_imgs/ --gin_config=envs/configs/active_vision_config.gin

"""

import matplotlib
# pylint: disable=g-import-not-at-top
# Need Tk for interactive plots.
matplotlib.use('TkAgg')
import tensorflow as tf
from matplotlib import pyplot as plt
import numpy as np
import os
from pyglib import app
from pyglib import flags
import gin
import cv2
from envs import active_vision_dataset_env
from envs import task_env


VIS_MODE = 'vis'
HUMAN_MODE = 'human'
BENCHMARK_MODE = 'benchmark'
GRAPH_MODE = 'graph'
EVAL_MODE = 'eval'

flags.DEFINE_enum('mode', VIS_MODE,
                  [VIS_MODE, HUMAN_MODE, BENCHMARK_MODE, GRAPH_MODE, EVAL_MODE],
                  'mode of the execution')
flags.DEFINE_integer('benchmark_iter', 1000,
                     'number of iterations for benchmarking')
flags.DEFINE_string('eval_folder', '', 'the path to the eval folder')
flags.DEFINE_string('output_folder', '',
                    'the path to which the images and gifs are written')
flags.DEFINE_multi_string('gin_config', [],
                          'List of paths to a gin config files for the env.')
flags.DEFINE_multi_string('gin_params', [],
                          'Newline separated list of Gin parameter bindings.')

mt = task_env.ModalityTypes
FLAGS = flags.FLAGS

def benchmark(env, targets):
  """Benchmarks the speed of sequence generation by env.

  Args:
    env: environment.
    targets: list of target classes.
  """
  episode_lengths = {}
  all_init_configs = {}
  all_actions = dict([(a, 0.) for a in env.actions])
  for i in range(FLAGS.benchmark_iter):
    path, actions, _, _ = env.random_step_sequence()
    selected_actions = np.argmax(actions, axis=-1)
    new_actions = dict([(a, 0.) for a in env.actions])
    for a in selected_actions:
      new_actions[env.actions[a]] += 1. / selected_actions.shape[0]
    for a in new_actions:
      all_actions[a] += new_actions[a] / FLAGS.benchmark_iter
    start_image_id, world, goal = env.get_init_config(path)
    print world
    if world not in all_init_configs:
      all_init_configs[world] = set()
    all_init_configs[world].add((start_image_id, goal, len(actions)))
    if env.goal_index not in episode_lengths:
      episode_lengths[env.goal_index] = []
    episode_lengths[env.goal_index].append(len(actions))
  for i, cls in enumerate(episode_lengths):
    plt.subplot(231 + i)
    plt.hist(episode_lengths[cls])
    plt.title(targets[cls])
  plt.show()


def human(env, targets):
  """Lets user play around the env manually."""
  string_key_map = {
      'a': 'left',
      'd': 'right',
      'w': 'forward',
      's': 'backward',
      'j': 'rotate_ccw',
      'l': 'rotate_cw',
      'n': 'stop'
  }
  integer_key_map = {
      'a': env.actions.index('left'),
      'd': env.actions.index('right'),
      'w': env.actions.index('forward'),
      's': env.actions.index('backward'),
      'j': env.actions.index('rotate_ccw'),
      'l': env.actions.index('rotate_cw'),
      'n': env.actions.index('stop')
  }
  for k in integer_key_map:
    integer_key_map[k] = np.int32(integer_key_map[k])
  plt.ion()
  for _ in range(20):
    obs = env.reset()
    steps = -1
    action = None
    while True:
      print 'distance = ', obs[task_env.ModalityTypes.DISTANCE]
      steps += 1
      depth_value = obs[task_env.ModalityTypes.DEPTH][:, :, 0]
      depth_mask = obs[task_env.ModalityTypes.DEPTH][:, :, 1]
      seg_mask = np.squeeze(obs[task_env.ModalityTypes.SEMANTIC_SEGMENTATION])
      det_mask = np.argmax(
          obs[task_env.ModalityTypes.OBJECT_DETECTION], axis=-1)
      img = obs[task_env.ModalityTypes.IMAGE]
      plt.subplot(231)
      plt.title('steps = {}'.format(steps))
      plt.imshow(img.astype(np.uint8))
      plt.subplot(232)
      plt.imshow(depth_value)
      plt.title('depth value')
      plt.subplot(233)
      plt.imshow(depth_mask)
      plt.title('depth mask')
      plt.subplot(234)
      plt.imshow(seg_mask)
      plt.title('seg')
      plt.subplot(235)
      plt.imshow(det_mask)
      plt.title('det')
      plt.subplot(236)
      plt.title('goal={}'.format(targets[env.goal_index]))
      plt.draw()
      while True:
        s = raw_input('key = ')
        if np.random.rand() > 0.5:
          key_map = string_key_map
        else:
          key_map = integer_key_map
        if s in key_map:
          action = key_map[s]
          break
        else:
          print 'invalid action'
      print 'action = {}'.format(action)
      if action == 'stop':
        print 'dist to goal: {}'.format(len(env.path_to_goal()) - 2)
        break
      obs, reward, done, info = env.step(action)
      print 'reward = {}, done = {}, success = {}'.format(
          reward, done, info['success'])


def visualize_random_step_sequence(env):
  """Visualizes random sequence of steps."""
  plt.ion()
  for _ in range(20):
    path, actions, _, step_outputs = env.random_step_sequence(max_len=30)
    print 'path = {}'.format(path)
    for action, step_output in zip(actions, step_outputs):
      obs, _, done, _ = step_output
      depth_value = obs[task_env.ModalityTypes.DEPTH][:, :, 0]
      depth_mask = obs[task_env.ModalityTypes.DEPTH][:, :, 1]
      seg_mask = np.squeeze(obs[task_env.ModalityTypes.SEMANTIC_SEGMENTATION])
      det_mask = np.argmax(
          obs[task_env.ModalityTypes.OBJECT_DETECTION], axis=-1)
      img = obs[task_env.ModalityTypes.IMAGE]
      plt.subplot(231)
      plt.imshow(img.astype(np.uint8))
      plt.subplot(232)
      plt.imshow(depth_value)
      plt.title('depth value')
      plt.subplot(233)
      plt.imshow(depth_mask)
      plt.title('depth mask')
      plt.subplot(234)
      plt.imshow(seg_mask)
      plt.title('seg')
      plt.subplot(235)
      plt.imshow(det_mask)
      plt.title('det')
      plt.subplot(236)
      print 'action = {}'.format(action)
      print 'done = {}'.format(done)
      plt.draw()
      if raw_input('press \'n\' to go to the next random sequence. Otherwise, '
                   'press any key to continue...') == 'n':
        break


def visualize(env, input_folder, output_root_folder):
  """visualizes images for sequence of steps from the evals folder."""
  def which_env(file_name):
    img_name = file_name.split('_')[0][2:5]
    env_dict = {'161': 'Home_016_1', '131': 'Home_013_1', '111': 'Home_011_1'}
    if img_name in env_dict:
      return env_dict[img_name]
    else:
      raise ValueError('could not resolve env: {} {}'.format(
          img_name, file_name))

  def which_goal(file_name):
    return file_name[file_name.find('_')+1:]

  output_images_folder = os.path.join(output_root_folder, 'images')
  output_gifs_folder = os.path.join(output_root_folder, 'gifs')
  if not tf.gfile.IsDirectory(output_images_folder):
    tf.gfile.MakeDirs(output_images_folder)
  if not tf.gfile.IsDirectory(output_gifs_folder):
    tf.gfile.MakeDirs(output_gifs_folder)
  npy_files = [
      os.path.join(input_folder, name)
      for name in tf.gfile.ListDirectory(input_folder)
      if name.find('npy') >= 0
  ]
  for i, npy_file in enumerate(npy_files):
    print 'saving images {}/{}'.format(i, len(npy_files))
    pure_name = npy_file[npy_file.rfind('/') + 1:-4]
    output_folder = os.path.join(output_images_folder, pure_name)
    if not tf.gfile.IsDirectory(output_folder):
      tf.gfile.MakeDirs(output_folder)
    print '*******'
    print pure_name[0:pure_name.find('_')]
    env.reset_for_eval(which_env(pure_name),
                       which_goal(pure_name),
                       pure_name[0:pure_name.find('_')],
                      )
    with tf.gfile.Open(npy_file) as h:
      states = np.load(h).item()['states']
      images = [
          env.observation(state)[mt.IMAGE] for state in states
      ]
      for j, img in enumerate(images):
        cv2.imwrite(os.path.join(output_folder, '{0:03d}'.format(j) + '.jpg'),
                    img[:, :, ::-1])
      print 'converting to gif'
      os.system(
          'convert -set delay 20 -colors 256 -dispose 1 {}/*.jpg {}.gif'.format(
              output_folder,
              os.path.join(output_gifs_folder, pure_name + '.gif')
          )
      )

def evaluate_folder(env, folder_path):
  """Evaluates the performance from the evals folder."""
  targets = ['fridge', 'dining_table', 'microwave', 'tv', 'couch']

  def compute_acc(npy_file):
    with tf.gfile.Open(npy_file) as h:
      data = np.load(h).item()
    if npy_file.find('dining_table') >= 0:
      category = 'dining_table'
    else:
      category = npy_file[npy_file.rfind('_') + 1:-4]
    return category, data['distance'][-1] - 2

  def evaluate_iteration(folder):
    """Evaluates the data from the folder of certain eval iteration."""
    print folder
    npy_files = [
        os.path.join(folder, name)
        for name in tf.gfile.ListDirectory(folder)
        if name.find('npy') >= 0
    ]
    eval_stats = {c: [] for c in targets}
    for npy_file in npy_files:
      try:
        category, dist = compute_acc(npy_file)
      except:  # pylint: disable=bare-except
        continue
      eval_stats[category].append(float(dist <= 5))
    for c in eval_stats:
      if not eval_stats[c]:
        print 'incomplete eval {}: empty class {}'.format(folder_path, c)
        return None
      eval_stats[c] = np.mean(eval_stats[c])

    eval_stats['mean'] = np.mean(eval_stats.values())
    return eval_stats

  checkpoint_folders = [
      folder_path + x
      for x in tf.gfile.ListDirectory(folder_path)
      if tf.gfile.IsDirectory(folder_path + x)
  ]

  print '{} folders found'.format(len(checkpoint_folders))
  print '------------------------'
  all_iters = []
  all_accs = []
  for i, folder in enumerate(checkpoint_folders):
    print 'processing {}/{}'.format(i, len(checkpoint_folders))
    eval_stats = evaluate_iteration(folder)
    if eval_stats is None:
      continue
    else:
      iter_no = int(folder[folder.rfind('/') + 1:])
      print 'result ', iter_no, eval_stats['mean']
      all_accs.append(eval_stats['mean'])
      all_iters.append(iter_no)

  all_accs = np.asarray(all_accs)
  all_iters = np.asarray(all_iters)
  idx = np.argmax(all_accs)
  print 'best result at iteration {} was {}'.format(all_iters[idx],
                                                    all_accs[idx])
  order = np.argsort(all_iters)
  all_iters = all_iters[order]
  all_accs = all_accs[order]
  #plt.plot(all_iters, all_accs)
  #plt.show()
  #print 'done plotting'

  best_iteration_folder = os.path.join(folder_path, str(all_iters[idx]))

  print 'generating gifs and images for {}'.format(best_iteration_folder)
  visualize(env, best_iteration_folder, FLAGS.output_folder)


def main(_):
  gin.parse_config_files_and_bindings(FLAGS.gin_config, FLAGS.gin_params)
  print('********')
  print(FLAGS.mode)
  print(FLAGS.gin_config)
  print(FLAGS.gin_params)

  env = active_vision_dataset_env.ActiveVisionDatasetEnv(modality_types=[
      task_env.ModalityTypes.IMAGE,
      task_env.ModalityTypes.SEMANTIC_SEGMENTATION,
      task_env.ModalityTypes.OBJECT_DETECTION, task_env.ModalityTypes.DEPTH,
      task_env.ModalityTypes.DISTANCE
  ])

  if FLAGS.mode == BENCHMARK_MODE:
    benchmark(env, env.possible_targets)
  elif FLAGS.mode == GRAPH_MODE:
    for loc in env.worlds:
      env.check_scene_graph(loc, 'fridge')
  elif FLAGS.mode == HUMAN_MODE:
    human(env, env.possible_targets)
  elif FLAGS.mode == VIS_MODE:
    visualize_random_step_sequence(env)
  elif FLAGS.mode == EVAL_MODE:
    evaluate_folder(env, FLAGS.eval_folder)

if __name__ == '__main__':
  app.run(main)
