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

r"""
Code for plotting trajectories in the top view, and also plot first person views
from saved trajectories. Does not run the network but only loads the mesh data
to plot the view points.
  CUDA_VISIBLE_DEVICES=0 LD_LIBRARY_PATH=/opt/cuda-8.0/lib64:/opt/cudnnv51/lib64 
  PYTHONPATH='.' PYOPENGL_PLATFORM=egl python scripts/script_plot_trajectory.py \
      --first_person --num_steps 40 \
      --config_name cmp.lmap_Msc.clip5.sbpd_d_r2r \
      --imset test --alsologtostderr --base_dir output --out_dir vis

"""
import os, sys, numpy as np, copy
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from matplotlib.gridspec import GridSpec

import tensorflow as tf
from tensorflow.contrib import slim
import cv2
import logging
from tensorflow.python.platform import gfile
from tensorflow.python.platform import app 
from tensorflow.python.platform import flags 

from datasets import nav_env
import scripts.script_nav_agent_release as sna 
import src.file_utils as fu
from src import graph_utils 
from src import utils
FLAGS = flags.FLAGS

flags.DEFINE_string('out_dir', 'vis', 'Directory where to store the output')
flags.DEFINE_string('type', '', 'Optional type.')
flags.DEFINE_bool('first_person', False, 'Visualize the first person view.')
flags.DEFINE_bool('top_view', False, 'Visualize the trajectory in the top view.')
flags.DEFINE_integer('num_steps', 40, 'Number of steps to run the model for.')
flags.DEFINE_string('imset', 'test', '')
flags.DEFINE_string('base_dir', 'output', 'Cache directory.')

def _get_suffix_str():
  return ''


def _load_trajectory():
  base_dir = FLAGS.base_dir
  config_name = FLAGS.config_name+_get_suffix_str()

  dir_name = os.path.join(base_dir, FLAGS.type, config_name)
  logging.info('Waiting for snapshot in directory %s.', dir_name)
  last_checkpoint = slim.evaluation.wait_for_new_checkpoint(dir_name, None)
  checkpoint_iter = int(os.path.basename(last_checkpoint).split('-')[1])

  # Load the distances.
  a = utils.load_variables(os.path.join(dir_name, 'bench_on_'+FLAGS.imset,
                                        'all_locs_at_t_{:d}.pkl'.format(checkpoint_iter)))
  return a

def _compute_hardness():
  # Load the stanford data to compute the hardness.
  if FLAGS.type == '':
    args = sna.get_args_for_config(FLAGS.config_name+'+bench_'+FLAGS.imset)
  else:
    args = sna.get_args_for_config(FLAGS.type+'.'+FLAGS.config_name+'+bench_'+FLAGS.imset)

  args.navtask.logdir = None
  R = lambda: nav_env.get_multiplexer_class(args.navtask, 0)
  R = R()

  rng_data = [np.random.RandomState(0), np.random.RandomState(0)]

  # Sample a room.
  h_dists = []
  gt_dists = []
  for i in range(250):
    e = R.sample_env(rng_data)
    nodes = e.task.nodes

    # Initialize the agent.
    init_env_state = e.reset(rng_data)

    gt_dist_to_goal = [e.episode.dist_to_goal[0][j][s] 
                       for j, s in enumerate(e.episode.start_node_ids)]

    for j in range(args.navtask.task_params.batch_size):
      start_node_id = e.episode.start_node_ids[j]
      end_node_id =e.episode.goal_node_ids[0][j]
      h_dist = graph_utils.heuristic_fn_vec(
          nodes[[start_node_id],:], nodes[[end_node_id], :],
          n_ori=args.navtask.task_params.n_ori,
          step_size=args.navtask.task_params.step_size)[0][0]
      gt_dist = e.episode.dist_to_goal[0][j][start_node_id]
      h_dists.append(h_dist)
      gt_dists.append(gt_dist)

  h_dists = np.array(h_dists)
  gt_dists = np.array(gt_dists)
  e = R.sample_env([np.random.RandomState(0), np.random.RandomState(0)])
  input = e.get_common_data()
  orig_maps = input['orig_maps'][0,0,:,:,0]
  return h_dists, gt_dists, orig_maps

def plot_trajectory_first_person(dt, orig_maps, out_dir):
  out_dir = os.path.join(out_dir, FLAGS.config_name+_get_suffix_str(),
                         FLAGS.imset)
  fu.makedirs(out_dir)
  
  # Load the model so that we can render.
  plt.set_cmap('gray')
  samples_per_action = 8; wait_at_action = 0;
  
  Writer = animation.writers['mencoder']
  writer = Writer(fps=3*(samples_per_action+wait_at_action), 
                  metadata=dict(artist='anonymous'), bitrate=1800)
  
  args = sna.get_args_for_config(FLAGS.config_name + '+bench_'+FLAGS.imset)
  args.navtask.logdir = None
  navtask_ = copy.deepcopy(args.navtask)
  navtask_.camera_param.modalities = ['rgb']
  navtask_.task_params.modalities = ['rgb']
  sz = 512
  navtask_.camera_param.height = sz
  navtask_.camera_param.width = sz
  navtask_.task_params.img_height = sz
  navtask_.task_params.img_width = sz
  R = lambda: nav_env.get_multiplexer_class(navtask_, 0)
  R = R()
  b = R.buildings[0]
  
  f = [0 for _ in range(wait_at_action)] + \
      [float(_)/samples_per_action for _ in range(samples_per_action)];
  
  # Generate things for it to render.
  inds_to_do = []
  inds_to_do += [1, 4, 10] #1291, 1268, 1273, 1289, 1302, 1426, 1413, 1449, 1399, 1390]

  for i in inds_to_do:
    fig = plt.figure(figsize=(10,8))
    gs = GridSpec(3,4)
    gs.update(wspace=0.05, hspace=0.05, left=0.0, top=0.97, right=1.0, bottom=0.)
    ax = fig.add_subplot(gs[:,:-1])
    ax1 = fig.add_subplot(gs[0,-1])
    ax2 = fig.add_subplot(gs[1,-1])
    ax3 = fig.add_subplot(gs[2,-1])
    axes = [ax, ax1, ax2, ax3]
    # ax = fig.add_subplot(gs[:,:])
    # axes = [ax]
    for ax in axes:
      ax.set_axis_off()
    
    node_ids = dt['all_node_ids'][i, :, 0]*1
    # Prune so that last node is not repeated more than 3 times?
    if np.all(node_ids[-4:] == node_ids[-1]):
      while node_ids[-4] == node_ids[-1]:
        node_ids = node_ids[:-1]
    num_steps = np.minimum(FLAGS.num_steps, len(node_ids))

    xyt = b.to_actual_xyt_vec(b.task.nodes[node_ids])
    xyt_diff = xyt[1:,:] - xyt[:-1:,:]
    xyt_diff[:,2] = np.mod(xyt_diff[:,2], 4)
    ind = np.where(xyt_diff[:,2] == 3)[0]
    xyt_diff[ind, 2] = -1
    xyt_diff = np.expand_dims(xyt_diff, axis=1)
    to_cat = [xyt_diff*_ for _ in f]
    perturbs_all = np.concatenate(to_cat, axis=1)
    perturbs_all = np.concatenate([perturbs_all, np.zeros_like(perturbs_all[:,:,:1])], axis=2)
    node_ids_all = np.expand_dims(node_ids, axis=1)*1
    node_ids_all = np.concatenate([node_ids_all for _ in f], axis=1)
    node_ids_all = np.reshape(node_ids_all[:-1,:], -1)
    perturbs_all = np.reshape(perturbs_all, [-1, 4])
    imgs = b.render_nodes(b.task.nodes[node_ids_all,:], perturb=perturbs_all)
  
    # Get action at each node.
    actions = []
    _, action_to_nodes = b.get_feasible_actions(node_ids)
    for j in range(num_steps-1):
      action_to_node = action_to_nodes[j]
      node_to_action = dict(zip(action_to_node.values(), action_to_node.keys()))
      actions.append(node_to_action[node_ids[j+1]])
    
    def init_fn():
      return fig,
    gt_dist_to_goal = []

    # Render trajectories.
    def worker(j):
      # Plot the image.
      step_number = j/(samples_per_action + wait_at_action)
      img = imgs[j]; ax = axes[0]; ax.clear(); ax.set_axis_off();
      img = img.astype(np.uint8); ax.imshow(img);
      tt = ax.set_title(
          "First Person View\n" + 
          "Top corners show diagnostics (distance, agents' action) not input to agent.", 
          fontsize=12)
      plt.setp(tt, color='white')

      # Distance to goal.
      t = 'Dist to Goal:\n{:2d} steps'.format(int(dt['all_d_at_t'][i, step_number]))
      t = ax.text(0.01, 0.99, t,
          horizontalalignment='left',
          verticalalignment='top',
          fontsize=20, color='red',
          transform=ax.transAxes, alpha=1.0)
      t.set_bbox(dict(color='white', alpha=0.85, pad=-0.1))
      
      # Action to take.
      action_latex = ['$\odot$ ', '$\curvearrowright$ ', '$\curvearrowleft$ ', '$\Uparrow$ ']
      t = ax.text(0.99, 0.99, action_latex[actions[step_number]],
          horizontalalignment='right',
          verticalalignment='top',
          fontsize=40, color='green',
          transform=ax.transAxes, alpha=1.0)
      t.set_bbox(dict(color='white', alpha=0.85, pad=-0.1))


      # Plot the map top view.
      ax = axes[-1]
      if j == 0:
        # Plot the map
        locs = dt['all_locs'][i,:num_steps,:]
        goal_loc = dt['all_goal_locs'][i,:,:]
        xymin = np.minimum(np.min(goal_loc, axis=0), np.min(locs, axis=0))
        xymax = np.maximum(np.max(goal_loc, axis=0), np.max(locs, axis=0))
        xy1 = (xymax+xymin)/2. - 0.7*np.maximum(np.max(xymax-xymin), 24)
        xy2 = (xymax+xymin)/2. + 0.7*np.maximum(np.max(xymax-xymin), 24)

        ax.set_axis_on()
        ax.patch.set_facecolor((0.333, 0.333, 0.333))
        ax.set_xticks([]); ax.set_yticks([]);
        ax.imshow(orig_maps, origin='lower', vmin=-1.0, vmax=2.0)
        ax.plot(goal_loc[:,0], goal_loc[:,1], 'g*', markersize=12)

        locs = dt['all_locs'][i,:1,:]
        ax.plot(locs[:,0], locs[:,1], 'b.', markersize=12)

        ax.set_xlim([xy1[0], xy2[0]])
        ax.set_ylim([xy1[1], xy2[1]])

      locs = dt['all_locs'][i,step_number,:]
      locs = np.expand_dims(locs, axis=0)
      ax.plot(locs[:,0], locs[:,1], 'r.', alpha=1.0, linewidth=0, markersize=4)
      tt = ax.set_title('Trajectory in topview', fontsize=14)
      plt.setp(tt, color='white') 
      return fig,

    line_ani = animation.FuncAnimation(fig, worker,
                                       (num_steps-1)*(wait_at_action+samples_per_action),
                                       interval=500, blit=True, init_func=init_fn)
    tmp_file_name = 'tmp.mp4'
    line_ani.save(tmp_file_name, writer=writer, savefig_kwargs={'facecolor':'black'})
    out_file_name = os.path.join(out_dir, 'vis_{:04d}.mp4'.format(i))
    print out_file_name

    if fu.exists(out_file_name):
      gfile.Remove(out_file_name)
    gfile.Copy(tmp_file_name, out_file_name)
    gfile.Remove(tmp_file_name)
    plt.close(fig)

def plot_trajectory(dt, hardness, orig_maps, out_dir):
  out_dir = os.path.join(out_dir, FLAGS.config_name+_get_suffix_str(),
                         FLAGS.imset)
  fu.makedirs(out_dir)
  out_file = os.path.join(out_dir, 'all_locs_at_t.pkl')
  dt['hardness'] = hardness
  utils.save_variables(out_file, dt.values(), dt.keys(), overwrite=True)
  
  #Plot trajectories onto the maps
  plt.set_cmap('gray')
  for i in range(4000):
    goal_loc = dt['all_goal_locs'][i, :, :]
    locs = np.concatenate((dt['all_locs'][i,:,:], 
                           dt['all_locs'][i,:,:]), axis=0)
    xymin = np.minimum(np.min(goal_loc, axis=0), np.min(locs, axis=0))
    xymax = np.maximum(np.max(goal_loc, axis=0), np.max(locs, axis=0))
    xy1 = (xymax+xymin)/2. - 1.*np.maximum(np.max(xymax-xymin), 24)
    xy2 = (xymax+xymin)/2. + 1.*np.maximum(np.max(xymax-xymin), 24)

    fig, ax = utils.tight_imshow_figure(plt, figsize=(6,6))
    ax.set_axis_on()
    ax.patch.set_facecolor((0.333, 0.333, 0.333))
    ax.set_xticks([])
    ax.set_yticks([])

    all_locs = dt['all_locs'][i,:,:]*1
    uniq = np.where(np.any(all_locs[1:,:] != all_locs[:-1,:], axis=1))[0]+1
    uniq = np.sort(uniq).tolist()
    uniq.insert(0,0)
    uniq = np.array(uniq)
    all_locs = all_locs[uniq, :]

    ax.plot(dt['all_locs'][i, 0, 0], 
            dt['all_locs'][i, 0, 1], 'b.', markersize=24)
    ax.plot(dt['all_goal_locs'][i, 0, 0], 
            dt['all_goal_locs'][i, 0, 1], 'g*', markersize=19)
    ax.plot(all_locs[:,0], all_locs[:,1], 'r', alpha=0.4, linewidth=2)
    ax.scatter(all_locs[:,0], all_locs[:,1],
               c=5+np.arange(all_locs.shape[0])*1./all_locs.shape[0], 
               cmap='Reds', s=30, linewidth=0)
    ax.imshow(orig_maps, origin='lower', vmin=-1.0, vmax=2.0, aspect='equal')
    ax.set_xlim([xy1[0], xy2[0]])
    ax.set_ylim([xy1[1], xy2[1]])
    
    file_name = os.path.join(out_dir, 'trajectory_{:04d}.png'.format(i))
    print file_name
    with fu.fopen(file_name, 'w') as f: 
      plt.savefig(f)
    plt.close(fig)
  

def main(_):
  a = _load_trajectory()
  h_dists, gt_dists, orig_maps = _compute_hardness()
  hardness = 1.-h_dists*1./ gt_dists
  
  if FLAGS.top_view:
    plot_trajectory(a, hardness, orig_maps, out_dir=FLAGS.out_dir)

  if FLAGS.first_person:
    plot_trajectory_first_person(a, orig_maps, out_dir=FLAGS.out_dir)
  
if __name__ == '__main__':
  app.run()
