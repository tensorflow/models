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

# pylint: disable=line-too-long
# pyformat: disable
"""Train and eval for supervised navigation training.

For training:
python train_supervised_active_vision.py \
  --mode='train' \
  --logdir=$logdir/checkin_log_det/ \
  --modality_types='det' \
  --batch_size=8 \
  --train_iters=200000 \
  --lstm_cell_size=2048 \
  --policy_fc_size=2048 \
  --sequence_length=20 \
  --max_eval_episode_length=100 \
  --test_iters=194 \
  --gin_config=envs/configs/active_vision_config.gin \
  --gin_params='ActiveVisionDatasetEnv.dataset_root="$datadir"' \
  --logtostderr

For testing:
python train_supervised_active_vision.py
  --mode='eval' \
  --logdir=$logdir/checkin_log_det/ \
  --modality_types='det' \
  --batch_size=8 \
  --train_iters=200000 \
  --lstm_cell_size=2048 \
  --policy_fc_size=2048 \
  --sequence_length=20 \
  --max_eval_episode_length=100 \
  --test_iters=194 \
  --gin_config=envs/configs/active_vision_config.gin \
  --gin_params='ActiveVisionDatasetEnv.dataset_root="$datadir"' \
  --logtostderr
"""

import collections
import os
import time
from absl import app
from absl import flags
from absl import logging
import networkx as nx
import numpy as np
import tensorflow as tf
import gin
import embedders
import policies
import tasks
from envs import active_vision_dataset_env
from envs import task_env

slim = tf.contrib.slim

flags.DEFINE_string('logdir', '',
                    'Path to a directory to write summaries and checkpoints')
# Parameters controlling the training setup. In general one would not need to
# modify them.
flags.DEFINE_string('master', 'local',
                    'BNS name of the TensorFlow master, or local.')
flags.DEFINE_integer('task_id', 0,
                     'Task id of the replica running the training.')
flags.DEFINE_integer('ps_tasks', 0,
                     'Number of tasks in the ps job. If 0 no ps job is used.')

flags.DEFINE_integer('decay_steps', 1000,
                     'Number of steps for exponential decay.')
flags.DEFINE_float('learning_rate', 0.0001, 'Learning rate.')
flags.DEFINE_integer('batch_size', 8, 'Batch size.')
flags.DEFINE_integer('sequence_length', 20, 'sequence length')
flags.DEFINE_integer('train_iters', 200000, 'number of training iterations.')
flags.DEFINE_integer('save_summaries_secs', 300,
                     'number of seconds between saving summaries')
flags.DEFINE_integer('save_interval_secs', 300,
                     'numer of seconds between saving variables')
flags.DEFINE_integer('log_every_n_steps', 20, 'number of steps between logging')
flags.DEFINE_string('modality_types', '',
                    'modality names in _ separated format')
flags.DEFINE_string('conv_window_sizes', '8_4_3',
                    'conv window size in separated by _')
flags.DEFINE_string('conv_strides', '4_2_1', '')
flags.DEFINE_string('conv_channels', '8_16_16', '')
flags.DEFINE_integer('embedding_fc_size', 128,
                     'size of embedding for each modality')
flags.DEFINE_integer('obs_resolution', 64,
                     'resolution of the input observations')
flags.DEFINE_integer('lstm_cell_size', 2048, 'size of lstm cell size')
flags.DEFINE_integer('policy_fc_size', 2048,
                     'size of fully connected layers for policy part')
flags.DEFINE_float('weight_decay', 0.0002, 'weight decay')
flags.DEFINE_integer('goal_category_count', 5, 'number of goal categories')
flags.DEFINE_integer('action_size', 7, 'number of possible actions')
flags.DEFINE_integer('max_eval_episode_length', 100,
                     'maximum sequence length for evaluation.')
flags.DEFINE_enum('mode', 'train', ['train', 'eval'],
                  'indicates whether it is in training or evaluation')
flags.DEFINE_integer('test_iters', 194,
                     'number of iterations that the eval needs to be run')
flags.DEFINE_multi_string('gin_config', [],
                          'List of paths to a gin config files for the env.')
flags.DEFINE_multi_string('gin_params', [],
                          'Newline separated list of Gin parameter bindings.')
flags.DEFINE_string(
    'resnet50_path', './resnet_v2_50_checkpoint/resnet_v2_50.ckpt', 'path to resnet50'
    'checkpoint')
flags.DEFINE_bool('freeze_resnet_weights', True, '')
flags.DEFINE_string(
    'eval_init_points_file_name', '',
    'Name of the file that containts the initial locations and'
    'worlds for each evalution point')

FLAGS = flags.FLAGS
TRAIN_WORLDS = [
    'Home_001_1', 'Home_001_2', 'Home_002_1', 'Home_003_1', 'Home_003_2',
    'Home_004_1', 'Home_004_2', 'Home_005_1', 'Home_005_2', 'Home_006_1',
    'Home_010_1'
]

TEST_WORLDS = ['Home_011_1', 'Home_013_1', 'Home_016_1']


def create_modality_types():
  """Parses the modality_types and returns a list of task_env.ModalityType."""
  if not FLAGS.modality_types:
    raise ValueError('there needs to be at least one modality type')
  modality_types = FLAGS.modality_types.split('_')
  for x in modality_types:
    if x not in ['image', 'sseg', 'det', 'depth']:
      raise ValueError('invalid modality type: {}'.format(x))

  conversion_dict = {
      'image': task_env.ModalityTypes.IMAGE,
      'sseg': task_env.ModalityTypes.SEMANTIC_SEGMENTATION,
      'depth': task_env.ModalityTypes.DEPTH,
      'det': task_env.ModalityTypes.OBJECT_DETECTION,
  }
  return [conversion_dict[k] for k in modality_types]


def create_task_io_config(
    modality_types,
    goal_category_count,
    action_size,
    sequence_length,
):
  """Generates task io config."""
  shape_prefix = [sequence_length, FLAGS.obs_resolution, FLAGS.obs_resolution]
  shapes = {
      task_env.ModalityTypes.IMAGE: [sequence_length, 224, 224, 3],
      task_env.ModalityTypes.DEPTH: shape_prefix + [
          2,
      ],
      task_env.ModalityTypes.SEMANTIC_SEGMENTATION: shape_prefix + [
          1,
      ],
      task_env.ModalityTypes.OBJECT_DETECTION: shape_prefix + [
          90,
      ]
  }
  types = {k: tf.float32 for k in shapes}
  types[task_env.ModalityTypes.IMAGE] = tf.uint8
  inputs = collections.OrderedDict(
      [[mtype, (types[mtype], shapes[mtype])] for mtype in modality_types])
  inputs[task_env.ModalityTypes.GOAL] = (tf.float32,
                                         [sequence_length, goal_category_count])
  inputs[task_env.ModalityTypes.PREV_ACTION] = (tf.float32, [
      sequence_length, action_size + 1
  ])
  print inputs
  return tasks.UnrolledTaskIOConfig(
      inputs=inputs,
      output=(tf.float32, [sequence_length, action_size]),
      query=None)


def map_to_embedder(modality_type):
  """Maps modality_type to its corresponding embedder."""
  if modality_type == task_env.ModalityTypes.PREV_ACTION:
    return None
  if modality_type == task_env.ModalityTypes.GOAL:
    return embedders.IdentityEmbedder()
  if modality_type == task_env.ModalityTypes.IMAGE:
    return embedders.ResNet50Embedder()
  conv_window_sizes = [int(x) for x in FLAGS.conv_window_sizes.split('_')]
  conv_channels = [int(x) for x in FLAGS.conv_channels.split('_')]
  conv_strides = [int(x) for x in FLAGS.conv_strides.split('_')]
  params = tf.contrib.training.HParams(
      to_one_hot=modality_type == task_env.ModalityTypes.SEMANTIC_SEGMENTATION,
      one_hot_length=10,
      conv_sizes=conv_window_sizes,
      conv_strides=conv_strides,
      conv_channels=conv_channels,
      embedding_size=FLAGS.embedding_fc_size,
      weight_decay_rate=FLAGS.weight_decay,
  )
  return embedders.SmallNetworkEmbedder(params)


def create_train_and_init_ops(policy, task):
  """Creates training ops given the arguments.

  Args:
    policy: the policy for the task.
    task: the task instance.

  Returns:
    train_op: the op that needs to be runned at each step.
    summaries_op: the summary op that is executed.
    init_fn: the op that initializes the variables if there is no previous
      checkpoint. If Resnet50 is not used in the model it is None, otherwise
      it reads the weights from FLAGS.resnet50_path and sets the init_fn
      to the op that initializes the ResNet50 with the pre-trained weights.
  """
  assert isinstance(task, tasks.GotoStaticXNoExplorationTask)
  assert isinstance(policy, policies.Policy)

  inputs, _, gt_outputs, masks = task.tf_episode_batch(FLAGS.batch_size)
  outputs, _ = policy.build(inputs, None)
  loss = task.target_loss(gt_outputs, outputs, masks)

  init_fn = None

  # If resnet is added to the graph, init_fn should initialize resnet weights
  # if there is no previous checkpoint.
  variables_assign_dict = {}
  vars_list = []
  for v in slim.get_model_variables():
    if v.name.find('resnet') >= 0:
      if not FLAGS.freeze_resnet_weights:
        vars_list.append(v)
      variables_assign_dict[v.name[v.name.find('resnet'):-2]] = v
    else:
      vars_list.append(v)
  
  global_step = tf.train.get_or_create_global_step()
  learning_rate = tf.train.exponential_decay(
      FLAGS.learning_rate,
      global_step,
      decay_steps=FLAGS.decay_steps,
      decay_rate=0.98,
      staircase=True)
  optimizer = tf.train.AdamOptimizer(learning_rate)
  train_op = slim.learning.create_train_op(
      loss,
      optimizer,
      global_step=global_step,
      variables_to_train=vars_list,
  )

  if variables_assign_dict:
    init_fn = slim.assign_from_checkpoint_fn(
        FLAGS.resnet50_path,
        variables_assign_dict,
        ignore_missing_vars=False)
  scalar_summaries = {}
  scalar_summaries['LR'] = learning_rate
  scalar_summaries['loss'] = loss

  for name, summary in scalar_summaries.iteritems():
    tf.summary.scalar(name, summary)
 
  return train_op, init_fn


def create_eval_ops(policy, config, possible_targets):
  """Creates the necessary ops for evaluation."""
  inputs_feed = collections.OrderedDict([[
      mtype,
      tf.placeholder(config.inputs[mtype].type,
                     [1] + config.inputs[mtype].shape)
  ] for mtype in config.inputs])
  inputs_feed[task_env.ModalityTypes.PREV_ACTION] = tf.placeholder(
      tf.float32, [1, 1] + [
          config.output.shape[-1] + 1,
      ])
  prev_state_feed = [
      tf.placeholder(
          tf.float32, [1, FLAGS.lstm_cell_size], name='prev_state_{}'.format(i))
      for i in range(2)
  ]
  policy_outputs = policy.build(inputs_feed, prev_state_feed)
  summary_feed = {}
  for c in possible_targets + ['mean']:
    summary_feed[c] = tf.placeholder(
        tf.float32, [], name='eval_in_range_{}_input'.format(c))
    tf.summary.scalar('eval_in_range_{}'.format(c), summary_feed[c])

  return inputs_feed, prev_state_feed, policy_outputs, (tf.summary.merge_all(),
                                                        summary_feed)


def unroll_policy_for_eval(
    sess,
    env,
    inputs_feed,
    prev_state_feed,
    policy_outputs,
    number_of_steps,
    output_folder,
):
  """unrolls the policy for testing.

  Args:
    sess: tf.Session
    env: The environment.
    inputs_feed: dictionary of placeholder for the input modalities.
    prev_state_feed: placeholder for the input to the prev_state of the model.
    policy_outputs: tensor that contains outputs of the policy.
    number_of_steps: maximum number of unrolling steps.
    output_folder: output_folder where the function writes a dictionary of
      detailed information about the path. The dictionary keys are 'states' and
      'distance'. The value for 'states' is the list of states that the agent
      goes along the path. The value for 'distance' contains the length of
      shortest path to the goal at each step.

  Returns:
    states: list of states along the path.
    distance: list of distances along the path.
  """
  prev_state = [
      np.zeros((1, FLAGS.lstm_cell_size), dtype=np.float32) for _ in range(2)
  ]
  prev_action = np.zeros((1, 1, FLAGS.action_size + 1), dtype=np.float32)
  obs = env.reset()
  distances_to_goal = []
  states = []
  unique_id = '{}_{}'.format(env.cur_image_id(), env.goal_string)
  for _ in range(number_of_steps):
    distances_to_goal.append(
        np.min([
            len(
                nx.shortest_path(env.graph, env.pose_to_vertex(env.state()),
                                 env.pose_to_vertex(target_view)))
            for target_view in env.targets()
        ]))
    states.append(env.state())
    feed_dict = {inputs_feed[mtype]: [[obs[mtype]]] for mtype in inputs_feed}
    feed_dict[prev_state_feed[0]] = prev_state[0]
    feed_dict[prev_state_feed[1]] = prev_state[1]
    action_values, prev_state = sess.run(policy_outputs, feed_dict=feed_dict)
    chosen_action = np.argmax(action_values[0])
    obs, _, done, info = env.step(np.int32(chosen_action))
    prev_action[0][0][chosen_action] = 1.
    prev_action[0][0][-1] = float(info['success'])
    # If the agent chooses action stop or the number of steps exceeeded
    # env._episode_length.
    if done:
      break

  # logging.info('distance = %d, id = %s, #steps = %d', distances_to_goal[-1],
  output_path = os.path.join(output_folder, unique_id + '.npy')
  with tf.gfile.Open(output_path, 'w') as f:
    print 'saving path information to {}'.format(output_path)
    np.save(f, {'states': states, 'distance': distances_to_goal})
  return states, distances_to_goal


def init(sequence_length, eval_init_points_file_name, worlds):
  """Initializes the common operations between train and test."""
  modality_types = create_modality_types()
  logging.info('modality types: %r', modality_types)
  # negative reward_goal_range prevents the env from terminating early when the
  # agent is close to the goal. The policy should keep the agent until the end
  # of the 100 steps either through chosing stop action or oscilating around
  # the target.

  env = active_vision_dataset_env.ActiveVisionDatasetEnv(
      modality_types=modality_types +
      [task_env.ModalityTypes.GOAL, task_env.ModalityTypes.PREV_ACTION],
      reward_goal_range=-1,
      eval_init_points_file_name=eval_init_points_file_name,
      worlds=worlds,
      output_size=FLAGS.obs_resolution,
  )

  config = create_task_io_config(
      modality_types=modality_types,
      goal_category_count=FLAGS.goal_category_count,
      action_size=FLAGS.action_size,
      sequence_length=sequence_length,
  )
  task = tasks.GotoStaticXNoExplorationTask(env=env, config=config)
  embedders_dict = {mtype: map_to_embedder(mtype) for mtype in config.inputs}
  policy_params = tf.contrib.training.HParams(
      lstm_state_size=FLAGS.lstm_cell_size,
      fc_channels=FLAGS.policy_fc_size,
      weight_decay=FLAGS.weight_decay,
      target_embedding_size=FLAGS.embedding_fc_size,
  )
  policy = policies.LSTMPolicy(
      modality_names=config.inputs.keys(),
      embedders_dict=embedders_dict,
      action_size=FLAGS.action_size,
      params=policy_params,
      max_episode_length=sequence_length)
  return env, config, task, policy


def test():
  """Contains all the operations for testing policies."""
  env, config, _, policy = init(1, 'all_init_configs', TEST_WORLDS)
  inputs_feed, prev_state_feed, policy_outputs, summary_op = create_eval_ops(
      policy, config, env.possible_targets)
  sv = tf.train.Supervisor(logdir=FLAGS.logdir)
  prev_checkpoint = None
  with sv.managed_session(
      start_standard_services=False,
      config=tf.ConfigProto(allow_soft_placement=True)) as sess:
    while not sv.should_stop():
      while True:
        new_checkpoint = tf.train.latest_checkpoint(FLAGS.logdir)
        print 'new_checkpoint ', new_checkpoint
        if not new_checkpoint:
          time.sleep(1)
          continue
        if prev_checkpoint is None:
          prev_checkpoint = new_checkpoint
          break
        if prev_checkpoint != new_checkpoint:
          prev_checkpoint = new_checkpoint
          break
        else:  # if prev_checkpoint == new_checkpoint, we have to wait more.
          time.sleep(1)

      checkpoint_step = int(new_checkpoint[new_checkpoint.rfind('-') + 1:])
      sv.saver.restore(sess, new_checkpoint)
      print '--------------------'
      print 'evaluating checkpoint {}'.format(new_checkpoint)
      folder_path = os.path.join(FLAGS.logdir, 'evals', str(checkpoint_step))
      if not tf.gfile.Exists(folder_path):
        tf.gfile.MakeDirs(folder_path)
      eval_stats = {c: [] for c in env.possible_targets}
      for test_iter in range(FLAGS.test_iters):
        print 'evaluating {} of {}'.format(test_iter, FLAGS.test_iters)
        _, distance_to_goal = unroll_policy_for_eval(
            sess,
            env,
            inputs_feed,
            prev_state_feed,
            policy_outputs,
            FLAGS.max_eval_episode_length,
            folder_path,
        )
        print 'goal = {}'.format(env.goal_string)
        eval_stats[env.goal_string].append(float(distance_to_goal[-1] <= 7))
      eval_stats = {k: np.mean(v) for k, v in eval_stats.iteritems()}
      eval_stats['mean'] = np.mean(eval_stats.values())
      print eval_stats
      feed_dict = {summary_op[1][c]: eval_stats[c] for c in eval_stats}
      summary_str = sess.run(summary_op[0], feed_dict=feed_dict)
      writer = sv.summary_writer
      writer.add_summary(summary_str, checkpoint_step)
      writer.flush()


def train():
  _, _, task, policy = init(FLAGS.sequence_length, None, TRAIN_WORLDS)
  print(FLAGS.save_summaries_secs)
  print(FLAGS.save_interval_secs)
  print(FLAGS.logdir)

  with tf.device(
      tf.train.replica_device_setter(ps_tasks=FLAGS.ps_tasks, merge_devices=True)):
    train_op, init_fn = create_train_and_init_ops(policy=policy, task=task)
    print(FLAGS.logdir)
    slim.learning.train(
        train_op=train_op,
        init_fn=init_fn,
        logdir=FLAGS.logdir,
        is_chief=FLAGS.task_id == 0,
        number_of_steps=FLAGS.train_iters,
        save_summaries_secs=FLAGS.save_summaries_secs,
        save_interval_secs=FLAGS.save_interval_secs,
        session_config=tf.ConfigProto(allow_soft_placement=True),
    )


def main(_):
  gin.parse_config_files_and_bindings(FLAGS.gin_config, FLAGS.gin_params)
  if FLAGS.mode == 'train':
    train()
  else:
    test()


if __name__ == '__main__':
  app.run(main)
