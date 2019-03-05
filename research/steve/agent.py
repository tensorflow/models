from __future__ import print_function
from builtins import zip
from builtins import range
from builtins import object
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

import numpy as np
import tensorflow as tf
import time, os, traceback, multiprocessing, portalocker

import envwrap
import valuerl
import util
from config import config


def run_env(pipe):
  env = envwrap.get_env(config["env"]["name"])
  reset = True
  while True:
    if reset is True: pipe.send(env.reset())
    action = pipe.recv()
    obs, reward, done, reset = env.step(action)
    pipe.send((obs, reward, done, reset))

class AgentManager(object):
  """
  Interact with the environment according to the learned policy,
  """
  def __init__(self, proc_num, evaluation, policy_lock, batch_size, config):
    self.evaluation = evaluation
    self.policy_lock = policy_lock
    self.batch_size = batch_size
    self.config = config

    self.log_path =  util.create_directory("%s/%s/%s/%s" % (config["output_root"], config["env"]["name"], config["name"], config["log_path"])) + "/%s" % config["name"]
    self.load_path = util.create_directory("%s/%s/%s/%s" % (config["output_root"], config["env"]["name"], config["name"], config["save_model_path"]))

    ## placeholders for intermediate states (basis for rollout)
    self.obs_loader = tf.placeholder(tf.float32, [self.batch_size, np.prod(self.config["env"]["obs_dims"])])

    ## build model
    self.valuerl =  valuerl.ValueRL(self.config["name"], self.config["env"], self.config["policy_config"])
    self.policy_actions = self.valuerl.build_evalution_graph(self.obs_loader, mode="exploit" if self.evaluation else "explore")

    # interactors
    self.agent_pipes, self.agent_child_pipes = list(zip(*[multiprocessing.Pipe() for _ in range(self.batch_size)]))
    self.agents = [multiprocessing.Process(target=run_env, args=(self.agent_child_pipes[i],)) for i in range(self.batch_size)]
    for agent in self.agents: agent.start()
    self.obs = [pipe.recv() for pipe in self.agent_pipes]
    self.total_rewards = [0. for _ in self.agent_pipes]
    self.loaded_policy = False

    self.sess = tf.Session()
    self.sess.run(tf.global_variables_initializer())

    self.rollout_i = 0
    self.proc_num = proc_num
    self.epoch = -1
    self.frame_total = 0
    self.hours = 0.

    self.first = True

  def get_action(self, obs):
    if self.loaded_policy:
      all_actions = self.sess.run(self.policy_actions, feed_dict={self.obs_loader: obs})
      all_actions = np.clip(all_actions, -1., 1.)
      return all_actions[:self.batch_size]
    else:
      return [self.get_random_action() for _ in range(obs.shape[0])]

  def get_random_action(self, *args, **kwargs):
    return np.random.random(self.config["env"]["action_dim"]) * 2 - 1

  def step(self):
    actions = self.get_action(np.stack(self.obs))
    self.first = False
    [pipe.send(action) for pipe, action in zip(self.agent_pipes, actions)]
    next_obs, rewards, dones, resets = list(zip(*[pipe.recv() for pipe in self.agent_pipes]))

    frames = list(zip(self.obs, next_obs, actions, rewards, dones))

    self.obs = [o if resets[i] is False else self.agent_pipes[i].recv() for i, o in enumerate(next_obs)]

    for i, (t,r,reset) in enumerate(zip(self.total_rewards, rewards, resets)):
      if reset:
        self.total_rewards[i] = 0.
        if self.evaluation and self.loaded_policy:
          with portalocker.Lock(self.log_path+'.greedy.csv', mode="a") as f: f.write("%2f,%d,%d,%2f\n" % (self.hours, self.epoch, self.frame_total, t+r))

      else:
        self.total_rewards[i] = t + r

    if self.evaluation and np.any(resets): self.reload()

    self.rollout_i += 1
    return frames

  def reload(self):
    if not os.path.exists("%s/%s.params.index" % (self.load_path ,self.valuerl.saveid)): return False
    with self.policy_lock:
      self.valuerl.load(self.sess, self.load_path)
      self.epoch, self.frame_total, self.hours = self.sess.run([self.valuerl.epoch_n, self.valuerl.frame_n, self.valuerl.hours])
    self.loaded_policy = True
    self.first = True
    return True

def main(proc_num, evaluation, policy_replay_frame_queue, model_replay_frame_queue, policy_lock, config):
  try:
    np.random.seed((proc_num * int(time.time())) % (2 ** 32 - 1))
    agentmanager = AgentManager(proc_num, evaluation, policy_lock, config["evaluator_config"]["batch_size"] if evaluation else config["agent_config"]["batch_size"], config)
    frame_i = 0
    while True:
      new_frames = agentmanager.step()
      if not evaluation:
        policy_replay_frame_queue.put(new_frames)
        if model_replay_frame_queue is not None: model_replay_frame_queue.put(new_frames)
        if frame_i % config["agent_config"]["reload_every_n"] == 0: agentmanager.reload()
        frame_i += len(new_frames)

  except Exception as e:
    print('Caught exception in agent process %d' % proc_num)
    traceback.print_exc()
    print()
    try:
      for i in agentmanager.agents: i.join()
    except:
      pass
    raise e
