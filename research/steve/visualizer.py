from __future__ import print_function
from builtins import range
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
# import moviepy.editor as mpy
import time, os, traceback, multiprocessing, portalocker, sys

import envwrap
import util
import valuerl, worldmodel
from config import config

MODEL_NAME = config["name"]
LOG_PATH = util.create_directory("output/" + config["env"] + "/" + MODEL_NAME + "/" + config["log_path"]) + "/" + MODEL_NAME
LOAD_PATH =    util.create_directory("output/" + config["env"] + "/" + MODEL_NAME + "/" + config["save_model_path"])
OBS_DIM =   np.prod(config["obs_dims"])
HIDDEN_DIM = config["hidden_dim"]
ACTION_DIM = config["action_dim"]
MAX_FRAMES = config["max_frames"]
REWARD_SCALE = config["reward_scale"]
DISCOUNT = config["discount"]
ALGO = config["policy_config"]["algo"]
AGENT_BATCH_SIZE = config["agent_config"]["batch_size"]
EVALUATOR_BATCH_SIZE = config["evaluator_config"]["batch_size"]
RELOAD_EVERY_N = config["agent_config"]["reload_every_n"]
FRAMES_BEFORE_LEARNING = config["policy_config"]["frames_before_learning"]
FRAMES_PER_UPDATE = config["policy_config"]["frames_per_update"]
LEARNER_EPOCH_N = config["policy_config"]["epoch_n"]
SYNC_UPDATES = config["policy_config"]["frames_per_update"] >= 0
POLICY_BAYESIAN_CONFIG = config["policy_config"]["bayesian"]
AUX_CONFIG = config["aux_config"]
DDPG_EXPLORE_CHANCE = config["policy_config"]["explore_chance"] if ALGO == "ddpg" else 0.
MODEL_AUGMENTED = config["model_config"] is not False
if MODEL_AUGMENTED: MODEL_BAYESIAN_CONFIG = config["model_config"]["bayesian"]

FILENAME = sys.argv[3]

if __name__ == '__main__':
    oprl = valuerl.ValueRL(MODEL_NAME, ALGO, OBS_DIM, ACTION_DIM, HIDDEN_DIM, REWARD_SCALE, DISCOUNT, POLICY_BAYESIAN_CONFIG, AUX_CONFIG, DDPG_EXPLORE_CHANCE)

    obs_loader = tf.placeholder(tf.float32, [1, OBS_DIM])
    policy_actions, _ = oprl.build_evalution_graph(obs_loader, mode="exploit")

    if MODEL_AUGMENTED:
        next_obs_loader = tf.placeholder(tf.float32, [1, OBS_DIM])
        reward_loader = tf.placeholder(tf.float32, [1])
        done_loader = tf.placeholder(tf.float32, [1])
        worldmodel = worldmodel.DeterministicWorldModel(MODEL_NAME, OBS_DIM, ACTION_DIM, HIDDEN_DIM, REWARD_SCALE, DISCOUNT, MODEL_BAYESIAN_CONFIG)
        _, _, _, _, _, confidence, _ = oprl.build_Q_expansion_graph(next_obs_loader, reward_loader, done_loader, worldmodel, rollout_len=3, model_ensembling=True)

    sess = tf.Session()
    sess.run(tf.global_variables_initializer())

    oprl.load(sess, FILENAME)
    if MODEL_AUGMENTED: worldmodel.load(sess, FILENAME)

    env = envwrap.get_env(config["env"])

    hist = np.zeros([4, 10])
    for _ in range(10):
        ts = 0
        rgb_frames = []
        obs, reward, done, reset = env.reset(), 0, False, False
        while not reset:
            # env.internal_env.render()
            # rgb_frames.append(env.internal_env.render(mode='rgb_array'))
            # action = env.action_space.sample()
            all_actions = sess.run(policy_actions, feed_dict={obs_loader: np.array([obs])})
            all_actions = np.clip(all_actions, -1., 1.)
            action = all_actions[0]
            obs, _reward, done, reset = env.step(action)

            if MODEL_AUGMENTED:
                _confidences = sess.run(confidence, feed_dict={next_obs_loader: np.expand_dims(obs,0),
                                                               reward_loader: np.expand_dims(_reward,0),
                                                               done_loader: np.expand_dims(done,0)})
                # print "%.02f %.02f %.02f %.02f" % tuple(_confidences[0,0])
                for h in range(4):
                    bucket = int((_confidences[0,0,h]-1e-5)*10)
                    hist[h,bucket] += 1

            reward += _reward
            ts += 1
            # print ts, _reward, reward
        print(ts, reward)
    hist /= np.sum(hist, axis=1, keepdims=True)
    for row in reversed(hist.T): print(' '.join(["%.02f"] * 4) % tuple(row))

    #clip = mpy.ImageSequenceClip(rgb_frames, fps=100)
    #clip.write_videofile(FILENAME + "/movie.mp4")


