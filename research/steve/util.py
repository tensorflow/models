from __future__ import division
from future import standard_library
standard_library.install_aliases()
from builtins import str
from builtins import range
from past.utils import old_div
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
import os, random, gc, math, re
import multiprocessing, types, shutil, pickle, json
from collections import defaultdict, MutableMapping

def tanh_sample_info(mu, logsigma, stop_action_gradient=False, n_samples=1):
    if n_samples > 1:
      mu = tf.expand_dims(mu, 2)
      logsigma = tf.expand_dims(logsigma, 2)
      sample_shape = tf.concat([tf.shape(mu), n_samples], 0)
    else:
      sample_shape = tf.shape(mu)

    flat_act = mu + tf.random_normal(sample_shape) * tf.exp(logsigma)
    if stop_action_gradient: flat_act = tf.stop_gradient(flat_act)
    normalized_dist_t = (flat_act - mu) * tf.exp(-logsigma)  # ... x D
    quadratic = - 0.5 * tf.reduce_sum(normalized_dist_t ** 2, axis=-1) # ... x (None)
    log_z = tf.reduce_sum(logsigma, axis=-1)  # ... x (None)
    D_t = tf.cast(tf.shape(mu)[-1], tf.float32)
    log_z += 0.5 * D_t * np.log(2 * np.pi)
    flat_ll = quadratic - log_z

    scaled_act = tf.tanh(flat_act)
    corr = tf.reduce_sum(tf.log(1. - tf.square(scaled_act) + 1e-6), axis=-1)
    scaled_ll = flat_ll - corr
    return flat_act, flat_ll, scaled_act, scaled_ll

def tf_cheating_contcartpole(state, action):
    gravity = 9.8
    masscart = 1.0
    masspole = 0.1
    total_mass = (masspole + masscart)
    length = 0.5 # actually half the pole's length
    polemass_length = (masspole * length)
    force_mag = 10.0
    tau = 0.02  # seconds between state updates

    # Angle at which to fail the episode
    theta_threshold_radians = 12 * 2 * math.pi / 360
    x_threshold = 2.4

    x, x_dot, theta, theta_dot = tf.split(state, 4, axis=-1)
    done =  tf.logical_or(x < -x_threshold,
                          tf.logical_or(x > x_threshold,
                          tf.logical_or(theta < -theta_threshold_radians,
                                        theta > theta_threshold_radians)))

    force = force_mag * action
    costheta = tf.cos(theta)
    sintheta = tf.sin(theta)
    temp = old_div((force + polemass_length * theta_dot * theta_dot * sintheta), total_mass)
    thetaacc = old_div((gravity * sintheta - costheta* temp), (length * (old_div(4.0,3.0) - masspole * costheta * costheta / total_mass)))
    xacc  = temp - polemass_length * thetaacc * costheta / total_mass
    x  = x + tau * x_dot
    x_dot = x_dot + tau * xacc
    theta = theta + tau * theta_dot
    theta_dot = theta_dot + tau * thetaacc
    state = tf.concat([x,x_dot,theta,theta_dot], -1)
    done = tf.squeeze(tf.cast(done, tf.float32), -1)
    reward = 1.0 - done
    done *= 0.
    return state, reward, done

def create_directory(dir):
    dir_chunks = dir.split("/")
    for i in range(len(dir_chunks)):
        partial_dir = "/".join(dir_chunks[:i+1])
        try:
            os.makedirs(partial_dir)
        except OSError:
            pass
    return dir

def create_and_wipe_directory(dir):
    shutil.rmtree(create_directory(dir))
    create_directory(dir)

def wipe_file(fname):
    with open(fname, "w") as f:
        f.write("")
    return fname

def get_largest_epoch_in_dir(dir, saveid):
    reg_matches = [re.findall('\d+_%s'%saveid,filename) for filename in os.listdir(dir)]
    epoch_labels = [int(regmatch[0].split("_")[0]) for regmatch in reg_matches if regmatch]
    if len(epoch_labels) == 0: return False
    return max(epoch_labels)

def wipe_all_but_largest_epoch_in_dir(dir, saveid):
    largest = get_largest_epoch_in_dir(dir, saveid)
    reg_matches = [(filename, re.findall('\d+_%s'%saveid,filename)) for filename in os.listdir(dir)]
    for filename, regmatch in reg_matches:
        if regmatch and int(regmatch[0].split("_")[0]) != largest:
            os.remove(os.path.join(dir,filename))

class ConfigDict(dict):
    def __init__(self, loc=None, ghost=False):
        self._dict = defaultdict(lambda :False)
        self.ghost = ghost
        if loc:
            with open(loc) as f: raw = json.load(f)
            if "inherits" in raw and raw["inherits"]:
                for dep_loc in raw["inherits"]:
                    self.update(ConfigDict(dep_loc))
            if "updates" in raw and raw["updates"]:
                self.update(raw["updates"], include_all=True)

    def __getitem__(self, key):
        return self._dict[key]

    def __setitem__(self, key, value):
        self._dict[key] = value

    def __str__(self):
        return str(dict(self._dict))

    def __repr__(self):
        return str(dict(self._dict))

    def __iter__(self):
        return self._dict.__iter__()

    def __bool__(self):
        return bool(self._dict)

    def __nonzero__(self):
        return bool(self._dict)

    def update(self, dictlike, include_all=False):
        for key in dictlike:
            value = dictlike[key]
            if isinstance(value, dict):
                if key[0] == "*": # this means only override, do not set
                    key = key[1:]
                    ghost = True
                else:
                    ghost = False
                if not include_all and isinstance(value, ConfigDict) and key not in self._dict and value.ghost: continue
                if key not in self._dict: self._dict[key] = ConfigDict(ghost=ghost)
                self._dict[key].update(value)
            else:
                self._dict[key] = value
