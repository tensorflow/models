# Copyright 2017 Google Inc. All Rights Reserved.
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
#
# ==============================================================================
from __future__ import print_function

import h5py
import numpy as np
import os
from six.moves import xrange
import tensorflow as tf

from utils import write_datasets
from synthetic_data_utils import normalize_rates
from synthetic_data_utils import get_train_n_valid_inds, nparray_and_transpose
from synthetic_data_utils import spikify_data, split_list_by_inds

DATA_DIR = "rnn_synth_data_v1.0"

flags = tf.app.flags
flags.DEFINE_string("save_dir", "/tmp/" + DATA_DIR + "/",
                    "Directory for saving data.")
flags.DEFINE_string("datafile_name", "itb_rnn",
                    "Name of data file for input case.")
flags.DEFINE_integer("synth_data_seed", 5, "Random seed for RNN generation.")
flags.DEFINE_float("T", 1.0, "Time in seconds to generate.")
flags.DEFINE_integer("C", 800, "Number of conditions")
flags.DEFINE_integer("N", 50, "Number of units for the RNN")
flags.DEFINE_float("train_percentage", 4.0/5.0,
                   "Percentage of train vs validation trials")
flags.DEFINE_integer("nspikifications", 5,
                     "Number of spikifications of the same underlying rates.")
flags.DEFINE_float("tau", 0.025, "Time constant of RNN")
flags.DEFINE_float("dt", 0.010, "Time bin")
flags.DEFINE_float("max_firing_rate", 30.0,
                   "Map 1.0 of RNN to a spikes per second")
flags.DEFINE_float("u_std", 0.25,
                   "Std dev of input to integration to bound model")
flags.DEFINE_string("checkpoint_path", "SAMPLE_CHECKPOINT",
                    """Path to directory with checkpoints of model
                    trained on integration to bound task. Currently this
                    is a placeholder which tells the code to grab the
                    checkpoint that is provided with the code
                    (in /trained_itb/..). If you have your own checkpoint
                    you would like to restore, you would point it to
                    that path.""")
FLAGS = flags.FLAGS


class IntegrationToBoundModel:
  def __init__(self, N):
    scale = 0.8 / float(N**0.5)
    self.N = N
    self.Wh_nxn = tf.Variable(tf.random_normal([N, N], stddev=scale))
    self.b_1xn = tf.Variable(tf.zeros([1, N]))
    self.Bu_1xn = tf.Variable(tf.zeros([1, N]))
    self.Wro_nxo = tf.Variable(tf.random_normal([N, 1], stddev=scale))
    self.bro_o = tf.Variable(tf.zeros([1]))

  def call(self, h_tm1_bxn, u_bx1):
    act_t_bxn = tf.matmul(h_tm1_bxn, self.Wh_nxn) + self.b_1xn + u_bx1 * self.Bu_1xn
    h_t_bxn = tf.nn.tanh(act_t_bxn)
    z_t = tf.nn.xw_plus_b(h_t_bxn, self.Wro_nxo, self.bro_o)
    return z_t, h_t_bxn

def get_data_batch(batch_size, T, rng, u_std):
  u_bxt = rng.randn(batch_size, T) * u_std
  running_sum_b = np.zeros([batch_size])
  labels_bxt = np.zeros([batch_size, T])
  for t in xrange(T):
    running_sum_b += u_bxt[:, t]
    labels_bxt[:, t] += running_sum_b
  labels_bxt = np.clip(labels_bxt, -1, 1)
  return u_bxt, labels_bxt


rng = np.random.RandomState(seed=FLAGS.synth_data_seed)
u_rng = np.random.RandomState(seed=FLAGS.synth_data_seed+1)
T = FLAGS.T
C = FLAGS.C
N = FLAGS.N  # must be same N as in trained model (provided example is N = 50)
nspikifications = FLAGS.nspikifications
E = nspikifications * C  # total number of trials
train_percentage = FLAGS.train_percentage
ntimesteps = int(T / FLAGS.dt)
batch_size = 1  # gives one example per ntrial

model = IntegrationToBoundModel(N)
inputs_ph_t = [tf.placeholder(tf.float32,
                              shape=[None, 1]) for _ in range(ntimesteps)]
state = tf.zeros([batch_size, N])
saver = tf.train.Saver()

P_nxn = rng.randn(N,N) / np.sqrt(N)  # random projections

# unroll RNN for T timesteps
outputs_t = []
states_t = []

for inp in inputs_ph_t:
  output, state = model.call(state, inp)
  outputs_t.append(output)
  states_t.append(state)

with tf.Session() as sess:
  # restore the latest model ckpt
  if FLAGS.checkpoint_path == "SAMPLE_CHECKPOINT":
    dir_path = os.path.dirname(os.path.realpath(__file__))
    model_checkpoint_path = os.path.join(dir_path, "trained_itb/model-65000")
  else:
    model_checkpoint_path = FLAGS.checkpoint_path
  try:
    saver.restore(sess, model_checkpoint_path)
    print ('Model restored from', model_checkpoint_path)
  except:
    assert False, ("No checkpoints to restore from, is the path %s correct?"
                   %model_checkpoint_path)

  # generate data for trials
  data_e = []
  u_e = []
  outs_e = []
  for c in range(C):
    u_1xt, outs_1xt = get_data_batch(batch_size, ntimesteps, u_rng, FLAGS.u_std)

    feed_dict = {}
    for t in xrange(ntimesteps):
      feed_dict[inputs_ph_t[t]] = np.reshape(u_1xt[:,t], (batch_size,-1))

    states_t_bxn, outputs_t_bxn = sess.run([states_t, outputs_t],
                                           feed_dict=feed_dict)
    states_nxt = np.transpose(np.squeeze(np.asarray(states_t_bxn)))
    outputs_t_bxn = np.squeeze(np.asarray(outputs_t_bxn))
    r_sxt = np.dot(P_nxn, states_nxt)

    for s in xrange(nspikifications):
      data_e.append(r_sxt)
      u_e.append(u_1xt)
      outs_e.append(outputs_t_bxn)

  truth_data_e = normalize_rates(data_e, E, N)

spiking_data_e = spikify_data(truth_data_e, rng, dt=FLAGS.dt,
                              max_firing_rate=FLAGS.max_firing_rate)
train_inds, valid_inds = get_train_n_valid_inds(E, train_percentage,
                                                nspikifications)

data_train_truth, data_valid_truth = split_list_by_inds(truth_data_e,
                                                        train_inds,
                                                        valid_inds)
data_train_spiking, data_valid_spiking = split_list_by_inds(spiking_data_e,
                                                            train_inds,
                                                            valid_inds)

data_train_truth = nparray_and_transpose(data_train_truth)
data_valid_truth = nparray_and_transpose(data_valid_truth)
data_train_spiking = nparray_and_transpose(data_train_spiking)
data_valid_spiking = nparray_and_transpose(data_valid_spiking)

# save down the inputs used to generate this data
train_inputs_u, valid_inputs_u = split_list_by_inds(u_e,
                                                    train_inds,
                                                    valid_inds)
train_inputs_u = nparray_and_transpose(train_inputs_u)
valid_inputs_u = nparray_and_transpose(valid_inputs_u)

# save down the network outputs (may be useful later)
train_outputs_u, valid_outputs_u = split_list_by_inds(outs_e,
                                                      train_inds,
                                                      valid_inds)
train_outputs_u = np.array(train_outputs_u)
valid_outputs_u = np.array(valid_outputs_u)


data = { 'train_truth': data_train_truth,
         'valid_truth': data_valid_truth,
         'train_data' : data_train_spiking,
         'valid_data' : data_valid_spiking,
         'train_percentage' : train_percentage,
         'nspikifications' : nspikifications,
         'dt' : FLAGS.dt,
         'u_std' : FLAGS.u_std,
         'max_firing_rate': FLAGS.max_firing_rate,
         'train_inputs_u': train_inputs_u,
         'valid_inputs_u': valid_inputs_u,
         'train_outputs_u': train_outputs_u,
         'valid_outputs_u': valid_outputs_u,
         'conversion_factor' : FLAGS.max_firing_rate/(1.0/FLAGS.dt) }

# just one dataset here
datasets = {}
dataset_name = 'dataset_N' + str(N)
datasets[dataset_name] = data

# write out the dataset
write_datasets(FLAGS.save_dir, FLAGS.datafile_name, datasets)
print ('Saved to ', os.path.join(FLAGS.save_dir,
                                 FLAGS.datafile_name + '_' + dataset_name))
