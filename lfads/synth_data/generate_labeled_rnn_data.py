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

import os
import h5py
import numpy as np

from synthetic_data_utils import generate_data, generate_rnn
from synthetic_data_utils import get_train_n_valid_inds
from synthetic_data_utils import nparray_and_transpose
from synthetic_data_utils import spikify_data, split_list_by_inds
import tensorflow as tf
from utils import write_datasets

DATA_DIR = "rnn_synth_data_v1.0"

flags = tf.app.flags
flags.DEFINE_string("save_dir", "/tmp/" + DATA_DIR + "/",
                    "Directory for saving data.")
flags.DEFINE_string("datafile_name", "conditioned_rnn_data",
                    "Name of data file for input case.")
flags.DEFINE_integer("synth_data_seed", 5, "Random seed for RNN generation.")
flags.DEFINE_float("T", 1.0, "Time in seconds to generate.")
flags.DEFINE_integer("C", 400, "Number of conditions")
flags.DEFINE_integer("N", 50, "Number of units for the RNN")
flags.DEFINE_float("train_percentage", 4.0/5.0,
                   "Percentage of train vs validation trials")
flags.DEFINE_integer("nspikifications", 10,
                     "Number of spikifications of the same underlying rates.")
flags.DEFINE_float("g", 1.5, "Complexity of dynamics")
flags.DEFINE_float("x0_std", 1.0,
                   "Volume from which to pull initial conditions (affects diversity of dynamics.")
flags.DEFINE_float("tau", 0.025, "Time constant of RNN")
flags.DEFINE_float("dt", 0.010, "Time bin")
flags.DEFINE_float("max_firing_rate", 30.0, "Map 1.0 of RNN to a spikes per second")
FLAGS = flags.FLAGS

rng = np.random.RandomState(seed=FLAGS.synth_data_seed)
rnn_rngs = [np.random.RandomState(seed=FLAGS.synth_data_seed+1),
            np.random.RandomState(seed=FLAGS.synth_data_seed+2)]
T = FLAGS.T
C = FLAGS.C
N = FLAGS.N
nspikifications = FLAGS.nspikifications
E = nspikifications * C
train_percentage = FLAGS.train_percentage
ntimesteps = int(T / FLAGS.dt)

rnn_a = generate_rnn(rnn_rngs[0], N, FLAGS.g, FLAGS.tau, FLAGS.dt,
                     FLAGS.max_firing_rate)
rnn_b = generate_rnn(rnn_rngs[1], N, FLAGS.g, FLAGS.tau, FLAGS.dt,
                     FLAGS.max_firing_rate)
rnns = [rnn_a, rnn_b]

# pick which RNN is used on each trial
rnn_to_use = rng.randint(2, size=E)
ext_input = np.repeat(np.expand_dims(rnn_to_use, axis=1), ntimesteps, axis=1)
ext_input = np.expand_dims(ext_input, axis=2)  # these are "a's" in the paper

x0s = []
condition_labels = []
condition_number = 0
for c in range(C):
  x0 = FLAGS.x0_std * rng.randn(N, 1)
  x0s.append(np.tile(x0, nspikifications))
  for ns in range(nspikifications):
    condition_labels.append(condition_number)
  condition_number += 1
x0s = np.concatenate(x0s, axis=1)

P_nxn = rng.randn(N, N) / np.sqrt(N)

# generate trials for both RNNs
rates_a, x0s_a, _ = generate_data(rnn_a, T=T, E=E, x0s=x0s, P_sxn=P_nxn,
                                  input_magnitude=0.0, input_times=None)
spikes_a = spikify_data(rates_a, rng, rnn_a['dt'], rnn_a['max_firing_rate'])

rates_b, x0s_b, _ = generate_data(rnn_b, T=T, E=E, x0s=x0s, P_sxn=P_nxn,
                                  input_magnitude=0.0, input_times=None)
spikes_b = spikify_data(rates_b, rng, rnn_b['dt'], rnn_b['max_firing_rate'])

# not the best way to do this but E is small enough
rates = []
spikes = []
for trial in xrange(E):
  if rnn_to_use[trial] == 0:
    rates.append(rates_a[trial])
    spikes.append(spikes_a[trial])
  else:
    rates.append(rates_b[trial])
    spikes.append(spikes_b[trial])

# split into train and validation sets
train_inds, valid_inds = get_train_n_valid_inds(E, train_percentage,
                                                nspikifications)

rates_train, rates_valid = split_list_by_inds(rates, train_inds, valid_inds)
spikes_train, spikes_valid = split_list_by_inds(spikes, train_inds, valid_inds)
condition_labels_train, condition_labels_valid = split_list_by_inds(
    condition_labels, train_inds, valid_inds)
ext_input_train, ext_input_valid = split_list_by_inds(
    ext_input, train_inds, valid_inds)

rates_train = nparray_and_transpose(rates_train)
rates_valid = nparray_and_transpose(rates_valid)
spikes_train = nparray_and_transpose(spikes_train)
spikes_valid = nparray_and_transpose(spikes_valid)

# add train_ext_input and valid_ext input
data = {'train_truth': rates_train,
        'valid_truth': rates_valid,
        'train_data' : spikes_train,
        'valid_data' : spikes_valid,
        'train_ext_input' : np.array(ext_input_train),
        'valid_ext_input': np.array(ext_input_valid),
        'train_percentage' : train_percentage,
        'nspikifications' : nspikifications,
        'dt' : FLAGS.dt,
        'P_sxn' : P_nxn,
        'condition_labels_train' : condition_labels_train,
        'condition_labels_valid' : condition_labels_valid,
        'conversion_factor': 1.0 / rnn_a['conversion_factor']}

# just one dataset here
datasets = {}
dataset_name = 'dataset_N' + str(N)
datasets[dataset_name] = data

# write out the dataset
write_datasets(FLAGS.save_dir, FLAGS.datafile_name, datasets)
print ('Saved to ', os.path.join(FLAGS.save_dir,
                                 FLAGS.datafile_name + '_' + dataset_name))
