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
import tensorflow as tf         # used for flags here

from utils import write_datasets
from synthetic_data_utils import add_alignment_projections, generate_data
from synthetic_data_utils import generate_rnn, get_train_n_valid_inds
from synthetic_data_utils import nparray_and_transpose
from synthetic_data_utils import spikify_data, gaussify_data, split_list_by_inds
import matplotlib
import matplotlib.pyplot as plt
import scipy.signal

matplotlib.rcParams['image.interpolation'] = 'nearest'
DATA_DIR = "rnn_synth_data_v1.0"

flags = tf.app.flags
flags.DEFINE_string("save_dir", "/tmp/" + DATA_DIR + "/",
                    "Directory for saving data.")
flags.DEFINE_string("datafile_name", "thits_data",
                    "Name of data file for input case.")
flags.DEFINE_string("noise_type", "poisson", "Noise type for data.")
flags.DEFINE_integer("synth_data_seed", 5, "Random seed for RNN generation.")
flags.DEFINE_float("T", 1.0, "Time in seconds to generate.")
flags.DEFINE_integer("C", 100, "Number of conditions")
flags.DEFINE_integer("N", 50, "Number of units for the RNN")
flags.DEFINE_integer("S", 50, "Number of sampled units from RNN")
flags.DEFINE_integer("npcs", 10, "Number of PCS for multi-session case.")
flags.DEFINE_float("train_percentage", 4.0/5.0,
                   "Percentage of train vs validation trials")
flags.DEFINE_integer("nreplications", 40,
                     "Number of noise replications of the same underlying rates.")
flags.DEFINE_float("g", 1.5, "Complexity of dynamics")
flags.DEFINE_float("x0_std", 1.0,
                   "Volume from which to pull initial conditions (affects diversity of dynamics.")
flags.DEFINE_float("tau", 0.025, "Time constant of RNN")
flags.DEFINE_float("dt", 0.010, "Time bin")
flags.DEFINE_float("input_magnitude", 20.0,
                   "For the input case, what is the value of the input?")
flags.DEFINE_float("max_firing_rate", 30.0, "Map 1.0 of RNN to a spikes per second")
FLAGS = flags.FLAGS


# Note that with N small, (as it is 25 above), the finite size effects
# will have pretty dramatic effects on the dynamics of the random RNN.
# If you want more complex dynamics, you'll have to run the script a
# lot, or increase N (or g).

# Getting hard vs. easy data can be a little stochastic, so we set the seed.

# Pull out some commonly used parameters.
# These are user parameters (configuration)
rng = np.random.RandomState(seed=FLAGS.synth_data_seed)
T = FLAGS.T
C = FLAGS.C
N = FLAGS.N
S = FLAGS.S
input_magnitude = FLAGS.input_magnitude
nreplications = FLAGS.nreplications
E = nreplications * C         # total number of trials
# S is the number of measurements in each datasets, w/ each
# dataset having a different set of observations.
ndatasets = N/S                 # ok if rounded down
train_percentage = FLAGS.train_percentage
ntime_steps = int(T / FLAGS.dt)
# End of user parameters

rnn = generate_rnn(rng, N, FLAGS.g, FLAGS.tau, FLAGS.dt, FLAGS.max_firing_rate)

# Check to make sure the RNN is the one we used in the paper.
if N == 50:
  assert abs(rnn['W'][0,0] - 0.06239899) < 1e-8, 'Error in random seed?'
  rem_check = nreplications * train_percentage
  assert  abs(rem_check - int(rem_check)) < 1e-8, \
    'Train percentage  * nreplications should be integral number.'


# Initial condition generation, and condition label generation.  This
# happens outside of the dataset loop, so that all datasets have the
# same conditions, which is similar to a neurophys setup.
condition_number = 0
x0s = []
condition_labels = []
for c in range(C):
  x0 = FLAGS.x0_std * rng.randn(N, 1)
  x0s.append(np.tile(x0, nreplications)) # replicate x0 nreplications times
  # replicate the condition label nreplications times
  for ns in range(nreplications):
    condition_labels.append(condition_number)
  condition_number += 1
x0s = np.concatenate(x0s, axis=1)

# Containers for storing data across data.
datasets = {}
for n in range(ndatasets):
  print(n+1, " of ", ndatasets)

  # First generate all firing rates. in the next loop, generate all
  # replications this allows the random state for rate generation to be
  # independent of n_replications.
  dataset_name = 'dataset_N' + str(N) + '_S' + str(S)
  if S < N:
    dataset_name += '_n' + str(n+1)

  # Sample neuron subsets.  The assumption is the PC axes of the RNN
  # are not unit aligned, so sampling units is adequate to sample all
  # the high-variance PCs.
  P_sxn = np.eye(S,N)
  for m in range(n):
    P_sxn = np.roll(P_sxn, S, axis=1)

  if input_magnitude > 0.0:
    # time of "hits" randomly chosen between [1/4 and 3/4] of total time
    input_times = rng.choice(int(ntime_steps/2), size=[E]) + int(ntime_steps/4)
  else:
    input_times = None

  rates, x0s, inputs = \
      generate_data(rnn, T=T, E=E, x0s=x0s, P_sxn=P_sxn,
                    input_magnitude=input_magnitude,
                    input_times=input_times)

  if FLAGS.noise_type == "poisson":
    noisy_data = spikify_data(rates, rng, rnn['dt'], rnn['max_firing_rate'])
  elif FLAGS.noise_type == "gaussian":
    noisy_data = gaussify_data(rates, rng, rnn['dt'], rnn['max_firing_rate'])
  else:
    raise ValueError("Only noise types supported are poisson or gaussian")

    # split into train and validation sets
  train_inds, valid_inds = get_train_n_valid_inds(E, train_percentage,
                                                  nreplications)

  # Split the data, inputs, labels and times into train vs. validation.
  rates_train, rates_valid = \
      split_list_by_inds(rates, train_inds, valid_inds)
  noisy_data_train, noisy_data_valid = \
      split_list_by_inds(noisy_data, train_inds, valid_inds)
  input_train, inputs_valid = \
      split_list_by_inds(inputs, train_inds, valid_inds)
  condition_labels_train, condition_labels_valid = \
      split_list_by_inds(condition_labels, train_inds, valid_inds)
  input_times_train, input_times_valid = \
      split_list_by_inds(input_times, train_inds, valid_inds)

  # Turn rates, noisy_data, and input into numpy arrays.
  rates_train = nparray_and_transpose(rates_train)
  rates_valid = nparray_and_transpose(rates_valid)
  noisy_data_train = nparray_and_transpose(noisy_data_train)
  noisy_data_valid = nparray_and_transpose(noisy_data_valid)
  input_train = nparray_and_transpose(input_train)
  inputs_valid = nparray_and_transpose(inputs_valid)

  # Note that we put these 'truth' rates and input into this
  # structure, the only data that is used in LFADS are the noisy
  # data e.g. spike trains.  The rest is either for printing or posterity.
  data = {'train_truth': rates_train,
          'valid_truth': rates_valid,
          'input_train_truth' : input_train,
          'input_valid_truth' : inputs_valid,
          'train_data' : noisy_data_train,
          'valid_data' : noisy_data_valid,
          'train_percentage' : train_percentage,
          'nreplications' : nreplications,
          'dt' : rnn['dt'],
          'input_magnitude' : input_magnitude,
          'input_times_train' : input_times_train,
          'input_times_valid' : input_times_valid,
          'P_sxn' : P_sxn,
          'condition_labels_train' : condition_labels_train,
          'condition_labels_valid' : condition_labels_valid,
          'conversion_factor': 1.0 / rnn['conversion_factor']}
  datasets[dataset_name] = data

if S < N:
  # Note that this isn't necessary for this synthetic example, but
  # it's useful to see how the input factor matrices were initialized
  # for actual neurophysiology data.
  datasets = add_alignment_projections(datasets, npcs=FLAGS.npcs)

# Write out the datasets.
write_datasets(FLAGS.save_dir, FLAGS.datafile_name, datasets)
