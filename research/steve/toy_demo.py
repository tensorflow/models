from __future__ import division
from __future__ import print_function
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
import scipy
import matplotlib.pyplot as plt
import seaborn as sns

### Hyperparameters

NONTERMINAL_STATE_COUNT = 100
NOISE_AMOUNT = 0.1
TRAIN_STEPS = 10000
Q_ENSEMBLE_SIZE = 8
MODEL_ENSEMBLE_SIZE = 8
HORIZON = 5
TRIAL_N = 10

### Helper functions

initial_state = 0
terminal_state = NONTERMINAL_STATE_COUNT + 1
nonterminal_state_count = NONTERMINAL_STATE_COUNT
state_count = NONTERMINAL_STATE_COUNT + 1
final_reward = NONTERMINAL_STATE_COUNT
colors = sns.color_palette('husl', 4)
plt.rcParams["figure.figsize"] = (6,5)

def step(state):
  if state == terminal_state: next_state = terminal_state
  else:                       next_state = state + 1

  if state == terminal_state:     reward = 0
  elif state+1 == terminal_state: reward = final_reward
  else:                           reward = -1

  return next_state, reward

def noisy_step(state):
  if state == terminal_state:               next_state = terminal_state
  elif np.random.random([]) < NOISE_AMOUNT: next_state = np.random.randint(0, state_count)
  else:                                     next_state = state + 1

  if state == terminal_state:     reward = 0
  elif state+1 == terminal_state: reward = final_reward
  else:                           reward = -1

  return next_state, reward

def get_error(Q):
  losses = np.square(np.arange(state_count) - Q[:-1])
  return np.mean(losses)

def downsample(array, factor):
  pad_size = np.ceil(old_div(float(array.size),factor))*factor - array.size
  array_padded = np.append(array, np.zeros([pad_size.astype(np.int64)])*np.NaN)
  return scipy.nanmean(array_padded.reshape(-1,factor), axis=1)


######################
### Main experiments
######################

# Basic Q
if True:
  print("Running basic Q-learning.")
  trial_results = []
  for run_i in range(TRIAL_N):
    print("Trial %d" % run_i)
    Q = np.random.randint(0,state_count,[state_count+1]).astype(np.float64)
    Q[state_count] = 0
    losses = []
    for step_i in range(TRAIN_STEPS):
      state = np.random.randint(0,state_count)
      next_state, reward = step(state)
      Q[state] = reward + Q[next_state]
      losses.append(get_error(Q))
    trial_results.append(losses)
  print("...complete.\n")

  result = np.stack(trial_results, axis=1)
  means = np.mean(result, axis=1)
  stdevs = np.std(result, axis=1)
  plt.plot(means, label="Basic Q-learning", color=colors[0])
  plt.fill_between(np.arange(TRAIN_STEPS), means - stdevs, means + stdevs, alpha=.2, color=colors[0])
  with open('Toy-v1/baseline.csv', 'w') as f:
    data = []
    for frame_i in range(result.shape[0]):
      for loss in result[frame_i]:
        data.append("%f,%f,%f,%f" % (frame_i, frame_i, frame_i, loss))
    f.write("\n".join(data))

# Ensemble Q
if True:
  print("Running ensemble Q-learning.")
  trial_results = []
  for run_i in range(TRIAL_N):
    print("Trial %d" % run_i)
    Q = np.random.randint(0,state_count,[Q_ENSEMBLE_SIZE, state_count+1]).astype(np.float64)
    Q[:, state_count] = 0
    losses = []
    for step_i in range(TRAIN_STEPS):
      for q_ensemble_i in range(Q_ENSEMBLE_SIZE):
        state = np.random.randint(0,state_count)
        next_state, reward = step(state)
        Q[q_ensemble_i, state] = reward + np.mean(Q[:, next_state])
      losses.append(get_error(np.mean(Q, axis=0)))
    trial_results.append(losses)
  print("...complete.\n")

  result = np.stack(trial_results, axis=1)
  means = np.mean(result, axis=1)
  stdevs = np.std(result, axis=1)
  plt.plot(means, label="Ensemble Q-learning", color=colors[1])
  plt.fill_between(np.arange(TRAIN_STEPS), means - stdevs, means + stdevs, alpha=.2, color=colors[1])

# Ensemble MVE-Oracle
if True:
  print("Running ensemble oracle MVE.")
  trial_results = []
  for run_i in range(TRIAL_N):
    print("Trial %d" % run_i)
    Q = np.random.randint(0,state_count,[Q_ENSEMBLE_SIZE, state_count+1]).astype(np.float64)
    Q[:, state_count] = 0
    losses = []
    for step_i in range(TRAIN_STEPS):
      for q_ensemble_i in range(Q_ENSEMBLE_SIZE):
        state = np.random.randint(0,state_count)
        next_state, reward = step(state)

        # MVE rollout
        target = reward
        for _ in range(HORIZON):
          next_state, reward = step(next_state)
          target += reward
        target += np.mean(Q[:,next_state])

        Q[q_ensemble_i, state] = target
      losses.append(get_error(np.mean(Q, axis=0)))
    trial_results.append(losses)
  print("...complete.\n")

  result = np.stack(trial_results, axis=1)
  means = np.mean(result, axis=1)
  stdevs = np.std(result, axis=1)
  plt.plot(means, label="MVE-oracle", color=colors[2])
  plt.fill_between(np.arange(TRAIN_STEPS), means - stdevs, means + stdevs, alpha=.2, color=colors[2])
  with open('Toy-v1/mve_oracle.csv', 'w') as f:
    data = []
    for frame_i in range(result.shape[0]):
      for loss in result[frame_i]:
        data.append("%f,%f,%f,%f" % (frame_i, frame_i, frame_i, loss))
    f.write("\n".join(data))

# Ensemble MVE-Noisy
if True:
  print("Running ensemble noisy MVE.")
  trial_results = []
  for run_i in range(TRIAL_N):
    print("Trial %d" % run_i)
    Q = np.random.randint(0,state_count,[Q_ENSEMBLE_SIZE, state_count+1]).astype(np.float64)
    Q[:, state_count] = 0
    losses = []
    for step_i in range(TRAIN_STEPS):
      for q_ensemble_i in range(Q_ENSEMBLE_SIZE):
        state = np.random.randint(0,state_count)
        next_state, reward = step(state)

        # MVE rollout
        targets = []
        first_next_state, first_reward = next_state, reward
        for model_ensemble_i in range(MODEL_ENSEMBLE_SIZE):
          next_state, reward = first_next_state, first_reward
          target = reward
          for _ in range(HORIZON):
            next_state, reward = noisy_step(next_state)
            target += reward
          target += np.mean(Q[:,next_state])
          targets.append(target)

        Q[q_ensemble_i, state] = np.mean(targets)
      losses.append(get_error(np.mean(Q, axis=0)))
    trial_results.append(losses)
  print("...complete.\n")

  result = np.stack(trial_results, axis=1)
  means = np.mean(result, axis=1)
  stdevs = np.std(result, axis=1)
  plt.plot(means, label="MVE-noisy", color=colors[2], linestyle='dotted')
  plt.fill_between(np.arange(TRAIN_STEPS), means - stdevs, means + stdevs, alpha=.2, color=colors[2])
  with open('Toy-v1/mve_noisy.csv', 'w') as f:
    data = []
    for frame_i in range(result.shape[0]):
      for loss in result[frame_i]:
        data.append("%f,%f,%f,%f" % (frame_i, frame_i, frame_i, loss))
    f.write("\n".join(data))

# STEVE-Oracle
if True:
  print("Running ensemble oracle STEVE.")
  trial_results = []

  oracle_q_estimate_errors = []
  oracle_mve_estimate_errors = []
  oracle_steve_estimate_errors = []
  oracle_opt_estimate_errors = []


  for run_i in range(TRIAL_N):
    print("Trial %d" % run_i)
    Q = np.random.randint(0,state_count,[Q_ENSEMBLE_SIZE, state_count+1]).astype(np.float64)
    Q[:, state_count] = 0
    losses = []

    q_estimate_errors = []
    mve_estimate_errors = []
    steve_estimate_errors = []
    opt_estimate_errors = []
    steve_beat_freq= []

    for step_i in range(TRAIN_STEPS):
      _q_estimate_errors = []
      _mve_estimate_errors = []
      _steve_estimate_errors = []
      _opt_estimate_errors = []
      _steve_beat_freq = []

      for q_ensemble_i in range(Q_ENSEMBLE_SIZE):
        state = np.random.randint(0,state_count)
        next_state, reward = step(state)

        # STEVE rollout
        Q_est_mat = np.zeros([HORIZON + 1, Q_ENSEMBLE_SIZE])
        reward_est_mat = np.zeros([HORIZON + 1, 1])
        first_next_state, first_reward = next_state, reward
        next_state, reward = first_next_state, first_reward
        Q_est_mat[0, :] = Q[:, next_state]
        reward_est_mat[0, 0] = reward
        for timestep_i in range(1,HORIZON+1):
          next_state, reward = step(next_state)
          Q_est_mat[timestep_i, :] = Q[:, next_state]
          reward_est_mat[timestep_i, 0] = reward
        all_targets = Q_est_mat + np.cumsum(reward_est_mat, axis=0)

        # STEVE weight calculation
        estimates = np.mean(all_targets, axis=1)
        confidences = old_div(1., (np.var(all_targets, axis=1) + 1e-8))
        coefficients = old_div(confidences, np.sum(confidences))
        target = np.sum(estimates * coefficients)

        Q[q_ensemble_i, state] = target

        true_target = state + 1. if state != terminal_state else 0.
        _q_estimate_errors.append(np.square(estimates[0] - true_target))
        _mve_estimate_errors.append(np.square(estimates[-1] - true_target))
        _steve_estimate_errors.append(np.square(np.sum(estimates * coefficients) - true_target))
        _opt_estimate_errors.append(np.min(np.square(estimates - true_target)))

      losses.append(get_error(np.mean(Q, axis=0)))
      q_estimate_errors.append(np.mean(_q_estimate_errors))
      mve_estimate_errors.append(np.mean(_mve_estimate_errors))
      steve_estimate_errors.append(np.mean(_steve_estimate_errors))
      opt_estimate_errors.append(np.mean(_opt_estimate_errors))
    trial_results.append(losses)
    oracle_q_estimate_errors.append(q_estimate_errors)
    oracle_mve_estimate_errors.append(mve_estimate_errors)
    oracle_steve_estimate_errors.append(steve_estimate_errors)
    oracle_opt_estimate_errors.append(opt_estimate_errors)
  print("...complete.\n")

  result = np.stack(trial_results, axis=1)
  means = np.mean(result, axis=1)
  stdevs = np.std(result, axis=1)
  plt.plot(means, label="STEVE-oracle", color=colors[3])
  plt.fill_between(np.arange(TRAIN_STEPS), means - stdevs, means + stdevs, alpha=.2, color=colors[3])
  with open('Toy-v1/steve_oracle.csv', 'w') as f:
    data = []
    for frame_i in range(result.shape[0]):
      for loss in result[frame_i]:
        data.append("%f,%f,%f,%f" % (frame_i, frame_i, frame_i, loss))
    f.write("\n".join(data))

# STEVE-Noisy
if True:
  print("Running ensemble noisy STEVE.")
  trial_results = []

  noisy_q_estimate_errors = []
  noisy_mve_estimate_errors = []
  noisy_steve_estimate_errors = []
  noisy_opt_estimate_errors = []
  noisy_steve_beat_freq = []

  for run_i in range(TRIAL_N):
    print("Trial %d" % run_i)
    Q = np.random.randint(0,state_count,[Q_ENSEMBLE_SIZE, state_count+1]).astype(np.float64)
    Q[:, state_count] = 0
    losses = []

    q_estimate_errors = []
    mve_estimate_errors = []
    steve_estimate_errors = []
    opt_estimate_errors = []
    steve_beat_freq= []

    for step_i in range(TRAIN_STEPS):
      _q_estimate_errors = []
      _mve_estimate_errors = []
      _steve_estimate_errors = []
      _opt_estimate_errors = []
      _steve_beat_freq = []
      for q_ensemble_i in range(Q_ENSEMBLE_SIZE):
        state = np.random.randint(0,state_count)
        next_state, reward = step(state)

        # STEVE rollout
        Q_est_mat = np.zeros([HORIZON + 1, MODEL_ENSEMBLE_SIZE, Q_ENSEMBLE_SIZE])
        reward_est_mat = np.zeros([HORIZON + 1, MODEL_ENSEMBLE_SIZE, 1])
        first_next_state, first_reward = next_state, reward
        for model_ensemble_i in range(MODEL_ENSEMBLE_SIZE):
          next_state, reward = first_next_state, first_reward
          Q_est_mat[0, model_ensemble_i, :] = Q[:, next_state]
          reward_est_mat[0, model_ensemble_i, 0] = reward
          for timestep_i in range(1,HORIZON+1):
            next_state, reward = noisy_step(next_state)
            Q_est_mat[timestep_i, model_ensemble_i, :] = Q[:, next_state]
            reward_est_mat[timestep_i, model_ensemble_i, 0] = reward
        all_targets = Q_est_mat + np.cumsum(reward_est_mat, axis=0)

        # STEVE weight calculation
        all_targets = np.reshape(all_targets, [HORIZON+1, MODEL_ENSEMBLE_SIZE * Q_ENSEMBLE_SIZE])
        estimates = np.mean(all_targets, axis=1)
        confidences = old_div(1., (np.var(all_targets, axis=1) + 1e-8))
        coefficients = old_div(confidences, np.sum(confidences))
        target = np.sum(estimates * coefficients)
        # target = estimates[0]

        Q[q_ensemble_i, state] = target

        true_target = state + 1. if state != terminal_state else 0.
        _q_estimate_errors.append(np.square(estimates[0] - true_target))
        _mve_estimate_errors.append(np.square(estimates[-1] - true_target))
        _steve_estimate_errors.append(np.square(np.sum(estimates * coefficients) - true_target))
        _opt_estimate_errors.append(np.min(np.square(estimates - true_target)))
        _steve_beat_freq.append(float(np.square(estimates[0] - true_target) > np.square(target - true_target)))

      losses.append(get_error(np.mean(Q, axis=0)))
      q_estimate_errors.append(np.mean(_q_estimate_errors))
      mve_estimate_errors.append(np.mean(_mve_estimate_errors))
      steve_estimate_errors.append(np.mean(_steve_estimate_errors))
      opt_estimate_errors.append(np.mean(_opt_estimate_errors))
      steve_beat_freq.append(np.mean(_steve_beat_freq))
    trial_results.append(losses)
    noisy_q_estimate_errors.append(q_estimate_errors)
    noisy_mve_estimate_errors.append(mve_estimate_errors)
    noisy_steve_estimate_errors.append(steve_estimate_errors)
    noisy_opt_estimate_errors.append(opt_estimate_errors)
    noisy_steve_beat_freq.append(steve_beat_freq)

  print("...complete.\n")

  result = np.stack(trial_results, axis=1)
  means = np.mean(result, axis=1)
  stdevs = np.std(result, axis=1)
  plt.plot(means, label="STEVE-noisy", color=colors[3], linestyle='dotted')
  plt.fill_between(np.arange(TRAIN_STEPS), means - stdevs, means + stdevs, alpha=.2, color=colors[3])
  with open('Toy-v1/steve_noisy.csv', 'w') as f:
    data = []
    for frame_i in range(result.shape[0]):
      for loss in result[frame_i]:
        data.append("%f,%f,%f,%f" % (frame_i, frame_i, frame_i, loss))
    f.write("\n".join(data))

# ### Display results
# plt.title("Comparison of convergence rates")
# plt.legend()
# plt.savefig("comparison.pdf")
# plt.show()
#
# ### Display secondary results - error comparison
# DOWNSAMPLE = 50
# colors = sns.color_palette('husl', 8)
# for i, (error_curve, label) in enumerate([
#                                           (oracle_q_estimate_errors, "Oracle Q error"),
#                                           (oracle_mve_estimate_errors, "Oracle MVE error"),
#                                           (oracle_steve_estimate_errors, "Oracle STEVE error"),
#                                           # (oracle_opt_estimate_errors, "Oracle minimum single-estimate error"),
#                                          ]):
#   result = np.stack(error_curve, axis=1)
#   means = downsample(np.mean(result, axis=1), DOWNSAMPLE)
#   stdevs = downsample(np.std(result, axis=1), DOWNSAMPLE)
#   plt.plot(means, label=label, color=colors[i])
#   plt.fill_between(np.arange(means.shape[0]), means - stdevs, means + stdevs, alpha=.2, color=colors[i])
#
# plt.title("Comparison of errors for oracle dynamics")
# plt.legend()
# plt.show()
#
# for i, (error_curve, label) in enumerate([
#                                           (noisy_q_estimate_errors, "Noisy Q error"),
#                                           (noisy_mve_estimate_errors, "Noisy MVE error"),
#                                           (noisy_steve_estimate_errors, "Noisy STEVE error"),
#                                           # (noisy_opt_estimate_errors, "Noisy minimum single-estimate error"),
#                                           # (trial_steve_beat_freq, "STEVE beat freq"),
#                                         ]):
#   result = np.stack(error_curve, axis=1)
#   means = downsample(np.mean(result, axis=1), DOWNSAMPLE)
#   stdevs = downsample(np.std(result, axis=1), DOWNSAMPLE)
#   plt.plot(means, label=label, color=colors[i])
#   plt.fill_between(np.arange(means.shape[0]), means - stdevs, means + stdevs, alpha=.2, color=colors[i])
#
# plt.title("Comparison of errors for noisy dynamics")
# plt.legend()
# plt.show()