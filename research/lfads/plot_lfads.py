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
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import matplotlib
matplotlib.use('Agg')
from matplotlib import pyplot as plt
import numpy as np
import tensorflow as tf

def _plot_item(W, name, full_name, nspaces):
  plt.figure()
  if W.shape == ():
    print(name, ": ", W)
  elif W.shape[0] == 1:
    plt.stem(W.T)
    plt.title(full_name)
  elif W.shape[1] == 1:
    plt.stem(W)
    plt.title(full_name)
  else:
    plt.imshow(np.abs(W), interpolation='nearest', cmap='jet');
    plt.colorbar()
    plt.title(full_name)


def all_plot(d, full_name="", exclude="", nspaces=0):
  """Recursively plot all the LFADS model parameters in the nested
  dictionary."""
  for k, v in d.iteritems():
    this_name = full_name+"/"+k
    if isinstance(v, dict):
      all_plot(v, full_name=this_name, exclude=exclude, nspaces=nspaces+4)
    else:
      if exclude == "" or exclude not in this_name:
        _plot_item(v, name=k, full_name=full_name+"/"+k, nspaces=nspaces+4)



def plot_time_series(vals_bxtxn, bidx=None, n_to_plot=np.inf, scale=1.0,
                     color='r', title=None):

  if bidx is None:
    vals_txn = np.mean(vals_bxtxn, axis=0)
  else:
    vals_txn = vals_bxtxn[bidx,:,:]

  T, N = vals_txn.shape
  if n_to_plot > N:
    n_to_plot = N

  plt.plot(vals_txn[:,0:n_to_plot] + scale*np.array(range(n_to_plot)),
           color=color, lw=1.0)
  plt.axis('tight')
  if title:
    plt.title(title)


def plot_lfads_timeseries(data_bxtxn, model_vals, ext_input_bxtxi=None,
                          truth_bxtxn=None, bidx=None, output_dist="poisson",
                          conversion_factor=1.0, subplot_cidx=0,
                          col_title=None):

  n_to_plot = 10
  scale = 1.0
  nrows = 7
  plt.subplot(nrows,2,1+subplot_cidx)

  if output_dist == 'poisson':
    rates = means = conversion_factor * model_vals['output_dist_params']
    plot_time_series(rates, bidx, n_to_plot=n_to_plot, scale=scale,
                     title=col_title + " rates (LFADS - red, Truth - black)")
  elif output_dist == 'gaussian':
    means_vars = model_vals['output_dist_params']
    means, vars = np.split(means_vars,2, axis=2) # bxtxn
    stds = np.sqrt(vars)
    plot_time_series(means, bidx, n_to_plot=n_to_plot, scale=scale,
                     title=col_title + " means (LFADS - red, Truth - black)")
    plot_time_series(means+stds, bidx, n_to_plot=n_to_plot, scale=scale,
                     color='c')
    plot_time_series(means-stds, bidx, n_to_plot=n_to_plot, scale=scale,
                     color='c')
  else:
    assert 'NIY'


  if truth_bxtxn is not None:
    plot_time_series(truth_bxtxn, bidx, n_to_plot=n_to_plot, color='k',
                     scale=scale)

  input_title = ""
  if "controller_outputs" in model_vals.keys():
    input_title += " Controller Output"
    plt.subplot(nrows,2,3+subplot_cidx)
    u_t = model_vals['controller_outputs'][0:-1]
    plot_time_series(u_t, bidx, n_to_plot=n_to_plot, color='c', scale=1.0,
                     title=col_title + input_title)

  if ext_input_bxtxi is not None:
    input_title += " External Input"
    plot_time_series(ext_input_bxtxi, n_to_plot=n_to_plot, color='b',
                     scale=scale, title=col_title + input_title)

  plt.subplot(nrows,2,5+subplot_cidx)
  plot_time_series(means, bidx,
                   n_to_plot=n_to_plot, scale=1.0,
                   title=col_title + " Spikes (LFADS - red, Spikes - black)")
  plot_time_series(data_bxtxn, bidx, n_to_plot=n_to_plot, color='k', scale=1.0)

  plt.subplot(nrows,2,7+subplot_cidx)
  plot_time_series(model_vals['factors'], bidx, n_to_plot=n_to_plot, color='b',
                   scale=2.0, title=col_title + " Factors")

  plt.subplot(nrows,2,9+subplot_cidx)
  plot_time_series(model_vals['gen_states'], bidx, n_to_plot=n_to_plot,
                   color='g', scale=1.0, title=col_title + " Generator State")

  if bidx is not None:
    data_nxt = data_bxtxn[bidx,:,:].T
    params_nxt = model_vals['output_dist_params'][bidx,:,:].T
  else:
    data_nxt = np.mean(data_bxtxn, axis=0).T
    params_nxt = np.mean(model_vals['output_dist_params'], axis=0).T
  if output_dist == 'poisson':
    means_nxt = params_nxt
  elif output_dist == 'gaussian': # (means+vars) x time
    means_nxt = np.vsplit(params_nxt,2)[0] # get means
  else:
    assert "NIY"

  plt.subplot(nrows,2,11+subplot_cidx)
  plt.imshow(data_nxt, aspect='auto', interpolation='nearest')
  plt.title(col_title + ' Data')

  plt.subplot(nrows,2,13+subplot_cidx)
  plt.imshow(means_nxt, aspect='auto', interpolation='nearest')
  plt.title(col_title + ' Means')


def plot_lfads(train_bxtxd, train_model_vals,
               train_ext_input_bxtxi=None, train_truth_bxtxd=None,
               valid_bxtxd=None, valid_model_vals=None,
               valid_ext_input_bxtxi=None, valid_truth_bxtxd=None,
               bidx=None, cf=1.0, output_dist='poisson'):

  # Plotting
  f = plt.figure(figsize=(18,20), tight_layout=True)
  plot_lfads_timeseries(train_bxtxd, train_model_vals,
                        train_ext_input_bxtxi,
                        truth_bxtxn=train_truth_bxtxd,
                        conversion_factor=cf, bidx=bidx,
                        output_dist=output_dist, col_title='Train')
  plot_lfads_timeseries(valid_bxtxd, valid_model_vals,
                        valid_ext_input_bxtxi,
                        truth_bxtxn=valid_truth_bxtxd,
                        conversion_factor=cf, bidx=bidx,
                        output_dist=output_dist,
                        subplot_cidx=1, col_title='Valid')

  # Convert from figure to an numpy array width x height x 3 (last for RGB)
  f.canvas.draw()
  data = np.fromstring(f.canvas.tostring_rgb(), dtype=np.uint8, sep='')
  data_wxhx3 = data.reshape(f.canvas.get_width_height()[::-1] + (3,))
  plt.close()

  return data_wxhx3
