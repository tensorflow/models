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

r"""Generaly Utilities.
"""

import numpy as np, cPickle, os, time
import src.file_utils as fu
import logging

class Timer():
  def __init__(self):
    self.calls = 0.
    self.start_time = 0.
    self.time_per_call = 0.
    self.total_time = 0.
    self.last_log_time = 0.

  def tic(self):
    self.start_time = time.time()

  def toc(self, average=True, log_at=-1, log_str='', type='calls'):
    if self.start_time == 0:
      logging.error('Timer not started by calling tic().')
    t = time.time()
    diff = time.time() - self.start_time
    self.total_time += diff
    self.calls += 1.
    self.time_per_call = self.total_time/self.calls

    if type == 'calls' and log_at > 0 and np.mod(self.calls, log_at) == 0:
      _ = []
      logging.info('%s: %f seconds.', log_str, self.time_per_call)
    elif type == 'time' and log_at > 0 and t - self.last_log_time >= log_at:
      _ = []
      logging.info('%s: %f seconds.', log_str, self.time_per_call)
      self.last_log_time = t

    if average:
      return self.time_per_call
    else:
      return diff

class Foo(object):
  def __init__(self, **kwargs):
    self.__dict__.update(kwargs)
  def __str__(self):
    str_ = ''
    for v in vars(self).keys():
      a = getattr(self, v)
      if True: #isinstance(v, object):
        str__ = str(a)
        str__ = str__.replace('\n', '\n  ')
      else:
        str__ = str(a)
      str_ += '{:s}: {:s}'.format(v, str__)
      str_ += '\n'
    return str_


def dict_equal(dict1, dict2):
  assert(set(dict1.keys()) == set(dict2.keys())), "Sets of keys between 2 dictionaries are different."
  for k in dict1.keys():
    assert(type(dict1[k]) == type(dict2[k])), "Type of key '{:s}' if different.".format(k)
    if type(dict1[k]) == np.ndarray:
      assert(dict1[k].dtype == dict2[k].dtype), "Numpy Type of key '{:s}' if different.".format(k)
      assert(np.allclose(dict1[k], dict2[k])), "Value for key '{:s}' do not match.".format(k)
    else:
      assert(dict1[k] == dict2[k]), "Value for key '{:s}' do not match.".format(k)
  return True

def subplot(plt, Y_X, sz_y_sz_x = (10, 10)):
  Y,X = Y_X
  sz_y, sz_x = sz_y_sz_x
  plt.rcParams['figure.figsize'] = (X*sz_x, Y*sz_y)
  fig, axes = plt.subplots(Y, X)
  plt.subplots_adjust(wspace=0.1, hspace=0.1)
  return fig, axes

def tic_toc_print(interval, string):
  global tic_toc_print_time_old
  if 'tic_toc_print_time_old' not in globals():
    tic_toc_print_time_old = time.time()
    print string
  else:
    new_time = time.time()
    if new_time - tic_toc_print_time_old > interval:
      tic_toc_print_time_old = new_time;
      print string

def mkdir_if_missing(output_dir):
  if not fu.exists(output_dir):
    fu.makedirs(output_dir)

def save_variables(pickle_file_name, var, info, overwrite = False):
  if fu.exists(pickle_file_name) and overwrite == False:
    raise Exception('{:s} exists and over write is false.'.format(pickle_file_name))
  # Construct the dictionary
  assert(type(var) == list); assert(type(info) == list);
  d = {}
  for i in xrange(len(var)):
    d[info[i]] = var[i]
  with fu.fopen(pickle_file_name, 'w') as f:
    cPickle.dump(d, f, cPickle.HIGHEST_PROTOCOL)

def load_variables(pickle_file_name):
  if fu.exists(pickle_file_name):
    with fu.fopen(pickle_file_name, 'r') as f:
      d = cPickle.load(f)
    return d
  else:
    raise Exception('{:s} does not exists.'.format(pickle_file_name))

def voc_ap(rec, prec):
  rec = rec.reshape((-1,1))
  prec = prec.reshape((-1,1))
  z = np.zeros((1,1)) 
  o = np.ones((1,1))
  mrec = np.vstack((z, rec, o))
  mpre = np.vstack((z, prec, z))
  for i in range(len(mpre)-2, -1, -1):
    mpre[i] = max(mpre[i], mpre[i+1])

  I = np.where(mrec[1:] != mrec[0:-1])[0]+1;
  ap = 0;
  for i in I:
    ap = ap + (mrec[i] - mrec[i-1])*mpre[i];
  return ap

def tight_imshow_figure(plt, figsize=None):
  fig = plt.figure(figsize=figsize)
  ax = plt.Axes(fig, [0,0,1,1])
  ax.set_axis_off()
  fig.add_axes(ax)
  return fig, ax

def calc_pr(gt, out, wt=None):
  if wt is None:
    wt = np.ones((gt.size,1))

  gt = gt.astype(np.float64).reshape((-1,1))
  wt = wt.astype(np.float64).reshape((-1,1))
  out = out.astype(np.float64).reshape((-1,1))

  gt = gt*wt
  tog = np.concatenate([gt, wt, out], axis=1)*1.
  ind = np.argsort(tog[:,2], axis=0)[::-1]
  tog = tog[ind,:]
  cumsumsortgt = np.cumsum(tog[:,0])
  cumsumsortwt = np.cumsum(tog[:,1])
  prec = cumsumsortgt / cumsumsortwt
  rec = cumsumsortgt / np.sum(tog[:,0])

  ap = voc_ap(rec, prec)
  return ap, rec, prec

