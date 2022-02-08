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
import json

import numpy as np
import tensorflow as tf


def log_sum_exp(x_k):
  """Computes log \sum exp in a numerically stable way.
    log ( sum_i exp(x_i) )
    log ( sum_i exp(x_i - m + m) ),       with m = max(x_i)
    log ( sum_i exp(x_i - m)*exp(m) )
    log ( sum_i exp(x_i - m) + m

  Args:
    x_k - k -dimensional list of arguments to log_sum_exp.

  Returns:
    log_sum_exp of the arguments.
  """
  m = tf.reduce_max(x_k)
  x1_k = x_k - m
  u_k = tf.exp(x1_k)
  z = tf.reduce_sum(u_k)
  return tf.log(z) + m


def linear(x, out_size, do_bias=True, alpha=1.0, identity_if_possible=False,
           normalized=False, name=None, collections=None):
  """Linear (affine) transformation, y = x W + b, for a variety of
  configurations.

  Args:
    x: input The tensor to tranformation.
    out_size: The integer size of non-batch output dimension.
    do_bias (optional): Add a learnable bias vector to the operation.
    alpha (optional): A multiplicative scaling for the weight initialization
      of the matrix, in the form \alpha * 1/\sqrt{x.shape[1]}.
    identity_if_possible (optional): just return identity,
      if x.shape[1] == out_size.
    normalized (optional): Option to divide out by the norms of the rows of W.
    name (optional): The name prefix to add to variables.
    collections (optional): List of additional collections. (Placed in
      tf.GraphKeys.GLOBAL_VARIABLES already, so no need for that.)

  Returns:
    In the equation, y = x W + b, returns the tensorflow op that yields y.
  """
  in_size = int(x.get_shape()[1]) # from Dimension(10) -> 10
  stddev = alpha/np.sqrt(float(in_size))
  mat_init = tf.random_normal_initializer(0.0, stddev)
  wname = (name + "/W") if name else "/W"

  if identity_if_possible and in_size == out_size:
    # Sometimes linear layers are nothing more than size adapters.
    return tf.identity(x, name=(wname+'_ident'))

  W,b = init_linear(in_size, out_size, do_bias=do_bias, alpha=alpha,
                    normalized=normalized, name=name, collections=collections)

  if do_bias:
    return tf.matmul(x, W) + b
  else:
    return tf.matmul(x, W)


def init_linear(in_size, out_size, do_bias=True, mat_init_value=None,
                bias_init_value=None, alpha=1.0, identity_if_possible=False,
                normalized=False, name=None, collections=None, trainable=True):
  """Linear (affine) transformation, y = x W + b, for a variety of
  configurations.

  Args:
    in_size: The integer size of the non-batc input dimension. [(x),y]
    out_size: The integer size of non-batch output dimension. [x,(y)]
    do_bias (optional): Add a (learnable) bias vector to the operation,
      if false, b will be None
    mat_init_value (optional): numpy constant for matrix initialization, if None
      , do random, with additional parameters.
    alpha (optional): A multiplicative scaling for the weight initialization
      of the matrix, in the form \alpha * 1/\sqrt{x.shape[1]}.
    identity_if_possible (optional): just return identity,
      if x.shape[1] == out_size.
    normalized (optional): Option to divide out by the norms of the rows of W.
    name (optional): The name prefix to add to variables.
    collections (optional): List of additional collections. (Placed in
      tf.GraphKeys.GLOBAL_VARIABLES already, so no need for that.)

  Returns:
    In the equation, y = x W + b, returns the pair (W, b).
  """

  if mat_init_value is not None and mat_init_value.shape != (in_size, out_size):
    raise ValueError(
        'Provided mat_init_value must have shape [%d, %d].'%(in_size, out_size))
  if bias_init_value is not None and bias_init_value.shape != (1,out_size):
    raise ValueError(
        'Provided bias_init_value must have shape [1,%d].'%(out_size,))

  if mat_init_value is None:
    stddev = alpha/np.sqrt(float(in_size))
    mat_init = tf.random_normal_initializer(0.0, stddev)

  wname = (name + "/W") if name else "/W"

  if identity_if_possible and in_size == out_size:
    return (tf.constant(np.eye(in_size).astype(np.float32)),
            tf.zeros(in_size))

  # Note the use of get_variable vs. tf.Variable.  this is because get_variable
  # does not allow the initialization of the variable with a value.
  if normalized:
    w_collections = [tf.GraphKeys.GLOBAL_VARIABLES, "norm-variables"]
    if collections:
      w_collections += collections
    if mat_init_value is not None:
      w = tf.Variable(mat_init_value, name=wname, collections=w_collections,
                      trainable=trainable)
    else:
      w = tf.get_variable(wname, [in_size, out_size], initializer=mat_init,
                          collections=w_collections, trainable=trainable)
    w = tf.nn.l2_normalize(w, dim=0) # x W, so xW_j = \sum_i x_bi W_ij
  else:
    w_collections = [tf.GraphKeys.GLOBAL_VARIABLES]
    if collections:
      w_collections += collections
    if mat_init_value is not None:
      w = tf.Variable(mat_init_value, name=wname, collections=w_collections,
                      trainable=trainable)
    else:
      w = tf.get_variable(wname, [in_size, out_size], initializer=mat_init,
                          collections=w_collections, trainable=trainable)
  b = None
  if do_bias:
    b_collections = [tf.GraphKeys.GLOBAL_VARIABLES]
    if collections:
      b_collections += collections
    bname = (name + "/b") if name else "/b"
    if bias_init_value is None:
      b = tf.get_variable(bname, [1, out_size],
                          initializer=tf.zeros_initializer(),
                          collections=b_collections,
                          trainable=trainable)
    else:
      b = tf.Variable(bias_init_value, name=bname,
                      collections=b_collections,
                      trainable=trainable)

  return (w, b)


def write_data(data_fname, data_dict, use_json=False, compression=None):
  """Write data in HD5F format.

  Args:
    data_fname: The filename of teh file in which to write the data.
    data_dict:  The dictionary of data to write. The keys are strings
      and the values are numpy arrays.
    use_json (optional): human readable format for simple items
    compression (optional): The compression to use for h5py (disabled by
      default because the library borks on scalars, otherwise try 'gzip').
  """

  dir_name = os.path.dirname(data_fname)
  if not os.path.exists(dir_name):
    os.makedirs(dir_name)

  if use_json:
    the_file = open(data_fname,'wb')
    json.dump(data_dict, the_file)
    the_file.close()
  else:
    try:
      with h5py.File(data_fname, 'w') as hf:
        for k, v in data_dict.items():
          clean_k = k.replace('/', '_')
          if clean_k is not k:
            print('Warning: saving variable with name: ', k, ' as ', clean_k)
          else:
            print('Saving variable with name: ', clean_k)
          hf.create_dataset(clean_k, data=v, compression=compression)
    except IOError:
      print("Cannot open %s for writing.", data_fname)
      raise


def read_data(data_fname):
  """ Read saved data in HDF5 format.

  Args:
    data_fname: The filename of the file from which to read the data.
  Returns:
    A dictionary whose keys will vary depending on dataset (but should
    always contain the keys 'train_data' and 'valid_data') and whose
    values are numpy arrays.
  """

  try:
    with h5py.File(data_fname, 'r') as hf:
      data_dict = {k: np.array(v) for k, v in hf.items()}
      return data_dict
  except IOError:
    print("Cannot open %s for reading." % data_fname)
    raise


def write_datasets(data_path, data_fname_stem, dataset_dict, compression=None):
  """Write datasets in HD5F format.

  This function assumes the dataset_dict is a mapping ( string ->
  to data_dict ).  It calls write_data for each data dictionary,
  post-fixing the data filename with the key of the dataset.

  Args:
    data_path: The path to the save directory.
    data_fname_stem: The filename stem of the file in which to write the data.
    dataset_dict:  The dictionary of datasets. The keys are strings
      and the values data dictionaries (str -> numpy arrays) associations.
    compression (optional): The compression to use for h5py (disabled by
      default because the library borks on scalars, otherwise try 'gzip').
  """

  full_name_stem = os.path.join(data_path, data_fname_stem)
  for s, data_dict in dataset_dict.items():
    write_data(full_name_stem + "_" + s, data_dict, compression=compression)


def read_datasets(data_path, data_fname_stem):
  """Read dataset sin HD5F format.

  This function assumes the dataset_dict is a mapping ( string ->
  to data_dict ).  It calls write_data for each data dictionary,
  post-fixing the data filename with the key of the dataset.

  Args:
    data_path: The path to the save directory.
    data_fname_stem: The filename stem of the file in which to write the data.
  """

  dataset_dict = {}
  fnames = os.listdir(data_path)

  print ('loading data from ' + data_path + ' with stem ' + data_fname_stem)
  for fname in fnames:
    if fname.startswith(data_fname_stem):
      data_dict = read_data(os.path.join(data_path,fname))
      idx = len(data_fname_stem) + 1
      key = fname[idx:]
      data_dict['data_dim'] = data_dict['train_data'].shape[2]
      data_dict['num_steps'] = data_dict['train_data'].shape[1]
      dataset_dict[key] = data_dict

  if len(dataset_dict) == 0:
    raise ValueError("Failed to load any datasets, are you sure that the "
                     "'--data_dir' and '--data_filename_stem' flag values "
                     "are correct?")

  print (str(len(dataset_dict)) + ' datasets loaded')
  return dataset_dict


# NUMPY utility functions
def list_t_bxn_to_list_b_txn(values_t_bxn):
  """Convert a length T list of BxN numpy tensors of length B list of TxN numpy
  tensors.

  Args:
    values_t_bxn: The length T list of BxN numpy tensors.

  Returns:
    The length B list of TxN numpy tensors.
  """
  T = len(values_t_bxn)
  B, N = values_t_bxn[0].shape
  values_b_txn = []
  for b in range(B):
    values_pb_txn = np.zeros([T,N])
    for t in range(T):
      values_pb_txn[t,:] = values_t_bxn[t][b,:]
    values_b_txn.append(values_pb_txn)

  return values_b_txn


def list_t_bxn_to_tensor_bxtxn(values_t_bxn):
  """Convert a length T list of BxN numpy tensors to single numpy tensor with
  shape BxTxN.

  Args:
    values_t_bxn: The length T list of BxN numpy tensors.

  Returns:
    values_bxtxn: The BxTxN numpy tensor.
  """

  T = len(values_t_bxn)
  B, N = values_t_bxn[0].shape
  values_bxtxn = np.zeros([B,T,N])
  for t in range(T):
    values_bxtxn[:,t,:] = values_t_bxn[t]

  return values_bxtxn


def tensor_bxtxn_to_list_t_bxn(tensor_bxtxn):
  """Convert a numpy tensor with shape BxTxN to a length T list of numpy tensors
  with shape BxT.

  Args:
    tensor_bxtxn: The BxTxN numpy tensor.

  Returns:
    A length T list of numpy tensors with shape BxT.
  """

  values_t_bxn = []
  B, T, N = tensor_bxtxn.shape
  for t in range(T):
    values_t_bxn.append(np.squeeze(tensor_bxtxn[:,t,:]))

  return values_t_bxn


def flatten(list_of_lists):
  """Takes a list of lists and returns a list of the elements.

  Args:
    list_of_lists: List of lists.

  Returns:
    flat_list: Flattened list.
    flat_list_idxs: Flattened list indices.
  """
  flat_list = []
  flat_list_idxs = []
  start_idx = 0
  for item in list_of_lists:
    if isinstance(item, list):
      flat_list += item
      l = len(item)
      idxs = range(start_idx, start_idx+l)
      start_idx = start_idx+l
    else:                   # a value
      flat_list.append(item)
      idxs = [start_idx]
      start_idx += 1
    flat_list_idxs.append(idxs)

  return flat_list, flat_list_idxs
