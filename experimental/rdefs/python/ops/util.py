# Copyright 2016 Google Inc. All Rights Reserved.
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
"""Utility functions.
"""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os
import sys

import h5py
import numpy as np
import tensorflow as tf

st = tf.contrib.bayesflow.stochastic_tensor
distributions = tf.contrib.distributions


def provide_tfrecords_data(path, split_name, batch_size, n_timesteps,
                           timestep_dim):
  """Provides batches of MNIST digits.

  Args:
    path: String specifying location of tf.records files.
    split_name: string. name of the split.
    batch_size: int. batch size.
    n_timesteps: int. number of timesteps.
    timestep_dim: int. dimension of each timestep.

  Returns:
    labels: minibatch tensor of the indices of each datapoint.
    images: minibatch tensor of images.
  """
  # Load the data:
  image, label = read_and_decode_single_example(
      os.path.join(path, 'binarized_mnist_{}.tfrecords'.format(split_name)))

  # Preprocess the images.
  image = tf.reshape(image, [28, 28])
  if n_timesteps < 28:
    image = image[0:n_timesteps, :]
  if timestep_dim < 28:
    image = image[:, 0:timestep_dim]
  image = tf.expand_dims(image, 2)

  # Creates a QueueRunner for the pre-fetching operation.
  images, labels = tf.train.batch(
      [image, label],
      batch_size=batch_size,
      num_threads=15,
      capacity=batch_size * 5000)

  return labels, images


def read_and_decode_single_example(filename):
  """Read and decode a single example.

  Args:
    filename: str. path to a tf.records file.

  Returns:
    image: tensor. a single image.
    label: tensor. the index for the image.
  """
  # first construct a queue containing a list of filenames.
  # this lets a user split up there dataset in multiple files to keep
  # size down
  filename_queue = tf.train.string_input_producer([filename],
                                                  num_epochs=None)
  # Unlike the TFRecordWriter, the TFRecordReader is symbolic
  reader = tf.TFRecordReader()
  # One can read a single serialized example from a filename
  # serialized_example is a Tensor of type string.
  _, serialized_example = reader.read(filename_queue)
  # The serialized example is converted back to actual values.
  # One needs to describe the format of the objects to be returned
  features = tf.parse_single_example(
      serialized_example,
      features={
          # We know the length of both fields. If not the
          # tf.VarLenFeature could be used
          'image': tf.FixedLenFeature([784], tf.float32),
          'label': tf.FixedLenFeature([], tf.int64)
      })
  # now return the converted data
  image = features['image']
  label = features['label']
  return image, label


def provide_hdf5_data(path, split_name, n_examples, batch_size, n_timesteps,
                      timestep_dim, dataset):
  """Provides batches of MNIST digits.

  Args:
   path: str. path to the  dataset.
   split_name: string. name of the split.
   n_examples: int. number of examples to serve from the dataset.
   batch_size: int. batch size.
   n_timesteps: int. number of timesteps.
   timestep_dim: int. dimension of each timestep.
   dataset: String specifying dataset.

  Returns:
    data_iterator: a generator of minibatches.
  """
  if dataset == 'alternating':
    data_list = []
    start_zeros = np.vstack([np.zeros(timestep_dim) if t % 2 == 0 else
                             np.ones(timestep_dim) for t in range(n_timesteps)])
    start_ones = np.roll(start_zeros, 1, axis=0)
    start_zeros = start_zeros.flatten()
    start_ones = start_ones.flatten()
    data_list = [start_zeros if n % 2 == 0 else
                 start_ones for n in range(n_examples)]
    data = np.vstack(data_list)
  elif dataset == 'MNIST':
    f = h5py.File(path, 'r')
    if split_name == 'train_and_valid':
      train = f['train'][:]
      valid = f['valid'][:]
      data = np.vstack([train, valid])
    else:
      data = f[split_name][:]

    data = data[0:n_examples]

  # create indexes for the data points.
  indexed_data = zip(range(len(data)), np.split(data, len(data)))
  def data_iterator():
    """Generate minibatches of examples from the dataset."""
    batch_idx = 0
    while True:
      # shuffle data
      idxs = np.arange(0, len(data))
      np.random.shuffle(idxs)
      shuf_data = [indexed_data[idx] for idx in idxs]
      for batch_idx in range(0, len(data), batch_size):
        indexed_images_batch = shuf_data[batch_idx:batch_idx+batch_size]
        indexes, images_batch = zip(*indexed_images_batch)
        images_batch = np.vstack(images_batch)
        if timestep_dim == 784:
          images_batch = images_batch.reshape(
              (batch_size, 1, 784, 1))
        else:
          if dataset == 'alternating':
            images_batch = images_batch.reshape(
                (batch_size, n_timesteps, timestep_dim, 1))
          else:
            images_batch = images_batch.reshape(
                (batch_size, 28, 28, 1))[:, :n_timesteps, :timestep_dim]
        yield indexes, images_batch

  return data_iterator()


def inv_softplus(x):
  """Inverse softplus."""
  return np.log(np.exp(x) - 1.)


def softplus(x):
  """Softplus."""
  return np.log(np.exp(x) + 1.)


def build_gamma(shape, init_shape=1., init_mean=1., x_indexes=None,
                fixed_mean=False, place_on_cpu=False, n_samples=1,
                dtype='float64'):
  """Builds a Gaussian DistributionTensor.

  Truncation: we truncate shape and mean parameters because gamma sampling is
  numerically unstable. Reference: http://ajbc.io/resources/bbvi_for_gammas.pdf

  Args:
    shape: list. shape of the distribution.
    init_shape: float. initial shape
    init_mean: float. initial standard deviation
    x_indexes: tensor. integer placeholder for mean-field parameters
    fixed_mean: bool. whether to learn mean
    place_on_cpu: bool. whether to place the op on cpu.
    n_samples: number of samples
    dtype: dtype

  Returns:
    A Gaussian DistributionTensor of the specified shape, with variables for
    mean and standard deviation safely parametrized to avoid over/underflow.
  """
  if place_on_cpu:
    with tf.device('/cpu:0'):
      shape_softplus_inv = tf.get_variable(
          'shape_softplus_inv', shape, dtype, tf.constant_initializer(
              inv_softplus(init_shape)), collections=[tf.GraphKeys.VARIABLES,
                                                      'non_reparam_variables'])
  else:
    shape_softplus_inv = tf.get_variable(
        'shape_softplus_inv', shape, dtype, tf.constant_initializer(
            inv_softplus(init_shape)), collections=[tf.GraphKeys.VARIABLES,
                                                    'non_reparam_variables'])
  if fixed_mean:
    mean_softplus_inv = None
  else:
    mean_softplus_arg = tf.constant_initializer(inv_softplus(init_mean))
    if place_on_cpu:
      with tf.device('/cpu:0'):
        mean_softplus_inv = tf.get_variable(
            'mean_softplus_inv', shape, dtype, mean_softplus_arg)
    else:
      mean_softplus_inv = tf.get_variable('mean_softplus_inv', shape,
                                          dtype, mean_softplus_arg,
                                          collections=[tf.GraphKeys.VARIABLES,
                                                       'non_reparam_variables'])

  if x_indexes is not None:
    shape_softplus_inv_batch = tf.nn.embedding_lookup(
        shape_softplus_inv, x_indexes)
    if not fixed_mean:
      mean_softplus_inv_batch = tf.nn.embedding_lookup(
          mean_softplus_inv, x_indexes)
  else:
    shape_softplus_inv_batch, mean_softplus_inv_batch = (shape_softplus_inv,
                                                         mean_softplus_inv)
  shape_batch = tf.nn.softplus(shape_softplus_inv_batch)

  if fixed_mean:
    mean_batch = tf.constant(init_mean)
  else:
    mean_batch = tf.nn.softplus(mean_softplus_inv_batch)

  with st.value_type(st.SampleValue(n=n_samples)):
    dist = st.StochasticTensor(distributions.Gamma,
                               alpha=shape_batch,
                               beta=shape_batch / mean_batch,
                               validate_args=False)
  return dist


def truncate(max_or_min, var, val):
  """Truncate variable to a max or min value."""
  if max_or_min == 'max':
    tf_fn = tf.minimum
  elif max_or_min == 'min':
    tf_fn = tf.maximum

  if isinstance(var, tf.IndexedSlices):
    assign_op = tf.assign(var.values, tf_fn(var.values, inv_softplus(val)))
  else:
    assign_op = tf.assign(var, tf_fn(var, inv_softplus(val)))
  return assign_op


def build_gaussian(shape, init_mu=0., init_sigma=1.0, x_indexes=None,
                   fixed_sigma=False, place_on_cpu=False, dtype='float64'):
  """Builds a Gaussian DistributionTensor.

  Args:
    shape: list. shape of the distribution.
    init_mu: float. initial mean
    init_sigma: float. initial standard deviation
    x_indexes: tensor. integer placeholder for mean-field parameters
    fixed_sigma: bool. whether to learn sigma
    place_on_cpu: bool. whether to place the op on cpu.
    dtype: dtpe

  Returns:
    A Gaussian DistributionTensor of the specified shape, with variables for
    mean and standard deviation safely parametrized to avoid over/underflow.
  """
  if place_on_cpu:
    with tf.device('/cpu:0'):
      mu = tf.get_variable(
          'mu', shape, dtype, tf.random_normal_initializer(
              mean=init_mu, stddev=0.1))
  else:
    mu = tf.get_variable('mu', shape, dtype,
                         tf.random_normal_initializer(mean=init_mu, stddev=0.1),
                         collections=[tf.GraphKeys.VARIABLES,
                                      'reparam_variables'])
  if fixed_sigma:
    sigma_softplus_inv = None
  else:
    sigma_softplus_arg = tf.truncated_normal_initializer(
        mean=inv_softplus(init_sigma), stddev=0.1)
    if place_on_cpu:
      with tf.device('/cpu:0'):
        sigma_softplus_inv = tf.get_variable(
            'sigma_softplus_inv', shape, dtype, sigma_softplus_arg)
    else:
      sigma_softplus_inv = tf.get_variable('sigma_softplus_inv', shape,
                                           dtype, sigma_softplus_arg,
                                           collections=[tf.GraphKeys.VARIABLES,
                                                        'reparam_variables'])

  if x_indexes is not None:
    mu_batch = tf.nn.embedding_lookup(mu, x_indexes)
    if not fixed_sigma:
      sigma_softplus_inv_batch = tf.nn.embedding_lookup(
          sigma_softplus_inv, x_indexes)
  else:
    mu_batch, sigma_softplus_inv_batch = mu, sigma_softplus_inv

  if fixed_sigma:
    sigma_batch = np.array(init_sigma, dtype)
  else:
    sigma_batch = tf.maximum(tf.nn.softplus(sigma_softplus_inv_batch), 1e-5)

  dist = st.StochasticTensor(distributions.Normal, mu=mu_batch,
                             sigma=sigma_batch, validate_args=False)
  return dist


def get_np_dtype(tensor):
  """Returns the numpy dtype."""
  return np.float32 if 'float32' in str(tensor.dtype) else np.float64


def build_bernoulli_log_likelihood(params, x, batch_size,
                                   n_samples_latents=1,
                                   use_bias_observations=False):
  """Builds the likelihood given stochastic latents and weights.

  Args:
    params: dict that contains:
        z_1 tensor. sampled latent variables
          [n_samples_latents] + [batch_size, n_timesteps, z_dim]
        w_0 tensor. sampled stochastic weights [z_dim, timestep_dim]
        b_0 optional tensor. biases [timestep_dim]
    x: tensor. minibatch of examples
    batch_size: integer number of minibatch examples.
    n_samples_latents: number of samples of latent variables
    use_bias_observations: use bias

  Returns:
    likelihood: the bernoulli likelihood distribution of the data.
        [n_samples, batch_size, n_timesteps, timestep_dim]
  """
  z_1 = params['z_1']
  w_0 = params['w_0']
  if use_bias_observations:
    b_0 = params['b_0']
  if n_samples_latents > 1:
    wz = tf.batch_matmul(z_1, tf.pack([tf.pack([w_0] * batch_size)]
                                      * n_samples_latents))
    if use_bias_observations:
      wz += b_0
    logits = tf.expand_dims(wz, 4)
    dims_to_reduce = [2, 3, 4]
  else:
    wz = tf.batch_matmul(z_1, tf.pack([w_0] * batch_size))
    if use_bias_observations:
      wz += b_0
    logits = tf.expand_dims(wz, 3)
    dims_to_reduce = [1, 2, 3]
  p_x_zw = distributions.Bernoulli(logits=logits, validate_args=False)
  log_p_x_zw = tf.reduce_sum(p_x_zw.log_pmf(x), dims_to_reduce)
  print('log_p_x_zw', log_p_x_zw.get_shape())
  print('logits', logits.get_shape())
  print('z_1', z_1.value().get_shape())
  return log_p_x_zw, p_x_zw.p


def clip_mean(mean):
  """Clip mean parameter of gamma."""
  return tf.clip_by_value(mean, clip_value_max=sys.float_info.max,
                          clip_value_min=1e-5)


def clip_shape(shape):
  """Clip shape parameter of gamma."""
  return tf.clip_by_value(shape, clip_value_max=sys.float_info.max,
                          clip_value_min=5e-3)


def bernoulli_likelihood_sample(params, z_1, n_samples,
                                use_bias_observations=False):
  """Sample from the model likelihood.

  Args:
    params: dict that contains
        w_0 tensor. sample of latent weights
        b_0 optional tensor. bias
    z_1: tensor. sample of latent variables
    n_samples: int. number of samples to draw
    use_bias_observations: use bias

  Returns:
    A tensor sample from the model likelihood.
  """
  w_0 = params['w_0']
  if isinstance(z_1, st.StochasticTensor):
    z_1 = z_1.value()
  if z_1.get_shape().ndims == 4:
    z_1 = z_1[0, :, :, :]
  wz = tf.batch_matmul(z_1, tf.pack([w_0] * n_samples))
  if use_bias_observations:
    wz += params['b_0']
  logits = tf.expand_dims(wz, 3)
  p_x_zw = distributions.Bernoulli(logits=logits, validate_args=False)
  return tf.cast(p_x_zw.sample_n(n=1)[0, :, :, :, :], logits.dtype)
