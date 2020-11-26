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

r"""Utility functions for Real NVP.
"""

# pylint: disable=dangerous-default-value

import numpy
from six.moves import xrange
import tensorflow as tf
from tensorflow.python.framework import ops

DEFAULT_BN_LAG = .0


def stable_var(input_, mean=None, axes=[0]):
    """Numerically more stable variance computation."""
    if mean is None:
        mean = tf.reduce_mean(input_, axes)
    res = tf.square(input_ - mean)
    max_sqr = tf.reduce_max(res, axes)
    res /= max_sqr
    res = tf.reduce_mean(res, axes)
    res *= max_sqr

    return res


def variable_on_cpu(name, shape, initializer, trainable=True):
    """Helper to create a Variable stored on CPU memory.

    Args:
            name: name of the variable
            shape: list of ints
            initializer: initializer for Variable
            trainable: boolean defining if the variable is for training
    Returns:
            Variable Tensor
    """
    var = tf.get_variable(
        name, shape, initializer=initializer, trainable=trainable)
    return var


# layers
def conv_layer(input_,
               filter_size,
               dim_in,
               dim_out,
               name,
               stddev=1e-2,
               strides=[1, 1, 1, 1],
               padding="SAME",
               nonlinearity=None,
               bias=False,
               weight_norm=False,
               scale=False):
    """Convolutional layer."""
    with tf.variable_scope(name) as scope:
        weights = variable_on_cpu(
            "weights",
            filter_size + [dim_in, dim_out],
            tf.random_uniform_initializer(
                minval=-stddev, maxval=stddev))
        # weight normalization
        if weight_norm:
            weights /= tf.sqrt(tf.reduce_sum(tf.square(weights), [0, 1, 2]))
            if scale:
                magnitude = variable_on_cpu(
                    "magnitude", [dim_out],
                    tf.constant_initializer(
                        stddev * numpy.sqrt(dim_in * numpy.prod(filter_size) / 12.)))
                weights *= magnitude
        res = input_
        # handling filter size bigger than image size
        if hasattr(input_, "shape"):
            if input_.get_shape().as_list()[1] < filter_size[0]:
                pad_1 = tf.zeros([
                    input_.get_shape().as_list()[0],
                    filter_size[0] - input_.get_shape().as_list()[1],
                    input_.get_shape().as_list()[2],
                    input_.get_shape().as_list()[3]
                ])
                pad_2 = tf.zeros([
                    input_.get_shape().as_list[0],
                    filter_size[0],
                    filter_size[1] - input_.get_shape().as_list()[2],
                    input_.get_shape().as_list()[3]
                ])
                res = tf.concat(axis=1, values=[pad_1, res])
                res = tf.concat(axis=2, values=[pad_2, res])
        res = tf.nn.conv2d(
            input=res,
            filter=weights,
            strides=strides,
            padding=padding,
            name=scope.name)

        if hasattr(input_, "shape"):
            if input_.get_shape().as_list()[1] < filter_size[0]:
                res = tf.slice(res, [
                    0, filter_size[0] - input_.get_shape().as_list()[1],
                    filter_size[1] - input_.get_shape().as_list()[2], 0
                ], [-1, -1, -1, -1])

        if bias:
            biases = variable_on_cpu("biases", [dim_out], tf.constant_initializer(0.))
            res = tf.nn.bias_add(res, biases)
        if nonlinearity is not None:
            res = nonlinearity(res)

    return res


def max_pool_2x2(input_):
    """Max pooling."""
    return tf.nn.max_pool(
        input_, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding="SAME")


def depool_2x2(input_, stride=2):
    """Depooling."""
    shape = input_.get_shape().as_list()
    batch_size = shape[0]
    height = shape[1]
    width = shape[2]
    channels = shape[3]
    res = tf.reshape(input_, [batch_size, height, 1, width, 1, channels])
    res = tf.concat(
        axis=2, values=[res, tf.zeros([batch_size, height, stride - 1, width, 1, channels])])
    res = tf.concat(axis=4, values=[
        res, tf.zeros([batch_size, height, stride, width, stride - 1, channels])
    ])
    res = tf.reshape(res, [batch_size, stride * height, stride * width, channels])

    return res


# random flip on a batch of images
def batch_random_flip(input_):
    """Simultaneous horizontal random flip."""
    if isinstance(input_, (float, int)):
        return input_
    shape = input_.get_shape().as_list()
    batch_size = shape[0]
    height = shape[1]
    width = shape[2]
    channels = shape[3]
    res = tf.split(axis=0, num_or_size_splits=batch_size, value=input_)
    res = [elem[0, :, :, :] for elem in res]
    res = [tf.image.random_flip_left_right(elem) for elem in res]
    res = [tf.reshape(elem, [1, height, width, channels]) for elem in res]
    res = tf.concat(axis=0, values=res)

    return res


# build a one hot representation corresponding to the integer tensor
# the one-hot dimension is appended to the integer tensor shape
def as_one_hot(input_, n_indices):
    """Convert indices to one-hot."""
    shape = input_.get_shape().as_list()
    n_elem = numpy.prod(shape)
    indices = tf.range(n_elem)
    indices = tf.cast(indices, tf.int64)
    indices_input = tf.concat(axis=0, values=[indices, tf.reshape(input_, [-1])])
    indices_input = tf.reshape(indices_input, [2, -1])
    indices_input = tf.transpose(indices_input)
    res = tf.sparse_to_dense(
        indices_input, [n_elem, n_indices], 1., 0., name="flat_one_hot")
    res = tf.reshape(res, [elem for elem in shape] + [n_indices])

    return res


def squeeze_2x2(input_):
    """Squeezing operation: reshape to convert space to channels."""
    return squeeze_nxn(input_, n_factor=2)


def squeeze_nxn(input_, n_factor=2):
    """Squeezing operation: reshape to convert space to channels."""
    if isinstance(input_, (float, int)):
        return input_
    shape = input_.get_shape().as_list()
    batch_size = shape[0]
    height = shape[1]
    width = shape[2]
    channels = shape[3]
    if height % n_factor != 0:
        raise ValueError("Height not divisible by %d." % n_factor)
    if width % n_factor != 0:
        raise ValueError("Width not divisible by %d." % n_factor)
    res = tf.reshape(
        input_,
        [batch_size,
         height // n_factor,
         n_factor, width // n_factor,
         n_factor, channels])
    res = tf.transpose(res, [0, 1, 3, 5, 2, 4])
    res = tf.reshape(
        res,
        [batch_size,
         height // n_factor,
         width // n_factor,
         channels * n_factor * n_factor])

    return res


def unsqueeze_2x2(input_):
    """Unsqueezing operation: reshape to convert channels into space."""
    if isinstance(input_, (float, int)):
        return input_
    shape = input_.get_shape().as_list()
    batch_size = shape[0]
    height = shape[1]
    width = shape[2]
    channels = shape[3]
    if channels % 4 != 0:
        raise ValueError("Number of channels not divisible by 4.")
    res = tf.reshape(input_, [batch_size, height, width, channels // 4, 2, 2])
    res = tf.transpose(res, [0, 1, 4, 2, 5, 3])
    res = tf.reshape(res, [batch_size, 2 * height, 2 * width, channels // 4])

    return res


# batch norm
def batch_norm(input_,
               dim,
               name,
               scale=True,
               train=True,
               epsilon=1e-8,
               decay=.1,
               axes=[0],
               bn_lag=DEFAULT_BN_LAG):
    """Batch normalization."""
    # create variables
    with tf.variable_scope(name):
        var = variable_on_cpu(
            "var", [dim], tf.constant_initializer(1.), trainable=False)
        mean = variable_on_cpu(
            "mean", [dim], tf.constant_initializer(0.), trainable=False)
        step = variable_on_cpu("step", [], tf.constant_initializer(0.), trainable=False)
        if scale:
            gamma = variable_on_cpu("gamma", [dim], tf.constant_initializer(1.))
        beta = variable_on_cpu("beta", [dim], tf.constant_initializer(0.))
    # choose the appropriate moments
    if train:
        used_mean, used_var = tf.nn.moments(input_, axes, name="batch_norm")
        cur_mean, cur_var = used_mean, used_var
        if bn_lag > 0.:
            used_mean -= (1. - bn_lag) * (used_mean - tf.stop_gradient(mean))
            used_var -= (1 - bn_lag) * (used_var - tf.stop_gradient(var))
            used_mean /= (1. - bn_lag**(step + 1))
            used_var /= (1. - bn_lag**(step + 1))
    else:
        used_mean, used_var = mean, var
        cur_mean, cur_var = used_mean, used_var

    # normalize
    res = (input_ - used_mean) / tf.sqrt(used_var + epsilon)
    # de-normalize
    if scale:
        res *= gamma
    res += beta

    # update variables
    if train:
        with tf.name_scope(name, "AssignMovingAvg", [mean, cur_mean, decay]):
            with ops.colocate_with(mean):
                new_mean = tf.assign_sub(
                    mean,
                    tf.check_numerics(decay * (mean - cur_mean), "NaN in moving mean."))
        with tf.name_scope(name, "AssignMovingAvg", [var, cur_var, decay]):
            with ops.colocate_with(var):
                new_var = tf.assign_sub(
                    var,
                    tf.check_numerics(decay * (var - cur_var),
                                      "NaN in moving variance."))
        with tf.name_scope(name, "IncrementTime", [step]):
            with ops.colocate_with(step):
                new_step = tf.assign_add(step, 1.)
        res += 0. * new_mean * new_var * new_step

    return res


# batch normalization taking into account the volume transformation
def batch_norm_log_diff(input_,
                        dim,
                        name,
                        train=True,
                        epsilon=1e-8,
                        decay=.1,
                        axes=[0],
                        reuse=None,
                        bn_lag=DEFAULT_BN_LAG):
    """Batch normalization with corresponding log determinant Jacobian."""
    if reuse is None:
        reuse = not train
    # create variables
    with tf.variable_scope(name) as scope:
        if reuse:
            scope.reuse_variables()
        var = variable_on_cpu(
            "var", [dim], tf.constant_initializer(1.), trainable=False)
        mean = variable_on_cpu(
            "mean", [dim], tf.constant_initializer(0.), trainable=False)
        step = variable_on_cpu("step", [], tf.constant_initializer(0.), trainable=False)
    # choose the appropriate moments
    if train:
        used_mean, used_var = tf.nn.moments(input_, axes, name="batch_norm")
        cur_mean, cur_var = used_mean, used_var
        if bn_lag > 0.:
            used_var = stable_var(input_=input_, mean=used_mean, axes=axes)
            cur_var = used_var
            used_mean -= (1 - bn_lag) * (used_mean - tf.stop_gradient(mean))
            used_mean /= (1. - bn_lag**(step + 1))
            used_var -= (1 - bn_lag) * (used_var - tf.stop_gradient(var))
            used_var /= (1. - bn_lag**(step + 1))
    else:
        used_mean, used_var = mean, var
        cur_mean, cur_var = used_mean, used_var

    # update variables
    if train:
        with tf.name_scope(name, "AssignMovingAvg", [mean, cur_mean, decay]):
            with ops.colocate_with(mean):
                new_mean = tf.assign_sub(
                    mean,
                    tf.check_numerics(
                        decay * (mean - cur_mean), "NaN in moving mean."))
        with tf.name_scope(name, "AssignMovingAvg", [var, cur_var, decay]):
            with ops.colocate_with(var):
                new_var = tf.assign_sub(
                    var,
                    tf.check_numerics(decay * (var - cur_var),
                                      "NaN in moving variance."))
        with tf.name_scope(name, "IncrementTime", [step]):
            with ops.colocate_with(step):
                new_step = tf.assign_add(step, 1.)
        used_var += 0. * new_mean * new_var * new_step
    used_var += epsilon

    return used_mean, used_var


def convnet(input_,
            dim_in,
            dim_hid,
            filter_sizes,
            dim_out,
            name,
            use_batch_norm=True,
            train=True,
            nonlinearity=tf.nn.relu):
    """Chaining of convolutional layers."""
    dims_in = [dim_in] + dim_hid[:-1]
    dims_out = dim_hid
    res = input_

    bias = (not use_batch_norm)
    with tf.variable_scope(name):
        for layer_idx in xrange(len(dim_hid)):
            res = conv_layer(
                input_=res,
                filter_size=filter_sizes[layer_idx],
                dim_in=dims_in[layer_idx],
                dim_out=dims_out[layer_idx],
                name="h_%d" % layer_idx,
                stddev=1e-2,
                nonlinearity=None,
                bias=bias)
            if use_batch_norm:
                res = batch_norm(
                    input_=res,
                    dim=dims_out[layer_idx],
                    name="bn_%d" % layer_idx,
                    scale=(nonlinearity == tf.nn.relu),
                    train=train,
                    epsilon=1e-8,
                    axes=[0, 1, 2])
            if nonlinearity is not None:
                res = nonlinearity(res)

        res = conv_layer(
            input_=res,
            filter_size=filter_sizes[-1],
            dim_in=dims_out[-1],
            dim_out=dim_out,
            name="out",
            stddev=1e-2,
            nonlinearity=None)

    return res


# distributions
# log-likelihood estimation
def standard_normal_ll(input_):
    """Log-likelihood of standard Gaussian distribution."""
    res = -.5 * (tf.square(input_) + numpy.log(2. * numpy.pi))

    return res


def standard_normal_sample(shape):
    """Samples from standard Gaussian distribution."""
    return tf.random_normal(shape)


SQUEEZE_MATRIX = numpy.array([[[[1., 0., 0., 0.]], [[0., 0., 1., 0.]]],
                              [[[0., 0., 0., 1.]], [[0., 1., 0., 0.]]]])


def squeeze_2x2_ordered(input_, reverse=False):
    """Squeezing operation with a controlled ordering."""
    shape = input_.get_shape().as_list()
    batch_size = shape[0]
    height = shape[1]
    width = shape[2]
    channels = shape[3]
    if reverse:
        if channels % 4 != 0:
            raise ValueError("Number of channels not divisible by 4.")
        channels /= 4
    else:
        if height % 2 != 0:
            raise ValueError("Height not divisible by 2.")
        if width % 2 != 0:
            raise ValueError("Width not divisible by 2.")
    weights = numpy.zeros((2, 2, channels, 4 * channels))
    for idx_ch in xrange(channels):
        slice_2 = slice(idx_ch, (idx_ch + 1))
        slice_3 = slice((idx_ch * 4), ((idx_ch + 1) * 4))
        weights[:, :, slice_2, slice_3] = SQUEEZE_MATRIX
    shuffle_channels = [idx_ch * 4 for idx_ch in xrange(channels)]
    shuffle_channels += [idx_ch * 4 + 1 for idx_ch in xrange(channels)]
    shuffle_channels += [idx_ch * 4 + 2 for idx_ch in xrange(channels)]
    shuffle_channels += [idx_ch * 4 + 3 for idx_ch in xrange(channels)]
    shuffle_channels = numpy.array(shuffle_channels)
    weights = weights[:, :, :, shuffle_channels].astype("float32")
    if reverse:
        res = tf.nn.conv2d_transpose(
            value=input_,
            filter=weights,
            output_shape=[batch_size, height * 2, width * 2, channels],
            strides=[1, 2, 2, 1],
            padding="SAME",
            name="unsqueeze_2x2")
    else:
        res = tf.nn.conv2d(
            input=input_,
            filter=weights,
            strides=[1, 2, 2, 1],
            padding="SAME",
            name="squeeze_2x2")

    return res
