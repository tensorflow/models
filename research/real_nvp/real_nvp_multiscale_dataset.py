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

r"""Script for training, evaluation and sampling for Real NVP.

$ python real_nvp_multiscale_dataset.py \
--alsologtostderr \
--image_size 64 \
--hpconfig=n_scale=5,base_dim=8 \
--dataset imnet \
--data_path [DATA_PATH]
"""

from __future__ import print_function

import time
from datetime import datetime
import os

import numpy
from six.moves import xrange
import tensorflow as tf

from tensorflow import gfile

from real_nvp_utils import (
    batch_norm, batch_norm_log_diff, conv_layer,
    squeeze_2x2, squeeze_2x2_ordered, standard_normal_ll,
    standard_normal_sample, unsqueeze_2x2, variable_on_cpu)


tf.flags.DEFINE_string("master", "local",
                       "BNS name of the TensorFlow master, or local.")

tf.flags.DEFINE_string("logdir", "/tmp/real_nvp_multiscale",
                       "Directory to which writes logs.")

tf.flags.DEFINE_string("traindir", "/tmp/real_nvp_multiscale",
                       "Directory to which writes logs.")

tf.flags.DEFINE_integer("train_steps", 1000000000000000000,
                        "Number of steps to train for.")

tf.flags.DEFINE_string("data_path", "", "Path to the data.")

tf.flags.DEFINE_string("mode", "train",
                       "Mode of execution. Must be 'train', "
                       "'sample' or 'eval'.")

tf.flags.DEFINE_string("dataset", "imnet",
                       "Dataset used. Must be 'imnet', "
                       "'celeba' or 'lsun'.")

tf.flags.DEFINE_integer("recursion_type", 2,
                        "Type of the recursion.")

tf.flags.DEFINE_integer("image_size", 64,
                        "Size of the input image.")

tf.flags.DEFINE_integer("eval_set_size", 0,
                        "Size of evaluation dataset.")

tf.flags.DEFINE_string(
    "hpconfig", "",
    "A comma separated list of hyperparameters for the model. Format is "
    "hp1=value1,hp2=value2,etc. If this FLAG is set, the model will be trained "
    "with the specified hyperparameters, filling in missing hyperparameters "
    "from the default_values in |hyper_params|.")

FLAGS = tf.flags.FLAGS

class HParams(object):
    """Dictionary of hyperparameters."""
    def __init__(self, **kwargs):
        self.dict_ = kwargs
        self.__dict__.update(self.dict_)

    def update_config(self, in_string):
        """Update the dictionary with a comma separated list."""
        pairs = in_string.split(",")
        pairs = [pair.split("=") for pair in pairs]
        for key, val in pairs:
            self.dict_[key] = type(self.dict_[key])(val)
        self.__dict__.update(self.dict_)
        return self

    def __getitem__(self, key):
        return self.dict_[key]

    def __setitem__(self, key, val):
        self.dict_[key] = val
        self.__dict__.update(self.dict_)


def get_default_hparams():
    """Get the default hyperparameters."""
    return HParams(
        batch_size=64,
        residual_blocks=2,
        n_couplings=2,
        n_scale=4,
        learning_rate=0.001,
        momentum=1e-1,
        decay=1e-3,
        l2_coeff=0.00005,
        clip_gradient=100.,
        optimizer="adam",
        dropout_mask=0,
        base_dim=32,
        bottleneck=0,
        use_batch_norm=1,
        alternate=1,
        use_aff=1,
        skip=1,
        data_constraint=.9,
        n_opt=0)


# RESNET UTILS
def residual_block(input_, dim, name, use_batch_norm=True,
                   train=True, weight_norm=True, bottleneck=False):
    """Residual convolutional block."""
    with tf.variable_scope(name):
        res = input_
        if use_batch_norm:
            res = batch_norm(
                input_=res, dim=dim, name="bn_in", scale=False,
                train=train, epsilon=1e-4, axes=[0, 1, 2])
        res = tf.nn.relu(res)
        if bottleneck:
            res = conv_layer(
                input_=res, filter_size=[1, 1], dim_in=dim, dim_out=dim,
                name="h_0", stddev=numpy.sqrt(2. / (dim)),
                strides=[1, 1, 1, 1], padding="SAME",
                nonlinearity=None, bias=(not use_batch_norm),
                weight_norm=weight_norm, scale=False)
            if use_batch_norm:
                res = batch_norm(
                    input_=res, dim=dim,
                    name="bn_0", scale=False, train=train,
                    epsilon=1e-4, axes=[0, 1, 2])
            res = tf.nn.relu(res)
            res = conv_layer(
                input_=res, filter_size=[3, 3], dim_in=dim,
                dim_out=dim, name="h_1", stddev=numpy.sqrt(2. / (1. * dim)),
                strides=[1, 1, 1, 1], padding="SAME", nonlinearity=None,
                bias=(not use_batch_norm),
                weight_norm=weight_norm, scale=False)
            if use_batch_norm:
                res = batch_norm(
                    input_=res, dim=dim, name="bn_1", scale=False,
                    train=train, epsilon=1e-4, axes=[0, 1, 2])
            res = tf.nn.relu(res)
            res = conv_layer(
                input_=res, filter_size=[1, 1], dim_in=dim, dim_out=dim,
                name="out", stddev=numpy.sqrt(2. / (1. * dim)),
                strides=[1, 1, 1, 1], padding="SAME", nonlinearity=None,
                bias=True, weight_norm=weight_norm, scale=True)
        else:
            res = conv_layer(
                input_=res, filter_size=[3, 3], dim_in=dim, dim_out=dim,
                name="h_0", stddev=numpy.sqrt(2. / (dim)),
                strides=[1, 1, 1, 1], padding="SAME",
                nonlinearity=None, bias=(not use_batch_norm),
                weight_norm=weight_norm, scale=False)
            if use_batch_norm:
                res = batch_norm(
                    input_=res, dim=dim, name="bn_0", scale=False,
                    train=train, epsilon=1e-4, axes=[0, 1, 2])
            res = tf.nn.relu(res)
            res = conv_layer(
                input_=res, filter_size=[3, 3], dim_in=dim, dim_out=dim,
                name="out", stddev=numpy.sqrt(2. / (1. * dim)),
                strides=[1, 1, 1, 1], padding="SAME", nonlinearity=None,
                bias=True, weight_norm=weight_norm, scale=True)
        res += input_

    return res


def resnet(input_, dim_in, dim, dim_out, name, use_batch_norm=True,
           train=True, weight_norm=True, residual_blocks=5,
           bottleneck=False, skip=True):
    """Residual convolutional network."""
    with tf.variable_scope(name):
        res = input_
        if residual_blocks != 0:
            res = conv_layer(
                input_=res, filter_size=[3, 3], dim_in=dim_in, dim_out=dim,
                name="h_in", stddev=numpy.sqrt(2. / (dim_in)),
                strides=[1, 1, 1, 1], padding="SAME",
                nonlinearity=None, bias=True,
                weight_norm=weight_norm, scale=False)
            if skip:
                out = conv_layer(
                    input_=res, filter_size=[1, 1], dim_in=dim, dim_out=dim,
                    name="skip_in", stddev=numpy.sqrt(2. / (dim)),
                    strides=[1, 1, 1, 1], padding="SAME",
                    nonlinearity=None, bias=True,
                    weight_norm=weight_norm, scale=True)

            # residual blocks
            for idx_block in xrange(residual_blocks):
                res = residual_block(res, dim, "block_%d" % idx_block,
                                     use_batch_norm=use_batch_norm, train=train,
                                     weight_norm=weight_norm,
                                     bottleneck=bottleneck)
                if skip:
                    out += conv_layer(
                        input_=res, filter_size=[1, 1], dim_in=dim, dim_out=dim,
                        name="skip_%d" % idx_block, stddev=numpy.sqrt(2. / (dim)),
                        strides=[1, 1, 1, 1], padding="SAME",
                        nonlinearity=None, bias=True,
                        weight_norm=weight_norm, scale=True)
            # outputs
            if skip:
                res = out
            if use_batch_norm:
                res = batch_norm(
                    input_=res, dim=dim, name="bn_pre_out", scale=False,
                    train=train, epsilon=1e-4, axes=[0, 1, 2])
            res = tf.nn.relu(res)
            res = conv_layer(
                input_=res, filter_size=[1, 1], dim_in=dim,
                dim_out=dim_out,
                name="out", stddev=numpy.sqrt(2. / (1. * dim)),
                strides=[1, 1, 1, 1], padding="SAME",
                nonlinearity=None, bias=True,
                weight_norm=weight_norm, scale=True)
        else:
            if bottleneck:
                res = conv_layer(
                    input_=res, filter_size=[1, 1], dim_in=dim_in, dim_out=dim,
                    name="h_0", stddev=numpy.sqrt(2. / (dim_in)),
                    strides=[1, 1, 1, 1], padding="SAME",
                    nonlinearity=None, bias=(not use_batch_norm),
                    weight_norm=weight_norm, scale=False)
                if use_batch_norm:
                    res = batch_norm(
                        input_=res, dim=dim, name="bn_0", scale=False,
                        train=train, epsilon=1e-4, axes=[0, 1, 2])
                res = tf.nn.relu(res)
                res = conv_layer(
                    input_=res, filter_size=[3, 3], dim_in=dim,
                    dim_out=dim, name="h_1", stddev=numpy.sqrt(2. / (1. * dim)),
                    strides=[1, 1, 1, 1], padding="SAME",
                    nonlinearity=None,
                    bias=(not use_batch_norm),
                    weight_norm=weight_norm, scale=False)
                if use_batch_norm:
                    res = batch_norm(
                        input_=res, dim=dim, name="bn_1", scale=False,
                        train=train, epsilon=1e-4, axes=[0, 1, 2])
                res = tf.nn.relu(res)
                res = conv_layer(
                    input_=res, filter_size=[1, 1], dim_in=dim, dim_out=dim_out,
                    name="out", stddev=numpy.sqrt(2. / (1. * dim)),
                    strides=[1, 1, 1, 1], padding="SAME",
                    nonlinearity=None, bias=True,
                    weight_norm=weight_norm, scale=True)
            else:
                res = conv_layer(
                    input_=res, filter_size=[3, 3], dim_in=dim_in, dim_out=dim,
                    name="h_0", stddev=numpy.sqrt(2. / (dim_in)),
                    strides=[1, 1, 1, 1], padding="SAME",
                    nonlinearity=None, bias=(not use_batch_norm),
                    weight_norm=weight_norm, scale=False)
                if use_batch_norm:
                    res = batch_norm(
                        input_=res, dim=dim, name="bn_0", scale=False,
                        train=train, epsilon=1e-4, axes=[0, 1, 2])
                res = tf.nn.relu(res)
                res = conv_layer(
                    input_=res, filter_size=[3, 3], dim_in=dim, dim_out=dim_out,
                    name="out", stddev=numpy.sqrt(2. / (1. * dim)),
                    strides=[1, 1, 1, 1], padding="SAME",
                    nonlinearity=None, bias=True,
                    weight_norm=weight_norm, scale=True)
        return res


# COUPLING LAYERS
# masked convolution implementations
def masked_conv_aff_coupling(input_, mask_in, dim, name,
                             use_batch_norm=True, train=True, weight_norm=True,
                             reverse=False, residual_blocks=5,
                             bottleneck=False, use_width=1., use_height=1.,
                             mask_channel=0., skip=True):
    """Affine coupling with masked convolution."""
    with tf.variable_scope(name) as scope:
        if reverse or (not train):
            scope.reuse_variables()
        shape = input_.get_shape().as_list()
        batch_size = shape[0]
        height = shape[1]
        width = shape[2]
        channels = shape[3]

        # build mask
        mask = use_width * numpy.arange(width)
        mask = use_height * numpy.arange(height).reshape((-1, 1)) + mask
        mask = mask.astype("float32")
        mask = tf.mod(mask_in + mask, 2)
        mask = tf.reshape(mask, [-1, height, width, 1])
        if mask.get_shape().as_list()[0] == 1:
            mask = tf.tile(mask, [batch_size, 1, 1, 1])
        res = input_ * tf.mod(mask_channel + mask, 2)

        # initial input
        if use_batch_norm:
            res = batch_norm(
                input_=res, dim=channels, name="bn_in", scale=False,
                train=train, epsilon=1e-4, axes=[0, 1, 2])
            res *= 2.
        res = tf.concat([res, -res], 3)
        res = tf.concat([res, mask], 3)
        dim_in = 2. * channels + 1
        res = tf.nn.relu(res)
        res = resnet(input_=res, dim_in=dim_in, dim=dim,
                     dim_out=2 * channels,
                     name="resnet", use_batch_norm=use_batch_norm,
                     train=train, weight_norm=weight_norm,
                     residual_blocks=residual_blocks,
                     bottleneck=bottleneck, skip=skip)
        mask = tf.mod(mask_channel + mask, 2)
        res = tf.split(axis=3, num_or_size_splits=2, value=res)
        shift, log_rescaling = res[-2], res[-1]
        scale = variable_on_cpu(
            "rescaling_scale", [],
            tf.constant_initializer(0.))
        shift = tf.reshape(
            shift, [batch_size, height, width, channels])
        log_rescaling = tf.reshape(
            log_rescaling, [batch_size, height, width, channels])
        log_rescaling = scale * tf.tanh(log_rescaling)
        if not use_batch_norm:
            scale_shift = variable_on_cpu(
                "scale_shift", [],
                tf.constant_initializer(0.))
            log_rescaling += scale_shift
        shift *= (1. - mask)
        log_rescaling *= (1. - mask)
        if reverse:
            res = input_
            if use_batch_norm:
                mean, var = batch_norm_log_diff(
                    input_=res * (1. - mask), dim=channels, name="bn_out",
                    train=False, epsilon=1e-4, axes=[0, 1, 2])
                log_var = tf.log(var)
                res *= tf.exp(.5 * log_var * (1. - mask))
                res += mean * (1. - mask)
            res *= tf.exp(-log_rescaling)
            res -= shift
            log_diff = -log_rescaling
            if use_batch_norm:
                log_diff += .5 * log_var * (1. - mask)
        else:
            res = input_
            res += shift
            res *= tf.exp(log_rescaling)
            log_diff = log_rescaling
            if use_batch_norm:
                mean, var = batch_norm_log_diff(
                    input_=res * (1. - mask), dim=channels, name="bn_out",
                    train=train, epsilon=1e-4, axes=[0, 1, 2])
                log_var = tf.log(var)
                res -= mean * (1. - mask)
                res *= tf.exp(-.5 * log_var * (1. - mask))
                log_diff -= .5 * log_var * (1. - mask)

    return res, log_diff


def masked_conv_add_coupling(input_, mask_in, dim, name,
                             use_batch_norm=True, train=True, weight_norm=True,
                             reverse=False, residual_blocks=5,
                             bottleneck=False, use_width=1., use_height=1.,
                             mask_channel=0., skip=True):
    """Additive coupling with masked convolution."""
    with tf.variable_scope(name) as scope:
        if reverse or (not train):
            scope.reuse_variables()
        shape = input_.get_shape().as_list()
        batch_size = shape[0]
        height = shape[1]
        width = shape[2]
        channels = shape[3]

        # build mask
        mask = use_width * numpy.arange(width)
        mask = use_height * numpy.arange(height).reshape((-1, 1)) + mask
        mask = mask.astype("float32")
        mask = tf.mod(mask_in + mask, 2)
        mask = tf.reshape(mask, [-1, height, width, 1])
        if mask.get_shape().as_list()[0] == 1:
            mask = tf.tile(mask, [batch_size, 1, 1, 1])
        res = input_ * tf.mod(mask_channel + mask, 2)

        # initial input
        if use_batch_norm:
            res = batch_norm(
                input_=res, dim=channels, name="bn_in", scale=False,
                train=train, epsilon=1e-4, axes=[0, 1, 2])
            res *= 2.
        res = tf.concat([res, -res], 3)
        res = tf.concat([res, mask], 3)
        dim_in = 2. * channels + 1
        res = tf.nn.relu(res)
        shift = resnet(input_=res, dim_in=dim_in, dim=dim, dim_out=channels,
                       name="resnet", use_batch_norm=use_batch_norm,
                       train=train, weight_norm=weight_norm,
                       residual_blocks=residual_blocks,
                       bottleneck=bottleneck, skip=skip)
        mask = tf.mod(mask_channel + mask, 2)
        shift *= (1. - mask)
        # use_batch_norm = False
        if reverse:
            res = input_
            if use_batch_norm:
                mean, var = batch_norm_log_diff(
                    input_=res * (1. - mask),
                    dim=channels, name="bn_out", train=False, epsilon=1e-4)
                log_var = tf.log(var)
                res *= tf.exp(.5 * log_var * (1. - mask))
                res += mean * (1. - mask)
            res -= shift
            log_diff = tf.zeros_like(res)
            if use_batch_norm:
                log_diff += .5 * log_var * (1. - mask)
        else:
            res = input_
            res += shift
            log_diff = tf.zeros_like(res)
            if use_batch_norm:
                mean, var = batch_norm_log_diff(
                    input_=res * (1. - mask), dim=channels,
                    name="bn_out", train=train, epsilon=1e-4, axes=[0, 1, 2])
                log_var = tf.log(var)
                res -= mean * (1. - mask)
                res *= tf.exp(-.5 * log_var * (1. - mask))
                log_diff -= .5 * log_var * (1. - mask)

    return res, log_diff


def masked_conv_coupling(input_, mask_in, dim, name,
                         use_batch_norm=True, train=True, weight_norm=True,
                         reverse=False, residual_blocks=5,
                         bottleneck=False, use_aff=True,
                         use_width=1., use_height=1.,
                         mask_channel=0., skip=True):
    """Coupling with masked convolution."""
    if use_aff:
        return masked_conv_aff_coupling(
            input_=input_, mask_in=mask_in, dim=dim, name=name,
            use_batch_norm=use_batch_norm, train=train, weight_norm=weight_norm,
            reverse=reverse, residual_blocks=residual_blocks,
            bottleneck=bottleneck, use_width=use_width, use_height=use_height,
            mask_channel=mask_channel, skip=skip)
    else:
        return masked_conv_add_coupling(
            input_=input_, mask_in=mask_in, dim=dim, name=name,
            use_batch_norm=use_batch_norm, train=train, weight_norm=weight_norm,
            reverse=reverse, residual_blocks=residual_blocks,
            bottleneck=bottleneck, use_width=use_width, use_height=use_height,
            mask_channel=mask_channel, skip=skip)


# channel-axis splitting implementations
def conv_ch_aff_coupling(input_, dim, name,
                         use_batch_norm=True, train=True, weight_norm=True,
                         reverse=False, residual_blocks=5,
                         bottleneck=False, change_bottom=True, skip=True):
    """Affine coupling with channel-wise splitting."""
    with tf.variable_scope(name) as scope:
        if reverse or (not train):
            scope.reuse_variables()

        if change_bottom:
            input_, canvas = tf.split(axis=3, num_or_size_splits=2, value=input_)
        else:
            canvas, input_ = tf.split(axis=3, num_or_size_splits=2, value=input_)
        shape = input_.get_shape().as_list()
        batch_size = shape[0]
        height = shape[1]
        width = shape[2]
        channels = shape[3]
        res = input_

        # initial input
        if use_batch_norm:
            res = batch_norm(
                input_=res, dim=channels, name="bn_in", scale=False,
                train=train, epsilon=1e-4, axes=[0, 1, 2])
        res = tf.concat([res, -res], 3)
        dim_in = 2. * channels
        res = tf.nn.relu(res)
        res = resnet(input_=res, dim_in=dim_in, dim=dim, dim_out=2 * channels,
                     name="resnet", use_batch_norm=use_batch_norm,
                     train=train, weight_norm=weight_norm,
                     residual_blocks=residual_blocks,
                     bottleneck=bottleneck, skip=skip)
        shift, log_rescaling = tf.split(axis=3, num_or_size_splits=2, value=res)
        scale = variable_on_cpu(
            "scale", [],
            tf.constant_initializer(1.))
        shift = tf.reshape(
            shift, [batch_size, height, width, channels])
        log_rescaling = tf.reshape(
            log_rescaling, [batch_size, height, width, channels])
        log_rescaling = scale * tf.tanh(log_rescaling)
        if not use_batch_norm:
            scale_shift = variable_on_cpu(
                "scale_shift", [],
                tf.constant_initializer(0.))
            log_rescaling += scale_shift
        if reverse:
            res = canvas
            if use_batch_norm:
                mean, var = batch_norm_log_diff(
                    input_=res, dim=channels, name="bn_out", train=False,
                    epsilon=1e-4, axes=[0, 1, 2])
                log_var = tf.log(var)
                res *= tf.exp(.5 * log_var)
                res += mean
            res *= tf.exp(-log_rescaling)
            res -= shift
            log_diff = -log_rescaling
            if use_batch_norm:
                log_diff += .5 * log_var
        else:
            res = canvas
            res += shift
            res *= tf.exp(log_rescaling)
            log_diff = log_rescaling
            if use_batch_norm:
                mean, var = batch_norm_log_diff(
                    input_=res, dim=channels, name="bn_out", train=train,
                    epsilon=1e-4, axes=[0, 1, 2])
                log_var = tf.log(var)
                res -= mean
                res *= tf.exp(-.5 * log_var)
                log_diff -= .5 * log_var
        if change_bottom:
            res = tf.concat([input_, res], 3)
            log_diff = tf.concat([tf.zeros_like(log_diff), log_diff], 3)
        else:
            res = tf.concat([res, input_], 3)
            log_diff = tf.concat([log_diff, tf.zeros_like(log_diff)], 3)

    return res, log_diff


def conv_ch_add_coupling(input_, dim, name,
                         use_batch_norm=True, train=True, weight_norm=True,
                         reverse=False, residual_blocks=5,
                         bottleneck=False, change_bottom=True, skip=True):
    """Additive coupling with channel-wise splitting."""
    with tf.variable_scope(name) as scope:
        if reverse or (not train):
            scope.reuse_variables()

        if change_bottom:
            input_, canvas = tf.split(axis=3, num_or_size_splits=2, value=input_)
        else:
            canvas, input_ = tf.split(axis=3, num_or_size_splits=2, value=input_)
        shape = input_.get_shape().as_list()
        channels = shape[3]
        res = input_

        # initial input
        if use_batch_norm:
            res = batch_norm(
                input_=res, dim=channels, name="bn_in", scale=False,
                train=train, epsilon=1e-4, axes=[0, 1, 2])
        res = tf.concat([res, -res], 3)
        dim_in = 2. * channels
        res = tf.nn.relu(res)
        shift = resnet(input_=res, dim_in=dim_in, dim=dim, dim_out=channels,
                       name="resnet", use_batch_norm=use_batch_norm,
                       train=train, weight_norm=weight_norm,
                       residual_blocks=residual_blocks,
                       bottleneck=bottleneck, skip=skip)
        if reverse:
            res = canvas
            if use_batch_norm:
                mean, var = batch_norm_log_diff(
                    input_=res, dim=channels, name="bn_out", train=False,
                    epsilon=1e-4, axes=[0, 1, 2])
                log_var = tf.log(var)
                res *= tf.exp(.5 * log_var)
                res += mean
            res -= shift
            log_diff = tf.zeros_like(res)
            if use_batch_norm:
                log_diff += .5 * log_var
        else:
            res = canvas
            res += shift
            log_diff = tf.zeros_like(res)
            if use_batch_norm:
                mean, var = batch_norm_log_diff(
                    input_=res, dim=channels, name="bn_out", train=train,
                    epsilon=1e-4, axes=[0, 1, 2])
                log_var = tf.log(var)
                res -= mean
                res *= tf.exp(-.5 * log_var)
                log_diff -= .5 * log_var
        if change_bottom:
            res = tf.concat([input_, res], 3)
            log_diff = tf.concat([tf.zeros_like(log_diff), log_diff], 3)
        else:
            res = tf.concat([res, input_], 3)
            log_diff = tf.concat([log_diff, tf.zeros_like(log_diff)], 3)

    return res, log_diff


def conv_ch_coupling(input_, dim, name,
                     use_batch_norm=True, train=True, weight_norm=True,
                     reverse=False, residual_blocks=5,
                     bottleneck=False, use_aff=True, change_bottom=True,
                     skip=True):
    """Coupling with channel-wise splitting."""
    if use_aff:
        return conv_ch_aff_coupling(
            input_=input_, dim=dim, name=name,
            use_batch_norm=use_batch_norm, train=train, weight_norm=weight_norm,
            reverse=reverse, residual_blocks=residual_blocks,
            bottleneck=bottleneck, change_bottom=change_bottom, skip=skip)
    else:
        return conv_ch_add_coupling(
            input_=input_, dim=dim, name=name,
            use_batch_norm=use_batch_norm, train=train, weight_norm=weight_norm,
            reverse=reverse, residual_blocks=residual_blocks,
            bottleneck=bottleneck, change_bottom=change_bottom, skip=skip)


# RECURSIVE USE OF COUPLING LAYERS
def rec_masked_conv_coupling(input_, hps, scale_idx, n_scale,
                             use_batch_norm=True, weight_norm=True,
                             train=True):
    """Recursion on coupling layers."""
    shape = input_.get_shape().as_list()
    channels = shape[3]
    residual_blocks = hps.residual_blocks
    base_dim = hps.base_dim
    mask = 1.
    use_aff = hps.use_aff
    res = input_
    skip = hps.skip
    log_diff = tf.zeros_like(input_)
    dim = base_dim
    if FLAGS.recursion_type < 4:
        dim *= 2 ** scale_idx
    with tf.variable_scope("scale_%d" % scale_idx):
        # initial coupling layers
        res, inc_log_diff = masked_conv_coupling(
            input_=res,
            mask_in=mask, dim=dim,
            name="coupling_0",
            use_batch_norm=use_batch_norm, train=train,
            weight_norm=weight_norm,
            reverse=False, residual_blocks=residual_blocks,
            bottleneck=hps.bottleneck, use_aff=use_aff,
            use_width=1., use_height=1., skip=skip)
        log_diff += inc_log_diff
        res, inc_log_diff = masked_conv_coupling(
            input_=res,
            mask_in=1. - mask, dim=dim,
            name="coupling_1",
            use_batch_norm=use_batch_norm, train=train,
            weight_norm=weight_norm,
            reverse=False, residual_blocks=residual_blocks,
            bottleneck=hps.bottleneck, use_aff=use_aff,
            use_width=1., use_height=1., skip=skip)
        log_diff += inc_log_diff
        res, inc_log_diff = masked_conv_coupling(
            input_=res,
            mask_in=mask, dim=dim,
            name="coupling_2",
            use_batch_norm=use_batch_norm, train=train,
            weight_norm=weight_norm,
            reverse=False, residual_blocks=residual_blocks,
            bottleneck=hps.bottleneck, use_aff=True,
            use_width=1., use_height=1., skip=skip)
        log_diff += inc_log_diff
    if scale_idx < (n_scale - 1):
        with tf.variable_scope("scale_%d" % scale_idx):
            res = squeeze_2x2(res)
            log_diff = squeeze_2x2(log_diff)
            res, inc_log_diff = conv_ch_coupling(
                input_=res,
                change_bottom=True, dim=2 * dim,
                name="coupling_4",
                use_batch_norm=use_batch_norm, train=train,
                weight_norm=weight_norm,
                reverse=False, residual_blocks=residual_blocks,
                bottleneck=hps.bottleneck, use_aff=use_aff, skip=skip)
            log_diff += inc_log_diff
            res, inc_log_diff = conv_ch_coupling(
                input_=res,
                change_bottom=False, dim=2 * dim,
                name="coupling_5",
                use_batch_norm=use_batch_norm, train=train,
                weight_norm=weight_norm,
                reverse=False, residual_blocks=residual_blocks,
                bottleneck=hps.bottleneck, use_aff=use_aff, skip=skip)
            log_diff += inc_log_diff
            res, inc_log_diff = conv_ch_coupling(
                input_=res,
                change_bottom=True, dim=2 * dim,
                name="coupling_6",
                use_batch_norm=use_batch_norm, train=train,
                weight_norm=weight_norm,
                reverse=False, residual_blocks=residual_blocks,
                bottleneck=hps.bottleneck, use_aff=True, skip=skip)
            log_diff += inc_log_diff
            res = unsqueeze_2x2(res)
            log_diff = unsqueeze_2x2(log_diff)
        if FLAGS.recursion_type > 1:
            res = squeeze_2x2_ordered(res)
            log_diff = squeeze_2x2_ordered(log_diff)
            if FLAGS.recursion_type > 2:
                res_1 = res[:, :, :, :channels]
                res_2 = res[:, :, :, channels:]
                log_diff_1 = log_diff[:, :, :, :channels]
                log_diff_2 = log_diff[:, :, :, channels:]
            else:
                res_1, res_2 = tf.split(axis=3, num_or_size_splits=2, value=res)
                log_diff_1, log_diff_2 = tf.split(axis=3, num_or_size_splits=2, value=log_diff)
            res_1, inc_log_diff = rec_masked_conv_coupling(
                input_=res_1, hps=hps, scale_idx=scale_idx + 1, n_scale=n_scale,
                use_batch_norm=use_batch_norm, weight_norm=weight_norm,
                train=train)
            res = tf.concat([res_1, res_2], 3)
            log_diff_1 += inc_log_diff
            log_diff = tf.concat([log_diff_1, log_diff_2], 3)
            res = squeeze_2x2_ordered(res, reverse=True)
            log_diff = squeeze_2x2_ordered(log_diff, reverse=True)
        else:
            res = squeeze_2x2_ordered(res)
            log_diff = squeeze_2x2_ordered(log_diff)
            res, inc_log_diff = rec_masked_conv_coupling(
                input_=res, hps=hps, scale_idx=scale_idx + 1, n_scale=n_scale,
                use_batch_norm=use_batch_norm, weight_norm=weight_norm,
                train=train)
            log_diff += inc_log_diff
            res = squeeze_2x2_ordered(res, reverse=True)
            log_diff = squeeze_2x2_ordered(log_diff, reverse=True)
    else:
        with tf.variable_scope("scale_%d" % scale_idx):
            res, inc_log_diff = masked_conv_coupling(
                input_=res,
                mask_in=1. - mask, dim=dim,
                name="coupling_3",
                use_batch_norm=use_batch_norm, train=train,
                weight_norm=weight_norm,
                reverse=False, residual_blocks=residual_blocks,
                bottleneck=hps.bottleneck, use_aff=True,
                use_width=1., use_height=1., skip=skip)
            log_diff += inc_log_diff
    return res, log_diff


def rec_masked_deconv_coupling(input_, hps, scale_idx, n_scale,
                               use_batch_norm=True, weight_norm=True,
                               train=True):
    """Recursion on inverting coupling layers."""
    shape = input_.get_shape().as_list()
    channels = shape[3]
    residual_blocks = hps.residual_blocks
    base_dim = hps.base_dim
    mask = 1.
    use_aff = hps.use_aff
    res = input_
    log_diff = tf.zeros_like(input_)
    skip = hps.skip
    dim = base_dim
    if FLAGS.recursion_type < 4:
        dim *= 2 ** scale_idx
    if scale_idx < (n_scale - 1):
        if FLAGS.recursion_type > 1:
            res = squeeze_2x2_ordered(res)
            log_diff = squeeze_2x2_ordered(log_diff)
            if FLAGS.recursion_type > 2:
                res_1 = res[:, :, :, :channels]
                res_2 = res[:, :, :, channels:]
                log_diff_1 = log_diff[:, :, :, :channels]
                log_diff_2 = log_diff[:, :, :, channels:]
            else:
                res_1, res_2 = tf.split(axis=3, num_or_size_splits=2, value=res)
                log_diff_1, log_diff_2 = tf.split(axis=3, num_or_size_splits=2, value=log_diff)
            res_1, log_diff_1 = rec_masked_deconv_coupling(
                input_=res_1, hps=hps,
                scale_idx=scale_idx + 1, n_scale=n_scale,
                use_batch_norm=use_batch_norm, weight_norm=weight_norm,
                train=train)
            res = tf.concat([res_1, res_2], 3)
            log_diff = tf.concat([log_diff_1, log_diff_2], 3)
            res = squeeze_2x2_ordered(res, reverse=True)
            log_diff = squeeze_2x2_ordered(log_diff, reverse=True)
        else:
            res = squeeze_2x2_ordered(res)
            log_diff = squeeze_2x2_ordered(log_diff)
            res, log_diff = rec_masked_deconv_coupling(
                input_=res, hps=hps,
                scale_idx=scale_idx + 1, n_scale=n_scale,
                use_batch_norm=use_batch_norm, weight_norm=weight_norm,
                train=train)
            res = squeeze_2x2_ordered(res, reverse=True)
            log_diff = squeeze_2x2_ordered(log_diff, reverse=True)
        with tf.variable_scope("scale_%d" % scale_idx):
            res = squeeze_2x2(res)
            log_diff = squeeze_2x2(log_diff)
            res, inc_log_diff = conv_ch_coupling(
                input_=res,
                change_bottom=True, dim=2 * dim,
                name="coupling_6",
                use_batch_norm=use_batch_norm, train=train,
                weight_norm=weight_norm,
                reverse=True, residual_blocks=residual_blocks,
                bottleneck=hps.bottleneck, use_aff=True, skip=skip)
            log_diff += inc_log_diff
            res, inc_log_diff = conv_ch_coupling(
                input_=res,
                change_bottom=False, dim=2 * dim,
                name="coupling_5",
                use_batch_norm=use_batch_norm, train=train,
                weight_norm=weight_norm,
                reverse=True, residual_blocks=residual_blocks,
                bottleneck=hps.bottleneck, use_aff=use_aff, skip=skip)
            log_diff += inc_log_diff
            res, inc_log_diff = conv_ch_coupling(
                input_=res,
                change_bottom=True, dim=2 * dim,
                name="coupling_4",
                use_batch_norm=use_batch_norm, train=train,
                weight_norm=weight_norm,
                reverse=True, residual_blocks=residual_blocks,
                bottleneck=hps.bottleneck, use_aff=use_aff, skip=skip)
            log_diff += inc_log_diff
            res = unsqueeze_2x2(res)
            log_diff = unsqueeze_2x2(log_diff)
    else:
        with tf.variable_scope("scale_%d" % scale_idx):
            res, inc_log_diff = masked_conv_coupling(
                input_=res,
                mask_in=1. - mask, dim=dim,
                name="coupling_3",
                use_batch_norm=use_batch_norm, train=train,
                weight_norm=weight_norm,
                reverse=True, residual_blocks=residual_blocks,
                bottleneck=hps.bottleneck, use_aff=True,
                use_width=1., use_height=1., skip=skip)
            log_diff += inc_log_diff

    with tf.variable_scope("scale_%d" % scale_idx):
        res, inc_log_diff = masked_conv_coupling(
            input_=res,
            mask_in=mask, dim=dim,
            name="coupling_2",
            use_batch_norm=use_batch_norm, train=train, weight_norm=weight_norm,
            reverse=True, residual_blocks=residual_blocks,
            bottleneck=hps.bottleneck, use_aff=True,
            use_width=1., use_height=1., skip=skip)
        log_diff += inc_log_diff
        res, inc_log_diff = masked_conv_coupling(
            input_=res,
            mask_in=1. - mask, dim=dim,
            name="coupling_1",
            use_batch_norm=use_batch_norm, train=train, weight_norm=weight_norm,
            reverse=True, residual_blocks=residual_blocks,
            bottleneck=hps.bottleneck, use_aff=use_aff,
            use_width=1., use_height=1., skip=skip)
        log_diff += inc_log_diff
        res, inc_log_diff = masked_conv_coupling(
            input_=res,
            mask_in=mask, dim=dim,
            name="coupling_0",
            use_batch_norm=use_batch_norm, train=train, weight_norm=weight_norm,
            reverse=True, residual_blocks=residual_blocks,
            bottleneck=hps.bottleneck, use_aff=use_aff,
            use_width=1., use_height=1., skip=skip)
        log_diff += inc_log_diff

    return res, log_diff


# ENCODER AND DECODER IMPLEMENTATIONS
# start the recursions
def encoder(input_, hps, n_scale, use_batch_norm=True,
            weight_norm=True, train=True):
    """Encoding/gaussianization function."""
    res = input_
    log_diff = tf.zeros_like(input_)
    res, inc_log_diff = rec_masked_conv_coupling(
        input_=res, hps=hps, scale_idx=0, n_scale=n_scale,
        use_batch_norm=use_batch_norm, weight_norm=weight_norm,
        train=train)
    log_diff += inc_log_diff

    return res, log_diff


def decoder(input_, hps, n_scale, use_batch_norm=True,
            weight_norm=True, train=True):
    """Decoding/generator function."""
    res, log_diff = rec_masked_deconv_coupling(
        input_=input_, hps=hps, scale_idx=0, n_scale=n_scale,
        use_batch_norm=use_batch_norm, weight_norm=weight_norm,
        train=train)

    return res, log_diff


class RealNVP(object):
    """Real NVP model."""

    def __init__(self, hps, sampling=False):
        # DATA TENSOR INSTANTIATION
        device = "/cpu:0"
        if FLAGS.dataset == "imnet":
            with tf.device(
                tf.train.replica_device_setter(0, worker_device=device)):
                filename_queue = tf.train.string_input_producer(
                    gfile.Glob(FLAGS.data_path), num_epochs=None)
                reader = tf.TFRecordReader()
                _, serialized_example = reader.read(filename_queue)
                features = tf.parse_single_example(
                    serialized_example,
                    features={
                        "image_raw": tf.FixedLenFeature([], tf.string),
                    })
                image = tf.decode_raw(features["image_raw"], tf.uint8)
                image.set_shape([FLAGS.image_size * FLAGS.image_size * 3])
                image = tf.cast(image, tf.float32)
                if FLAGS.mode == "train":
                    images = tf.train.shuffle_batch(
                        [image], batch_size=hps.batch_size, num_threads=1,
                        capacity=1000 + 3 * hps.batch_size,
                        # Ensures a minimum amount of shuffling of examples.
                        min_after_dequeue=1000)
                else:
                    images = tf.train.batch(
                        [image], batch_size=hps.batch_size, num_threads=1,
                        capacity=1000 + 3 * hps.batch_size)
            self.x_orig = x_orig = images
            image_size = FLAGS.image_size
            x_in = tf.reshape(
                x_orig,
                [hps.batch_size, FLAGS.image_size, FLAGS.image_size, 3])
            x_in = tf.clip_by_value(x_in, 0, 255)
            x_in = (tf.cast(x_in, tf.float32)
                    + tf.random_uniform(tf.shape(x_in))) / 256.
        elif FLAGS.dataset == "celeba":
            with tf.device(
                tf.train.replica_device_setter(0, worker_device=device)):
                filename_queue = tf.train.string_input_producer(
                    gfile.Glob(FLAGS.data_path), num_epochs=None)
                reader = tf.TFRecordReader()
                _, serialized_example = reader.read(filename_queue)
                features = tf.parse_single_example(
                    serialized_example,
                    features={
                        "image_raw": tf.FixedLenFeature([], tf.string),
                    })
                image = tf.decode_raw(features["image_raw"], tf.uint8)
                image.set_shape([218 * 178 * 3])  # 218, 178
                image = tf.cast(image, tf.float32)
                image = tf.reshape(image, [218, 178, 3])
                image = image[40:188, 15:163, :]
                if FLAGS.mode == "train":
                    image = tf.image.random_flip_left_right(image)
                    images = tf.train.shuffle_batch(
                        [image], batch_size=hps.batch_size, num_threads=1,
                        capacity=1000 + 3 * hps.batch_size,
                        min_after_dequeue=1000)
                else:
                    images = tf.train.batch(
                        [image], batch_size=hps.batch_size, num_threads=1,
                        capacity=1000 + 3 * hps.batch_size)
            self.x_orig = x_orig = images
            image_size = 64
            x_in = tf.reshape(x_orig, [hps.batch_size, 148, 148, 3])
            x_in = tf.image.resize_images(
                x_in, [64, 64], method=0, align_corners=False)
            x_in = (tf.cast(x_in, tf.float32)
                    + tf.random_uniform(tf.shape(x_in))) / 256.
        elif FLAGS.dataset == "lsun":
            with tf.device(
                tf.train.replica_device_setter(0, worker_device=device)):
                filename_queue = tf.train.string_input_producer(
                    gfile.Glob(FLAGS.data_path), num_epochs=None)
                reader = tf.TFRecordReader()
                _, serialized_example = reader.read(filename_queue)
                features = tf.parse_single_example(
                    serialized_example,
                    features={
                        "image_raw": tf.FixedLenFeature([], tf.string),
                        "height": tf.FixedLenFeature([], tf.int64),
                        "width": tf.FixedLenFeature([], tf.int64),
                        "depth": tf.FixedLenFeature([], tf.int64)
                    })
                image = tf.decode_raw(features["image_raw"], tf.uint8)
                height = tf.reshape((features["height"], tf.int64)[0], [1])
                height = tf.cast(height, tf.int32)
                width = tf.reshape((features["width"], tf.int64)[0], [1])
                width = tf.cast(width, tf.int32)
                depth = tf.reshape((features["depth"], tf.int64)[0], [1])
                depth = tf.cast(depth, tf.int32)
                image = tf.reshape(image, tf.concat([height, width, depth], 0))
                image = tf.random_crop(image, [64, 64, 3])
                if FLAGS.mode == "train":
                    image = tf.image.random_flip_left_right(image)
                    images = tf.train.shuffle_batch(
                        [image], batch_size=hps.batch_size, num_threads=1,
                        capacity=1000 + 3 * hps.batch_size,
                        # Ensures a minimum amount of shuffling of examples.
                        min_after_dequeue=1000)
                else:
                    images = tf.train.batch(
                        [image], batch_size=hps.batch_size, num_threads=1,
                        capacity=1000 + 3 * hps.batch_size)
            self.x_orig = x_orig = images
            image_size = 64
            x_in = tf.reshape(x_orig, [hps.batch_size, 64, 64, 3])
            x_in = (tf.cast(x_in, tf.float32)
                    + tf.random_uniform(tf.shape(x_in))) / 256.
        else:
            raise ValueError("Unknown dataset.")
        x_in = tf.reshape(x_in, [hps.batch_size, image_size, image_size, 3])
        side_shown = int(numpy.sqrt(hps.batch_size))
        shown_x = tf.transpose(
            tf.reshape(
                x_in[:(side_shown * side_shown), :, :, :],
                [side_shown, image_size * side_shown, image_size, 3]),
            [0, 2, 1, 3])
        shown_x = tf.transpose(
            tf.reshape(
                shown_x,
                [1, image_size * side_shown, image_size * side_shown, 3]),
            [0, 2, 1, 3]) * 255.
        tf.summary.image(
            "inputs",
            tf.cast(shown_x, tf.uint8),
            max_outputs=1)

        # restrict the data
        FLAGS.image_size = image_size
        data_constraint = hps.data_constraint
        pre_logit_scale = numpy.log(data_constraint)
        pre_logit_scale -= numpy.log(1. - data_constraint)
        pre_logit_scale = tf.cast(pre_logit_scale, tf.float32)
        logit_x_in = 2. * x_in  # [0, 2]
        logit_x_in -= 1.  # [-1, 1]
        logit_x_in *= data_constraint  # [-.9, .9]
        logit_x_in += 1.  # [.1, 1.9]
        logit_x_in /= 2.  # [.05, .95]
        # logit the data
        logit_x_in = tf.log(logit_x_in) - tf.log(1. - logit_x_in)
        transform_cost = tf.reduce_sum(
            tf.nn.softplus(logit_x_in) + tf.nn.softplus(-logit_x_in)
            - tf.nn.softplus(-pre_logit_scale),
            [1, 2, 3])

        # INFERENCE AND COSTS
        z_out, log_diff = encoder(
            input_=logit_x_in, hps=hps, n_scale=hps.n_scale,
            use_batch_norm=hps.use_batch_norm, weight_norm=True,
            train=True)
        if FLAGS.mode != "train":
              z_out, log_diff = encoder(
                  input_=logit_x_in, hps=hps, n_scale=hps.n_scale,
                  use_batch_norm=hps.use_batch_norm, weight_norm=True,
                  train=False)
        final_shape = [image_size, image_size, 3]
        prior_ll = standard_normal_ll(z_out)
        prior_ll = tf.reduce_sum(prior_ll, [1, 2, 3])
        log_diff = tf.reduce_sum(log_diff, [1, 2, 3])
        log_diff += transform_cost
        cost = -(prior_ll + log_diff)

        self.x_in = x_in
        self.z_out = z_out
        self.cost = cost = tf.reduce_mean(cost)

        l2_reg = sum(
            [tf.reduce_sum(tf.square(v)) for v in tf.trainable_variables()
             if ("magnitude" in v.name) or ("rescaling_scale" in v.name)])

        bit_per_dim = ((cost + numpy.log(256.) * image_size * image_size * 3.)
                       / (image_size * image_size * 3. * numpy.log(2.)))
        self.bit_per_dim = bit_per_dim

        # OPTIMIZATION
        momentum = 1. - hps.momentum
        decay = 1. - hps.decay
        if hps.optimizer == "adam":
            optimizer = tf.train.AdamOptimizer(
                learning_rate=hps.learning_rate,
                beta1=momentum, beta2=decay, epsilon=1e-08,
                use_locking=False, name="Adam")
        elif hps.optimizer == "rmsprop":
            optimizer = tf.train.RMSPropOptimizer(
                learning_rate=hps.learning_rate, decay=decay,
                momentum=momentum, epsilon=1e-04,
                use_locking=False, name="RMSProp")
        else:
            optimizer = tf.train.MomentumOptimizer(hps.learning_rate,
                                                   momentum=momentum)

        step = tf.get_variable(
            "global_step", [], tf.int64,
            tf.zeros_initializer(),
            trainable=False)
        self.step = step
        grads_and_vars = optimizer.compute_gradients(
            cost + hps.l2_coeff * l2_reg,
            tf.trainable_variables())
        grads, vars_ = zip(*grads_and_vars)
        capped_grads, gradient_norm = tf.clip_by_global_norm(
            grads, clip_norm=hps.clip_gradient)
        gradient_norm = tf.check_numerics(gradient_norm,
                                          "Gradient norm is NaN or Inf.")

        l2_z = tf.reduce_sum(tf.square(z_out), [1, 2, 3])
        if not sampling:
            tf.summary.scalar("negative_log_likelihood", tf.reshape(cost, []))
            tf.summary.scalar("gradient_norm", tf.reshape(gradient_norm, []))
            tf.summary.scalar("bit_per_dim", tf.reshape(bit_per_dim, []))
            tf.summary.scalar("log_diff", tf.reshape(tf.reduce_mean(log_diff), []))
            tf.summary.scalar("prior_ll", tf.reshape(tf.reduce_mean(prior_ll), []))
            tf.summary.scalar(
                "log_diff_var",
                tf.reshape(tf.reduce_mean(tf.square(log_diff))
                           - tf.square(tf.reduce_mean(log_diff)), []))
            tf.summary.scalar(
                "prior_ll_var",
                tf.reshape(tf.reduce_mean(tf.square(prior_ll))
                           - tf.square(tf.reduce_mean(prior_ll)), []))
            tf.summary.scalar("l2_z_mean", tf.reshape(tf.reduce_mean(l2_z), []))
            tf.summary.scalar(
                "l2_z_var",
                tf.reshape(tf.reduce_mean(tf.square(l2_z))
                           - tf.square(tf.reduce_mean(l2_z)), []))


        capped_grads_and_vars = zip(capped_grads, vars_)
        self.train_step = optimizer.apply_gradients(
            capped_grads_and_vars, global_step=step)

        # SAMPLING AND VISUALIZATION
        if sampling:
            # SAMPLES
            sample = standard_normal_sample([100] + final_shape)
            sample, _ = decoder(
                input_=sample, hps=hps, n_scale=hps.n_scale,
                use_batch_norm=hps.use_batch_norm, weight_norm=True,
                train=True)
            sample = tf.nn.sigmoid(sample)

            sample = tf.clip_by_value(sample, 0, 1) * 255.
            sample = tf.reshape(sample, [100, image_size, image_size, 3])
            sample = tf.transpose(
                tf.reshape(sample, [10, image_size * 10, image_size, 3]),
                [0, 2, 1, 3])
            sample = tf.transpose(
                tf.reshape(sample, [1, image_size * 10, image_size * 10, 3]),
                [0, 2, 1, 3])
            tf.summary.image(
                "samples",
                tf.cast(sample, tf.uint8),
                max_outputs=1)

            # CONCATENATION
            concatenation, _ = encoder(
                input_=logit_x_in, hps=hps,
                n_scale=hps.n_scale,
                use_batch_norm=hps.use_batch_norm, weight_norm=True,
                train=False)
            concatenation = tf.reshape(
                concatenation,
                [(side_shown * side_shown), image_size, image_size, 3])
            concatenation = tf.transpose(
                tf.reshape(
                    concatenation,
                    [side_shown, image_size * side_shown, image_size, 3]),
                [0, 2, 1, 3])
            concatenation = tf.transpose(
                tf.reshape(
                    concatenation,
                    [1, image_size * side_shown, image_size * side_shown, 3]),
                [0, 2, 1, 3])
            concatenation, _ = decoder(
                input_=concatenation, hps=hps, n_scale=hps.n_scale,
                use_batch_norm=hps.use_batch_norm, weight_norm=True,
                train=False)
            concatenation = tf.nn.sigmoid(concatenation) * 255.
            tf.summary.image(
                "concatenation",
                tf.cast(concatenation, tf.uint8),
                max_outputs=1)

            # MANIFOLD

            # Data basis
            z_u, _ = encoder(
                input_=logit_x_in[:8, :, :, :], hps=hps,
                n_scale=hps.n_scale,
                use_batch_norm=hps.use_batch_norm, weight_norm=True,
                train=False)
            u_1 = tf.reshape(z_u[0, :, :, :], [-1])
            u_2 = tf.reshape(z_u[1, :, :, :], [-1])
            u_3 = tf.reshape(z_u[2, :, :, :], [-1])
            u_4 = tf.reshape(z_u[3, :, :, :], [-1])
            u_5 = tf.reshape(z_u[4, :, :, :], [-1])
            u_6 = tf.reshape(z_u[5, :, :, :], [-1])
            u_7 = tf.reshape(z_u[6, :, :, :], [-1])
            u_8 = tf.reshape(z_u[7, :, :, :], [-1])

            # 3D dome
            manifold_side = 8
            angle_1 = numpy.arange(manifold_side) * 1. / manifold_side
            angle_2 = numpy.arange(manifold_side) * 1. / manifold_side
            angle_1 *= 2. * numpy.pi
            angle_2 *= 2. * numpy.pi
            angle_1 = angle_1.astype("float32")
            angle_2 = angle_2.astype("float32")
            angle_1 = tf.reshape(angle_1, [1, -1, 1])
            angle_1 += tf.zeros([manifold_side, manifold_side, 1])
            angle_2 = tf.reshape(angle_2, [-1, 1, 1])
            angle_2 += tf.zeros([manifold_side, manifold_side, 1])
            n_angle_3 = 40
            angle_3 = numpy.arange(n_angle_3) * 1. / n_angle_3
            angle_3 *= 2 * numpy.pi
            angle_3 = angle_3.astype("float32")
            angle_3 = tf.reshape(angle_3, [-1, 1, 1, 1])
            angle_3 += tf.zeros([n_angle_3, manifold_side, manifold_side, 1])
            manifold = tf.cos(angle_1) * (
                tf.cos(angle_2) * (
                    tf.cos(angle_3) * u_1 + tf.sin(angle_3) * u_2)
                + tf.sin(angle_2) * (
                    tf.cos(angle_3) * u_3 + tf.sin(angle_3) * u_4))
            manifold += tf.sin(angle_1) * (
                tf.cos(angle_2) * (
                    tf.cos(angle_3) * u_5 + tf.sin(angle_3) * u_6)
                + tf.sin(angle_2) * (
                    tf.cos(angle_3) * u_7 + tf.sin(angle_3) * u_8))
            manifold = tf.reshape(
                manifold,
                [n_angle_3 * manifold_side * manifold_side] + final_shape)
            manifold, _ = decoder(
                input_=manifold, hps=hps, n_scale=hps.n_scale,
                use_batch_norm=hps.use_batch_norm, weight_norm=True,
                train=False)
            manifold = tf.nn.sigmoid(manifold)

            manifold = tf.clip_by_value(manifold, 0, 1) * 255.
            manifold = tf.reshape(
                manifold,
                [n_angle_3,
                 manifold_side * manifold_side,
                 image_size,
                 image_size,
                 3])
            manifold = tf.transpose(
                tf.reshape(
                    manifold,
                    [n_angle_3, manifold_side,
                     image_size * manifold_side, image_size, 3]), [0, 1, 3, 2, 4])
            manifold = tf.transpose(
                tf.reshape(
                    manifold,
                    [n_angle_3, image_size * manifold_side,
                     image_size * manifold_side, 3]),
                [0, 2, 1, 3])
            manifold = tf.transpose(manifold, [1, 2, 0, 3])
            manifold = tf.reshape(
                manifold,
                [1, image_size * manifold_side,
                 image_size * manifold_side, 3 * n_angle_3])
            tf.summary.image(
                "manifold",
                tf.cast(manifold[:, :, :, :3], tf.uint8),
                max_outputs=1)

            # COMPRESSION
            z_complete, _ = encoder(
                input_=logit_x_in[:hps.n_scale, :, :, :], hps=hps,
                n_scale=hps.n_scale,
                use_batch_norm=hps.use_batch_norm, weight_norm=True,
                train=False)
            z_compressed_list = [z_complete]
            z_noisy_list = [z_complete]
            z_lost = z_complete
            for scale_idx in xrange(hps.n_scale - 1):
                z_lost = squeeze_2x2_ordered(z_lost)
                z_lost, _ = tf.split(axis=3, num_or_size_splits=2, value=z_lost)
                z_compressed = z_lost
                z_noisy = z_lost
                for _ in xrange(scale_idx + 1):
                    z_compressed = tf.concat(
                        [z_compressed, tf.zeros_like(z_compressed)], 3)
                    z_compressed = squeeze_2x2_ordered(
                        z_compressed, reverse=True)
                    z_noisy = tf.concat(
                        [z_noisy, tf.random_normal(
                            z_noisy.get_shape().as_list())], 3)
                    z_noisy = squeeze_2x2_ordered(z_noisy, reverse=True)
                z_compressed_list.append(z_compressed)
                z_noisy_list.append(z_noisy)
            self.z_reduced = z_lost
            z_compressed = tf.concat(z_compressed_list, 0)
            z_noisy = tf.concat(z_noisy_list, 0)
            noisy_images, _ = decoder(
                input_=z_noisy, hps=hps, n_scale=hps.n_scale,
                use_batch_norm=hps.use_batch_norm, weight_norm=True,
                train=False)
            compressed_images, _ = decoder(
                input_=z_compressed, hps=hps, n_scale=hps.n_scale,
                use_batch_norm=hps.use_batch_norm, weight_norm=True,
                train=False)
            noisy_images = tf.nn.sigmoid(noisy_images)
            compressed_images = tf.nn.sigmoid(compressed_images)

            noisy_images = tf.clip_by_value(noisy_images, 0, 1) * 255.
            noisy_images = tf.reshape(
                noisy_images,
                [(hps.n_scale * hps.n_scale), image_size, image_size, 3])
            noisy_images = tf.transpose(
                tf.reshape(
                    noisy_images,
                    [hps.n_scale, image_size * hps.n_scale, image_size, 3]),
                [0, 2, 1, 3])
            noisy_images = tf.transpose(
                tf.reshape(
                    noisy_images,
                    [1, image_size * hps.n_scale, image_size * hps.n_scale, 3]),
                [0, 2, 1, 3])
            tf.summary.image(
                "noise",
                tf.cast(noisy_images, tf.uint8),
                max_outputs=1)
            compressed_images = tf.clip_by_value(compressed_images, 0, 1) * 255.
            compressed_images = tf.reshape(
                compressed_images,
                [(hps.n_scale * hps.n_scale), image_size, image_size, 3])
            compressed_images = tf.transpose(
                tf.reshape(
                    compressed_images,
                    [hps.n_scale, image_size * hps.n_scale, image_size, 3]),
                [0, 2, 1, 3])
            compressed_images = tf.transpose(
                tf.reshape(
                    compressed_images,
                    [1, image_size * hps.n_scale, image_size * hps.n_scale, 3]),
                [0, 2, 1, 3])
            tf.summary.image(
                "compression",
                tf.cast(compressed_images, tf.uint8),
                max_outputs=1)

            # SAMPLES x2
            final_shape[0] *= 2
            final_shape[1] *= 2
            big_sample = standard_normal_sample([25] + final_shape)
            big_sample, _ = decoder(
                input_=big_sample, hps=hps, n_scale=hps.n_scale,
                use_batch_norm=hps.use_batch_norm, weight_norm=True,
                train=True)
            big_sample = tf.nn.sigmoid(big_sample)

            big_sample = tf.clip_by_value(big_sample, 0, 1) * 255.
            big_sample = tf.reshape(
                big_sample,
                [25, image_size * 2, image_size * 2, 3])
            big_sample = tf.transpose(
                tf.reshape(
                    big_sample,
                    [5, image_size * 10, image_size * 2, 3]), [0, 2, 1, 3])
            big_sample = tf.transpose(
                tf.reshape(
                    big_sample,
                    [1, image_size * 10, image_size * 10, 3]),
                [0, 2, 1, 3])
            tf.summary.image(
                "big_sample",
                tf.cast(big_sample, tf.uint8),
                max_outputs=1)

            # SAMPLES x10
            final_shape[0] *= 5
            final_shape[1] *= 5
            extra_large = standard_normal_sample([1] + final_shape)
            extra_large, _ = decoder(
                input_=extra_large, hps=hps, n_scale=hps.n_scale,
                use_batch_norm=hps.use_batch_norm, weight_norm=True,
                train=True)
            extra_large = tf.nn.sigmoid(extra_large)

            extra_large = tf.clip_by_value(extra_large, 0, 1) * 255.
            tf.summary.image(
                "extra_large",
                tf.cast(extra_large, tf.uint8),
                max_outputs=1)

    def eval_epoch(self, hps):
        """Evaluate bits/dim."""
        n_eval_dict = {
            "imnet": 50000,
            "lsun": 300,
            "celeba": 19962,
            "svhn": 26032,
        }
        if FLAGS.eval_set_size == 0:
            num_examples_eval = n_eval_dict[FLAGS.dataset]
        else:
            num_examples_eval = FLAGS.eval_set_size
        n_epoch = num_examples_eval / hps.batch_size
        eval_costs = []
        bar_len = 70
        for epoch_idx in xrange(n_epoch):
            n_equal = epoch_idx * bar_len * 1. / n_epoch
            n_equal = numpy.ceil(n_equal)
            n_equal = int(n_equal)
            n_dash = bar_len - n_equal
            progress_bar = "[" + "=" * n_equal + "-" * n_dash + "]\r"
            print(progress_bar, end=' ')
            cost = self.bit_per_dim.eval()
            eval_costs.append(cost)
        print("")
        return float(numpy.mean(eval_costs))


def train_model(hps, logdir):
    """Training."""
    with tf.Graph().as_default():
        with tf.device(tf.train.replica_device_setter(0)):
            with tf.variable_scope("model"):
                model = RealNVP(hps)

            saver = tf.train.Saver(tf.global_variables())

            # Build the summary operation from the last tower summaries.
            summary_op = tf.summary.merge_all()

            # Build an initialization operation to run below.
            init = tf.global_variables_initializer()

            # Start running operations on the Graph. allow_soft_placement must be set to
            # True to build towers on GPU, as some of the ops do not have GPU
            # implementations.
            sess = tf.Session(config=tf.ConfigProto(
                allow_soft_placement=True,
                log_device_placement=True))
            sess.run(init)

            ckpt_state = tf.train.get_checkpoint_state(logdir)
            if ckpt_state and ckpt_state.model_checkpoint_path:
                print("Loading file %s" % ckpt_state.model_checkpoint_path)
                saver.restore(sess, ckpt_state.model_checkpoint_path)

            # Start the queue runners.
            tf.train.start_queue_runners(sess=sess)

            summary_writer = tf.summary.FileWriter(
                logdir,
                graph=sess.graph)

            local_step = 0
            while True:
                fetches = [model.step, model.bit_per_dim, model.train_step]
                # The chief worker evaluates the summaries every 10 steps.
                should_eval_summaries = local_step % 100 == 0
                if should_eval_summaries:
                    fetches += [summary_op]


                start_time = time.time()
                outputs = sess.run(fetches)
                global_step_val = outputs[0]
                loss = outputs[1]
                duration = time.time() - start_time
                assert not numpy.isnan(
                    loss), 'Model diverged with loss = NaN'

                if local_step % 10 == 0:
                    examples_per_sec = hps.batch_size / float(duration)
                    format_str = ('%s: step %d, loss = %.2f '
                                  '(%.1f examples/sec; %.3f '
                                  'sec/batch)')
                    print(format_str % (datetime.now(), global_step_val, loss,
                                        examples_per_sec, duration))

                if should_eval_summaries:
                    summary_str = outputs[-1]
                    summary_writer.add_summary(summary_str, global_step_val)

                # Save the model checkpoint periodically.
                if local_step % 1000 == 0 or (local_step + 1) == FLAGS.train_steps:
                    checkpoint_path = os.path.join(logdir, 'model.ckpt')
                    saver.save(
                        sess,
                        checkpoint_path,
                        global_step=global_step_val)

                if outputs[0] >= FLAGS.train_steps:
                    break

                local_step += 1


def evaluate(hps, logdir, traindir, subset="valid", return_val=False):
    """Evaluation."""
    hps.batch_size = 100
    with tf.Graph().as_default():
        with tf.device("/cpu:0"):
            with tf.variable_scope("model") as var_scope:
                eval_model = RealNVP(hps)
                summary_writer = tf.summary.FileWriter(logdir)
                var_scope.reuse_variables()

            saver = tf.train.Saver()
            sess = tf.Session(config=tf.ConfigProto(
                allow_soft_placement=True,
                log_device_placement=True))
            tf.train.start_queue_runners(sess)

            previous_global_step = 0  # don"t run eval for step = 0

            with sess.as_default():
                while True:
                    ckpt_state = tf.train.get_checkpoint_state(traindir)
                    if not (ckpt_state and ckpt_state.model_checkpoint_path):
                        print("No model to eval yet at %s" % traindir)
                        time.sleep(30)
                        continue
                    print("Loading file %s" % ckpt_state.model_checkpoint_path)
                    saver.restore(sess, ckpt_state.model_checkpoint_path)

                    current_step = tf.train.global_step(sess, eval_model.step)
                    if current_step == previous_global_step:
                        print("Waiting for the checkpoint to be updated.")
                        time.sleep(30)
                        continue
                    previous_global_step = current_step

                    print("Evaluating...")
                    bit_per_dim = eval_model.eval_epoch(hps)
                    print("Epoch: %d, %s -> %.3f bits/dim"
                          % (current_step, subset, bit_per_dim))
                    print("Writing summary...")
                    summary = tf.Summary()
                    summary.value.extend(
                        [tf.Summary.Value(
                            tag="bit_per_dim",
                            simple_value=bit_per_dim)])
                    summary_writer.add_summary(summary, current_step)

                    if return_val:
                        return current_step, bit_per_dim


def sample_from_model(hps, logdir, traindir):
    """Sampling."""
    hps.batch_size = 100
    with tf.Graph().as_default():
        with tf.device("/cpu:0"):
            with tf.variable_scope("model") as var_scope:
                eval_model = RealNVP(hps, sampling=True)
                summary_writer = tf.summary.FileWriter(logdir)
                var_scope.reuse_variables()

                summary_op = tf.summary.merge_all()
            saver = tf.train.Saver()
            sess = tf.Session(config=tf.ConfigProto(
                allow_soft_placement=True,
                log_device_placement=True))
            coord = tf.train.Coordinator()
            threads = tf.train.start_queue_runners(sess=sess, coord=coord)

            previous_global_step = 0  # don"t run eval for step = 0

            initialized = False
            with sess.as_default():
                while True:
                    ckpt_state = tf.train.get_checkpoint_state(traindir)
                    if not (ckpt_state and ckpt_state.model_checkpoint_path):
                        if not initialized:
                            print("No model to eval yet at %s" % traindir)
                            time.sleep(30)
                            continue
                    else:
                        print ("Loading file %s"
                               % ckpt_state.model_checkpoint_path)
                        saver.restore(sess, ckpt_state.model_checkpoint_path)

                    current_step = tf.train.global_step(sess, eval_model.step)
                    if current_step == previous_global_step:
                        print("Waiting for the checkpoint to be updated.")
                        time.sleep(30)
                        continue
                    previous_global_step = current_step

                    fetches = [summary_op]

                    outputs = sess.run(fetches)
                    summary_writer.add_summary(outputs[0], current_step)
            coord.request_stop()
            coord.join(threads)


def main(unused_argv):
    hps = get_default_hparams().update_config(FLAGS.hpconfig)
    if FLAGS.mode == "train":
        train_model(hps=hps, logdir=FLAGS.logdir)
    elif FLAGS.mode == "sample":
        sample_from_model(hps=hps, logdir=FLAGS.logdir,
                          traindir=FLAGS.traindir)
    else:
        hps.batch_size = 100
        evaluate(hps=hps, logdir=FLAGS.logdir,
                 traindir=FLAGS.traindir, subset=FLAGS.mode)

if __name__ == "__main__":
    tf.app.run()
