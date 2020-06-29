# Copyright 2020 The TensorFlow Authors. All Rights Reserved.
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
"""Unsupervised Hierarchical Disentanglement for Fine Grained Object Generation.
"""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os
import time
import pickle
import random

import numpy as np
import pandas as pd
import tensorflow as tf

assert tf.version.VERSION.startswith('2.2')

from config.config import Config
from tensorflow.keras import Input, Model, Sequential
from tensorflow.keras.layers import LeakyReLU, BatchNormalization, ReLU, Activation
from tensorflow.keras.layers import UpSampling2D, Conv2D, Concatenate, Dense, concatenate
from tensorflow.keras.layers import Flatten, Lambda, Reshape, ZeroPadding2D, add
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.preprocessing.image import ImageDataGenerator


# TODO: Add appropriate comments and information where necessary

class GLU(tf.keras.layers.Layer):
    def __init__(self, **kwargs):
        super(GLU, self).__init__(**kwargs)

    def call(self, x):
        num_channels = x.shape[-1]
        assert num_channels % 2 == 0, "Channels don't divide by 2"
        num_channels /= 2
        return x[:, :, :, :num_channels] * Activation('sigmoid')(x[:, :, :, num_channels:])


def child_to_parent(child_code, child_classes, parent_classes):
    """Returns the parent conditional code"""
    ratio = child_classes/parent_classes
    arg_parent = tf.math.argmax(child_code, axis=1)/ratio
    parent_code = tf.zeros([child_code.shape[0], parent_classes])
    for i in range(child_code.shape[0]):
        parent_code[i][arg_parent[i]] = 1
    return parent_code

def conv3x3(filters=16):
    return Conv2D(filters=filters, kernel_size=3, strides=1, kernel_initializer="he_normal", 
            use_bias=False)


class UpSampleBlock(tf.keras.layers.Layer):
    def __init__(self, filters=16, **kwargs):
        super(UpSampleBlock, self).__init__(**kwargs)
        self.filters = filters
        
    @tf.function
    def call(self, inputs):
        x = UpSampling2D(size=2, interpolation="nearest")(inputs)
        x = conv3x3(self.filters * 2)(x)
        x = BatchNormalization()(x)
        return  GLU()(x)


class KeepDimsBlock(tf.keras.layers.Layer):
    def __init__(self, filters=16, **kwargs):
        super(KeepDimsBlock, self).__init__(**kwargs)
        self.filters = filters

    @tf.function
    def call(self, inputs):
        x = conv3x3(self.filters*2)(inputs)
        BatchNormalization()(x)
        return GLU()(x)


class ResidualBlock(tf.keras.layers.Layer):
    def __init__(self, channels=16, **kwargs):
        super(ResidualBlock, self).__init__(**kwargs)
        self.channels = channels

    @tf.function
    def call(self, inputs):
        residual = inputs
        x = conv3x3(self.channels * 2)(inputs)
        x = BatchNormalization()(x)
        x = GLU()(x)
        x = conv3x3(self.channels)
        x = BatchNormalization()(x)
        return tf.keras.layers.Add()([x, residual])


class InitGenerator(tf.keras.Model):
    def __init__(self, cfg, gen_dims, condition_flag, **kwargs):
        super(InitGenerator, self).__init__(**kwargs)
        self.gf_dim = gen_dims
        self.condition_flag = condition_flag

        if self.condition_flag==1 :
            self.input_dims = cfg.GAN['Z_DIM'] + cfg.SUPER_CATEGORIES
        elif self.condition_flag==2:
            self.input_dims = cfg.GAN['Z_DIM'] + cfg.FINE_GRAINED_CATEGORIES 

        self.layer1 = UpSampleBlock(self.gf_dim // 2)
        self.layer2 = UpSampleBlock(self.gf_dim // 4)
        self.layer3 = UpSampleBlock(self.gf_dim // 8)
        self.layer4 = UpSampleBlock(self.gf_dim // 16)
        self.layer5 = UpSampleBlock(self.gf_dim // 16)

    def call(self, z_code, h_code):
        x = Concatenate()([z_code, h_code])
        x = Dense(self.gf_dim*4*4*2, use_bias=False)(x)
        x = BatchNormalization()(x)
        x = GLU()(x)
        x = Reshape((-1, self.gf_dim, 4, 4))(x)
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        return self.layer5(x)

class CustomConfig(Config):
    def __init__(self, batch_size=16, **kwargs):
        super(CustomConfig, self).__init__(batch_size)

if __name__== '__main__':
    cfg = CustomConfig(16)
    temp_g = InitGenerator(cfg=cfg, gen_dims=cfg.GAN, condition_flag=1)
    temp_interim_g = IntermediateGenerator(cfg=cfg)

