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
from tensorflow.keras.layers import LeakyReLU, BatchNormalization, ReLU, Activation, LeakyReLU
from tensorflow.keras.layers import UpSampling2D, Conv2D, Concatenate, Dense, concatenate
from tensorflow.keras.layers import Flatten, Lambda, Reshape, ZeroPadding2D, add, dot
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.preprocessing.image import ImageDataGenerator


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


class ParentChildEncoder(tf.keras.layers.Layer):
    """Encoder for parent and child images"""
    def __init__(self, num_disc_features, **kwargs):
        super(ParentChildEncoder, self).__init__(**kwargs)
        self.num_disc_features = num_disc_features

    def call(self, inputs):
        x = ZeroPadding2D(1)(inputs)
        x = Conv2D(self.num_disc_features, 4, 2, use_bias=False)(x)
        x = LeakyReLU(alpha=0.2)(x)
        x = ZeroPadding2D(1)(x)
        x = Conv2D(self.num_disc_features*2, 4, 2, use_bias=False)(x)
        x = BatchNormalization()(x)
        x = LeakyReLU(alpha=0.2)(x)
        x = ZeroPadding2D(1)(x)
        x = Conv2d(self.num_disc_features*4, 4, 2, use_bias=False)(x)
        x = BatchNormalization()(x)
        x = LeakyReLU(alpha=0.2)(x)
        x = ZeroPadding2D(1)(x)
        x = Conv2d(self.num_disc_features*8, 4, 2, use_bias=False)(x)
        x = BatchNormalization()(x)
        return LeakyReLU(alpha=0.2)(x)


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


class DownSampleBlock(tf.keras.layers.Layer):
    def __init__(self, filters=16, **kwargs):
        super(DownSampleBlock, self).__init__(**kwargs)
        self.filters = filters

    def call(self, inputs):
        x = ZeroPadding2D(1)(inputs)
        x = Conv2D(self.filters, 4, 2, use_bias=False)(x)
        x = BatchNormalization()(x)
        return LeakyReLU(alpha=0.2)(x)


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


class IntermediateGenerator(tf.keras.Model):
    def __init__(self, cfg, gen_dims, hrc=1, num_residual=2, **kwargs):
        super(IntermediateGenerator, self).__init__(**kwargs)
        self.gf_dim = gen_dims
        self.res = num_residual
        if hrc == 1:
            self.ef_dim = cfg.SUPER_CATEGORIES
        else:
            self.ef_dim = cfg.FINE_GRAINED_CATEGORIES

        self.convblock = Sequential([
            conv3x3(self.gf_dim*2),
            Activation('relu'),
            BatchNormalization(self.gf_dim // 2),
            GLU()
        ])

        self.residual = self._make_layer(ResidualBlock, self.gf_dim)
        self.keepdims = KeepDimsBlock(self.gf_dim // 2)

    def _make_layer(self, block, channel_num):
        layers = []
        for _ in range(self.res):
            layers.append(block(channel_num))
        return Sequential(*layers)

    def call(self, h_code, code):
        # TODO: Fix the Dimension errors
        # s_size = h_code.shape[2]
        # code = tf.reshape((-1, self.ef_dim, 1, 1))
        # code = tf.repeat(1, 1, s_size, s_size)
        x = Concatenate([code, h_code], axis=1)   
        x = self.convblock(x)
        x = self.residual(x)
        return self.keepdims(x)


class GetImage(tf.keras.Model):
    def __init__(self, gen_dims, **kwargs):
        super(GetImage, self).__init__(**kwargs)
        self.out_image = Sequential([
            conv3x3(3),
            Activation('tanh')
        ])

    def call(self, inputs):
        # The inputs need to be h_code
        return self.out_image(inputs)


class GetMask(tf.keras.Model):
    def __init__(self, gen_dims, **kwargs):
        super(GetMask, self).__init__(**kwargs)
        self.out_mask = Sequential([
            conv3x3(1),
            Activation('sigmoid')
        ])

    def call(self, inputs):
        # The inputs need to be h_code
        return self.out_mask(inputs)

    
class GeneratorArchitecture(tf.keras.Model):
    def __init__(self, cfg, **kwargs):
        super(GeneratorArchitecture, self).__init__(**kwargs)
        self.gen_dims = cfg.GAN['GF_DIM']
        self.upsampling = UpSampling2D(size=2, interpolation='bilinear')
        self.scale_foreground = UpSampling2D(size=2, interpolation='bilinear') 
        # TODO: Assert that scaled foreground needs to be of size [126, 126]

        # Background Stage
        self.background_gen = InitGenerator(cfg, self.gen_dims*16, 2)
        self.image_bg = GetImage(self.gen_dims) # Background Image

        # Parent Stage
        self.parent_gen1 = InitGenerator(cfg, self.gen_dims*16, 1)
        self.parent_gen2 = IntermediateGenerator(cfg, self.gen_dims, 1)
        self.image_gen2 = GetImage(self.gen_dims // 2) # Parent Foreground
        self.mask_gen2 = GetMask(self.gen_dims // 2) # Parent Mask

        # Child Stage
        # TODO: Include the ResidualGen before IntermediateGen
        self.child_gen = IntermediateGenerator(cfg, self.gen_dims // 2, 0)
        self.image_child = GetImage(self.gen_dims // 4) # Child Foreground
        self.mask_child = GetMask(self.gen_dims // 4) # Child Mask

    def call(self, z_code, c_code, p_code=None, bg_code=None):
        fake_images = [] # [Background images, Parent images, Child images]
        foreground_images = [] # [Parent foreground, Child foreground]
        masks = [] # [Parent masks, Child masks]
        foreground_masks = [] # [Parent foreground mask, Child foreground mask]

        # Background Stage
        bg_stage_code = self.background_gen(z_code, bg_code) # Upsampled Background
        fake_bg = self.image_bg(bg_stage_code)
        fake_images.append(self.scale_foreground(fake_bg))

        # Parent Stage
        fp_dims = self.parent_gen1(z_code, p_code)
        p_dims = self.parent_gen2(fp_dims, p_code) # Feature Representation (F_p)
        fake_parent_fg = self.image_gen2(p_dims) # Parent Foreground (P_f)
        fake_parent_mask = self.mask_gen2(p_dims) # Parent Mask (P_m)
        inverse_ones = tf.ones_like(fake_parent_mask)
        inverse_mask = inverse_ones - fake_parent_mask # (1-P_m)
        parent_foreground_mask = tf.math.multiply(fake_parent_fg, fake_parent_mask) # Parent Foreground Mask (P_fm)
        background_mask = tf.math.multiply(fake_bg, inverse_mask) # Background Mask (B_m)
        fake_parent_image = parent_foreground_mask + background_mask # Parent Image (P)
        fake_images.append(fake_parent_image)
        foreground_images.append(fake_parent_fg)
        masks.append(fake_parent_mask)
        foreground_masks.append(parent_foreground_mask)

        # Child Stage
        # TODO: Test whether inclusion of the ResidualGen is necessary
        fc_dims = self.child_gen(p_dims, c_code)
        fake_child_fg = self.image_child(fc_dims) # Child Foreground (C_f)
        fake_child_mask = self.mask_child(fc_dims) # Child Mask (C_m)
        inverse_ones = tf.ones_like(fake_child_mask)
        inverse_mask = inverse_ones - fake_child_mask # (1-C_m)
        child_foreground_mask = tf.math.multiply(fake_child_fg, fake_child_mask) # Child Foreground mask (C_fm)
        child_parent_mask = tf.math.multiply(fake_parent_image, inverse_mask) # Parent Mask (P_m)
        fake_child_image = child_foreground_mask + child_parent_mask # Child Image (C)
        fake_images.append(fake_child_image)
        foreground_images.append(fake_child_fg)
        masks.append(fake_child_mask)
        foreground_masks.append(child_foreground_mask)

        return fake_images, foreground_images, masks, foreground_masks       




class CustomConfig(Config):
    def __init__(self, batch_size=16, **kwargs):
        super(CustomConfig, self).__init__(batch_size)

if __name__== '__main__':
    cfg = CustomConfig(16)
    temp_g = InitGenerator(cfg=cfg, gen_dims=cfg.GAN, condition_flag=1)
    temp_interim_g = IntermediateGenerator(cfg=cfg)

