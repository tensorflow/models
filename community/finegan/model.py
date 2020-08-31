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

assert tf.version.VERSION.startswith('2.')

from tensorflow.keras import Input, Model, Sequential
from tensorflow.keras.layers import LeakyReLU, BatchNormalization, ReLU, Activation, LeakyReLU
from tensorflow.keras.layers import UpSampling2D, Conv2D, Concatenate, Dense, concatenate, Conv2DTranspose
from tensorflow.keras.layers import Flatten, Lambda, Reshape, ZeroPadding2D, add, dot, ELU
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.preprocessing.image import ImageDataGenerator


class GLU(tf.keras.layers.Layer):
    def __init__(self):
        super(GLU, self).__init__()

    def call(self, inputs):
        nc = inputs.shape[-1]
        assert nc % 2 == 0, 'Channels are not divisible by 2.'
        nc = int(nc/2)
        if len(inputs.shape) == 2:
            val = inputs[:,:nc] * tf.math.sigmoid(inputs[:,nc:])
        else:
            val = inputs[:,:,:,:nc] * tf.math.sigmoid(inputs[:,:,:,nc:])
        return val
    

class ParentChildEncoder(tf.keras.layers.Layer):
    """Encoder for parent and child images"""
    def __init__(self, num_disc_features, **kwargs):
        super(ParentChildEncoder, self).__init__(**kwargs)
        self.num_disc_features = num_disc_features
        
        self.conv1 = Conv2D(self.num_disc_features, 4, 2, use_bias=False)
        self.conv2 = Conv2D(self.num_disc_features*2, 4, 2, use_bias=False)
        self.batchnorm1 = BatchNormalization()
        self.conv3 = Conv2D(self.num_disc_features*4, 4, 2, use_bias=False)
        self.batchnorm2 = BatchNormalization()
        self.conv4 = Conv2D(self.num_disc_features*8, 4, 2, use_bias=False)
        self.batchnorm3 = BatchNormalization()

    def call(self, inputs):
        x = ZeroPadding2D(1)(inputs)
        x = self.conv1(x)
        x = LeakyReLU(alpha=0.2)(x)
        x = ZeroPadding2D(1)(x)
        x = self.conv2(x)
        x = self.batchnorm1(x)
        x = LeakyReLU(alpha=0.2)(x)
        x = ZeroPadding2D(1)(x)
        x = self.conv3(x)
        x = self.batchnorm2(x)
        x = LeakyReLU(alpha=0.2)(x)
        x = ZeroPadding2D(1)(x)
        x = self.conv4(x)
        x = self.batchnorm3(x)
        return LeakyReLU(alpha=0.2)(x)


class BackgroundEncoder(tf.keras.layers.Layer):
    """Encoder for the background image"""
    def __init__(self, num_disc_features, **kwargs):
        super(BackgroundEncoder, self).__init__(**kwargs)
        self.num_disc_features = num_disc_features
        
        self.conv1 = Conv2D(self.num_disc_features, 4, 2, use_bias=False)
        self.conv2 = Conv2D(self.num_disc_features*2, 4, 2, use_bias=False)
        self.conv3 = Conv2D(self.num_disc_features*4, 4, 1, use_bias=False)
    
    def call(self, inputs):
        x = self.conv1(inputs)
        x = LeakyReLU(alpha=0.2)(x)
        x = self.conv2(x)
        x = LeakyReLU(alpha=0.2)(x)
        x = self.conv3(x)
        return LeakyReLU(alpha=0.2)(x)


class UpSampleBlock(tf.keras.layers.Layer):
    def __init__(self, filters=16, **kwargs):
        super(UpSampleBlock, self).__init__(**kwargs)
        self.filters = filters
        
        self.upsample1 = Conv2DTranspose(self.filters*2, 3, strides=2, padding='same',
                                         kernel_initializer="orthogonal", use_bias=False)
        self.batchnorm1 = BatchNormalization()

    def call(self, inputs):
        x = self.upsample1(inputs)
        x = self.batchnorm1(x)
        return GLU()(x)


class DownSampleBlock(tf.keras.layers.Layer):
    def __init__(self, filters=16, **kwargs):
        super(DownSampleBlock, self).__init__(**kwargs)
        self.filters = filters
        
        self.conv1 = Conv2D(self.filters, 4, 2, use_bias=False)
        self.batchnorm1 = BatchNormalization()

    def call(self, inputs):
        x = ZeroPadding2D(1)(inputs)
        x = self.conv1(x)
        x = self.batchnorm1(x)
        return LeakyReLU(alpha=0.2)(x)


class KeepDimsBlock(tf.keras.layers.Layer):
    def __init__(self, filters=16, **kwargs):
        super(KeepDimsBlock, self).__init__(**kwargs)
        self.filters = filters
        
        self.conv1 = Conv2D(self.filters*2, 3, kernel_initializer='orthogonal', use_bias=False)
        self.batchnorm1 = BatchNormalization()

    def call(self, inputs):
        x = ZeroPadding2D(1)(inputs)
        x = self.conv1(x)
        x = self.batchnorm1(x)
        return GLU()(x)


class ResidualBlock(tf.keras.layers.Layer):
    def __init__(self, channels=16, **kwargs):
        super(ResidualBlock, self).__init__(**kwargs)
        self.channels = channels
        
        self.conv1 = Conv2D(filters=self.channels * 2, kernel_size=3, strides=1, kernel_initializer='orthogonal', 
            use_bias=False)
        self.batchnorm1 = BatchNormalization()
        self.conv2 = Conv2D(filters=self.channels, kernel_size=3, strides=1, kernel_initializer='orthogonal', 
            use_bias=False)
        self.batchnorm2 = BatchNormalization()
        
    def call(self, inputs):
        residual = inputs
        x = ZeroPadding2D(1)(inputs)
        x = self.conv1(x)
        x = self.batchnorm1(x)
        x = GLU()(x)
        x = ZeroPadding2D(1)(x)
        x = self.conv2(x)
        x = self.batchnorm2(x)
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
        
        self.dense1 = Dense(self.gf_dim*4*4*2, kernel_initializer='orthogonal', use_bias=False)
        self.batchnorm1 = BatchNormalization()

    def call(self, z_code, h_code):
        z_code = tf.cast(z_code, dtype=tf.float32)
        h_code = tf.cast(h_code, dtype=tf.float32)
        x = Concatenate()([z_code, h_code])
        x = self.dense1(x)
        x = self.batchnorm1(x)
        x = GLU()(x)
        x = Reshape((4, 4, self.gf_dim))(x)
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        return self.layer5(x)


class IntermediateGenerator(tf.keras.Model):
    def __init__(self, cfg, gen_dims, hrc=1, num_residual=cfg.GAN['R_NUM'], **kwargs):
        super(IntermediateGenerator, self).__init__(**kwargs)
        self.gf_dim = gen_dims
        self.res = num_residual
        if hrc == 1:
            self.ef_dim = cfg.SUPER_CATEGORIES
        else:
            self.ef_dim = cfg.FINE_GRAINED_CATEGORIES

        self.convblock = Sequential([
            ZeroPadding2D(1),
            Conv2D(self.gf_dim*2, 3, 1, kernel_initializer='orthogonal', use_bias=False),
            BatchNormalization(),
            GLU()
        ])

        self.residual = self.make_layer(ResidualBlock, self.gf_dim)
        self.keepdims = KeepDimsBlock(self.gf_dim // 2)

    def make_layer(self, block, channel_num):
        return Sequential([block(channel_num),
                          block(channel_num)])

    def call(self, h_code, code):
        s_size = h_code.shape[1]
        code = Reshape([1, 1, self.ef_dim])(code)
        code = tf.tile(code, tf.constant([1, s_size, s_size, 1]))
        h_code = tf.cast(h_code, dtype=tf.float32)
        code = tf.cast(code, dtype=tf.float32)
        x = Concatenate(axis=-1)([code, h_code])
        x = self.convblock(x)
        x = self.residual(x)
        return self.keepdims(x)


class GetImage(tf.keras.Model):
    def __init__(self, **kwargs):
        super(GetImage, self).__init__(**kwargs)
        self.out_image = Sequential([
            ZeroPadding2D(1),
            Conv2D(filters=3, kernel_size=3, strides=1, kernel_initializer='orthogonal', 
            use_bias=False),
            Activation('tanh')
        ])

    def call(self, inputs):
        return self.out_image(inputs)


class GetMask(tf.keras.Model):
    def __init__(self, **kwargs):
        super(GetMask, self).__init__(**kwargs)
        self.out_mask = Sequential([
            ZeroPadding2D(1),
            Conv2D(filters=1, kernel_size=3, strides=1, kernel_initializer='orthogonal', 
            use_bias=False),
            Activation('sigmoid')
        ])

    def call(self, inputs):
        return self.out_mask(inputs)

    
class GeneratorArchitecture(tf.keras.Model):
    def __init__(self, cfg, **kwargs):
        super(GeneratorArchitecture, self).__init__(**kwargs)
        self.cfg = cfg
        self.gen_dims = cfg.GAN['GF_DIM']
                
        # Background Stage
        self.background_gen = InitGenerator(cfg, self.gen_dims*16, 2)
        self.image_bg = GetImage() # Background Image

        # Parent Stage
        self.parent_gen1 = InitGenerator(cfg, self.gen_dims*16, 1)
        self.parent_gen2 = IntermediateGenerator(cfg, self.gen_dims, 1)
        self.image_gen2 = GetImage() # Parent Foreground
        self.mask_gen2 = GetMask() # Parent Mask

        # Child Stage
        self.child_gen = IntermediateGenerator(cfg, self.gen_dims // 2, 0)
        self.image_child = GetImage() # Child Foreground
        self.mask_child = GetMask() # Child Mask

    def call(self, z_code, c_code, p_code=None, bg_code=None):
        fake_images = [] # [Background images, Parent images, Child images]
        foreground_images = [] # [Parent foreground, Child foreground]
        masks = [] # [Parent masks, Child masks]
        foreground_masks = [] # [Parent foreground mask, Child foreground mask]

        # Set only during training
        bg_code = tf.cast(c_code, dtype=tf.float32)

        # Background Stage
        bg_stage_code = self.background_gen(z_code, bg_code) # Upsampled Background
        fake_bg = self.image_bg(bg_stage_code)
        fake_img_126 = tf.image.resize(fake_bg,(126, 126))
        fake_images.append(fake_img_126)

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


class DiscriminatorArchitecture(tf.keras.Model):
    def __init__(self, cfg, stage_num, **kwargs):
        super(DiscriminatorArchitecture, self).__init__(**kwargs)
        self.disc_dims = cfg.GAN['DF_DIM']
        self.stage_num = stage_num

        if self.stage_num == 0:
            self.encoder_dims = 1
        elif self.stage_num == 1:
            self.encoder_dims = cfg.SUPER_CATEGORIES
        elif self.stage_num == 2:
            self.encoder_dims = cfg.FINE_GRAINED_CATEGORIES

        if self.stage_num == 0:
            # Background Stage
            self.patchgan_16 = BackgroundEncoder(self.disc_dims)
            self.logits1 = Sequential([
                Conv2D(1, 4, 1),
                Activation('sigmoid')  
            ])
            self.logits2 = Sequential([
                Conv2D(1, 4, 1),
                Activation('sigmoid')
            ])

        else:
            self.code_16 = ParentChildEncoder(self.disc_dims)
            self.code_32 = DownSampleBlock(self.disc_dims*16)
            self.code = Sequential([
                ZeroPadding2D(1),
                Conv2D(self.disc_dims*8, 3, kernel_initializer='orthogonal', use_bias=False),
                BatchNormalization(),
                LeakyReLU(alpha=0.2)
            ])
            # Pass gradients through
            self.logits_pc = Sequential([
                Conv2D(self.encoder_dims, 4, 4, name=f'logits_pc_{self.stage_num}')
            ])
            # Pass gradients through
            self.jointConv = Sequential([
                ZeroPadding2D(1),
                Conv2D(self.disc_dims*8, 3, kernel_initializer='orthogonal', use_bias=False, name=f'joint_conv_{self.stage_num}'),
                BatchNormalization(),
                LeakyReLU(alpha=0.2)
            ])
            self.logits_pc1 = Sequential([
                Conv2D(1, 4, 4, use_bias=False),
                Activation('sigmoid')
            ])


    def call(self, inputs):
        if self.stage_num == 0:
            x = self.patchgan_16(inputs)
            back_fore = self.logits1(x) # Background/Foreground classification (D_aux)
            real_fake = self.logits2(x) # Real/Fake classification (D_adv)
            return [back_fore, real_fake]
            
        else:
            x = self.code_16(inputs)
            x = self.code_32(x)
            x = self.code(x)
            x = self.jointConv(x)
            p_c = self.logits_pc(x) # Information maximising code (D_pinfo or D_cinfo)
            real_fake_child = self.logits_pc1(x) # Real/Fake classification - child (D_adv)
            return [Reshape([self.encoder_dims])(p_c), Reshape([-1])(real_fake_child)]
