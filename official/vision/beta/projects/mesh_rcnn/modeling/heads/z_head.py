# Copyright 2021 The TensorFlow Authors. All Rights Reserved.
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
"""Mesh R-CNN Heads."""

import tensorflow as tf

class ZHead(tf.keras.layers.Layer):
    '''Depth prediction Z Head for Mesh R-CNN model'''
    def __init__(self,
        num_fc: int,
        fc_dim: int,
        cls_agnostic: bool,
        num_classes: int,
        **kwargs):
        """
        Initialize Z-head
        Args:
            num_fc: number of fully connected layers
            fc_dim: dimension of fully connected layers
            cls_agnostic:
            num_classes: number of prediction classes
        """
        super(ZHead, self).__init__(**kwargs)

        self._num_fc = num_fc
        self._fc_dim = fc_dim
        self._cls_agnostic = cls_agnostic
        self._num_classes = num_classes

    def build(self, input_shape: tf.TensorShape) -> None:
        '''Build Z Head'''
        self.flatten = tf.keras.layers.Flatten()

        self.fcs = []
        for _ in range(self._num_fc):
            layer = tf.keras.layers.Dense(self._fc_dim,
                activation='relu',
                kernel_initializer='he_uniform')
            self.fcs.append(layer)
        num_z_reg_classes = 1 if self._cls_agnostic else self._num_classes
        pred_init = tf.keras.initializers.RandomNormal(stddev=0.001)
        self.z_pred = tf.keras.layers.Dense(num_z_reg_classes,
            kernel_initializer=pred_init,
            bias_initializer='zeros')

    def call(self, features):
        '''Forward pass of Z head'''
        out = self.flatten(features)
        for layer in self.fcs:
            out = layer(out)
        out = self.z_pred(out)
        return out

    def get_config(self):
        """Get config dict of the ZHead layer."""
        config = dict(
            num_fc = self._num_fc,
            fc_dim = self._fc_dim,
            cls_agnostic = self._cls_agnostic,
            num_classes = self._num_classes
        )
        return config

    @classmethod
    def from_config(cls, config):
        '''Initialize Z head from config'''
        return cls(**config)
