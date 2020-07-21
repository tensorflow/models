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
"""Trains the FineGAN Model for Fine Grained Object Generation and Discovery.
"""

import os
import numpy as np
import matplotlib.pyplot as plt

import tensorflow as tf
assert tf.version.VERSION.startswith('2.')

from .model import GeneratorArchitecture, DiscriminatorArchitecture
from .model import child_to_parent
from .config.config import Config

AUTOTUNE = tf.data.experimental.AUTOTUNE

def save_images():
    # TODO: Save images to the folder
    pass


class FineGAN(object):
    """The FineGAN Architecture"""
    def __init__(self, cfg, out_path, img_path, dataset, **kwargs):
        super(FineGAN, self).__init__(**kwargs)
        self.batch_size = cfg.TRAIN['BATCH_SIZE']
        self.num_epochs = cfg.TRAIN['MAX_EPOCH']
        self.dataset = dataset
        self.num_batches = len(dataset)
        self.summary_writer = tf.summary.create_file_writer('./logs')

    def prepare_dataset(self, batch_size=32):
        data = tf.data.Dataset.list_files('./dataset/*')
        data = data.map(lambda image: tf.image.resize(image, (128,128,3)), num_parallel_calls=AUTOTUNE)
        data = data.cache().shuffle(1000).repeat(2)
        return data.batch(self.batch_size).prefetch(num_parallel_calls=AUTOTUNE)

    def plot_images(self, images, num_cols=None):
        num_cols = num_cols or len(images)
        num_rows = (len(images) - 1) // num_cols + 1
        if images.shape[-1] == 1:
            images = np.squeeze(images, axis=-1)
        plt.figure(figsize=(num_cols, num_rows))
        for index, image in enumerate(images):
            plt.subplot(num_rows, num_cols, index + 1)
            plt.imshow(image)
            plt.axis("off")

    def prepare_data(self, data):
        foreground_img, c_img, conditioning_code, _, bbox = data
        real_vfimages, real_vcimages = [], []
        vc_code = tf.Variable(conditioning_code)
        for i in range(len(bbox)):
            bbox[i] = tf.Variable(bbox[i])

        real_vfimages.append(tf.Variable(foreground_img))
        real_vcimages.append(tf.Variable(c_img))

        return foreground_img, real_vfimages, real_vcimages, vc_code, bbox

    def train_generator(self):
        # TODO: Train the Generators
        pass

    def train_discriminator(self, stage, count):
        # TODO: Train the Discriminator
        if stage==0 or stage==2:

            with tf.GradientTape() as tape:
                batch_size = self.batch_size
                loss, loss1 = self.loss, self.loss1 # Binary Crossentropy

                # TODO: Load the network and optimizer for Discriminator stage
                # optimizer = optimizer_disc_list[stage] # stage_wise optimizer
                disc_network = tf.keras.models.load_model('temp.h5')
                optimizer = tf.keras.optimizers.Adam(learning_rate=0.01)

                if stage==0:
                    real_images = self.real_fimages[0]
                elif stage==2:
                    real_images = self.real_cimages[0]

                fake_images = self.fake_images[stage]
                real_logits = disc_network(real_images) # Labels?

                if stage==2:
                    fake_labels = tf.zeros_like(real_logits[1])
                    real_labels = tf.ones_like(real_logits[1])
                elif stage==0:
                    fake_labels = tf.zeros_like(real_logits[1])
                    ext, output = real_logits
                    real_weights = tf.ones_like(output)
                    real_labels = tf.ones_like(real_logits[1])

                    for i in range(batch_size):
                        x1 =  self.bbox[0][i]
                        x2 =  self.bbox[2][i]
                        y1 =  self.bbox[1][i]
                        y2 =  self.bbox[3][i]

                        """All the patches in NxN from a1:a2 (along rows) and b1:b2 (along columns) will be masked, and loss will only be computed from remaining members in NxN"""
                        a1 = tf.math.maximum(tf.Variable(0.0), tf.math.ceil((x1 - self.receptive_field)/(self.patch_stride)))
                        a2 = tf.math.minimum(tf.Variable(tf.cast(self.num_out-1, tf.float32)), tf.math.floor((self.num_out-1) - ((126 - self.receptive_field) - x2)/(self.patch_stride))) + 1
                        b1 = tf.math.maximum(tf.Variable(0.0), tf.math.ceil((y1 - self.receptive_field)/self.patch_stride))
                        b2 = tf.math.minimum(tf.Variable(tf.cast(self.num_out-1, tf.float32)), tf.math.floor((self.num_out-1) - ((126 - self.receptive_field) - y2)/(self.patch_stride))) + 1

                        if x1 != x2 and y1 != y2:
                            real_weights[i, a1:a2, b1:b2, :] = 0.0

                    norm_real = tf.reduce_sum(real_weights)
                    norm_fake = real_weights.shape[0] * real_weights.shape[1] * real_weights.shape[2] * real_weights.shape[3]
                    real_logits = ext, output
                
                fake_logits = disc_network(tf.stop_gradient(fake_images))

                if stage==0: 
                    error_disc_real = loss(real_logits[1], real_labels)
                    error_disc_real = tf.keras.backend.mean(tf.math.multiply(error_disc_real, real_weights))
                    error_disc_classification = tf.keras.backend.mean(loss(real_logits[0], real_weights))
                    error_disc_fake = loss(fake_logits[1], fake_labels)

                    if norm_real > 0:
                        error_real = error_disc_real * ((norm_fake * 1.0) /(norm_real * 1.0))
                    else:
                        error_real = error_disc_real

                    error_fake = error_disc_fake
                    discriminator_error = ((error_real + error_fake) * cfg.TRAIN['BG_LOSS_WT']) + error_disc_classification
                elif stage==2:
                    error_real = loss1(real_logits[1], real_labels) # Real/Fake loss for the real image
                    error_fake = loss1(fake_logits[1], fake_labels) # Real/Fake loss for the fake image   
                    discriminator_error = error_real + error_fake

            grads = tape.gradient(discriminator_error, disc_network.trainable_variables)
            optimizer.apply_gradients(zip(grads, disc_network.trainable_variables))



    @tf.function
    def train(self):
        # TODO: Train the entire FineGAN
        # TODO: Only for Background and Child phases

        self.patch_stride = 4.0 # Receptive field stride for Backround Stage Discriminator 
        self.num_out = 24 # Patch output size in NxN
        self.receptive_field = 34 # Receptive field of every patch in NxN

if __name__ == '__main__':
    cfg = Config(32)
    gen = GeneratorArchitecture(cfg, '', '')
