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
import time
import PIL
import pickle
import numpy as np
import pandas as pd

from PIL import Image

import matplotlib.pyplot as plt

import tensorflow as tf
assert tf.version.VERSION.startswith('2.')

from .model import GeneratorArchitecture, DiscriminatorArchitecture
from .model import child_to_parent
from .config.config import Config

AUTOTUNE = tf.data.experimental.AUTOTUNE

def save_images(imgs_tcpu, fake_imgs, epoch):

    num = cfg.TRAIN['VIS_COUNT']
    real_img = imgs_tcpu[-1][0:num]

    image = plt.figure()
    ax = image.add_subplot(1,1,1)
    ax.imshow(real_img[0])
    ax.axis("off")
    plt.savefig(f'test/{epoch}_real_sample.png')

    for i in range(len(fake_imgs)):
        fake_img = fake_imgs[i][0:num]

        image = plt.figure()
        ax = image.add_subplot(1,1,1)
        ax.imshow(fake_img)
        ax.axis("off")
        plt.savefig(f'test/{epoch}_fake_sample_{i}.png')

def normalize(input_image):
    input_image = (input_image / 127.5) - 1
    return input_image

def load_class_ids_filenames(class_id_path, filename_path):
    with open(class_id_path, 'rb') as file:
        class_id = pickle.load(file, encoding='latin1')

    with open(filename_path, 'rb') as file:
        filename = pickle.load(file, encoding='latin1')

    return class_id, filename

def load_bbox(data_path='../../CUB data/CUB_200_2011'):
    bbox_path = data_path + '/bounding_boxes.txt'
    image_path = data_path + '/images.txt'

    bbox_df = pd.read_csv(bbox_path, delim_whitespace=True, header=None).astype(int)
    filename_df = pd.read_csv(image_path, delim_whitespace=True, header=None)

    filenames = filename_df[1].tolist()
    bbox_dict = {i[:-4]:[] for i in filenames[:2]}

    for i in range(0, len(filenames)):
        bbox = bbox_df.iloc[i][1:].tolist()
        dict_key = filenames[i][:-4]
        bbox_dict[dict_key] = bbox

    return bbox_dict

def load_images(image_path, bounding_box, size):
    """Crops the image to the bounding box and then resizes it.
    """
    base_size=64

    imsize = []
    for _ in range(3):
        imsize.append(base_size)
        base_size *= 2

    image = Image.open(image_path).convert('RGB')

    w, h = image.size

    if bounding_box is not None:
        r = int(np.maximum(bounding_box[2], bounding_box[3]) * 0.75)
        c_x = int((bounding_box[0] + bounding_box[2]) / 2)
        c_y = int((bounding_box[1] + bounding_box[3]) / 2)
        y1 = np.maximum(0, c_y - r)
        y2 = np.minimum(h, c_y + r)
        x1 = np.maximum(0, c_x - r)
        x2 = np.minimum(w, c_x + r)
        fimg = image.copy()
        fimg_arr = np.array(fimg)
        fimg = Image.fromarray(fimg_arr)
        cimg = image.crop([x1, y1, x2, y2])
        
    retf = []
    retc = []
    re_cimg = cimg.resize([imsize[1], imsize[1]])
    retc.append(re_cimg)
    
    # TODO: Random Crop + Flip and Modify bbox accordingly
    # re_fimg = tf.image.resize(fimg, size=(126*76/64))
    # re_width, re_height = re_fimg.size
    # TODO: Normalize before append
    fimg = fimg.resize([126, 126])
    retf.append(fimg)

    return fimg, re_cimg, bounding_box

def load_data(filename_path, class_id_path, dataset_path, size):
    """Loads the Dataset.
    """
    _, filenames = load_class_ids_filenames(class_id_path, filename_path)
    bbox_dict = load_bbox(dataset_path)

    fimgs_list, cimgs_list, child_code_list, key_list, mod_bbox_list = [], [], [], [], []

    for _, filename in enumerate(filenames):
        bbox = bbox_dict[filename]

        try:
            image_path = f'{dataset_path}/images/{filename}.jpg'
            fimgs, cimgs, mod_bbox = load_images(image_path, bbox, size)

            rand_class = list(np.random.choice(range(200), 1))
            child_code = np.zeros([200,])
            child_code[rand_class] = 1

            fimgs_list.append(normalize(np.array(fimgs)))
            cimgs_list.append(normalize(np.array(cimgs)))
            child_code_list.append(child_code)
            key_list.append(filename)
            mod_bbox_list.append(mod_bbox)

        except Exception as e:
            print(f'{e}')

    fimgs_list = np.array(fimgs_list)
    cimgs_list = np.array(cimgs_list)
    child_code_list = np.array(child_code_list)
    key_list = np.array(key_list)
    mod_bbox_list = np.array(mod_bbox_list)

    return (fimgs_list, cimgs_list, child_code_list, key_list, mod_bbox_list)

def load_finegan_network():
    # TODO: Load the FineGAN network
    start_epoch = 0
    return GeneratorArchitecture(), [DiscriminatorArchitecture() for i in range(2)], len(list(range(2))), start_epoch

def define_optimizers(gen, disc):
    # TODO: Define the appropriate optimizers to be used
    return tf.keras.optimizers.Adam(), tf.keras.optimizers.Adam()


class FineGAN(object):
    """The FineGAN Architecture"""
    def __init__(self, cfg, img_path, train_dataset, **kwargs):
        super(FineGAN, self).__init__(**kwargs)
        self.batch_size = cfg.TRAIN['BATCH_SIZE']
        self.num_epochs = cfg.TRAIN['MAX_EPOCH']
        self.data_dir = img_path
        self.train_dataset = train_dataset
        # self.summary_writer = tf.summary.create_file_writer('./logs')

        # TODO: Define in the train step
        self.discriminators = []
        self.generator = GeneratorArchitecture()

        self.optimizer_gen_list = []
        self.optimizer_disc_list = []

        self.foreground_masks = []
        self.fake_images = []
        self.num_disc = 2
        self.parent_code = None


    # def prepare_dataset(self, batch_size=64):
    #     data_dir = self.data_dir
    #     data = tf.data.Dataset.list_files(str(data_dir + '/images/*/*'))
    #     data = data.map(lambda image: tf.image.resize(image, (128,128,3)), num_parallel_calls=AUTOTUNE)
    #     data = data.cache().shuffle(1000).repeat(2)
    #     return data.batch(self.batch_size).prefetch(num_parallel_calls=AUTOTUNE)

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

    def load_models(self):
        # TODO: Load all the weights
        pass

    @tf.function
    def train_generator(self):
        generator_error = 0.0
        loss1, class_loss, c_code, parent_code = self.loss1, self.class_loss, self.c_code, self.parent_code

        with tf.GradientTape() as tape:
            for i in range(self.num_disc):
                outputs = self.discriminators[i](self.fake_images[1])

                if i==0 or i==2:
                    real_labels = tf.ones_like(outputs[1])
                    gen_loss = loss1(outputs[1], real_labels)

                    if i==0:
                        gen_loss *= cfg.TRAIN['BG_LOSS_WT']
                        # Background/Foreground classification loss for the fake background
                        gen_class = class_loss(outputs[0], real_labels)
                        gen_loss += gen_class

                    generator_error += gen_loss

                if i==1:
                    # Information maximizing loss for parent stage
                    parent_prediction = self.discriminators[i](self.foreground_masks[i-1])
                    gen_info_loss = class_loss(parent_prediction[0], tf.where(parent_code)[:, 1])
                elif i==2:
                    # Information maximizing loss for child stage
                    child_prediction = self.discriminators[2](self.foreground_masks[i-1])
                    gen_info_loss = class_loss(child_prediction[0], tf.where(c_code)[:, 1])

                if i>0:
                    generator_error += gen_info_loss

            # TODO: Tensorboard Summary for train_generator

        grads = tape.gradient(generator_error, self.generator.trainable_variables)

        for index in range(len(self.discriminators)):
            self.optimizer_gen_list[index].apply_gradients(zip(grads, self.generator.trainable_variables))           

        return generator_error

    @tf.function
    def train_discriminator(self, stage, count=0):
        if stage==0 or stage==2:

            with tf.GradientTape() as tape:
                batch_size = tf.shape(self.real_fimages)[0]
                loss, loss1 = self.loss, self.loss1 # Binary Crossentropy

                optimizer = self.optimizer_disc_list[stage] # stage_wise optimizer
                disc_network = self.discriminators[stage]

                if stage==0:
                    real_images = self.real_fimages[0]
                elif stage==2:
                    real_images = self.real_cimages[0]

                fake_images = self.fake_images[stage]
                real_logits = disc_network(real_images)

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

                        """All the patches in NxN from a1:a2 (along rows) and b1:b2 (along columns) will be masked, 
                        and loss will only be computed from remaining members in NxN"""
                        a1 = tf.math.maximum(tf.Variable(0.0), 
                                tf.math.ceil((x1 - self.receptive_field)/(self.patch_stride)))
                        a2 = tf.math.minimum(tf.Variable(tf.cast(self.num_out-1, tf.float32)), 
                                tf.math.floor((self.num_out-1) - ((126 - self.receptive_field) - x2)/(self.patch_stride))) + 1
                        b1 = tf.math.maximum(tf.Variable(0.0), 
                                tf.math.ceil((y1 - self.receptive_field)/self.patch_stride))
                        b2 = tf.math.minimum(tf.Variable(tf.cast(self.num_out-1, tf.float32)), 
                                tf.math.floor((self.num_out-1) - ((126 - self.receptive_field) - y2)/(self.patch_stride))) + 1

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

                # TODO: Tensorboard Summary for train_discriminator

            grads = tape.gradient(discriminator_error, disc_network.trainable_variables)
            optimizer.apply_gradients(zip(grads, disc_network.trainable_variables))

            return discriminator_error


    @tf.function
    def train(self):
        
        self.patch_stride = 4.0 # Receptive field stride for Backround Stage Discriminator 
        self.num_out = 24 # Patch output size in NxN
        self.receptive_field = 34 # Receptive field of every patch in NxN

        self.generator, self.discriminators, self.num_disc, start_epoch = load_finegan_network()
        # TODO: Deepcopy the weights for generator?
        # Deepcopy here:

        self.optimizer_gen_list, self.optimizer_disc_list = define_optimizers(self.generator, self.discriminators)

        self.loss = tf.keras.losses.BinaryCrossentropy(reduction=False)
        self.loss1 = tf.keras.losses.BinaryCrossentropy()
        self.class_loss = tf.keras.losses.CategoricalCrossentropy()

        self.real_labels = tf.Variable(tf.ones_like(self.batch_size, dtype=tf.float32))
        self.fake_labels = tf.Variable(tf.zeros_like(self.batch_size, dtype=tf.float32))

        z_dims = cfg.GAN['Z_DIM']
        noise = tf.Variable(tf.random.normal(shape=(self.batch_size, z_dims)))
        # latent_noise = tf.Variable(tf.random.normal(shape=(self.batch_size, z_dims)))
        # fixed_noise = tf.Variable(tf.random.normal(shape=(self.batch_size, z_dims)))
        # hard_noise = tf.Variable(tf.random.normal(shape=(self.batch_size, z_dims)))

        print(f'[INFO] Starting FineGAN Training...')

        for epoch in range(start_epoch, self.num_epochs):
            start_time = time.time()

            for _, data in enumerate(self.train_dataset):
                self.imgs_tcpu, self.real_fimages, self.real_cimages, self.c_code, self.bbox = self.prepare_data(data)

                self.fake_images, self.foreground_images, self.mask_images, self.foreground_masks = self.generator(noise, self.c_code)

                self.parent_code = child_to_parent(self.c_code, cfg.FINE_GRAINED_CATEGORIES, cfg.SUPER_CATEGORIES) 

                total_discriminator_error = 0.0
                for index in range(self.num_disc):
                    if index==0 or index==2:
                        discriminator_error = self.train_discriminator(index)
                        total_discriminator_error += discriminator_error

                total_generator_error = self.train_generator()
                # TODO: 0.999 * avg_weights + 0.1 * actual_weights ---> Stable model
                print(f'[INFO] FineGAN Gen Error: {total_generator_error}. Disc Error: {total_discriminator_error}')

            end_time = time.time()
            print(f'[INFO] Epoch: {epoch}/{self.num_epochs} took {(end_time-start_time):.2}s.') 

        print(f'[INFO] Saving model after {self.num_epochs} epochs')
        # TODO: save the model


        # TODO [OPTIONAL]: Hard Negative Mining


if __name__ == '__main__':
    cfg = Config(32)
    print(f'[INFO] Initialize CUB Dataset...')

    data_dir = '../../CUB data/CUB_200_2011'
    filename_path = data_dir + "/filenames.pickle"
    class_id_path = data_dir + "/class_info.pickle"
    dataset_path = "../../CUB data/CUB_200_2011"
    print(f'[INFO] Before loading CUB Dataset...')
    train_dataset = load_data(filename_path, class_id_path, dataset_path, size=(128,128))

    print(f'[INFO] After loading CUB Dataset...')

    # algo = FineGAN(cfg, data_dir, train_dataset)

    # print(f'[INFO] CUB Dataset Initialized...')

    # print(f'[INFO] FineGAN Initialization Complete...')
    # print(f'[INFO] FineGAN Training starts...')
    # start_t = time.time()
    # algo.train()
    # end_t = time.time()
    # print(f'Total time for training: {end_t - start_t}')
    # print(f'[INFO] FineGAN Training Complete...')
