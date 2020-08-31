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

from model import GeneratorArchitecture, DiscriminatorArchitecture
from model import child_to_parent
from config.config import Config

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
    input_image = np.array(input_image)
    input_image = (input_image / 127.5) - 1.
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
        
    re_cimg = cimg.resize([imsize[1], imsize[1]])
        
    re_fimg = fimg.resize([int(126 * 76/64), int(126 * 76/64)])
    re_w, re_h = re_fimg.size
    
    x_crop = re_w - 126
    y_crop = re_h - 126
    
    # Return cropped image
    cropped_re_fimg = re_fimg.crop([x_crop, y_crop, x_crop+126, y_crop+126])
    
    mod_x1 = bounding_box[0] * re_w / w
    mod_y1 = bounding_box[1] * re_h / h
    mod_x2 = mod_x1 + (bounding_box[2] * re_w / w)
    mod_y2 = mod_y1 + (bounding_box[3] * re_h / h)
    
    mod_x1 = min(max(0, mod_x1 - x_crop), 126)
    mod_y1 = min(max(0, mod_y1 - y_crop), 126)
    mod_x2 = max(min(126, mod_x2 - x_crop),0)
    mod_y2 = max(min(126, mod_y2 - y_crop),0)

    random_flag = np.random.randint(2)
    if(random_flag == 0):
        cropped_re_fimg = cropped_re_fimg.transpose(Image.FLIP_LEFT_RIGHT)
        flipped_x1 = 126 - mod_x2
        flipped_x2 = 126 - mod_x1
        mod_x1 = flipped_x1
        mod_x2 = flipped_x2
    
    modified_bbox = []
    modified_bbox.append(mod_y1)
    modified_bbox.append(mod_x1)
    modified_bbox.append(mod_y2)
    modified_bbox.append(mod_x2)

    return cropped_re_fimg, re_cimg, modified_bbox


def load_data(filename_path, class_id_path, dataset_path, size):
    """Loads the Dataset.
    """
    class_id, filenames = load_class_ids_filenames(class_id_path, filename_path)
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

            fimgs_list.append(normalize(fimgs))
            cimgs_list.append(normalize(np.asarray(cimgs)))
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


def load_finegan_network(cfg):
    generator = GeneratorArchitecture(cfg)
    print(f'[INFO] Initialized Generator...')
    
    discriminators = []
    for i in range(3): # 3 discriminators for background, parent and child stage
        discriminators.append(DiscriminatorArchitecture(cfg, i))
    print(f'[INFO] Initialized Discriminators...')
        
    start_epoch = 0    
    return generator, discriminators, len(discriminators), start_epoch


def define_optimizers(gen, disc):   
    optimizers_disc = []
    num_disc = len(disc)
    
    for i in range(3):      
        opt = tf.keras.optimizers.Adam(learning_rate=cfg.TRAIN['DISCRIMINATOR_LR'], beta_1=0.5, beta_2=0.999)
        optimizers_disc.append(opt)

    optimizers_gen = []
    optimizers_gen.append(tf.keras.optimizers.Adam(learning_rate=cfg.TRAIN['GENERATOR_LR'], beta_1=0.5, beta_2=0.999))

    for i in range(1,3):
        optimizers_gen.append(tf.keras.optimizers.Adam(learning_rate=cfg.TRAIN['GENERATOR_LR'], beta_1=0.5, beta_2=0.999))

    return optimizers_gen, optimizers_disc


def casting_func(fimg, cimg, child_code, mod_bbox):
    fimg = tf.cast(fimg, dtype=tf.float32)
    cimg = tf.cast(cimg, dtype=tf.float32)
    child_code = tf.cast(child_code, dtype=tf.float32)
    mod_bbox = tf.cast(mod_bbox, dtype=tf.float32)
    return fimg, cimg, child_code, mod_bbox


class FineGAN(object):
    """The FineGAN Architecture"""
    def __init__(self, cfg, img_path, train_dataset, **kwargs):
        super(FineGAN, self).__init__(**kwargs)
        self.batch_size = cfg.TRAIN['BATCH_SIZE']
        self.num_epochs = cfg.TRAIN['MAX_EPOCH']
        self.data_dir = img_path
        self.train_dataset = train_dataset
        self.num_disc = 3
        
        self.num_batches = int(train_dataset[0].shape[0] / self.batch_size)

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
    
    
    @tf.function
    def train_generator(self):
        generator_error = 0.0
        loss1, class_loss, c_code, parent_code = self.loss1, self.class_loss, self.c_code, self.parent_code

        with tf.GradientTape(persistent=True) as tape:
            
            z_dims = cfg.GAN['Z_DIM']
            noise = tf.random.normal(shape=(self.batch_size, z_dims))
            self.fake_images, self.foreground_images, self.mask_images, self.foreground_masks = self.generator(noise, self.c_code, p_code=parent_code)
            
            for i in range(self.num_disc):
                outputs = self.discriminators[i](self.fake_images[i])

                if i==0 or i==2:
                    real_labels = tf.ones_like(outputs[1])
                    gen_loss = loss1(real_labels, outputs[1])

                    if i==0:
                        gen_loss *= cfg.TRAIN['BG_LOSS_WT']
                        # Background/Foreground classification loss for the fake background
                        gen_class = loss1(real_labels, outputs[0])
                        gen_loss += gen_class

                    generator_error += gen_loss
                
                if i==1:
                    # Information maximizing loss for parent stage
                    parent_prediction = self.discriminators[1](self.foreground_masks[i-1])                   
                    parent_code = tf.cast(parent_code, dtype=tf.float32)
                    gen_info_loss = class_loss(parent_code, parent_prediction[0])
                    stage_1_info_loss = gen_info_loss * 2
                                        
                elif i==2:
                    # Information maximizing loss for child stage
                    child_prediction = self.discriminators[2](self.foreground_masks[i-1])
                    c_code = tf.cast(c_code, dtype=tf.float32)
                    gen_info_loss = class_loss(c_code, child_prediction[0])
                    stage_2_info_loss = gen_info_loss * 2

                if i>0:
                    generator_error += gen_info_loss

        grads_1 = tape.gradient(generator_error, self.generator.trainable_variables)
        grads_2 = tape.gradient(generator_error, self.discriminators[1].trainable_variables[:-1])
        #grads_gen_2 = tape.gradient(stage_1_info_loss, self.generator.trainable_variables[19:57])
        grads_3 = tape.gradient(generator_error, self.discriminators[2].trainable_variables[-3:-1])
        #grads_gen_3 = tape.gradient(stage_2_info_loss, self.generator.trainable_variables[57:])
        
        for index in range(3):
            if index == 0:
                self.optimizer_gen_list[index].apply_gradients(zip(grads_1, self.generator.trainable_variables))
            elif index == 1:
                self.optimizer_gen_list[index].apply_gradients(zip(grads_2, self.discriminators[index].trainable_variables[:-1]))
                #self.optimizer_gen_list[index].apply_gradients(zip(grads_gen_2, self.generator.trainable_variables[19:57]))
            elif index == 2:
                self.optimizer_gen_list[index].apply_gradients(zip(grads_3, self.discriminators[index].trainable_variables[-3:-1]))
                #self.optimizer_gen_list[index].apply_gradients(zip(grads_gen_3, self.generator.trainable_variables[57:]))

        return generator_error
    

    def train_discriminator(self, stage):
        loss, loss1 = self.loss, self.loss1 # Binary Crossentropy

        if stage==0:
            real_images = self.real_fimages
        elif stage==2:
            real_images = self.real_cimages

        fake_images = self.fake_images[stage]
        real_logits = self.discriminators[stage](real_images)
        
        if stage==2:
            fake_labels = tf.zeros_like(real_logits[1])
            real_labels = tf.ones_like(real_logits[1])
            disc_error = self.disc_step_stage2(stage, real_images, real_labels, fake_images, fake_labels, loss, loss1)
        elif stage==0:
            fake_labels = tf.zeros_like(real_logits[1])
            ext, output = real_logits
            real_labels = tf.ones_like(real_logits[1])
            real_weights = np.ones([self.batch_size, 24, 24, 1])
            
            for i in range(self.batch_size):
                
                bbox = self.bbox[i].numpy()
                x1 =  bbox[0]
                x2 =  bbox[2]
                y1 =  bbox[1]
                y2 =  bbox[3]

                """All the patches in NxN from a1:a2 (along rows) and b1:b2 (along columns) will be masked, 
                and loss will only be computed from remaining members in NxN"""

                a1 = (np.maximum(0.0, np.ceil((x1 - self.receptive_field)/(float(self.patch_stride))))).astype(int)
                a2 = (np.minimum(self.num_out-1, np.floor((self.num_out - 1) - ((126 - self.receptive_field) - x2)/self.patch_stride)) + 1).astype(int)
                b1 = (np.maximum(0.0, np.ceil((y1 - self.receptive_field)/self.patch_stride))).astype(int)
                b2 = (np.minimum(self.num_out-1, np.floor((self.num_out-1) - ((126 - self.receptive_field) - y2)/self.patch_stride)) + 1).astype(int)
                
                if x1 != x2 and y1 != y2:
                    real_weights[i, a1:a2, b1:b2, :] = 0.0
            
            real_weights = tf.cast(real_weights, dtype=tf.float32)
            disc_error = self.disc_step_stage0(stage, real_images, real_labels, fake_images, fake_labels, loss, loss1, real_weights)
        return disc_error
    
        
    @tf.function
    def disc_step_stage0(self, stage, real_images, real_labels, fake_images, fake_labels, loss, loss1, real_weights):
        with tf.GradientTape() as tape:
            real_logits = self.discriminators[stage](real_images)
            fake_logits = self.discriminators[stage](tf.stop_gradient(fake_images))
            ext, output = real_logits

            norm_real = tf.reduce_sum(real_weights)
            norm_fake = self.batch_size * real_weights.shape[1] * real_weights.shape[2] * real_weights.shape[3]
            real_logits = ext, output
        
            error_disc_real = loss(real_labels, real_logits[1])
            error_disc_real = tf.keras.backend.mean(tf.math.multiply(error_disc_real, real_weights))
            error_disc_classification = tf.keras.backend.mean(loss(real_weights, real_logits[0]))
            error_disc_fake = loss(fake_labels, fake_logits[1])
            error_disc_fake = tf.keras.backend.mean(error_disc_fake)

            if norm_real > 0:
                error_real = error_disc_real * ((norm_fake * 1.0) / (norm_real * 1.0))
            else:
                error_real = error_disc_real
            
            error_real = error_disc_real
            error_fake = error_disc_fake
            discriminator_error = ((error_real + error_fake) * cfg.TRAIN['BG_LOSS_WT']) + error_disc_classification

        grads = tape.gradient(discriminator_error, self.discriminators[stage].trainable_variables)
        self.optimizer_disc_list[stage].apply_gradients(zip(grads, self.discriminators[stage].trainable_variables))
        return discriminator_error
    
    
    @tf.function
    def disc_step_stage2(self, stage, real_images, real_labels, fake_images, fake_labels, loss, loss1):
        with tf.GradientTape() as tape:
            real_logits = self.discriminators[stage](real_images)
            fake_logits = self.discriminators[stage](fake_images)
            
            error_real = loss1(real_labels, real_logits[1]) # Real/Fake loss for the real image
            error_fake = loss1(fake_labels, fake_logits[1]) # Real/Fake loss for the fake image   
            discriminator_error = error_real + error_fake

        grads = tape.gradient(discriminator_error, self.discriminators[stage].trainable_variables)
        self.optimizer_disc_list[stage].apply_gradients(zip(grads, self.discriminators[stage].trainable_variables))
        return discriminator_error
        
    
    def train_model(self, start=True):
        self.patch_stride = 4.0 # Receptive field stride for Backround Stage Discriminator 
        self.num_out = 24 # Patch output size in NxN
        
        self.receptive_field = 34 # Receptive field of every patch in NxN

        print(f'[INFO] Starting FineGAN Training...')
        self.generator, self.discriminators, self.num_disc, start_epoch = load_finegan_network(cfg)
        
        if start == False:
            self.generator.load_weights('./Checkpoints/generator_225_epochs.ckpt')
            self.discriminators[0].load_weights('./Checkpoints/discriminator_0_225_epochs.ckpt')
            self.discriminators[1].load_weights('./Checkpoints/discriminator_1_225_epochs.ckpt')
            self.discriminators[2].load_weights('./Checkpoints/discriminator_2_225_epochs.ckpt')
            print(f'[INFO] Loaded FineGAN Weights...')
            print(f'[INFO] Continuing FineGAN Training...')
            
        self.optimizer_gen_list, self.optimizer_disc_list = define_optimizers(self.generator, self.discriminators)

        self.loss = tf.keras.losses.BinaryCrossentropy()
        self.loss1 = tf.keras.losses.BinaryCrossentropy()
        self.class_loss = tf.keras.losses.CategoricalCrossentropy()

        self.real_labels = tf.ones_like(self.batch_size, dtype=tf.float32)
        self.fake_labels = tf.zeros_like(self.batch_size, dtype=tf.float32)
        
        fimgs_list, cimgs_list, child_code_list, key_list, mod_bbox_list = self.train_dataset
        
        data = tf.data.Dataset.from_tensor_slices((fimgs_list, cimgs_list, child_code_list, mod_bbox_list))
        data = data.batch(16)
        data = data.map(lambda fimgs, cimgs, child_code, mod_bbox: casting_func(fimgs, cimgs, child_code, mod_bbox), num_parallel_calls=AUTOTUNE)
        data = data.shuffle(1000).prefetch(2)
                    
        start_epoch = 0
        self.num_epochs = 100
                
        for epoch in range(start_epoch, self.num_epochs):
                
            print(f'[Train] FineGAN Epoch: {epoch+1}/{self.num_epochs}')
            start_time = time.time()
            count = 0
            for i, batch in enumerate(data):
                self.real_fimages, self.real_cimages, self.c_code, self.bbox = batch
                
                if self.c_code.shape[0] != self.batch_size:
                    continue
                                        
                ratio = 10
                child_code = self.c_code.numpy()
                arg_parent = (np.argmax(child_code, axis=1)/int(ratio)).astype(int)
                parent_code = np.zeros([child_code.shape[0], 20])
                
                for i in range(child_code.shape[0]):
                    parent_code[i][arg_parent[i]] = 1
                
                parent_code = tf.Variable(parent_code)
                self.parent_code = tf.cast(parent_code, dtype=tf.float32)
                                
                z_dims = cfg.GAN['Z_DIM']
                noise = tf.random.normal(shape=(self.batch_size, z_dims))
                self.fake_images, self.foreground_images, self.mask_images, self.foreground_masks = self.generator(noise, self.c_code, p_code=self.parent_code)

                total_discriminator_error = 0.0
                for index in range(self.num_disc):
                    if index==0 or index==2:
                        discriminator_error = self.train_discriminator(index)
                        total_discriminator_error += discriminator_error
                
                total_generator_error = self.train_generator()
                
                if count%50 == 0:
                    print(f'Epoch {epoch+1} Batch: {count+1}')
                    print(f'Discriminator Error: {total_discriminator_error}')
                    print(f'Generator Error: {total_generator_error}')
                count += 1
            end_time = time.time()
            print(f'[INFO] Epoch: {epoch+1}/{self.num_epochs} took {(end_time-start_time):.2}s.')
            
            
        print(f'[INFO] Saving model after {self.num_epochs} epochs')
        self.generator.save_weights('./Checkpoints/generator_300_epochs.ckpt')
        self.discriminators[0].save_weights('./Checkpoints/discriminator_0_300_epochs.ckpt')
        self.discriminators[1].save_weights('./Checkpoints/discriminator_1_300_epochs.ckpt')
        self.discriminators[2].save_weights('./Checkpoints/discriminator_2_300_epochs.ckpt')
        return

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

    algo = FineGAN(cfg, data_dir, train_dataset)

    print(f'[INFO] FineGAN Training starts...')
    start_t = time.time()
    algo.train_model(start=True)
    end_t = time.time()
    print(f'Total time for training: {end_t - start_t}s')
    print(f'[INFO] FineGAN Training Complete...')
