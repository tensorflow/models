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
        
    def prepare_dataset(self):
        data = tf.data.Dataset.list_files('./dataset/*'))
        data = data.map(lambda image: tf.image.resize(image, (128,128,3)), num_parallel_calls=AUTOTUNE)
        data = data.cache().shuffle(1000).repeat(2)
        return data.batch(self.batch_size).prefetch(num_parallel_calls=AUTOTUNE)

    def plot_images(images, num_cols=None):
	num_cols = num_cols or len(images)
	num_rows = (len(images) - 1) // num_cols + 1
	if images.shape[-1] == 1:
		images = np.squeeze(images, axis=-1)
	plt.figure(figsize=(num_cols, num_rows))
	for index, image in enumerate(images):
		plt.subplot(num_rows, num_cols, index + 1)
		plt.imshow(image)
		plt.axis("off")

    def train_generator():
        # TODO: Train the Generators
        pass

    def train_discriminator():
        # TODO: Train the Discriminator
        pass

    def train():
        # TODO: Train the entire FineGAN
        # TODO: Only for Background and Child phases
        pass

if __name__ == '__main__':
    cfg = Config(32)
    gen = GeneratorArchitecture(cfg, '', '')
