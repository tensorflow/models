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
"""Training BatchNorm and Only BatchNorm: On the Expressive Power of Random 
Features in CNNs - Experiments


@misc{frankle2020training,
    title={Training BatchNorm and Only BatchNorm: On the Expressive Power of Random Features in CNNs},
    author={Jonathan Frankle and David J. Schwab and Ari S. Morcos},
    year={2020},
    eprint={2003.00152},
    archivePrefix={arXiv},
    primaryClass={cs.LG}
}

"""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import random
import tensorflow as tf
import tensorflow_addons as tfa
import matplotlib.pyplot as plt
print(tf.version.VERSION)

from tensorflow.keras.layers import Conv2D, BatchNormalization, Activation, Input
from tensorflow.keras.layers import add, AveragePooling2D, Dense, Flatten
from tensorflow.keras.regularizers import l2
from tensorflow.keras.optimizers import SGD, Adam
from tensorflow_addons.optimizers import SGDW
from tensorflow.keras.initializers import RandomNormal
from tensorflow.keras.losses import CategoricalCrossentropy
from tensorflow.keras.datasets.cifar10 import load_data

AUTOTUNE = tf.data.experimental.AUTOTUNE

# Image augmentation
def augment(img, label):
    image = tf.cast(img, tf.float32)
    image = tf.image.random_flip_left_right(image)
    image = tf.image.resize_with_pad(image, 36, 36)
    image = tf.image.random_crop(image, size=[32, 32, 3])
    image = (image / 255.0)
    return image, label

def normalize(img, label):
    image = tf.cast(img, tf.float32)
    image = (image / 255.0)
    return image, label

# Creating a custom callback to change the learning rate at 80 and 120 epochs.

def scheduler(epoch, lr):
    if epoch in [80, 120]:
        return lr * 0.1
    return lr
    
# OnlyBN ResNet Architecture
class ResNet:
    
    def residual_block(data, filters, strides, transition):
        shortcut = data

        x = Conv2D(filters, 3, strides, padding="same", kernel_initializer="he_normal", 
                   use_bias=False)(data)
        x = BatchNormalization(beta_initializer='zeros', gamma_initializer=RandomNormal(mean=0.0, stddev=1.0))(x)
        x = Activation("relu")(x)
        
        x = Conv2D(filters, 3, 1, padding="same", kernel_initializer="he_normal", 
                   use_bias=False)(x)
        x = BatchNormalization(beta_initializer='zeros', gamma_initializer=RandomNormal(mean=0.0, stddev=1.0))(x)
        x = Activation("relu")(x)
        
        if transition:
            shortcut = Conv2D(filters, 1, 2, padding="valid", kernel_initializer="he_normal", 
                              use_bias=False)(shortcut)
            shortcut = BatchNormalization(beta_initializer='zeros', 
                                          gamma_initializer=RandomNormal(mean=0.0, stddev=1.0))(shortcut)

        return add([shortcut, x])
    
    def build(num_blocks=2, filters_block=[16,32,64]):
        inputs = Input(shape=(32,32,3))
        x = Conv2D(16, 3, padding="same", kernel_initializer="he_normal", use_bias=False)(inputs)
        
        for i in range(3):
            for j in range(num_blocks):
                if j==0:
                    transition = True
                    strides = 2
                else:
                    transition = False
                    strides = 1
                    
                x = ResNet.residual_block(x, filters_block[i], strides, transition)
                
        avg_pool = AveragePooling2D(3)(x)
        x = Dense(10, use_bias=False, kernel_initializer='he_normal')(avg_pool)
        x = Flatten()(x)        
        outputs = Activation("softmax")(x)
                
        return tf.keras.Model(inputs=inputs, outputs=outputs)


if __name__ == '__main__':

    # tf.data Training pipeline
    (X_train, y_train), (X_test, y_test) = load_data()

    X_train, X_val = X_train[5000:], X_train[:5000]
    y_train, y_val = y_train[5000:], y_train[:5000]

    train_dataset = tf.data.Dataset.from_tensor_slices((X_train, y_train))
    train_dataset = train_dataset.cache().shuffle(2048).map(augment, AUTOTUNE)
    train_dataset = train_dataset.batch(128).prefetch(AUTOTUNE)

    val_dataset = tf.data.Dataset.from_tensor_slices((X_val, y_val))
    val_dataset = val_dataset.cache().shuffle(2048).map(augment, AUTOTUNE)
    val_dataset = val_dataset.batch(128).prefetch(AUTOTUNE)

    test_dataset = tf.data.Dataset.from_tensor_slices((X_test, y_test))
    test_dataset = test_dataset.cache().shuffle(2048).map(augment, AUTOTUNE)
    test_dataset = test_dataset.batch(128).prefetch(AUTOTUNE)

    callbacks = [tf.keras.callbacks.LearningRateScheduler(scheduler, verbose=1)]

    # Build the model
    """
    The parameter count matches the exact count in the paper. Please use the appropriate value for `num_blocks` to train the required ResNet architecture.
    - ResNet-14 (use `num_blocks=2`)
    - ResNet-32 (use `num_blocks=5`)
    - ResNet-56 (use `num_blocks=9`)
    - ResNet-110 (use `num_blocks=18`)
    - ResNet-218 (use `num_blocks=36`)
    - ResNet-434 (use `num_blocks=72`)
    - ResNet-866 (use `num_blocks=144`)
    """
    model = ResNet.build(num_blocks=2) # This trains a ResNet-14 model

    # Set only batchnorm layers to be trainable
    count_conv = 0
    for layer in model.layers:
        if not isinstance(layer, BatchNormalization):
            if hasattr(layer, 'trainable'):
                layer.trainable = False
        if isinstance(layer, Conv2D):
            count_conv += 1
    print(f'Total Number of Conv layers: {count_conv - 2}')


    # Compile the model
    step = tf.Variable(0, trainable=False)
    schedule = tf.optimizers.schedules.PiecewiseConstantDecay(
        [28125, 42185], [1e-0, 1e-1, 1e-2])

    wd = lambda: 1e-4 * schedule(step)

    model.compile(optimizer=SGD(learning_rate=0.1, momentum=0.9), loss="sparse_categorical_crossentropy", metrics=['accuracy'])

    # Train the model"""
    history = model.fit(train_dataset, validation_data=val_dataset, epochs=160, batch_size=128, callbacks=callbacks)
    model.evaluate(test_dataset)

    # Visualize the training
    plt.plot(history.history['accuracy'])
    plt.title('model accuracy')
    plt.ylabel('accuracy')
    plt.xlabel('epoch')
    plt.legend(['train', 'test'], loc='upper left')
    plt.show()

    plt.plot(history.history['loss'])
    plt.title('model loss')
    plt.ylabel('loss')
    plt.xlabel('epoch')
    plt.legend(['train', 'test'], loc='upper left')
    plt.show()
