from __future__ import print_function

import keras
from keras import layers

# Model adapted from keras/examples
# Model parameter
# ----------------------------------------------------------------------------
#           |      | 200-epoch | Orig Paper| 200-epoch | Orig Paper| sec/epoch
# Model     |  n   | ResNet v1 | ResNet v1 | ResNet v2 | ResNet v2 | GTX1090Ti
#           |      | %Accuracy | %Accuracy | %Accuracy | %Accuracy | v1 (v2)
# ----------------------------------------------------------------------------
# ResNet20  |  3   | 92.16     | 91.25     | -----     | NA        | 35
# ResNet32  |  5   | 92.46     | 92.49     | -----     | NA        | 50
# ResNet44  |  7   | 92.50     | 92.83     | -----     | NA        | 70
# ResNet56  |  9   | 92.71     | 93.03     | 92.60     | NA        | 90 (100)
# ResNet110 |  18  | 92.65     | 93.39     | 93.03     | 93.63     | 165(180)
# ---------------------------------------------------------------------------
N = 8

# Model version
# Orig paper: version = 1 (ResNet v1), Improved ResNet: version = 2 (ResNet v2)
VERSION = 1

# Computed depth from supplied model parameter n
DEPTH = N * 6 + 2

def resnet_block(inputs, num_filters=16, kernel_size=3, strides=1,
    activation='relu', conv_first=True):
  """2D Convolution-Batch Normalization-Activation stack builder

  # Arguments
      inputs (tensor): input tensor from input image or previous layer
      num_filters (int): Conv2D number of filters
      kernel_size (int): Conv2D square kernel dimensions
      strides (int): Conv2D square stride dimensions
      activation (string): activation name
      conv_first (bool): conv-bn-activation (True) or
          activation-bn-conv (False)

  # Returns
      x (tensor): tensor as input to the next layer
  """
  if conv_first:
    x = layers.Conv2D(num_filters,
                      kernel_size=kernel_size,
                      strides=strides,
                      padding='same',
                      kernel_initializer='he_normal',
                      kernel_regularizer=keras.regularizers.l2(1e-4))(inputs)
    x = layers.BatchNormalization()(x)
    if activation:
      x = layers.Activation(activation)(x)
    return x

  x = layers.BatchNormalization()(inputs)
  if activation:
    x = layers.Activation('relu')(x)
  x = layers.Conv2D(num_filters,
                    kernel_size=kernel_size,
                    strides=strides,
                    padding='same',
                    kernel_initializer='he_normal',
                    kernel_regularizer=keras.regularizers.l2(1e-4))(x)
  return x


def resnet_v1(input_shape, depth, num_classes=10, input_tensor=None):
  """ResNet Version 1 Model builder [a]

  Stacks of 2 x (3 x 3) Conv2D-BN-ReLU
  Last ReLU is after the shortcut connection.
  The number of filters doubles when the feature maps size
  is halved.
  The Number of parameters is approx the same as Table 6 of [a]:
  ResNet20 0.27M
  ResNet32 0.46M
  ResNet44 0.66M
  ResNet56 0.85M
  ResNet110 1.7M

  # Arguments
      input_shape (tensor): shape of input image tensor
      depth (int): number of core convolutional layers
      num_classes (int): number of classes (CIFAR10 has 10)

  # Returns
      model (Model): Keras model instance
  """
  if (depth - 2) % 6 != 0:
    raise ValueError('depth should be 6n+2 (eg 20, 32, 44 in [a])')
  # Start model definition.
  if input_tensor is None:
    img_input = layers.Input(shape=input_shape)
  else:
    if not keras.backend.is_keras_tensor(input_tensor):
      img_input = layers.Input(tensor=input_tensor, shape=input_shape)
    else:
      img_input = input_tensor
  num_filters = 16
  num_sub_blocks = int((depth - 2) / 6)

  x = resnet_block(inputs=img_input)
  # Instantiate convolutional base (stack of blocks).
  for i in range(3):
    for j in range(num_sub_blocks):
      strides = 1
      is_first_layer_but_not_first_block = j == 0 and i > 0
      if is_first_layer_but_not_first_block:
        strides = 2
      y = resnet_block(inputs=x,
                       num_filters=num_filters,
                       strides=strides)
      y = resnet_block(inputs=y,
                       num_filters=num_filters,
                       activation=None)
      if is_first_layer_but_not_first_block:
        x = resnet_block(inputs=x,
                         num_filters=num_filters,
                         kernel_size=1,
                         strides=strides,
                         activation=None)
      x = keras.layers.add([x, y])
      x = layers.Activation('relu')(x)
    num_filters = 2 * num_filters

  # Add classifier on top.
  # v1 does not use BN after last shortcut connection-ReLU
  x = layers.AveragePooling2D(pool_size=8)(x)
  y = layers.Flatten()(x)
  outputs = layers.Dense(num_classes,
                  activation='softmax_crossentropy',
                  kernel_initializer='he_normal')(y)

  # Instantiate model.
  model = keras.engine.Model(inputs=img_input, outputs=outputs)
  return model
