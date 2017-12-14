from __future__ import print_function
import keras
from keras.layers import Dense, Conv2D, BatchNormalization, Activation
from keras.layers import AveragePooling2D, Input, Flatten
from keras.regularizers import l2
from keras import backend as K
from keras.models import Model
from keras.engine.topology import get_source_inputs

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
n = 8

# Model version
# Orig paper: version = 1 (ResNet v1), Improved ResNet: version = 2 (ResNet v2)
version = 1

# Computed depth from supplied model parameter n
depth = n * 6 + 2

# Model name, depth and version
model_type = 'ResNet%dv%d' % (depth, version)

def resnet_block(inputs,
    num_filters=16,
    kernel_size=3,
    strides=1,
    activation='relu',
    conv_first=True):
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
    x = Conv2D(num_filters,
               kernel_size=kernel_size,
               strides=strides,
               padding='same',
               kernel_initializer='he_normal',
               kernel_regularizer=l2(1e-4))(inputs)
    x = BatchNormalization()(x)
    if activation:
      x = Activation(activation)(x)
    return x
  x = BatchNormalization()(inputs)
  if activation:
    x = Activation('relu')(x)
  x = Conv2D(num_filters,
             kernel_size=kernel_size,
             strides=strides,
             padding='same',
             kernel_initializer='he_normal',
             kernel_regularizer=l2(1e-4))(x)
  return x


def resnet_v1(input_shape, depth, num_classes=10,input_tensor=None):
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
    img_input = Input(shape=input_shape)
  else:
    if not K.is_keras_tensor(input_tensor):
      img_input = Input(tensor=input_tensor, shape=input_shape)
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
      x = Activation('relu')(x)
    num_filters = 2 * num_filters

  # Add classifier on top.
  # v1 does not use BN after last shortcut connection-ReLU
  x = AveragePooling2D(pool_size=8)(x)
  y = Flatten()(x)
  outputs = Dense(num_classes,
                  activation='softmax',
                  kernel_initializer='he_normal')(y)

  # Ensure that the model takes into account
  # any potential predecessors of `input_tensor`.
  if input_tensor is not None:
    inputs = get_source_inputs(input_tensor)
  else:
    inputs = img_input
  # Instantiate model.
  model = Model(inputs=inputs, outputs=outputs)
  return model