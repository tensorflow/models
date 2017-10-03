# Copyright 2017 The TensorFlow Authors. All Rights Reserved.
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

"""YOLOFeatureExtractor for YOLO v1"""

import tensorflow as tf

from object_detection.meta_architectures import yolo_meta_arch

slim = tf.contrib.slim

class YOLOv1FeatureExtractor(yolo_meta_arch.YOLOFeatureExtractor):
  """
    YOLO Feature Extractor
    Written with reference to Darknet and YoloTensorFlow229
  """

  def __init__(self,
               is_training,
               reuse_weights=None):
    # TODO : find out the parameters
    super(YOLOFeatureExtractor, self).__init__(
      is_training, reuse_weights)


  def preprocess(self, resized_inputs):
    """Preprocesses images for feature extraction (minus image resizing).

    Args:
      resized_inputs: a [batch, height, width, channels] float tensor
        representing a batch of images.

    Returns:
      preprocessed_inputs: a [batch, height, width, channels] float tensor
        representing a batch of images.
    """
    pass


  def extract_features(self, preprocessed_inputs):
    """Extracts features from preprocessed inputs.

    This function is responsible for extracting the YOLO feature map from preprocessed
    images.

    Args:
      preprocessed_inputs: a [batch, height, width, channels] float tensor
        representing a batch of images.

    Returns:
      feature_maps: a tensor where the tensor has shape
        [batch, grid_size, grid_size, depth]
    """
    if len(preprocessed_inputs.get_shape().as_list()) != 4:
      raise ValueError('`preprocessed_inputs` must be 4 dimensional, got a '
                       'tensor of shape %s' % preprocessed_inputs.get_shape())

    with tf.variable_scope("YOLOv1", reuse=self.reuse_weights) as scope:
      conv0 = self._create_conv_layer(preprocessed_inputs, 7, 64, 2, "conv0")
      maxpool1 = self._create_max_pool_layer(conv0, 2, 2)
      conv2 = self._create_conv_layer(maxpool1, 3, 192, 1, "conv2")
      maxpool3 = self._create_max_pool_layer(conv2, 2, 2)
      conv4 = self._create_conv_layer(maxpool3, 1, 128, 1, "conv4")
      conv5 = self._create_conv_layer(conv4, 3, 256, 1, "conv5")
      conv6 = self._create_conv_layer(conv5, 1, 256, 1, "conv6")
      conv7 = self._create_conv_layer(conv6, 3, 512, 1, "conv7")
      maxpool8 = self.create_maxpool_layer(conv7, 2, 2)
      conv9 = self.create_conv(maxpool8, 1, 256, 1, 'conv9')
      conv10 = self.create_conv_layer(conv9, 3, 512, 1, 'conv10')
      conv11 = self.create_conv_layer(conv10, 1, 256, 1, 'conv11')
      conv12 = self.create_conv_layer(conv11, 3, 512, 1, 'conv12')
      conv13 = self.create_conv_layer(conv12, 1, 256, 1, 'conv13')
      conv14 = self.create_conv_layer(conv13, 3, 512, 1, 'conv14')
      conv15 = self.create_conv_layer(conv14, 1, 256, 1, 'conv15')
      conv16 = self.create_conv_layer(conv15, 3, 512, 1, 'conv16')
      conv17 = self.create_conv_layer(conv16, 1, 512, 1, 'conv17')
      conv18 = self.create_conv_layer(conv17, 3, 1024, 1, 'conv18')
      maxpool19 = self.create_maxpool_layer(conv18, 2, 2)
      conv20 = self.create_conv(maxpool19, 1, 512, 1, 'conv20')
      conv21 = self.create_conv_layer(conv20, 3, 1024, 1, 'conv21')
      conv22 = self.create_conv_layer(conv21, 1, 512, 1, 'conv22')
      conv23 = self.create_conv_layer(conv22, 3, 1024, 1, 'conv23')
      conv24 = self.create_conv_layer(conv23, 3, 1024, 1, 'conv24')
      conv25 = self.create_conv_layer(conv24, 3, 1024, 2, 'conv25')
      conv26 = self.create_conv_layer(conv25, 3, 1024, 1, 'conv26')
      conv27 = self.create_conv_layer(conv26, 3, 1024, 1, 'conv27')
      # flatten layer for connection to fully connected layer
      conv27_flatten_dim = int(reduce(lambda a, b: a * b, conv27.get_shape()[1:]))
      conv27_flatten = tf.reshape(tf.transpose(conv27, (0, 3, 1, 2)), [-1, conv27_flatten_dim])
      fc28 = self.create_connected_layer(conv27_flatten, 512, True, 'fc28')
      fc29 = self.create_connected_layer(fc28, 4096, True, 'fc29')

      dropout30 = self.create_dropout_layer(fc29, self.dropout_prob)
      fc31 = self.create_connected_layer(dropout30, 1470, False, 'fc31')
      self.feature_map = fc31

    return self.feature_map

  # TODO change the functions below and understand padding
  # TODO go through YoloTensorFlow229
  def _create_conv_layer(self, input_layer, size, filters, stride, name):
    channels = int(input_layer.get_shape()[3])
    weight_shape = [size, size, channels, filters]
    bias_shape = [filters]
    with tf.variable_scope(name + '_conv_weights'):
      weight = tf.get_variable('w_%s' % (name), weight_shape, initializer=tf.contrib.layers.xavier_initializer_conv2d())
      bias = tf.get_variable('b_%s' % (name), bias_shape, initializer=tf.constant_initializer(0.0))

    pad = int(size / 2)
    input_layer_padded = tf.pad(input_layer, paddings=[[0, 0], [pad, pad], [pad, pad], [0, 0]])
    convolution = tf.nn.conv2d(input=input_layer_padded, filter=weight, strides=[1, stride, stride, 1], padding='VALID')
    convolution_bias = tf.add(convolution, bias)
    return self._leaky_RELU(convolution_bias)

  def _create_fc_layer(self, input_layer, size, name):
    weight_shape = [int(input_layer.get_shape()[1]), size]
    bias_shape = [size]

    with tf.variable_scope(name + '_fully_connected_weights'):
      weight = tf.get_variable('w_%s' % (name), weight_shape, initializer=tf.contrib.layers.xavier_initializer())
      bias = tf.get_variable('b_%s' % (name), bias_shape, initializer=tf.constant_initializer(0.0))

    return self._leaky_RELU(tf.add(tf.matmul(input_layer, weight), bias))

  def _create_max_pool_layer(self, input_layer, size, stride):
    return tf.nn.max_pool(input_layer, ksize=[1, size, size, 1],
                          strides=[1, stride, stride, 1], padding='SAME')

  def _create_dropout_layer(self, input_layer, probability):
    return tf.nn.dropout(input_layer,probability)

  def _leaky_RELU(self, input_layer):
    return tf.maximum(input_layer, tf.scalar_mul(0.1, input_layer))