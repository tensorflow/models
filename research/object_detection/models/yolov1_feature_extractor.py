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

"""YOLOv1FeatureExtractor for YOLO v1 models"""

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
    super(YOLOv1FeatureExtractor, self).__init__(is_training, reuse_weights)


  def preprocess(self, resized_inputs):
    """Preprocesses images for feature extraction (minus image resizing).

    Args:
      resized_inputs: a [batch, height, width, channels] float tensor
        representing a batch of images.

    Returns:
      preprocessed_inputs: a [batch, height, width, channels] float tensor
        representing a batch of images.
    """
    return (2.0 / 255.0) * resized_inputs - 1.0


  def extract_features(self, preprocessed_inputs):
    """Extracts features from preprocessed inputs.

    This function is responsible for extracting the YOLO feature map from preprocessed
    images.

    Args:
      preprocessed_inputs: a [batch, height, width, channels] float tensor
        representing a batch of images. height = width = 448, channels = 3

    Returns:
      feature_maps: a tensor where the tensor has shape
        [batch, grid_size, grid_size, depth]
    """
    if len(preprocessed_inputs.get_shape().as_list()) != 4:
      raise ValueError('`preprocessed_inputs` must be 4 dimensional, got a '
                       'tensor of shape %s' % preprocessed_inputs.get_shape())

    with tf.variable_scope("YOLOv1FeatureExtractor", reuse=self.reuse_weights):
      with slim.arg_scope([slim.conv2d], padding='SAME', activation_fn=self._leaky_RELU):
        conv0 = slim.conv2d(preprocessed_inputs, 64, 7, 2, scope='Conv2d_0_7x7x64-s-2')
        maxpool1 = slim.max_pool2d(conv0, 2, 2, padding='SAME', scope='MaxPool_1_2x2-s-2')
        conv2 = slim.conv2d(maxpool1, 192, 3, 1, scope='Conv2d_2_3x3x192-s-1')
        maxpool3 = slim.max_pool2d(conv2, 2, 2, padding='SAME', scope='MaxPool_3_2x2-s-2')

        conv4 = slim.conv2d(maxpool3, 128, 1, 1, scope='Conv2d_4_1x1x128-s-1')
        conv5 = slim.conv2d(conv4, 256, 3, 1, scope='Conv2d_5_3x3x256-s-1')
        conv6 = slim.conv2d(conv5, 256, 1, 1, scope='Conv2d_6_1x1x256-s-1')
        conv7 = slim.conv2d(conv6, 512, 3, 1, scope='Conv2d_7_3x3x512-s-1')

        maxpool8 = slim.max_pool2d(conv7, 2, 2, padding='SAME', scope='MaxPool_8_2x2-s-2')

        conv9 = slim.conv2d(maxpool8, 256, 1, 1, scope='Conv2d_9_1x1x256-s-1')
        conv10 = slim.conv2d(conv9, 512, 3, 1, scope='Conv2d_10_3x3x512-s-1')
        conv11 = slim.conv2d(conv10, 256, 1, 1, scope='Conv2d_11_1x1x256-s-1')
        conv12 = slim.conv2d(conv11, 512, 3, 1, scope='Conv2d_12_3x3x512-s-1')
        conv13 = slim.conv2d(conv12, 256, 1, 1, scope='Conv2d_13_1x1x256-s-1')
        conv14 = slim.conv2d(conv13, 512, 3, 1, scope='Conv2d_14_3x3x512-s-1')
        conv15 = slim.conv2d(conv14, 256, 1, 1, scope='Conv2d_15_1x1x256-s-1')
        conv16 = slim.conv2d(conv15, 512, 3, 1, scope='Conv2d_16_3x3x512-s-1')
        conv17 = slim.conv2d(conv16, 512, 1, 1, scope='Conv2d_17_1x1x512-s-1')
        conv18 = slim.conv2d(conv17, 1024, 3, 1, scope='Conv2d_18_3x3x1024-s-1')

        maxpool19 = slim.max_pool2d(conv18, 2, 2, padding='SAME', scope='MaxPool_19_2x2-s-2')

        conv20 = slim.conv2d(maxpool19, 512, 1, 1, scope='Conv2d_20_1x1x512-s-1')
        conv21 = slim.conv2d(conv20, 1024, 3, 1, scope='Conv2d_21_3x3x1024-s-1')
        conv22 = slim.conv2d(conv21, 512, 1, 1, scope='Conv2d_22_1x1x512-s-1')
        conv23 = slim.conv2d(conv22, 1024, 3, 1, scope='Conv2d_23_3x3x1024-s-1')
        conv24 = slim.conv2d(conv23, 1024, 3, 1, scope='Conv2d_24_3x3x1024-s-1')
        conv25 = slim.conv2d(conv24, 1024, 3, 2, scope='Conv2d_25_3x3x1024-s-2')

        conv26 = slim.conv2d(conv25, 1024, 3, 1, scope='Conv2d_26_3x3x1024-s-1')
        conv27 = slim.conv2d(conv26, 1024, 3, 1, scope='Conv2d_27_3x3x1024-s-1')

        # flatten layer before connecting it to a fully connected layer
        conv27_flatten_dim = int(reduce(lambda a, b: a * b, conv27.get_shape()[1:]))
        conv27_flatten = tf.reshape(tf.transpose(conv27, (0, 3, 1, 2)),
                                    [-1, conv27_flatten_dim])

        fc28 = slim.fully_connected(conv27_flatten, 512,
                                    activation_fn=self._leaky_RELU, scope='Conn_28_512')
        fc29 = slim.fully_connected(fc28, 4096,
                                    activation_fn=self._leaky_RELU, scope='Conn_29_4096')

        dropout30 = slim.dropout(fc29, self.dropout_prob, scope='Dropout_30')
        fc31 = slim.fully_connected(dropout30,
                                    activation_fn=self._leaky_RELU, scope='Conn_31_1470')

        self._feature_map = fc31

    return self._feature_map

  def _leaky_RELU(self, inputs):
    return tf.maximum(inputs, tf.scalar_mul(0.1, inputs))