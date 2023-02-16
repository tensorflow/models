# Copyright 2022 The TensorFlow Authors. All Rights Reserved.
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

"""Contains definitions of instance prediction heads."""

from typing import List, Union, Optional
# Import libraries
import tensorflow as tf

from official.modeling import tf_utils


@tf.keras.utils.register_keras_serializable(package='Vision')
class DetectionHead(tf.keras.layers.Layer):
  """Creates a detection head."""

  def __init__(
      self,
      num_classes: int,
      num_convs: int = 0,
      num_filters: int = 256,
      use_separable_conv: bool = False,
      num_fcs: int = 2,
      fc_dims: int = 1024,
      class_agnostic_bbox_pred: bool = False,
      activation: str = 'relu',
      use_sync_bn: bool = False,
      norm_momentum: float = 0.99,
      norm_epsilon: float = 0.001,
      kernel_regularizer: Optional[tf.keras.regularizers.Regularizer] = None,
      bias_regularizer: Optional[tf.keras.regularizers.Regularizer] = None,
      **kwargs):
    """Initializes a detection head.

    Args:
      num_classes: An `int` for the number of classes.
      num_convs: An `int` number that represents the number of the intermediate
        convolution layers before the FC layers.
      num_filters: An `int` number that represents the number of filters of the
        intermediate convolution layers.
      use_separable_conv: A `bool` that indicates whether the separable
        convolution layers is used.
      num_fcs: An `int` number that represents the number of FC layers before
        the predictions.
      fc_dims: An `int` number that represents the number of dimension of the FC
        layers.
      class_agnostic_bbox_pred: `bool`, indicating whether bboxes should be
        predicted for every class or not.
      activation: A `str` that indicates which activation is used, e.g. 'relu',
        'swish', etc.
      use_sync_bn: A `bool` that indicates whether to use synchronized batch
        normalization across different replicas.
      norm_momentum: A `float` of normalization momentum for the moving average.
      norm_epsilon: A `float` added to variance to avoid dividing by zero.
      kernel_regularizer: A `tf.keras.regularizers.Regularizer` object for
        Conv2D. Default is None.
      bias_regularizer: A `tf.keras.regularizers.Regularizer` object for Conv2D.
      **kwargs: Additional keyword arguments to be passed.
    """
    super(DetectionHead, self).__init__(**kwargs)
    self._config_dict = {
        'num_classes': num_classes,
        'num_convs': num_convs,
        'num_filters': num_filters,
        'use_separable_conv': use_separable_conv,
        'num_fcs': num_fcs,
        'fc_dims': fc_dims,
        'class_agnostic_bbox_pred': class_agnostic_bbox_pred,
        'activation': activation,
        'use_sync_bn': use_sync_bn,
        'norm_momentum': norm_momentum,
        'norm_epsilon': norm_epsilon,
        'kernel_regularizer': kernel_regularizer,
        'bias_regularizer': bias_regularizer,
    }

    if tf.keras.backend.image_data_format() == 'channels_last':
      self._bn_axis = -1
    else:
      self._bn_axis = 1
    self._activation = tf_utils.get_activation(activation)

  def build(self, input_shape: Union[tf.TensorShape, List[tf.TensorShape]]):
    """Creates the variables of the head."""
    conv_op = (tf.keras.layers.SeparableConv2D
               if self._config_dict['use_separable_conv']
               else tf.keras.layers.Conv2D)
    conv_kwargs = {
        'filters': self._config_dict['num_filters'],
        'kernel_size': 3,
        'padding': 'same',
    }
    if self._config_dict['use_separable_conv']:
      conv_kwargs.update({
          'depthwise_initializer': tf.keras.initializers.VarianceScaling(
              scale=2, mode='fan_out', distribution='untruncated_normal'),
          'pointwise_initializer': tf.keras.initializers.VarianceScaling(
              scale=2, mode='fan_out', distribution='untruncated_normal'),
          'bias_initializer': tf.zeros_initializer(),
          'depthwise_regularizer': self._config_dict['kernel_regularizer'],
          'pointwise_regularizer': self._config_dict['kernel_regularizer'],
          'bias_regularizer': self._config_dict['bias_regularizer'],
      })
    else:
      conv_kwargs.update({
          'kernel_initializer': tf.keras.initializers.VarianceScaling(
              scale=2, mode='fan_out', distribution='untruncated_normal'),
          'bias_initializer': tf.zeros_initializer(),
          'kernel_regularizer': self._config_dict['kernel_regularizer'],
          'bias_regularizer': self._config_dict['bias_regularizer'],
      })
    bn_op = (tf.keras.layers.experimental.SyncBatchNormalization
             if self._config_dict['use_sync_bn']
             else tf.keras.layers.BatchNormalization)
    bn_kwargs = {
        'axis': self._bn_axis,
        'momentum': self._config_dict['norm_momentum'],
        'epsilon': self._config_dict['norm_epsilon'],
    }

    self._convs = []
    self._conv_norms = []
    for i in range(self._config_dict['num_convs']):
      conv_name = 'detection-conv_{}'.format(i)
      if 'kernel_initializer' in conv_kwargs:
        conv_kwargs['kernel_initializer'] = tf_utils.clone_initializer(
            conv_kwargs['kernel_initializer'])
      self._convs.append(conv_op(name=conv_name, **conv_kwargs))
      bn_name = 'detection-conv-bn_{}'.format(i)
      self._conv_norms.append(bn_op(name=bn_name, **bn_kwargs))

    self._fcs = []
    self._fc_norms = []
    for i in range(self._config_dict['num_fcs']):
      fc_name = 'detection-fc_{}'.format(i)
      self._fcs.append(
          tf.keras.layers.Dense(
              units=self._config_dict['fc_dims'],
              kernel_initializer=tf.keras.initializers.VarianceScaling(
                  scale=1 / 3.0, mode='fan_out', distribution='uniform'),
              kernel_regularizer=self._config_dict['kernel_regularizer'],
              bias_regularizer=self._config_dict['bias_regularizer'],
              name=fc_name))
      bn_name = 'detection-fc-bn_{}'.format(i)
      self._fc_norms.append(bn_op(name=bn_name, **bn_kwargs))

    self._classifier = tf.keras.layers.Dense(
        units=self._config_dict['num_classes'],
        kernel_initializer=tf.keras.initializers.RandomNormal(stddev=0.01),
        bias_initializer=tf.zeros_initializer(),
        kernel_regularizer=self._config_dict['kernel_regularizer'],
        bias_regularizer=self._config_dict['bias_regularizer'],
        name='detection-scores')

    num_box_outputs = (4 if self._config_dict['class_agnostic_bbox_pred'] else
                       self._config_dict['num_classes'] * 4)
    self._box_regressor = tf.keras.layers.Dense(
        units=num_box_outputs,
        kernel_initializer=tf.keras.initializers.RandomNormal(stddev=0.001),
        bias_initializer=tf.zeros_initializer(),
        kernel_regularizer=self._config_dict['kernel_regularizer'],
        bias_regularizer=self._config_dict['bias_regularizer'],
        name='detection-boxes')

    super(DetectionHead, self).build(input_shape)

  def call(self, inputs: tf.Tensor, training: bool = None):
    """Forward pass of box and class branches for the Mask-RCNN model.

    Args:
      inputs: A `tf.Tensor` of the shape [batch_size, num_instances, roi_height,
        roi_width, roi_channels], representing the ROI features.
      training: a `bool` indicating whether it is in `training` mode.

    Returns:
      class_outputs: A `tf.Tensor` of the shape
        [batch_size, num_rois, num_classes], representing the class predictions.
      box_outputs: A `tf.Tensor` of the shape
        [batch_size, num_rois, num_classes * 4], representing the box
        predictions.
    """
    roi_features = inputs
    _, num_rois, height, width, filters = roi_features.get_shape().as_list()

    x = tf.reshape(roi_features, [-1, height, width, filters])
    for conv, bn in zip(self._convs, self._conv_norms):
      x = conv(x)
      x = bn(x)
      x = self._activation(x)

    _, _, _, filters = x.get_shape().as_list()
    x = tf.reshape(x, [-1, num_rois, height * width * filters])

    for fc, bn in zip(self._fcs, self._fc_norms):
      x = fc(x)
      x = bn(x)
      x = self._activation(x)

    classes = self._classifier(x)
    boxes = self._box_regressor(x)
    return classes, boxes

  def get_config(self):
    return self._config_dict

  @classmethod
  def from_config(cls, config):
    return cls(**config)


@tf.keras.utils.register_keras_serializable(package='Vision')
class MaskHead(tf.keras.layers.Layer):
  """Creates a mask head."""

  def __init__(
      self,
      num_classes: int,
      upsample_factor: int = 2,
      num_convs: int = 4,
      num_filters: int = 256,
      use_separable_conv: bool = False,
      activation: str = 'relu',
      use_sync_bn: bool = False,
      norm_momentum: float = 0.99,
      norm_epsilon: float = 0.001,
      kernel_regularizer: Optional[tf.keras.regularizers.Regularizer] = None,
      bias_regularizer: Optional[tf.keras.regularizers.Regularizer] = None,
      class_agnostic: bool = False,
      **kwargs):
    """Initializes a mask head.

    Args:
      num_classes: An `int` of the number of classes.
      upsample_factor: An `int` that indicates the upsample factor to generate
        the final predicted masks. It should be >= 1.
      num_convs: An `int` number that represents the number of the intermediate
        convolution layers before the mask prediction layers.
      num_filters: An `int` number that represents the number of filters of the
        intermediate convolution layers.
      use_separable_conv: A `bool` that indicates whether the separable
        convolution layers is used.
      activation: A `str` that indicates which activation is used, e.g. 'relu',
        'swish', etc.
      use_sync_bn: A `bool` that indicates whether to use synchronized batch
        normalization across different replicas.
      norm_momentum: A `float` of normalization momentum for the moving average.
      norm_epsilon: A `float` added to variance to avoid dividing by zero.
      kernel_regularizer: A `tf.keras.regularizers.Regularizer` object for
        Conv2D. Default is None.
      bias_regularizer: A `tf.keras.regularizers.Regularizer` object for Conv2D.
      class_agnostic: A `bool`. If set, we use a single channel mask head that
        is shared between all classes.
      **kwargs: Additional keyword arguments to be passed.
    """
    super(MaskHead, self).__init__(**kwargs)
    self._config_dict = {
        'num_classes': num_classes,
        'upsample_factor': upsample_factor,
        'num_convs': num_convs,
        'num_filters': num_filters,
        'use_separable_conv': use_separable_conv,
        'activation': activation,
        'use_sync_bn': use_sync_bn,
        'norm_momentum': norm_momentum,
        'norm_epsilon': norm_epsilon,
        'kernel_regularizer': kernel_regularizer,
        'bias_regularizer': bias_regularizer,
        'class_agnostic': class_agnostic
    }

    if tf.keras.backend.image_data_format() == 'channels_last':
      self._bn_axis = -1
    else:
      self._bn_axis = 1
    self._activation = tf_utils.get_activation(activation)

  def build(self, input_shape: Union[tf.TensorShape, List[tf.TensorShape]]):
    """Creates the variables of the head."""
    conv_op = (tf.keras.layers.SeparableConv2D
               if self._config_dict['use_separable_conv']
               else tf.keras.layers.Conv2D)
    conv_kwargs = {
        'filters': self._config_dict['num_filters'],
        'kernel_size': 3,
        'padding': 'same',
    }
    if self._config_dict['use_separable_conv']:
      conv_kwargs.update({
          'depthwise_initializer': tf.keras.initializers.VarianceScaling(
              scale=2, mode='fan_out', distribution='untruncated_normal'),
          'pointwise_initializer': tf.keras.initializers.VarianceScaling(
              scale=2, mode='fan_out', distribution='untruncated_normal'),
          'bias_initializer': tf.zeros_initializer(),
          'depthwise_regularizer': self._config_dict['kernel_regularizer'],
          'pointwise_regularizer': self._config_dict['kernel_regularizer'],
          'bias_regularizer': self._config_dict['bias_regularizer'],
      })
    else:
      conv_kwargs.update({
          'kernel_initializer': tf.keras.initializers.VarianceScaling(
              scale=2, mode='fan_out', distribution='untruncated_normal'),
          'bias_initializer': tf.zeros_initializer(),
          'kernel_regularizer': self._config_dict['kernel_regularizer'],
          'bias_regularizer': self._config_dict['bias_regularizer'],
      })
    bn_op = (tf.keras.layers.experimental.SyncBatchNormalization
             if self._config_dict['use_sync_bn']
             else tf.keras.layers.BatchNormalization)
    bn_kwargs = {
        'axis': self._bn_axis,
        'momentum': self._config_dict['norm_momentum'],
        'epsilon': self._config_dict['norm_epsilon'],
    }

    self._convs = []
    self._conv_norms = []
    for i in range(self._config_dict['num_convs']):
      conv_name = 'mask-conv_{}'.format(i)
      for initializer_name in ['kernel_initializer', 'depthwise_initializer',
                               'pointwise_initializer']:
        if initializer_name in conv_kwargs:
          conv_kwargs[initializer_name] = tf_utils.clone_initializer(
              conv_kwargs[initializer_name])
      self._convs.append(conv_op(name=conv_name, **conv_kwargs))
      bn_name = 'mask-conv-bn_{}'.format(i)
      self._conv_norms.append(bn_op(name=bn_name, **bn_kwargs))

    self._deconv = tf.keras.layers.Conv2DTranspose(
        filters=self._config_dict['num_filters'],
        kernel_size=self._config_dict['upsample_factor'],
        strides=self._config_dict['upsample_factor'],
        padding='valid',
        kernel_initializer=tf.keras.initializers.VarianceScaling(
            scale=2, mode='fan_out', distribution='untruncated_normal'),
        bias_initializer=tf.zeros_initializer(),
        kernel_regularizer=self._config_dict['kernel_regularizer'],
        bias_regularizer=self._config_dict['bias_regularizer'],
        name='mask-upsampling')
    self._deconv_bn = bn_op(name='mask-deconv-bn', **bn_kwargs)

    if self._config_dict['class_agnostic']:
      num_filters = 1
    else:
      num_filters = self._config_dict['num_classes']

    conv_kwargs = {
        'filters': num_filters,
        'kernel_size': 1,
        'padding': 'valid',
    }
    if self._config_dict['use_separable_conv']:
      conv_kwargs.update({
          'depthwise_initializer': tf.keras.initializers.VarianceScaling(
              scale=2, mode='fan_out', distribution='untruncated_normal'),
          'pointwise_initializer': tf.keras.initializers.VarianceScaling(
              scale=2, mode='fan_out', distribution='untruncated_normal'),
          'bias_initializer': tf.zeros_initializer(),
          'depthwise_regularizer': self._config_dict['kernel_regularizer'],
          'pointwise_regularizer': self._config_dict['kernel_regularizer'],
          'bias_regularizer': self._config_dict['bias_regularizer'],
      })
    else:
      conv_kwargs.update({
          'kernel_initializer': tf.keras.initializers.VarianceScaling(
              scale=2, mode='fan_out', distribution='untruncated_normal'),
          'bias_initializer': tf.zeros_initializer(),
          'kernel_regularizer': self._config_dict['kernel_regularizer'],
          'bias_regularizer': self._config_dict['bias_regularizer'],
      })
    self._mask_regressor = conv_op(name='mask-logits', **conv_kwargs)

    super(MaskHead, self).build(input_shape)

  def call(self, inputs: List[tf.Tensor], training: bool = None):
    """Forward pass of mask branch for the Mask-RCNN model.

    Args:
      inputs: A `list` of two tensors where
        inputs[0]: A `tf.Tensor` of shape [batch_size, num_instances,
          roi_height, roi_width, roi_channels], representing the ROI features.
        inputs[1]: A `tf.Tensor` of shape [batch_size, num_instances],
          representing the classes of the ROIs.
      training: A `bool` indicating whether it is in `training` mode.

    Returns:
      mask_outputs: A `tf.Tensor` of shape
        [batch_size, num_instances, roi_height * upsample_factor,
         roi_width * upsample_factor], representing the mask predictions.
    """
    roi_features, roi_classes = inputs
    _, num_rois, height, width, filters = roi_features.get_shape().as_list()

    x = tf.reshape(roi_features, [-1, height, width, filters])
    for conv, bn in zip(self._convs, self._conv_norms):
      x = conv(x)
      x = bn(x)
      x = self._activation(x)

    x = self._deconv(x)
    x = self._deconv_bn(x)
    x = self._activation(x)

    logits = self._mask_regressor(x)

    mask_height = height * self._config_dict['upsample_factor']
    mask_width = width * self._config_dict['upsample_factor']

    if self._config_dict['class_agnostic']:
      return tf.reshape(logits, [-1, num_rois, mask_height, mask_width])
    else:
      logits = tf.reshape(
          logits,
          [-1, num_rois, mask_height, mask_width,
           self._config_dict['num_classes']])
      return tf.gather(
          logits, tf.cast(roi_classes, dtype=tf.int32), axis=-1, batch_dims=2
      )

  def get_config(self):
    return self._config_dict

  @classmethod
  def from_config(cls, config):
    return cls(**config)
