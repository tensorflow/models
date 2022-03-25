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

"""Contains definitions for 'Representation Flow' layer [1].

Representation flow layer is a generalization of optical flow extraction; the
layer could be inserted anywhere within a CNN to capture feature movements. This
is the version taking 4D tensor with the shape [batch*time, height, width,
channels], to make this run on TPU.

[1] AJ Piergiovanni and Michael S. Ryoo,
    Representation Flow for Action Recognition. CVPR 2019.
"""

import numpy as np
import tensorflow as tf

layers = tf.keras.layers
BATCH_NORM_DECAY = 0.99
BATCH_NORM_EPSILON = 1e-5


def build_batch_norm(init_zero: bool = False,
                     bn_decay: float = BATCH_NORM_DECAY,
                     bn_epsilon: float = BATCH_NORM_EPSILON,
                     use_sync_bn: bool = False):
  """Performs a batch normalization followed by a ReLU.

  Args:
    init_zero: `bool` if True, initializes scale parameter of batch
      normalization with 0 instead of 1 (default).
    bn_decay: `float` batch norm decay parameter to use.
    bn_epsilon: `float` batch norm epsilon parameter to use.
    use_sync_bn: use synchronized batch norm for TPU.

  Returns:
    A normalized `Tensor` with the same `data_format`.
  """

  if init_zero:
    gamma_initializer = tf.zeros_initializer()
  else:
    gamma_initializer = tf.ones_initializer()

  data_format = tf.keras.backend.image_data_format()
  assert data_format == 'channels_last'

  if data_format == 'channels_first':
    axis = 1
  else:
    axis = -1

  if use_sync_bn:
    batch_norm = layers.experimental.SyncBatchNormalization(
        axis=axis,
        momentum=bn_decay,
        epsilon=bn_epsilon,
        gamma_initializer=gamma_initializer)
  else:
    batch_norm = layers.BatchNormalization(
        axis=axis,
        momentum=bn_decay,
        epsilon=bn_epsilon,
        fused=True,
        gamma_initializer=gamma_initializer)

  return batch_norm


def divergence(p1, p2, f_grad_x, f_grad_y, name):
  """Computes the divergence value used with TV-L1 optical flow algorithm.

  Args:
    p1: 'Tensor' input.
    p2: 'Tensor' input in the next frame.
    f_grad_x: 'Tensor' x gradient of F value used in TV-L1.
    f_grad_y: 'Tensor' y gradient of F value used in TV-L1.
    name: 'str' name for the variable scope.

  Returns:
    A `Tensor` with the same `data_format` and shape as input.
  """
  data_format = tf.keras.backend.image_data_format()
  df = 'NHWC' if data_format == 'channels_last' else 'NCHW'

  with tf.name_scope('divergence_' + name):
    if data_format == 'channels_last':
      p1 = tf.pad(p1[:, :, :-1, :], [[0, 0], [0, 0], [1, 0], [0, 0]])
      p2 = tf.pad(p2[:, :-1, :, :], [[0, 0], [1, 0], [0, 0], [0, 0]])
    else:
      p1 = tf.pad(p1[:, :, :, :-1], [[0, 0], [0, 0], [0, 0], [1, 0]])
      p2 = tf.pad(p2[:, :, :-1, :], [[0, 0], [0, 0], [1, 0], [0, 0]])

    grad_x = tf.nn.conv2d(p1, f_grad_x, [1, 1, 1, 1], 'SAME', data_format=df)
    grad_y = tf.nn.conv2d(p2, f_grad_y, [1, 1, 1, 1], 'SAME', data_format=df)
    return grad_x + grad_y


def forward_grad(x, f_grad_x, f_grad_y, name):
  data_format = tf.keras.backend.image_data_format()
  with tf.name_scope('forward_grad_' + name):
    df = 'NHWC' if data_format == 'channels_last' else 'NCHW'
    grad_x = tf.nn.conv2d(x, f_grad_x, [1, 1, 1, 1], 'SAME', data_format=df)
    grad_y = tf.nn.conv2d(x, f_grad_y, [1, 1, 1, 1], 'SAME', data_format=df)
    return grad_x, grad_y


def norm_img(x):
  mx = tf.reduce_max(x)
  mn = tf.reduce_min(x)
  if mx == mn:
    return x
  else:
    return 255 * (x - mn) / (mx - mn)


class RepresentationFlow(layers.Layer):
  """Computes the representation flow motivated by TV-L1 optical flow."""

  def __init__(self,
               time: int,
               depth: int,
               num_iter: int = 20,
               bottleneck: int = 32,
               train_feature_grad: bool = False,
               train_divergence: bool = False,
               train_flow_grad: bool = False,
               train_hyper: bool = False,
               **kwargs):
    """Constructor.

    Args:
      time: 'int' number of frames in the input tensor.
      depth: channel depth of the input tensor.
      num_iter: 'int' number of iterations to use for the flow computation.
      bottleneck: 'int' number of filters to be used for the flow computation.
      train_feature_grad: Train image grad params.
      train_divergence: train divergence params
      train_flow_grad: train flow grad params.
      train_hyper: train rep flow hyperparams.
      **kwargs: keyword arguments to be passed to the parent constructor.

    Returns:
      A `Tensor` with the same `data_format` and shape as input.
    """
    super(RepresentationFlow, self).__init__(**kwargs)

    self._time = time
    self._depth = depth
    self._num_iter = num_iter
    self._bottleneck = bottleneck
    self._train_feature_grad = train_feature_grad
    self._train_divergence = train_divergence
    self._train_flow_grad = train_flow_grad
    self._train_hyper = train_hyper

  def get_config(self):
    config = {
        'time': self._time,
        'num_iter': self._num_iter,
        'bottleneck': self._bottleneck,
        'train_feature_grad': self._train_feature_grad,
        'train_divergence': self._train_divergence,
        'train_flow_grad': self._train_flow_grad,
        'train_hyper': self._train_hyper,
    }
    base_config = super(RepresentationFlow, self).get_config()
    return dict(list(base_config.items()) + list(config.items()))

  def build(self, input_shape: tf.TensorShape):
    img_grad = np.array([-0.5, 0, 0.5], dtype='float32')
    img_grad_x = np.repeat(
        np.reshape(img_grad, (1, 3, 1, 1)), self._bottleneck, axis=2) * np.eye(
            self._bottleneck, dtype='float32')
    self.img_grad_x = self.add_weight(
        shape=img_grad_x.shape,
        initializer=tf.constant_initializer(img_grad_x),
        trainable=self._train_feature_grad,
        name='img_grad_x')
    img_grad_y = np.repeat(
        np.reshape(img_grad, (3, 1, 1, 1)), self._bottleneck, axis=2) * np.eye(
            self._bottleneck, dtype='float32')
    self.img_grad_y = self.add_weight(
        shape=img_grad_y.shape,
        initializer=tf.constant_initializer(img_grad_y),
        trainable=self._train_feature_grad,
        name='img_grad_y')

    f_grad = np.array([-1, 1], dtype='float32')
    f_grad_x = np.repeat(
        np.reshape(f_grad, (1, 2, 1, 1)), self._bottleneck, axis=2) * np.eye(
            self._bottleneck, dtype='float32')
    self.f_grad_x = self.add_weight(
        shape=f_grad_x.shape,
        initializer=tf.constant_initializer(f_grad_x),
        trainable=self._train_divergence,
        name='f_grad_x')
    f_grad_y = np.repeat(
        np.reshape(f_grad, (2, 1, 1, 1)), self._bottleneck, axis=2) * np.eye(
            self._bottleneck, dtype='float32')
    self.f_grad_y = self.add_weight(
        shape=f_grad_y.shape,
        initializer=tf.constant_initializer(f_grad_y),
        trainable=self._train_divergence,
        name='f_grad_y')

    f_grad_x2 = np.repeat(
        np.reshape(f_grad, (1, 2, 1, 1)), self._bottleneck, axis=2) * np.eye(
            self._bottleneck, dtype='float32')
    self.f_grad_x2 = self.add_weight(
        shape=f_grad_x2.shape,
        initializer=tf.constant_initializer(f_grad_x2),
        trainable=self._train_flow_grad,
        name='f_grad_x2')
    f_grad_y2 = np.repeat(
        np.reshape(f_grad, (2, 1, 1, 1)), self._bottleneck, axis=2) * np.eye(
            self._bottleneck, dtype='float32')
    self.f_grad_y2 = self.add_weight(
        shape=f_grad_y2.shape,
        initializer=tf.constant_initializer(f_grad_y2),
        trainable=self._train_flow_grad,
        name='f_grad_y2')

    self.t = self.add_weight(
        name='theta',
        initializer=tf.constant_initializer(0.3),
        trainable=self._train_hyper)
    self.l = self.add_weight(
        name='lambda',
        initializer=tf.constant_initializer(0.15),
        trainable=self._train_hyper)
    self.a = self.add_weight(
        name='tau',
        initializer=tf.constant_initializer(0.25),
        trainable=self._train_hyper)
    self.t = tf.abs(self.t) + 1e-12
    self.l_t = self.l * self.t
    self.taut = self.a / self.t

    self._bottleneck_conv2 = None
    self._bottleneck_conv2 = None
    if self._bottleneck > 1:
      self._bottleneck_conv1 = layers.Conv2D(
          filters=self._bottleneck,
          kernel_size=1,
          strides=1,
          padding='same',
          use_bias=False,
          kernel_initializer=tf.keras.initializers.VarianceScaling(),
          name='rf/bottleneck1')
      self._bottleneck_conv2 = layers.Conv2D(
          filters=self._depth,
          kernel_size=1,
          strides=1,
          padding='same',
          use_bias=False,
          kernel_initializer=tf.keras.initializers.VarianceScaling(),
          name='rf/bottleneck2')
      self._batch_norm = build_batch_norm(init_zero=True)

  def call(self, inputs: tf.Tensor, training: bool = None) -> tf.Tensor:
    """Perform representation flows.

    Args:
      inputs: list of `Tensors` of shape `[batch*time, height, width,
        channels]`.
      training: True for training phase.

    Returns:
      A tensor of the same shape as the inputs.
    """
    data_format = tf.keras.backend.image_data_format()
    df = 'NHWC' if data_format == 'channels_last' else 'NCHW'
    axis = 3 if data_format == 'channels_last' else 1  # channel axis
    dtype = inputs.dtype
    residual = inputs
    depth = inputs.shape.as_list()[axis]
    # assert depth == self._depth, f'rep_flow {depth} != {self._depth}'

    if self._bottleneck == 1:
      inputs = tf.reduce_mean(inputs, axis=axis)
      inputs = tf.expand_dims(inputs, -1)
    elif depth != self._bottleneck:
      inputs = self._bottleneck_conv1(inputs)

    input_shape = inputs.shape.as_list()
    inp = norm_img(inputs)
    inp = tf.reshape(
        inp,
        (-1, self._time, inputs.shape[1], inputs.shape[2], inputs.shape[3]))
    inp = tf.ensure_shape(
        inp, (None, self._time, input_shape[1], input_shape[2], input_shape[3]))
    img1 = tf.reshape(
        inp[:, :-1], (-1, tf.shape(inp)[2], tf.shape(inp)[3], tf.shape(inp)[4]))
    img2 = tf.reshape(
        inp[:, 1:], (-1, tf.shape(inp)[2], tf.shape(inp)[3], tf.shape(inp)[4]))
    img1 = tf.ensure_shape(
        img1, (None, inputs.shape[1], inputs.shape[2], inputs.shape[3]))
    img2 = tf.ensure_shape(
        img2, (None, inputs.shape[1], inputs.shape[2], inputs.shape[3]))

    u1 = tf.zeros_like(img1, dtype=dtype)
    u2 = tf.zeros_like(img2, dtype=dtype)

    l_t = self.l_t
    taut = self.taut

    grad2_x = tf.nn.conv2d(
        img2, self.img_grad_x, [1, 1, 1, 1], 'SAME', data_format=df)
    grad2_y = tf.nn.conv2d(
        img2, self.img_grad_y, [1, 1, 1, 1], 'SAME', data_format=df)

    p11 = tf.zeros_like(img1, dtype=dtype)
    p12 = tf.zeros_like(img1, dtype=dtype)
    p21 = tf.zeros_like(img1, dtype=dtype)
    p22 = tf.zeros_like(img1, dtype=dtype)

    gsqx = grad2_x**2
    gsqy = grad2_y**2

    grad = gsqx + gsqy + 1e-12

    rho_c = img2 - grad2_x * u1 - grad2_y * u2 - img1

    for _ in range(self._num_iter):
      rho = rho_c + grad2_x * u1 + grad2_y * u2 + 1e-12

      v1 = tf.zeros_like(img1, dtype=dtype)
      v2 = tf.zeros_like(img2, dtype=dtype)

      mask1 = rho < -l_t * grad
      tmp11 = tf.where(mask1, l_t * grad2_x,
                       tf.zeros_like(grad2_x, dtype=dtype))
      tmp12 = tf.where(mask1, l_t * grad2_y,
                       tf.zeros_like(grad2_y, dtype=dtype))

      mask2 = rho > l_t * grad
      tmp21 = tf.where(mask2, -l_t * grad2_x,
                       tf.zeros_like(grad2_x, dtype=dtype))
      tmp22 = tf.where(mask2, -l_t * grad2_y,
                       tf.zeros_like(grad2_y, dtype=dtype))

      mask3 = (~mask1) & (~mask2) & (grad > 1e-12)
      tmp31 = tf.where(mask3, (-rho / grad) * grad2_x,
                       tf.zeros_like(grad2_x, dtype=dtype))
      tmp32 = tf.where(mask3, (-rho / grad) * grad2_y,
                       tf.zeros_like(grad2_y, dtype=dtype))

      v1 = tmp11 + tmp21 + tmp31 + u1
      v2 = tmp12 + tmp22 + tmp32 + u2

      u1 = v1 + self.t * divergence(p11, p12, self.f_grad_x, self.f_grad_y,
                                    'div_p1')
      u2 = v2 + self.t * divergence(p21, p22, self.f_grad_x, self.f_grad_y,
                                    'div_p2')

      u1x, u1y = forward_grad(u1, self.f_grad_x2, self.f_grad_y2, 'u1')
      u2x, u2y = forward_grad(u2, self.f_grad_x2, self.f_grad_y2, 'u2')

      p11 = (p11 + taut * u1x) / (1. + taut * tf.sqrt(u1x**2 + u1y**2 + 1e-12))
      p12 = (p12 + taut * u1y) / (1. + taut * tf.sqrt(u1x**2 + u1y**2 + 1e-12))
      p21 = (p21 + taut * u2x) / (1. + taut * tf.sqrt(u2x**2 + u2y**2 + 1e-12))
      p22 = (p22 + taut * u2y) / (1. + taut * tf.sqrt(u2x**2 + u2y**2 + 1e-12))

    u1 = tf.reshape(u1, (-1, self._time - 1, tf.shape(u1)[1],
                         tf.shape(u1)[2], tf.shape(u1)[3]))
    u2 = tf.reshape(u2, (-1, self._time - 1, tf.shape(u2)[1],
                         tf.shape(u2)[2], tf.shape(u2)[3]))
    flow = tf.concat([u1, u2], axis=axis + 1)
    flow = tf.concat([
        flow,
        tf.reshape(
            flow[:, -1, :, :, :],
            (-1, 1, tf.shape(u1)[2], tf.shape(u1)[3], tf.shape(u1)[4] * 2))
    ],
                     axis=1)
    # padding: [bs, 1, w, h, 2*c] -> [bs, 1, w, h, 2*c]
    # flow is [bs, t, w, h, 2*c]
    flow = tf.reshape(
        flow, (-1, tf.shape(u1)[2], tf.shape(u2)[3], tf.shape(u1)[4] * 2))
    # folwo is [bs*t, w, h, 2*c]

    if self._bottleneck == 1:
      output_shape = residual.shape.as_list()
      output_shape[-1] = self._bottleneck * 2
      flow = tf.ensure_shape(flow, output_shape)
      return flow
    else:
      flow = self._bottleneck_conv2(flow)

      flow = self._batch_norm(flow)
      flow = tf.ensure_shape(flow, residual.shape)
      return tf.nn.relu(flow + residual)
