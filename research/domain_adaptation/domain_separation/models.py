# Copyright 2016 The TensorFlow Authors All Rights Reserved.
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
"""Contains different architectures for the different DSN parts.

We define here the modules that can be used in the different parts of the DSN
model.
- shared encoder (dsn_cropped_linemod, dann_xxxx)
- private encoder (default_encoder)
- decoder (large_decoder, gtsrb_decoder, small_decoder)
"""
import tensorflow as tf

#from models.domain_adaptation.domain_separation
import utils

slim = tf.contrib.slim


def default_batch_norm_params(is_training=False):
  """Returns default batch normalization parameters for DSNs.

  Args:
    is_training: whether or not the model is training.

  Returns:
    a dictionary that maps batch norm parameter names (strings) to values.
  """
  return {
      # Decay for the moving averages.
      'decay': 0.5,
      # epsilon to prevent 0s in variance.
      'epsilon': 0.001,
      'is_training': is_training
  }


################################################################################
# PRIVATE ENCODERS
################################################################################
def default_encoder(images, code_size, batch_norm_params=None,
                    weight_decay=0.0):
  """Encodes the given images to codes of the given size.

  Args:
    images: a tensor of size [batch_size, height, width, 1].
    code_size: the number of hidden units in the code layer of the classifier.
    batch_norm_params: a dictionary that maps batch norm parameter names to
      values.
    weight_decay: the value for the weight decay coefficient.

  Returns:
    end_points: the code of the input.
  """
  end_points = {}
  with slim.arg_scope(
      [slim.conv2d, slim.fully_connected],
      weights_regularizer=slim.l2_regularizer(weight_decay),
      activation_fn=tf.nn.relu,
      normalizer_fn=slim.batch_norm,
      normalizer_params=batch_norm_params):
    with slim.arg_scope([slim.conv2d], kernel_size=[5, 5], padding='SAME'):
      net = slim.conv2d(images, 32, scope='conv1')
      net = slim.max_pool2d(net, [2, 2], 2, scope='pool1')
      net = slim.conv2d(net, 64, scope='conv2')
      net = slim.max_pool2d(net, [2, 2], 2, scope='pool2')

      net = slim.flatten(net)
      end_points['flatten'] = net
      net = slim.fully_connected(net, code_size, scope='fc1')
      end_points['fc3'] = net
  return end_points


################################################################################
# DECODERS
################################################################################
def large_decoder(codes,
                  height,
                  width,
                  channels,
                  batch_norm_params=None,
                  weight_decay=0.0):
  """Decodes the codes to a fixed output size.

  Args:
    codes: a tensor of size [batch_size, code_size].
    height: the height of the output images.
    width: the width of the output images.
    channels: the number of the output channels.
    batch_norm_params: a dictionary that maps batch norm parameter names to
      values.
    weight_decay: the value for the weight decay coefficient.

  Returns:
    recons: the reconstruction tensor of shape [batch_size, height, width, 3].
  """
  with slim.arg_scope(
      [slim.conv2d, slim.fully_connected],
      weights_regularizer=slim.l2_regularizer(weight_decay),
      activation_fn=tf.nn.relu,
      normalizer_fn=slim.batch_norm,
      normalizer_params=batch_norm_params):
    net = slim.fully_connected(codes, 600, scope='fc1')
    batch_size = net.get_shape().as_list()[0]
    net = tf.reshape(net, [batch_size, 10, 10, 6])

    net = slim.conv2d(net, 32, [5, 5], scope='conv1_1')

    net = tf.image.resize_nearest_neighbor(net, (16, 16))

    net = slim.conv2d(net, 32, [5, 5], scope='conv2_1')

    net = tf.image.resize_nearest_neighbor(net, (32, 32))

    net = slim.conv2d(net, 32, [5, 5], scope='conv3_2')

    output_size = [height, width]
    net = tf.image.resize_nearest_neighbor(net, output_size)

    with slim.arg_scope([slim.conv2d], kernel_size=[3, 3]):
      net = slim.conv2d(net, channels, activation_fn=None, scope='conv4_1')

  return net


def gtsrb_decoder(codes,
                  height,
                  width,
                  channels,
                  batch_norm_params=None,
                  weight_decay=0.0):
  """Decodes the codes to a fixed output size. This decoder is specific to GTSRB

  Args:
    codes: a tensor of size [batch_size, 100].
    height: the height of the output images.
    width: the width of the output images.
    channels: the number of the output channels.
    batch_norm_params: a dictionary that maps batch norm parameter names to
      values.
    weight_decay: the value for the weight decay coefficient.

  Returns:
    recons: the reconstruction tensor of shape [batch_size, height, width, 3].

  Raises:
    ValueError: When the input code size is not 100.
  """
  batch_size, code_size = codes.get_shape().as_list()
  if code_size != 100:
    raise ValueError('The code size used as an input to the GTSRB decoder is '
                     'expected to be 100.')

  with slim.arg_scope(
      [slim.conv2d, slim.fully_connected],
      weights_regularizer=slim.l2_regularizer(weight_decay),
      activation_fn=tf.nn.relu,
      normalizer_fn=slim.batch_norm,
      normalizer_params=batch_norm_params):
    net = codes
    net = tf.reshape(net, [batch_size, 10, 10, 1])
    net = slim.conv2d(net, 32, [3, 3], scope='conv1_1')

    # First upsampling 20x20
    net = tf.image.resize_nearest_neighbor(net, [20, 20])

    net = slim.conv2d(net, 32, [3, 3], scope='conv2_1')

    output_size = [height, width]
    # Final upsampling 40 x 40
    net = tf.image.resize_nearest_neighbor(net, output_size)

    with slim.arg_scope([slim.conv2d], kernel_size=[3, 3]):
      net = slim.conv2d(net, 16, scope='conv3_1')
      net = slim.conv2d(net, channels, activation_fn=None, scope='conv3_2')

  return net


def small_decoder(codes,
                  height,
                  width,
                  channels,
                  batch_norm_params=None,
                  weight_decay=0.0):
  """Decodes the codes to a fixed output size.

  Args:
    codes: a tensor of size [batch_size, code_size].
    height: the height of the output images.
    width: the width of the output images.
    channels: the number of the output channels.
    batch_norm_params: a dictionary that maps batch norm parameter names to
      values.
    weight_decay: the value for the weight decay coefficient.

  Returns:
    recons: the reconstruction tensor of shape [batch_size, height, width, 3].
  """
  with slim.arg_scope(
      [slim.conv2d, slim.fully_connected],
      weights_regularizer=slim.l2_regularizer(weight_decay),
      activation_fn=tf.nn.relu,
      normalizer_fn=slim.batch_norm,
      normalizer_params=batch_norm_params):
    net = slim.fully_connected(codes, 300, scope='fc1')
    batch_size = net.get_shape().as_list()[0]
    net = tf.reshape(net, [batch_size, 10, 10, 3])

    net = slim.conv2d(net, 16, [3, 3], scope='conv1_1')
    net = slim.conv2d(net, 16, [3, 3], scope='conv1_2')

    output_size = [height, width]
    net = tf.image.resize_nearest_neighbor(net, output_size)

    with slim.arg_scope([slim.conv2d], kernel_size=[3, 3]):
      net = slim.conv2d(net, 16, scope='conv2_1')
      net = slim.conv2d(net, channels, activation_fn=None, scope='conv2_2')

  return net


################################################################################
# SHARED ENCODERS
################################################################################
def dann_mnist(images,
               weight_decay=0.0,
               prefix='model',
               num_classes=10,
               **kwargs):
  """Creates a convolution MNIST model.

  Note that this model implements the architecture for MNIST proposed in:
   Y. Ganin et al., Domain-Adversarial Training of Neural Networks (DANN),
   JMLR 2015

  Args:
    images: the MNIST digits, a tensor of size [batch_size, 28, 28, 1].
    weight_decay: the value for the weight decay coefficient.
    prefix: name of the model to use when prefixing tags.
    num_classes: the number of output classes to use.
    **kwargs: Placeholder for keyword arguments used by other shared encoders.

  Returns:
    the output logits, a tensor of size [batch_size, num_classes].
    a dictionary with key/values the layer names and tensors.
  """
  end_points = {}

  with slim.arg_scope(
      [slim.conv2d, slim.fully_connected],
      weights_regularizer=slim.l2_regularizer(weight_decay),
      activation_fn=tf.nn.relu,):
    with slim.arg_scope([slim.conv2d], padding='SAME'):
      end_points['conv1'] = slim.conv2d(images, 32, [5, 5], scope='conv1')
      end_points['pool1'] = slim.max_pool2d(
          end_points['conv1'], [2, 2], 2, scope='pool1')
      end_points['conv2'] = slim.conv2d(
          end_points['pool1'], 48, [5, 5], scope='conv2')
      end_points['pool2'] = slim.max_pool2d(
          end_points['conv2'], [2, 2], 2, scope='pool2')
      end_points['fc3'] = slim.fully_connected(
          slim.flatten(end_points['pool2']), 100, scope='fc3')
      end_points['fc4'] = slim.fully_connected(
          slim.flatten(end_points['fc3']), 100, scope='fc4')

  logits = slim.fully_connected(
      end_points['fc4'], num_classes, activation_fn=None, scope='fc5')

  return logits, end_points


def dann_svhn(images,
              weight_decay=0.0,
              prefix='model',
              num_classes=10,
              **kwargs):
  """Creates the convolutional SVHN model.

  Note that this model implements the architecture for MNIST proposed in:
   Y. Ganin et al., Domain-Adversarial Training of Neural Networks (DANN),
   JMLR 2015

  Args:
    images: the SVHN digits, a tensor of size [batch_size, 32, 32, 3].
    weight_decay: the value for the weight decay coefficient.
    prefix: name of the model to use when prefixing tags.
    num_classes: the number of output classes to use.
    **kwargs: Placeholder for keyword arguments used by other shared encoders.

  Returns:
    the output logits, a tensor of size [batch_size, num_classes].
    a dictionary with key/values the layer names and tensors.
  """

  end_points = {}

  with slim.arg_scope(
      [slim.conv2d, slim.fully_connected],
      weights_regularizer=slim.l2_regularizer(weight_decay),
      activation_fn=tf.nn.relu,):
    with slim.arg_scope([slim.conv2d], padding='SAME'):

      end_points['conv1'] = slim.conv2d(images, 64, [5, 5], scope='conv1')
      end_points['pool1'] = slim.max_pool2d(
          end_points['conv1'], [3, 3], 2, scope='pool1')
      end_points['conv2'] = slim.conv2d(
          end_points['pool1'], 64, [5, 5], scope='conv2')
      end_points['pool2'] = slim.max_pool2d(
          end_points['conv2'], [3, 3], 2, scope='pool2')
      end_points['conv3'] = slim.conv2d(
          end_points['pool2'], 128, [5, 5], scope='conv3')

      end_points['fc3'] = slim.fully_connected(
          slim.flatten(end_points['conv3']), 3072, scope='fc3')
      end_points['fc4'] = slim.fully_connected(
          slim.flatten(end_points['fc3']), 2048, scope='fc4')

  logits = slim.fully_connected(
      end_points['fc4'], num_classes, activation_fn=None, scope='fc5')

  return logits, end_points


def dann_gtsrb(images,
               weight_decay=0.0,
               prefix='model',
               num_classes=43,
               **kwargs):
  """Creates the convolutional GTSRB model.

  Note that this model implements the architecture for MNIST proposed in:
   Y. Ganin et al., Domain-Adversarial Training of Neural Networks (DANN),
   JMLR 2015

  Args:
    images: the GTSRB images, a tensor of size [batch_size, 40, 40, 3].
    weight_decay: the value for the weight decay coefficient.
    prefix: name of the model to use when prefixing tags.
    num_classes: the number of output classes to use.
    **kwargs: Placeholder for keyword arguments used by other shared encoders.

  Returns:
    the output logits, a tensor of size [batch_size, num_classes].
    a dictionary with key/values the layer names and tensors.
  """

  end_points = {}

  with slim.arg_scope(
      [slim.conv2d, slim.fully_connected],
      weights_regularizer=slim.l2_regularizer(weight_decay),
      activation_fn=tf.nn.relu,):
    with slim.arg_scope([slim.conv2d], padding='SAME'):

      end_points['conv1'] = slim.conv2d(images, 96, [5, 5], scope='conv1')
      end_points['pool1'] = slim.max_pool2d(
          end_points['conv1'], [2, 2], 2, scope='pool1')
      end_points['conv2'] = slim.conv2d(
          end_points['pool1'], 144, [3, 3], scope='conv2')
      end_points['pool2'] = slim.max_pool2d(
          end_points['conv2'], [2, 2], 2, scope='pool2')
      end_points['conv3'] = slim.conv2d(
          end_points['pool2'], 256, [5, 5], scope='conv3')
      end_points['pool3'] = slim.max_pool2d(
          end_points['conv3'], [2, 2], 2, scope='pool3')

      end_points['fc3'] = slim.fully_connected(
          slim.flatten(end_points['pool3']), 512, scope='fc3')

  logits = slim.fully_connected(
      end_points['fc3'], num_classes, activation_fn=None, scope='fc4')

  return logits, end_points


def dsn_cropped_linemod(images,
                        weight_decay=0.0,
                        prefix='model',
                        num_classes=11,
                        batch_norm_params=None,
                        is_training=False):
  """Creates the convolutional pose estimation model for Cropped Linemod.

  Args:
    images: the Cropped Linemod samples, a tensor of size
      [batch_size, 64, 64, 4].
    weight_decay: the value for the weight decay coefficient.
    prefix: name of the model to use when prefixing tags.
    num_classes: the number of output classes to use.
    batch_norm_params: a dictionary that maps batch norm parameter names to
      values.
    is_training: specifies whether or not we're currently training the model.
      This variable will determine the behaviour of the dropout layer.

  Returns:
    the output logits, a tensor of size [batch_size, num_classes].
    a dictionary with key/values the layer names and tensors.
  """

  end_points = {}

  tf.summary.image('{}/input_images'.format(prefix), images)
  with slim.arg_scope(
      [slim.conv2d, slim.fully_connected],
      weights_regularizer=slim.l2_regularizer(weight_decay),
      activation_fn=tf.nn.relu,
      normalizer_fn=slim.batch_norm if batch_norm_params else None,
      normalizer_params=batch_norm_params):
    with slim.arg_scope([slim.conv2d], padding='SAME'):
      end_points['conv1'] = slim.conv2d(images, 32, [5, 5], scope='conv1')
      end_points['pool1'] = slim.max_pool2d(
          end_points['conv1'], [2, 2], 2, scope='pool1')
      end_points['conv2'] = slim.conv2d(
          end_points['pool1'], 64, [5, 5], scope='conv2')
      end_points['pool2'] = slim.max_pool2d(
          end_points['conv2'], [2, 2], 2, scope='pool2')
      net = slim.flatten(end_points['pool2'])
      end_points['fc3'] = slim.fully_connected(net, 128, scope='fc3')
      net = slim.dropout(
          end_points['fc3'], 0.5, is_training=is_training, scope='dropout')

      with tf.variable_scope('quaternion_prediction'):
        predicted_quaternion = slim.fully_connected(
            net, 4, activation_fn=tf.nn.tanh)
        predicted_quaternion = tf.nn.l2_normalize(predicted_quaternion, 1)
      logits = slim.fully_connected(
          net, num_classes, activation_fn=None, scope='fc4')
  end_points['quaternion_pred'] = predicted_quaternion

  return logits, end_points
