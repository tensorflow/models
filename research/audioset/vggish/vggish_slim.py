# Copyright 2017 The TensorFlow Authors All Rights Reserved.
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

"""Defines the 'VGGish' model used to generate AudioSet embedding features.

The public AudioSet release (https://research.google.com/audioset/download.html)
includes 128-D features extracted from the embedding layer of a VGG-like model
that was trained on a large Google-internal YouTube dataset. Here we provide
a TF-Slim definition of the same model, without any dependences on libraries
internal to Google. We call it 'VGGish'.

Note that we only define the model up to the embedding layer, which is the
penultimate layer before the final classifier layer. We also provide various
hyperparameter values (in vggish_params.py) that were used to train this model
internally.

For comparison, here is TF-Slim's VGG definition:
https://github.com/tensorflow/models/blob/master/research/slim/nets/vgg.py
"""

import tensorflow.compat.v1 as tf
import tf_slim as slim

import vggish_params as params


def define_vggish_slim(features_tensor=None, training=False):
  """Defines the VGGish TensorFlow model.

  All ops are created in the current default graph, under the scope 'vggish/'.

  The input is either a tensor passed in via the optional 'features_tensor'
  argument or a placeholder created below named 'vggish/input_features'. The
  input is expected to have dtype float32 and shape [batch_size, num_frames,
  num_bands] where batch_size is variable and num_frames and num_bands are
  constants, and [num_frames, num_bands] represents a log-mel-scale spectrogram
  patch covering num_bands frequency bands and num_frames time frames (where
  each frame step is usually 10ms). This is produced by computing the stabilized
  log(mel-spectrogram + params.LOG_OFFSET).  The output is a tensor named
  'vggish/embedding' which produces the pre-activation values of a 128-D
  embedding layer, which is usually the penultimate layer when used as part of a
  full model with a final classifier layer.

  Args:
    features_tensor: If not None, the tensor containing the input features.
      If None, a placeholder input is created.
    training: If true, all parameters are marked trainable.

  Returns:
    The op 'vggish/embeddings'.
  """
  # Defaults:
  # - All weights are initialized to N(0, INIT_STDDEV).
  # - All biases are initialized to 0.
  # - All activations are ReLU.
  # - All convolutions are 3x3 with stride 1 and SAME padding.
  # - All max-pools are 2x2 with stride 2 and SAME padding.
  with slim.arg_scope([slim.conv2d, slim.fully_connected],
                      weights_initializer=tf.truncated_normal_initializer(
                          stddev=params.INIT_STDDEV),
                      biases_initializer=tf.zeros_initializer(),
                      activation_fn=tf.nn.relu,
                      trainable=training), \
       slim.arg_scope([slim.conv2d],
                      kernel_size=[3, 3], stride=1, padding='SAME'), \
       slim.arg_scope([slim.max_pool2d],
                      kernel_size=[2, 2], stride=2, padding='SAME'), \
       tf.variable_scope('vggish'):
    # Input: a batch of 2-D log-mel-spectrogram patches.
    if features_tensor is None:
      features_tensor = tf.placeholder(
          tf.float32, shape=(None, params.NUM_FRAMES, params.NUM_BANDS),
          name='input_features')
    # Reshape to 4-D so that we can convolve a batch with conv2d().
    net = tf.reshape(features_tensor,
                     [-1, params.NUM_FRAMES, params.NUM_BANDS, 1])

    # The VGG stack of alternating convolutions and max-pools.
    net = slim.conv2d(net, 64, scope='conv1')
    net = slim.max_pool2d(net, scope='pool1')
    net = slim.conv2d(net, 128, scope='conv2')
    net = slim.max_pool2d(net, scope='pool2')
    net = slim.repeat(net, 2, slim.conv2d, 256, scope='conv3')
    net = slim.max_pool2d(net, scope='pool3')
    net = slim.repeat(net, 2, slim.conv2d, 512, scope='conv4')
    net = slim.max_pool2d(net, scope='pool4')

    # Flatten before entering fully-connected layers
    net = slim.flatten(net)
    net = slim.repeat(net, 2, slim.fully_connected, 4096, scope='fc1')
    # The embedding layer.
    net = slim.fully_connected(net, params.EMBEDDING_SIZE, scope='fc2',
                               activation_fn=None)
    return tf.identity(net, name='embedding')


def load_vggish_slim_checkpoint(session, checkpoint_path):
  """Loads a pre-trained VGGish-compatible checkpoint.

  This function can be used as an initialization function (referred to as
  init_fn in TensorFlow documentation) which is called in a Session after
  initializing all variables. When used as an init_fn, this will load
  a pre-trained checkpoint that is compatible with the VGGish model
  definition. Only variables defined by VGGish will be loaded.

  Args:
    session: an active TensorFlow session.
    checkpoint_path: path to a file containing a checkpoint that is
      compatible with the VGGish model definition.
  """
  # Get the list of names of all VGGish variables that exist in
  # the checkpoint (i.e., all inference-mode VGGish variables).
  with tf.Graph().as_default():
    define_vggish_slim(training=False)
    vggish_var_names = [v.name for v in tf.global_variables()]

  # Get the list of all currently existing variables that match
  # the list of variable names we just computed.
  vggish_vars = [v for v in tf.global_variables() if v.name in vggish_var_names]

  # Use a Saver to restore just the variables selected above.
  saver = tf.train.Saver(vggish_vars, name='vggish_load_pretrained',
                         write_version=1)
  saver.restore(session, checkpoint_path)
