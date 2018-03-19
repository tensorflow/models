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

"""Model implementations."""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from abc import ABCMeta
from abc import abstractmethod
import tensorflow as tf
import tensorflow.contrib.slim as slim
from tensorflow.contrib.slim.python.slim.nets import inception
from tensorflow.contrib.slim.python.slim.nets import resnet_v2 as resnet_v2
from tensorflow.contrib.slim.python.slim.nets import resnet_utils as resnet_utils


def get_embedder(
    embedder_strategy, config, images, is_training, reuse=False,
    l2_normalize_embedding=True):
  """Returns an embedder based on config.

  Args:
    embedder_strategy: String, name of embedder version to return.
    config: LuaTable object, training config.
    images: 4-D float `Tensor` containing batch images.
    is_training: Boolean or placeholder for boolean,
      indicator for whether or not we're training.
    reuse: Boolean: Reuse embedder variable scope.
    l2_normalize_embedding: Boolean, whether or not to l2 normalize the
      embedding.
  Returns:
    embedder: An `Embedder` object.
  Raises:
    ValueError: if unknown embedder_strategy specified.
  """
  if embedder_strategy == 'inception_baseline':
    pretrained_ckpt = config.inception_conv_ss_fc.pretrained_checkpoint
    return InceptionBaselineEmbedder(
        images,
        pretrained_ckpt,
        config.random_projection,
        config.random_projection_dim)

  strategy_to_embedder = {
      'inception_conv_ss_fc': InceptionConvSSFCEmbedder,
      'resnet': ResnetEmbedder,
  }
  if embedder_strategy not in strategy_to_embedder:
    raise ValueError('unknown embedder_strategy', embedder_strategy)

  embedding_size = config.embedding_size
  l2_reg_weight = config.learning.l2_reg_weight
  embedder = strategy_to_embedder[embedder_strategy](
      config[embedder_strategy], images, embedding_size,
      is_training, embedding_l2=l2_normalize_embedding,
      l2_reg_weight=l2_reg_weight, reuse=reuse)
  return embedder


def build_inceptionv3_graph(images, endpoint, is_training, checkpoint,
                            reuse=False):
  """Builds an InceptionV3 model graph.

  Args:
    images: A 4-D float32 `Tensor` of batch images.
    endpoint: String, name of the InceptionV3 endpoint.
    is_training: Boolean, whether or not to build a training or inference graph.
    checkpoint: String, path to the pretrained model checkpoint.
    reuse: Boolean, whether or not we are reusing the embedder.
  Returns:
    inception_output: `Tensor` holding the InceptionV3 output.
    inception_variables: List of inception variables.
    init_fn: Function to initialize the weights (if not reusing, then None).
  """
  with slim.arg_scope(inception.inception_v3_arg_scope()):
    _, endpoints = inception.inception_v3(
        images, num_classes=1001, is_training=is_training)
    inception_output = endpoints[endpoint]
    inception_variables = slim.get_variables_to_restore()
    inception_variables = [
        i for i in inception_variables if 'global_step' not in i.name]
    if is_training and not reuse:
      init_saver = tf.train.Saver(inception_variables)
      def init_fn(scaffold, sess):
        del scaffold
        init_saver.restore(sess, checkpoint)
    else:
      init_fn = None
    return inception_output, inception_variables, init_fn


class InceptionBaselineEmbedder(object):
  """Produces pre-trained InceptionV3 embeddings."""

  def __init__(self, images, pretrained_ckpt, reuse=False,
               random_projection=False, random_projection_dim=32):
    # Build InceptionV3 graph.
    (inception_output,
     self.inception_variables,
     self.init_fn) = build_inceptionv3_graph(
         images, 'Mixed_7c', False, pretrained_ckpt, reuse)

    # Pool 8x8x2048 -> 1x1x2048.
    embedding = slim.avg_pool2d(inception_output, [8, 8], stride=1)
    embedding = tf.squeeze(embedding, [1, 2])

    if random_projection:
      embedding = tf.matmul(
          embedding, tf.random_normal(
              shape=[2048, random_projection_dim], seed=123))
    self.embedding = embedding


class PretrainedEmbedder(object):
  """Base class for embedders that take pre-trained networks as input."""
  __metaclass__ = ABCMeta

  def __init__(self, config, images, embedding_size, is_training,
               embedding_l2=True, l2_reg_weight=1e-6, reuse=False):
    """Constructor.

    Args:
      config: A T object holding training config.
      images: A 4-D float32 `Tensor` holding images to embed.
      embedding_size: Int, the size of the embedding.
      is_training: Boolean, whether or not this is a training or inference-time
        graph.
      embedding_l2: Boolean, whether or not to l2 normalize the embedding.
      l2_reg_weight: Float, weight applied to l2 weight regularization.
      reuse: Boolean, whether or not we're reusing this graph.
    """
    # Pull out all the embedder hyperparameters.
    self._config = config
    self._embedding_size = embedding_size
    self._l2_reg_weight = l2_reg_weight
    self._embedding_l2 = embedding_l2
    self._is_training = is_training
    self._reuse = reuse

    # Pull out pretrained hparams.
    pretrained_checkpoint = config.pretrained_checkpoint
    pretrained_layer = config.pretrained_layer
    pretrained_keep_prob = config.dropout.keep_pretrained

    # Build pretrained graph.
    (pretrained_output,
     self._pretrained_variables,
     self.init_fn) = self.build_pretrained_graph(
         images, pretrained_layer, pretrained_checkpoint, is_training, reuse)

    # Optionally drop out the activations.
    pretrained_output = slim.dropout(
        pretrained_output, keep_prob=pretrained_keep_prob,
        is_training=is_training)
    self._pretrained_output = pretrained_output

  @abstractmethod
  def build_pretrained_graph(self, images, layer, pretrained_checkpoint,
                             is_training, reuse):
    """Builds the graph for the pre-trained network.

    Method to be overridden by implementations.

    Args:
      images: A 4-D tf.float32 `Tensor` holding images to embed.
      layer: String, defining which pretrained layer to take as input
        to adaptation layers.
      pretrained_checkpoint: String, path to a checkpoint used to load
        pretrained weights.
      is_training: Boolean, whether or not we're in training mode.
      reuse: Boolean, whether or not to reuse embedder weights.

    Returns:
      pretrained_output: A 2 or 3-d tf.float32 `Tensor` holding pretrained
        activations.
    """
    pass

  @abstractmethod
  def construct_embedding(self):
    """Builds an embedding function on top of images.

    Method to be overridden by implementations.

    Returns:
      embeddings: A 2-d float32 `Tensor` of shape [batch_size, embedding_size]
        holding the embedded images.
    """
    pass

  def get_trainable_variables(self):
    """Gets a list of variables to optimize."""
    if self._config.finetune:
      return tf.trainable_variables()
    else:
      adaptation_only_vars = tf.get_collection(
          tf.GraphKeys.TRAINABLE_VARIABLES, scope=self._adaptation_scope)
      return adaptation_only_vars


class ResnetEmbedder(PretrainedEmbedder):
  """Resnet TCN.

  ResnetV2 -> resnet adaptation layers -> optional l2 normalize -> embedding.
  """

  def __init__(self, config, images, embedding_size, is_training,
               embedding_l2=True, l2_reg_weight=1e-6, reuse=False):
    super(ResnetEmbedder, self).__init__(
        config, images, embedding_size, is_training, embedding_l2,
        l2_reg_weight, reuse)

  def build_pretrained_graph(
      self, images, resnet_layer, checkpoint, is_training, reuse=False):
    """See baseclass."""
    with slim.arg_scope(resnet_v2.resnet_arg_scope()):
      _, endpoints = resnet_v2.resnet_v2_50(
          images, is_training=is_training, reuse=reuse)
      resnet_layer = 'resnet_v2_50/block%d' % resnet_layer
      resnet_output = endpoints[resnet_layer]
      resnet_variables = slim.get_variables_to_restore()
      resnet_variables = [
          i for i in resnet_variables if 'global_step' not in i.name]
      if is_training and not reuse:
        init_saver = tf.train.Saver(resnet_variables)
        def init_fn(scaffold, sess):
          del scaffold
          init_saver.restore(sess, checkpoint)
      else:
        init_fn = None

      return resnet_output, resnet_variables, init_fn

  def construct_embedding(self):
    """Builds an embedding function on top of images.

    Method to be overridden by implementations.

    Returns:
      embeddings: A 2-d float32 `Tensor` of shape [batch_size, embedding_size]
        holding the embedded images.
    """
    with tf.variable_scope('tcn_net', reuse=self._reuse) as vs:
      self._adaptation_scope = vs.name
      net = self._pretrained_output

      # Define some adaptation blocks on top of the pre-trained resnet output.
      adaptation_blocks = []
      adaptation_block_params = [map(
          int, i.split('_')) for i in self._config.adaptation_blocks.split('-')]
      for i, (depth, num_units) in enumerate(adaptation_block_params):
        block = resnet_v2.resnet_v2_block(
            'adaptation_block_%d' % i, base_depth=depth, num_units=num_units,
            stride=1)
        adaptation_blocks.append(block)

      # Stack them on top of the resent output.
      net = resnet_utils.stack_blocks_dense(
          net, adaptation_blocks, output_stride=None)

      # Average pool the output.
      net = tf.reduce_mean(net, [1, 2], name='adaptation_pool', keep_dims=True)

      if self._config.emb_connection == 'fc':
        # Use fully connected layer to project to embedding layer.
        fc_hidden_sizes = self._config.fc_hidden_sizes
        if fc_hidden_sizes == 'None':
          fc_hidden_sizes = []
        else:
          fc_hidden_sizes = map(int, fc_hidden_sizes.split('_'))
        fc_hidden_keep_prob = self._config.dropout.keep_fc
        net = tf.squeeze(net)
        for fc_hidden_size in fc_hidden_sizes:
          net = slim.layers.fully_connected(net, fc_hidden_size)
          if fc_hidden_keep_prob < 1.0:
            net = slim.dropout(net, keep_prob=fc_hidden_keep_prob,
                               is_training=self._is_training)

        # Connect last FC layer to embedding.
        embedding = slim.layers.fully_connected(net, self._embedding_size,
                                                activation_fn=None)
      else:
        # Use 1x1 conv layer to project to embedding layer.
        embedding = slim.conv2d(
            net, self._embedding_size, [1, 1], activation_fn=None,
            normalizer_fn=None, scope='embedding')
        embedding = tf.squeeze(embedding)

      # Optionally L2 normalize the embedding.
      if self._embedding_l2:
        embedding = tf.nn.l2_normalize(embedding, dim=1)

      return embedding

  def get_trainable_variables(self):
    """Gets a list of variables to optimize."""
    if self._config.finetune:
      return tf.trainable_variables()
    else:
      adaptation_only_vars = tf.get_collection(
          tf.GraphKeys.TRAINABLE_VARIABLES, scope=self._adaptation_scope)
      return adaptation_only_vars


class InceptionEmbedderBase(PretrainedEmbedder):
  """Base class for embedders that take pre-trained InceptionV3 activations."""

  def __init__(self, config, images, embedding_size, is_training,
               embedding_l2=True, l2_reg_weight=1e-6, reuse=False):
    super(InceptionEmbedderBase, self).__init__(
        config, images, embedding_size, is_training, embedding_l2,
        l2_reg_weight, reuse)

  def build_pretrained_graph(
      self, images, inception_layer, checkpoint, is_training, reuse=False):
    """See baseclass."""
    # Build InceptionV3 graph.
    inception_output, inception_variables, init_fn = build_inceptionv3_graph(
        images, inception_layer, is_training, checkpoint, reuse)
    return inception_output, inception_variables, init_fn


class InceptionConvSSFCEmbedder(InceptionEmbedderBase):
  """TCN Embedder V1.

  InceptionV3 (mixed_5d) -> conv layers -> spatial softmax ->
    fully connected -> optional l2 normalize -> embedding.
  """

  def __init__(self, config, images, embedding_size, is_training,
               embedding_l2=True, l2_reg_weight=1e-6, reuse=False):
    super(InceptionConvSSFCEmbedder, self).__init__(
        config, images, embedding_size, is_training, embedding_l2,
        l2_reg_weight, reuse)

    # Pull out all the hyperparameters specific to this embedder.
    self._additional_conv_sizes = config.additional_conv_sizes
    self._conv_hidden_keep_prob = config.dropout.keep_conv
    self._fc_hidden_sizes = config.fc_hidden_sizes
    self._fc_hidden_keep_prob = config.dropout.keep_fc

  def construct_embedding(self):
    """Builds a conv -> spatial softmax -> FC adaptation network."""
    is_training = self._is_training
    normalizer_params = {'is_training': is_training}
    with tf.variable_scope('tcn_net', reuse=self._reuse) as vs:
      self._adaptation_scope = vs.name
      with slim.arg_scope(
          [slim.layers.conv2d],
          activation_fn=tf.nn.relu,
          normalizer_fn=slim.batch_norm, normalizer_params=normalizer_params,
          weights_regularizer=slim.regularizers.l2_regularizer(
              self._l2_reg_weight),
          biases_regularizer=slim.regularizers.l2_regularizer(
              self._l2_reg_weight)):
        with slim.arg_scope(
            [slim.layers.fully_connected],
            activation_fn=tf.nn.relu,
            normalizer_fn=slim.batch_norm, normalizer_params=normalizer_params,
            weights_regularizer=slim.regularizers.l2_regularizer(
                self._l2_reg_weight),
            biases_regularizer=slim.regularizers.l2_regularizer(
                self._l2_reg_weight)):

          # Input to embedder is pre-trained inception output.
          net = self._pretrained_output

          # Optionally add more conv layers.
          for num_filters in self._additional_conv_sizes:
            net = slim.layers.conv2d(
                net, num_filters, kernel_size=[3, 3], stride=[1, 1])
            net = slim.dropout(net, keep_prob=self._conv_hidden_keep_prob,
                               is_training=is_training)

          # Take the spatial soft arg-max of the last convolutional layer.
          # This is a form of spatial attention over the activations.
          # See more here: http://arxiv.org/abs/1509.06113.
          net = tf.contrib.layers.spatial_softmax(net)
          self.spatial_features = net

          # Add fully connected layers.
          net = slim.layers.flatten(net)
          for fc_hidden_size in self._fc_hidden_sizes:
            net = slim.layers.fully_connected(net, fc_hidden_size)
            if self._fc_hidden_keep_prob < 1.0:
              net = slim.dropout(net, keep_prob=self._fc_hidden_keep_prob,
                                 is_training=is_training)

          # Connect last FC layer to embedding.
          net = slim.layers.fully_connected(net, self._embedding_size,
                                            activation_fn=None)

          # Optionally L2 normalize the embedding.
          if self._embedding_l2:
            net = tf.nn.l2_normalize(net, dim=1)

          return net
