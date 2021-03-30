# Lint as: python3
# Copyright 2021 The TensorFlow Authors. All Rights Reserved.
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
"""CNN Image Retrieval model implementation based on the following papers:

  [1] Fine-tuning CNN Image Retrieval with No Human Annotation,
    Radenović F., Tolias G., Chum O., TPAMI 2018 [arXiv]
    https://arxiv.org/abs/1711.02512

  [2] CNN Image Retrieval Learns from BoW: Unsupervised Fine-Tuning with Hard
    Examples, Radenović F., Tolias G., Chum O., ECCV 2016 [arXiv]
    https://arxiv.org/abs/1604.02426
"""

import os

from absl import flags

import numpy as np
import pickle
import tensorflow as tf

from delf.python.pooling_layers import pooling as pooling_layers
from delf.python.normalization_layers import normalization
from delf.python.datasets import generic_dataset
from delf.python.training import global_features_utils

FLAGS = flags.FLAGS

# Pre-computed global whitening, for most commonly used architectures.
# Using pre-computed whitening improves the speed of the convergence and the
# performance.
_WHITENING_CONFIG = {
  'ResNet50': 'http://cmp.felk.cvut.cz/cnnimageretrieval_tf'
              '/SFM120k_ResNet50_gem_learned_whitening_config.pkl',
  'ResNet101': 'http://cmp.felk.cvut.cz/cnnimageretrieval_tf'
               '/SFM120k_ResNet101_gem_learned_whitening_config.pkl',
  'ResNet152': 'http://cmp.felk.cvut.cz/cnnimageretrieval_tf'
               '/SFM120k_ResNet152_gem_learned_whitening_config.pkl',
  'VGG19': 'http://cmp.felk.cvut.cz/cnnimageretrieval_tf'
           '/SFM120k_VGG19_gem_learned_whitening_config.pkl'
}

# Possible global pooling layers.
_POOLING = {
  'mac': pooling_layers.MAC,
  'spoc': pooling_layers.SPoC,
  'gem': pooling_layers.GeM
}

# Output dimensionality for supported architectures.
_OUTPUT_DIM = {
  'VGG16': 512,
  'VGG19': 512,
  'ResNet50': 2048,
  'ResNet101': 2048,
  'ResNet101V2': 2048,
  'ResNet152': 2048,
  'DenseNet121': 1024,
  'DenseNet169': 1664,
  'DenseNet201': 1920,
  'EfficientNetB5': 2048,
  'EfficientNetB7': 2560
}


class GlobalFeatureNet(tf.keras.Model):
  """Instantiates global model for image retrieval.

  This class implements the [GlobalFeatureNet](
  https://arxiv.org/abs/1711.02512) for image retrieval. The model uses a
  user-defined model as a backbone.
  """

  def __init__(self, architecture='ResNet101', pooling='gem',
               whitening=False, pretrained=True):
    """GlobalFeatureNet network initialization.

    Args:
      architecture: Network backbone 'VGG16'/'VGG19'/'ResNet50'/'ResNet101/
        'ResNet101V2'/'ResNet152'/'DenseNet121'/'DenseNet169'/'DenseNet201'.
      pooling: Pooling method used 'mac'/'spoc'/'gem'.
      whitening: Bool, whether to use whitening.
      pretrained: Bool, whether to initialize the network with pretrained
        weights.
    """
    super(GlobalFeatureNet, self).__init__()

    # Get standard output dimensionality size.
    dim = _OUTPUT_DIM[architecture]

    if pretrained:
      # Initialize with network pretrained on imagenet.
      net_in = getattr(tf.keras.applications, architecture)(include_top=False,
                                                            weights="imagenet")
    else:
      # Initialize with random weights.
      net_in = getattr(tf.keras.applications, architecture)(include_top=False,
                                                            weights=None)

    # Initialize `feature_extractor`. Take only convolutions for
    # `feature_extractor`, always end with ReLU to make last activations
    # non-negative.
    if architecture.lower().startswith('densenet'):
      net_in.add(tf.keras.layers.ReLU())

    # Initialize pooling.
    pool = _POOLING[pooling]()

    # Initialize whitening.
    if whitening:
      if pretrained and architecture in _WHITENING_CONFIG:
        # If precomputed whitening for the architecture exists,
        # the fully-connected layer is going to be initialized according to
        # the precomputed layer configuration.
        global_features_utils.debug_and_log(
          ">> {}: for '{}' custom computed whitening '{}' is used."
            .format(os.getcwd(), architecture,
                    os.path.basename(_WHITENING_CONFIG[architecture])))
        # The layer configuration is downloaded to the `data_root` folder.
        whiten_dir = os.path.join(FLAGS.data_root, architecture)
        path = tf.keras.utils.get_file(fname=whiten_dir,
                                       origin=_WHITENING_CONFIG[architecture])
        # Whitening configuration is loaded.
        with tf.io.gfile.GFile(path, 'rb') as learned_whitening_file:
          whitening_config = pickle.load(learned_whitening_file)
        # Whitening layer is initialized according to the configuration.
        whiten = tf.keras.layers.Dense.from_config(whitening_config)
      else:
        # In case if no precomputed whitening exists for the chosen
        # architecture, the fully-connected whitening layer is initialized
        # with the random weights.
        whiten = tf.keras.layers.Dense(dim, activation=None, use_bias=True)
        global_features_utils.debug_and_log(
          ">> {}: for '{}' there is either no whitening computed or the "
          "pretrained is False, random weights are used.".format(
            os.getcwd(), w))
    else:
      whiten = None

    # Create meta information to be stored in the network.
    self.meta = {
      'architecture': architecture,
      'pooling': pooling,
      'whitening': whitening,
      'outputdim': dim
    }

    # Freeze running mean and std in batch normalization layers.
    # We do training one image at a time to improve memory requirements of
    # the network; therefore, the computed statistics would not be per a
    # batch. Instead, we choose freezing - setting the parameters of all
    # batch norm layers in the network to non-trainable (i.e., using original
    # imagenet statistics).
    for layer in net_in.layers:
      if isinstance(layer, tf.keras.layers.BatchNormalization):
        layer.trainable = False

    self.feature_extractor = net_in
    self.pool = pool
    self.whiten = whiten
    self.norm = normalization.L2Normalization()

  def call(self, x, training=False):
    """Invokes the GlobalFeatureNet instance.

    Args:
      x: [B, H, W, C] Tensor with a batch of images.
      training: Indicator of whether the forward pass is running in training
      mode or not.

    Returns:
      out: [B, out_dim] Global descriptor.
    """
    # Forward pass through the fully-convolutional backbone.
    o = self.feature_extractor(x, training)
    # Pooling.
    o = self.pool(o)
    # Normalization.
    o = self.norm(o)

    # If whitening exists: the pooled global descriptor is whitened and
    # re-normalized.
    if self.whiten is not None:
      o = self.whiten(o)
      o = tf.nn.l2_normalize(o, axis=1, epsilon=self.norm.eps)
    return o

  def meta_repr(self):
    '''Provides high-level information about the network.

    Returns:
      meta: string with the information about the network (used
        architecture, pooling type, whitening, outputdim).
    '''
    tmpstr = '(meta):\n'
    tmpstr += '\tarchitecture: {}\n'.format(self.meta['architecture'])
    tmpstr += '\tpooling: {}\n'.format(self.meta['pooling'])
    tmpstr += '\twhitening: {}\n'.format(self.meta['whitening'])
    tmpstr += '\toutputdim: {}\n'.format(self.meta['outputdim'])
    return tmpstr


def extract_global_descriptors_from_list(net, images, image_size,
                                         bounding_boxes=None, ms=[1.],
                                         msp=1., print_freq=10):
  """Extracting global descriptors from a list of images.

  Args:
    net: Model object, network for the forward pass.
    images: Absolute image paths as strings.
    image_size: Integer, defines the maximum size of longer image side.
    bounding_boxes: List of (x1,y1,x2,y2) tuples to crop the query images.
    ms: List of float scales.
    msp: Float, multi-scale normalization power parameter.
    print_freq: Printing frequency for debugging.

  Returns:
    descriptors: Global descriptors for the input images.
  """
  # Creating dataset loader.
  data = generic_dataset.ImagesFromList(root='', image_paths=images,
                                        imsize=image_size,
                                        bounding_boxes=bounding_boxes)

  def data_gen():
    return (inst for inst in data)

  loader = tf.data.Dataset.from_generator(data_gen, output_types=(tf.float32))
  loader = loader.batch(1)

  # Extracting vectors.
  descriptors = tf.zeros((0, net.meta['outputdim']))
  for i, input in enumerate(loader):
    if len(ms) == 1 and ms[0] == 1:
      descriptors = tf.concat([descriptors, net(input)], 0)
    else:
      descriptors = tf.concat([descriptors, extract_ms(net, input, ms, msp)], 0)

    if (i + 1) % print_freq == 0 or (i + 1) == len(images):
      global_features_utils.debug_and_log(
        '\r>>>> {}/{} done...'.format((i + 1), len(images)),
        debug_on_the_same_line=True)
  global_features_utils.debug_and_log('', log=False)

  descriptors = tf.transpose(descriptors, perm=[1, 0])
  return descriptors


def extract_ms(net, input, ms, msp):
  """Extracts the global descriptor multi scale.

  Args:
    net: Model object, network for the forward pass.
    input: [B, H, W, C] input tensor in channel-last (BHWC) configuration.
    ms: List of float scales.
    msp: Float, multi-scale normalization power parameter.

  Returns:
    descriptors: Multi-scale global descriptors for the input images.
  """
  descriptors = tf.zeros(net.meta['outputdim'])

  for s in ms:
    if s == 1:
      input_t = input
    else:
      output_shape = s * tf.shape(input)[1:3].numpy()
      input_t = tf.image.resize(input, output_shape,
                                method='bilinear',
                                preserve_aspect_ratio=True)
    descriptors += tf.pow(net(input_t), msp)

  descriptors /= len(ms)
  descriptors = tf.pow(descriptors, 1. / msp)
  descriptors /= tf.norm(descriptors)

  return descriptors
