# Copyright 2019 The TensorFlow Authors All Rights Reserved.
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

"""Core model definition of YAMNet."""

import csv

import numpy as np
import tensorflow as tf
from tensorflow.keras import Model, layers

import features as features_lib


def _batch_norm(name, params):
  return layers.BatchNormalization(
      name=name,
      center=params.batchnorm_center,
      scale=params.batchnorm_scale,
      epsilon=params.batchnorm_epsilon)


def _conv(name, kernel, stride, filters, params):
  return [
      layers.Conv2D(name='{}/conv'.format(name),
                    filters=filters,
                    kernel_size=kernel,
                    strides=stride,
                    padding=params.conv_padding,
                    use_bias=False,
                    activation=None),
      _batch_norm('{}/conv/bn'.format(name), params),
      layers.ReLU(name='{}/relu'.format(name)),
  ]


def _separable_conv(name, kernel, stride, filters, params):
  return [
      layers.DepthwiseConv2D(
          name='{}/depthwise_conv'.format(name),
          kernel_size=kernel,
          strides=stride,
          depth_multiplier=1,
          padding=params.conv_padding,
          use_bias=False,
          activation=None),
      _batch_norm('{}/depthwise_conv/bn'.format(name), params),
      layers.ReLU(name='{}/depthwise_conv/relu'.format(name)),
      layers.Conv2D(
          name='{}/pointwise_conv'.format(name),
          filters=filters,
          kernel_size=(1, 1),
          strides=1,
          padding=params.conv_padding,
          use_bias=False,
          activation=None),
      _batch_norm('{}/pointwise_conv/bn'.format(name), params),
      layers.ReLU(name='{}/pointwise_conv/relu'.format(name)),
  ]


_YAMNET_LAYER_DEFS = [
    # (layer_function, kernel, stride, num_filters)
    (_conv,           [3, 3], 2,   32),
    (_separable_conv, [3, 3], 1,   64),
    (_separable_conv, [3, 3], 2,  128),
    (_separable_conv, [3, 3], 1,  128),
    (_separable_conv, [3, 3], 2,  256),
    (_separable_conv, [3, 3], 1,  256),
    (_separable_conv, [3, 3], 2,  512),
    (_separable_conv, [3, 3], 1,  512),
    (_separable_conv, [3, 3], 1,  512),
    (_separable_conv, [3, 3], 1,  512),
    (_separable_conv, [3, 3], 1,  512),
    (_separable_conv, [3, 3], 1,  512),
    (_separable_conv, [3, 3], 2, 1024),
    (_separable_conv, [3, 3], 1, 1024),
]


class YAMNetBase(tf.keras.Model):
  """Define the core YAMNet mode in Keras."""

  def __init__(self, params):
    super().__init__()
    self._params = params

    self.reshape = layers.Lambda(lambda x: tf.expand_dims(x, axis=-1))

    self.stack = []
    for (i, (layer_fun, kernel, stride,
             filters)) in enumerate(_YAMNET_LAYER_DEFS):
      new_layers = layer_fun('layer{}'.format(i + 1), kernel, stride, filters,
                             params)
      self.stack.extend(new_layers)
    self.pool = layers.GlobalAveragePooling2D()
    self.logits_from_embedding = layers.Dense(
        units=params.num_classes, use_bias=True)
    self.predictions_from_logits = layers.Activation(
        activation=params.classifier_activation)

  def call(self, features):
    print('features', features.shape)
    net = self.reshape(features)
    print('net', net.shape)

    # The inner 3-axes are the items. Any outer axes are the batch. Flatten the
    # iuter batch axes.
    shape = tf.shape(net)
    batch_shape = shape[:-3]
    num_items = tf.reduce_prod(batch_shape)
    item_shape = shape[-3:]
    flattened_batch_shape = tf.concat([[num_items], item_shape], axis=-1)
    net = tf.reshape(net, flattened_batch_shape)

    print('net', net.shape)

    for layer in self.stack:
      net = layer(net)

    # Unflatten the batch axes back to its shape from earlier.
    def fold_batch(arg):
      item_shape = tf.shape(arg)[1:]
      new_shape = tf.concat([batch_shape, item_shape], axis=-1)
      return tf.reshape(arg, new_shape)

    embeddings = self.pool(net)
    embeddings = fold_batch(embeddings)
    print("embeddings", embeddings.shape)

    logits = self.logits_from_embedding(embeddings)
    print("logits", logits.shape)

    predictions = self.predictions_from_logits(logits)
    print("predictions", predictions.shape)

    return predictions, embeddings


def yamnet(features, params):
  """Define the core YAMNet mode in Keras."""
  model = YAMNetBase.call(params)
  return model(features)


class YAMNetFrames(tf.keras.Model):
  """Defines the YAMNet waveform-to-class-scores model.

  Args:
    params: An instance of `params.Params` containing hyperparameters.
  """

  def __init__(self, params):
    super().__init__()
    self._params = params
    self._yamnet_base = YAMNetBase(params)

  @property
  def layers(self):
    return self._yamnet_base.layers

  def call(self, waveforms):
    """Runs the waveform-to-class-scores model.

    Args:
      waveforms: A tensor containing the input waverform(s) with shape
        `(samples,)` or `(batch, samples)`.

    Returns:
      A tuple of results (predictions, embeddings, log_mel_spectrograms). The
      results will have an outer `batch` axis if the `waveforms` input has
      a `batch` axis.

      predictions: (batch?, num_patches, num_classes) matrix of class scores per
        time frame
      embeddings: (batch?, num_patches, embedding size) matrix of embeddings per
        time frame
      log_mel_spectrogram: (batch?, num_spectrogram_frames, num_mel_bins)
        spectrogram feature matrix
    """
    # Fix shapes
    """if len(waveforms.shape) == 1:

      # A single waveform, add the batch axis.
      waveforms = waveforms[tf.newaxis, :]
      squeeze = True
    else:
      squeeze = False
    """
    squeeze = False
    print()
    print('waveforms', waveforms.shape)
    waveform_padded = features_lib.pad_waveform(waveforms, self._params)
    print('waveform_padded', waveform_padded.shape)
    log_mel_spectrogram, features = features_lib.waveform_to_log_mel_spectrogram_patches(
        waveform_padded, self._params)
    print('log_mel_spectrogram', log_mel_spectrogram.shape)

    predictions, embeddings = self._yamnet_base.call(features)
    """
    if squeeze:
      predictions = tf.squeeze(predictions, axis=0)
      embeddings = tf.squeeze(embeddings, axis=0)
      log_mel_spectrogram = tf.squeeze(log_mel_spectrogram, axis=0)
    """

    return predictions, embeddings, log_mel_spectrogram


def yamnet_frames_model(params):
  """Defines the YAMNet waveform-to-class-scores model.

  Args:
    params: An instance of Params containing hyperparameters.

  Returns:
    A model accepting (num_samples,) waveform input and emitting:
    - predictions: (num_patches, num_classes) matrix of class scores per time frame
    - embeddings: (num_patches, embedding size) matrix of embeddings per time frame
    - log_mel_spectrogram: (num_spectrogram_frames, num_mel_bins) spectrogram feature matrix
  """
  frames_model = YAMNetFrames(params).to_keras()
  return frames_model


def class_names(class_map_csv):
  """Read the class name definition file and return a list of strings."""
  if tf.is_tensor(class_map_csv):
    class_map_csv = class_map_csv.numpy()
  with open(class_map_csv) as csv_file:
    reader = csv.reader(csv_file)
    next(reader)   # Skip header
    return np.array([display_name for (_, _, display_name) in reader])
