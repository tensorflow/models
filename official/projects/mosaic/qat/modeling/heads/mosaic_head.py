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

"""Contains definitions of segmentation head of the MOSAIC model."""
from typing import List, Optional

import tensorflow as tf

import tensorflow_model_optimization as tfmot
from official.modeling import tf_utils
from official.projects.mosaic.modeling import mosaic_head
from official.projects.mosaic.qat.modeling.layers import nn_blocks
from official.projects.qat.vision.quantization import configs
from official.projects.qat.vision.quantization import helper


@tf.keras.utils.register_keras_serializable(package='Vision')
class MosaicDecoderHeadQuantized(mosaic_head.MosaicDecoderHead):
  """Creates a quantized MOSAIC decoder in segmentation head.

  Reference:
   [MOSAIC: Mobile Segmentation via decoding Aggregated Information and encoded
   Context](https://arxiv.org/pdf/2112.11623.pdf)
  """

  def __init__(
      self,
      num_classes: int,
      decoder_input_levels: Optional[List[str]] = None,
      decoder_stage_merge_styles: Optional[List[str]] = None,
      decoder_filters: Optional[List[int]] = None,
      decoder_projected_filters: Optional[List[int]] = None,
      encoder_end_level: Optional[int] = 4,
      use_additional_classifier_layer: bool = False,
      classifier_kernel_size: int = 1,
      activation: str = 'relu',
      use_sync_bn: bool = False,
      batchnorm_momentum: float = 0.99,
      batchnorm_epsilon: float = 0.001,
      kernel_initializer: str = 'GlorotUniform',
      kernel_regularizer: Optional[tf.keras.regularizers.Regularizer] = None,
      interpolation: str = 'bilinear',
      bias_regularizer: Optional[tf.keras.regularizers.Regularizer] = None,
      **kwargs):
    """Initializes a MOSAIC segmentation head.

    Args:
      num_classes: An `int` number of mask classification categories. The number
        of classes does not include background class.
      decoder_input_levels: A list of `str` specifying additional
        input levels from the backbone outputs for mask refinement in decoder.
      decoder_stage_merge_styles: A list of `str` specifying the merge style at
        each stage of the decoder, merge styles can be 'concat_merge' or
        'sum_merge'.
      decoder_filters: A list of integers specifying the number of channels used
        at each decoder stage. Note: this only has affects if the decoder merge
          style is 'concat_merge'.
      decoder_projected_filters: A list of integers specifying the number of
        projected channels at the end of each decoder stage.
      encoder_end_level: An optional integer specifying the output level of the
        encoder stage, which is used if the input from the encoder to the
        decoder head is a dictionary.
      use_additional_classifier_layer: A `bool` specifying whether to use an
        additional classifier layer or not. It must be True if the final decoder
        projected filters does not match the `num_classes`.
      classifier_kernel_size: An `int` number to specify the kernel size of the
        classifier layer.
      activation: A `str` that indicates which activation is used, e.g. 'relu',
        'swish', etc.
      use_sync_bn: A `bool` that indicates whether to use synchronized batch
        normalization across different replicas.
      batchnorm_momentum: A `float` of normalization momentum for the moving
        average.
      batchnorm_epsilon: A `float` added to variance to avoid dividing by zero.
      kernel_initializer: Kernel initializer for conv layers. Defaults to
        `glorot_uniform`.
      kernel_regularizer: A `tf.keras.regularizers.Regularizer` object for
        Conv2D. Default is None.
      interpolation: The interpolation method for upsampling. Defaults to
        `bilinear`.
      bias_regularizer: A `tf.keras.regularizers.Regularizer` object for Conv2D.
      **kwargs: Additional keyword arguments to be passed.
    """
    super().__init__(
        num_classes=num_classes,
        decoder_input_levels=decoder_input_levels,
        decoder_stage_merge_styles=decoder_stage_merge_styles,
        decoder_filters=decoder_filters,
        decoder_projected_filters=decoder_projected_filters,
        encoder_end_level=encoder_end_level,
        use_additional_classifier_layer=use_additional_classifier_layer,
        classifier_kernel_size=classifier_kernel_size,
        activation=activation,
        use_sync_bn=use_sync_bn,
        batchnorm_momentum=batchnorm_momentum,
        batchnorm_epsilon=batchnorm_epsilon,
        kernel_initializer=kernel_initializer,
        kernel_regularizer=kernel_regularizer,
        interpolation=interpolation,
        bias_regularizer=bias_regularizer,
        **kwargs)

    # Assuming decoder_input_levels and the following lists are sorted and
    # follow the same order.
    if decoder_input_levels is None:
      decoder_input_levels = ['3', '2']
    if decoder_stage_merge_styles is None:
      decoder_stage_merge_styles = ['concat_merge', 'sum_merge']
    if decoder_filters is None:
      decoder_filters = [64, 64]
    if decoder_projected_filters is None:
      decoder_projected_filters = [32, 32]
    self._decoder_input_levels = decoder_input_levels
    self._decoder_stage_merge_styles = decoder_stage_merge_styles
    self._decoder_filters = decoder_filters
    self._decoder_projected_filters = decoder_projected_filters
    if (len(decoder_input_levels) != len(decoder_stage_merge_styles) or
        len(decoder_input_levels) != len(decoder_filters) or
        len(decoder_input_levels) != len(decoder_projected_filters)):
      raise ValueError('The number of Decoder inputs and settings must match.')
    self._merge_stages = []
    for (stage_merge_style, decoder_filter,
         decoder_projected_filter) in zip(decoder_stage_merge_styles,
                                          decoder_filters,
                                          decoder_projected_filters):
      if stage_merge_style == 'concat_merge':
        concat_merge_stage = nn_blocks.DecoderConcatMergeBlockQuantized(
            decoder_internal_depth=decoder_filter,
            decoder_projected_depth=decoder_projected_filter,
            output_size=(0, 0),
            use_sync_bn=use_sync_bn,
            batchnorm_momentum=batchnorm_momentum,
            batchnorm_epsilon=batchnorm_epsilon,
            activation=activation,
            kernel_initializer=kernel_initializer,
            kernel_regularizer=kernel_regularizer,
            interpolation=interpolation)
        self._merge_stages.append(concat_merge_stage)
      elif stage_merge_style == 'sum_merge':
        sum_merge_stage = nn_blocks.DecoderSumMergeBlockQuantized(
            decoder_projected_depth=decoder_projected_filter,
            output_size=(0, 0),
            use_sync_bn=use_sync_bn,
            batchnorm_momentum=batchnorm_momentum,
            batchnorm_epsilon=batchnorm_epsilon,
            activation=activation,
            kernel_initializer=kernel_initializer,
            kernel_regularizer=kernel_regularizer,
            interpolation=interpolation)
        self._merge_stages.append(sum_merge_stage)
      else:
        raise ValueError(
            'A stage merge style in MOSAIC Decoder can only be concat_merge '
            'or sum_merge.')

    # Concat merge or sum merge does not require an additional classifer layer
    # unless the final decoder projected filter does not match num_classes.
    final_decoder_projected_filter = decoder_projected_filters[-1]
    if (final_decoder_projected_filter != num_classes and
        not use_additional_classifier_layer):
      raise ValueError('Additional classifier layer is needed if final decoder '
                       'projected filters does not match num_classes!')
    self._use_additional_classifier_layer = use_additional_classifier_layer
    if use_additional_classifier_layer:
      # This additional classification layer uses different kernel
      # initializers and bias compared to earlier blocks.
      self._pixelwise_classifier = helper.Conv2DQuantized(
          name='pixelwise_classifier',
          filters=num_classes,
          kernel_size=classifier_kernel_size,
          padding='same',
          bias_initializer=tf.zeros_initializer(),
          kernel_initializer=tf.keras.initializers.RandomNormal(stddev=0.01),
          kernel_regularizer=kernel_regularizer,
          bias_regularizer=bias_regularizer,
          activation=helper.NoOpActivation(),
          use_bias=True)

      self._activation_fn = tfmot.quantization.keras.QuantizeWrapperV2(
          tf_utils.get_activation(activation, use_keras_layer=True),
          configs.Default8BitActivationQuantizeConfig())

    self._config_dict = {
        'num_classes': num_classes,
        'decoder_input_levels': decoder_input_levels,
        'decoder_stage_merge_styles': decoder_stage_merge_styles,
        'decoder_filters': decoder_filters,
        'decoder_projected_filters': decoder_projected_filters,
        'encoder_end_level': encoder_end_level,
        'use_additional_classifier_layer': use_additional_classifier_layer,
        'classifier_kernel_size': classifier_kernel_size,
        'activation': activation,
        'use_sync_bn': use_sync_bn,
        'batchnorm_momentum': batchnorm_momentum,
        'batchnorm_epsilon': batchnorm_epsilon,
        'kernel_initializer': kernel_initializer,
        'kernel_regularizer': kernel_regularizer,
        'interpolation': interpolation,
        'bias_regularizer': bias_regularizer
    }
