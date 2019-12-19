# Lint as: python2, python3
# Copyright 2018 The TensorFlow Authors All Rights Reserved.
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
r"""Provides DeepLab model definition and helper functions.

DeepLab is a deep learning system for semantic image segmentation with
the following features:

(1) Atrous convolution to explicitly control the resolution at which
feature responses are computed within Deep Convolutional Neural Networks.

(2) Atrous spatial pyramid pooling (ASPP) to robustly segment objects at
multiple scales with filters at multiple sampling rates and effective
fields-of-views.

(3) ASPP module augmented with image-level feature and batch normalization.

(4) A simple yet effective decoder module to recover the object boundaries.

See the following papers for more details:

"Encoder-Decoder with Atrous Separable Convolution for Semantic Image
Segmentation"
Liang-Chieh Chen, Yukun Zhu, George Papandreou, Florian Schroff, Hartwig Adam.
(https://arxiv.org/abs/1802.02611)

"Rethinking Atrous Convolution for Semantic Image Segmentation,"
Liang-Chieh Chen, George Papandreou, Florian Schroff, Hartwig Adam
(https://arxiv.org/abs/1706.05587)

"DeepLab: Semantic Image Segmentation with Deep Convolutional Nets,
Atrous Convolution, and Fully Connected CRFs",
Liang-Chieh Chen*, George Papandreou*, Iasonas Kokkinos, Kevin Murphy,
Alan L Yuille (* equal contribution)
(https://arxiv.org/abs/1606.00915)

"Semantic Image Segmentation with Deep Convolutional Nets and Fully Connected
CRFs"
Liang-Chieh Chen*, George Papandreou*, Iasonas Kokkinos, Kevin Murphy,
Alan L. Yuille (* equal contribution)
(https://arxiv.org/abs/1412.7062)
"""
import tensorflow as tf
from tensorflow.contrib import slim as contrib_slim
from deeplab.core import dense_prediction_cell
from deeplab.core import feature_extractor
from deeplab.core import utils

slim = contrib_slim

LOGITS_SCOPE_NAME = 'logits'
MERGED_LOGITS_SCOPE = 'merged_logits'
IMAGE_POOLING_SCOPE = 'image_pooling'
ASPP_SCOPE = 'aspp'
CONCAT_PROJECTION_SCOPE = 'concat_projection'
DECODER_SCOPE = 'decoder'
META_ARCHITECTURE_SCOPE = 'meta_architecture'

PROB_SUFFIX = '_prob'

_resize_bilinear = utils.resize_bilinear
scale_dimension = utils.scale_dimension
split_separable_conv2d = utils.split_separable_conv2d


def get_extra_layer_scopes(last_layers_contain_logits_only=False):
  """Gets the scopes for extra layers.

  Args:
    last_layers_contain_logits_only: Boolean, True if only consider logits as
    the last layer (i.e., exclude ASPP module, decoder module and so on)

  Returns:
    A list of scopes for extra layers.
  """
  if last_layers_contain_logits_only:
    return [LOGITS_SCOPE_NAME]
  else:
    return [
        LOGITS_SCOPE_NAME,
        IMAGE_POOLING_SCOPE,
        ASPP_SCOPE,
        CONCAT_PROJECTION_SCOPE,
        DECODER_SCOPE,
        META_ARCHITECTURE_SCOPE,
    ]


def predict_labels_multi_scale(images,
                               model_options,
                               eval_scales=(1.0,),
                               add_flipped_images=False):
  """Predicts segmentation labels.

  Args:
    images: A tensor of size [batch, height, width, channels].
    model_options: A ModelOptions instance to configure models.
    eval_scales: The scales to resize images for evaluation.
    add_flipped_images: Add flipped images for evaluation or not.

  Returns:
    A dictionary with keys specifying the output_type (e.g., semantic
      prediction) and values storing Tensors representing predictions (argmax
      over channels). Each prediction has size [batch, height, width].
  """
  outputs_to_predictions = {
      output: []
      for output in model_options.outputs_to_num_classes
  }

  for i, image_scale in enumerate(eval_scales):
    with tf.variable_scope(tf.get_variable_scope(), reuse=True if i else None):
      outputs_to_scales_to_logits = multi_scale_logits(
          images,
          model_options=model_options,
          image_pyramid=[image_scale],
          is_training=False,
          fine_tune_batch_norm=False)

    if add_flipped_images:
      with tf.variable_scope(tf.get_variable_scope(), reuse=True):
        outputs_to_scales_to_logits_reversed = multi_scale_logits(
            tf.reverse_v2(images, [2]),
            model_options=model_options,
            image_pyramid=[image_scale],
            is_training=False,
            fine_tune_batch_norm=False)

    for output in sorted(outputs_to_scales_to_logits):
      scales_to_logits = outputs_to_scales_to_logits[output]
      logits = _resize_bilinear(
          scales_to_logits[MERGED_LOGITS_SCOPE],
          tf.shape(images)[1:3],
          scales_to_logits[MERGED_LOGITS_SCOPE].dtype)
      outputs_to_predictions[output].append(
          tf.expand_dims(tf.nn.softmax(logits), 4))

      if add_flipped_images:
        scales_to_logits_reversed = (
            outputs_to_scales_to_logits_reversed[output])
        logits_reversed = _resize_bilinear(
            tf.reverse_v2(scales_to_logits_reversed[MERGED_LOGITS_SCOPE], [2]),
            tf.shape(images)[1:3],
            scales_to_logits_reversed[MERGED_LOGITS_SCOPE].dtype)
        outputs_to_predictions[output].append(
            tf.expand_dims(tf.nn.softmax(logits_reversed), 4))

  for output in sorted(outputs_to_predictions):
    predictions = outputs_to_predictions[output]
    # Compute average prediction across different scales and flipped images.
    predictions = tf.reduce_mean(tf.concat(predictions, 4), axis=4)
    outputs_to_predictions[output] = tf.argmax(predictions, 3)
    predictions[output + PROB_SUFFIX] = tf.nn.softmax(predictions)

  return outputs_to_predictions


def predict_labels(images, model_options, image_pyramid=None):
  """Predicts segmentation labels.

  Args:
    images: A tensor of size [batch, height, width, channels].
    model_options: A ModelOptions instance to configure models.
    image_pyramid: Input image scales for multi-scale feature extraction.

  Returns:
    A dictionary with keys specifying the output_type (e.g., semantic
      prediction) and values storing Tensors representing predictions (argmax
      over channels). Each prediction has size [batch, height, width].
  """
  outputs_to_scales_to_logits = multi_scale_logits(
      images,
      model_options=model_options,
      image_pyramid=image_pyramid,
      is_training=False,
      fine_tune_batch_norm=False)

  predictions = {}
  for output in sorted(outputs_to_scales_to_logits):
    scales_to_logits = outputs_to_scales_to_logits[output]
    logits = scales_to_logits[MERGED_LOGITS_SCOPE]
    # There are two ways to obtain the final prediction results: (1) bilinear
    # upsampling the logits followed by argmax, or (2) argmax followed by
    # nearest neighbor upsampling. The second option may introduce the "blocking
    # effect" but is computationally efficient.
    if model_options.prediction_with_upsampled_logits:
      logits = _resize_bilinear(logits,
                                tf.shape(images)[1:3],
                                scales_to_logits[MERGED_LOGITS_SCOPE].dtype)
      predictions[output] = tf.argmax(logits, 3)
      predictions[output + PROB_SUFFIX] = tf.nn.softmax(logits)
    else:
      argmax_results = tf.argmax(logits, 3)
      argmax_results = tf.image.resize_nearest_neighbor(
          tf.expand_dims(argmax_results, 3),
          tf.shape(images)[1:3],
          align_corners=True,
          name='resize_prediction')
      predictions[output] = tf.squeeze(argmax_results, 3)
      predictions[output + PROB_SUFFIX] = tf.image.resize_bilinear(
          tf.nn.softmax(logits),
          tf.shape(images)[1:3],
          align_corners=True,
          name='resize_prob')
  return predictions


def multi_scale_logits(images,
                       model_options,
                       image_pyramid,
                       weight_decay=0.0001,
                       is_training=False,
                       fine_tune_batch_norm=False,
                       nas_training_hyper_parameters=None):
  """Gets the logits for multi-scale inputs.

  The returned logits are all downsampled (due to max-pooling layers)
  for both training and evaluation.

  Args:
    images: A tensor of size [batch, height, width, channels].
    model_options: A ModelOptions instance to configure models.
    image_pyramid: Input image scales for multi-scale feature extraction.
    weight_decay: The weight decay for model variables.
    is_training: Is training or not.
    fine_tune_batch_norm: Fine-tune the batch norm parameters or not.
    nas_training_hyper_parameters: A dictionary storing hyper-parameters for
      training nas models. Its keys are:
      - `drop_path_keep_prob`: Probability to keep each path in the cell when
        training.
      - `total_training_steps`: Total training steps to help drop path
        probability calculation.

  Returns:
    outputs_to_scales_to_logits: A map of maps from output_type (e.g.,
      semantic prediction) to a dictionary of multi-scale logits names to
      logits. For each output_type, the dictionary has keys which
      correspond to the scales and values which correspond to the logits.
      For example, if `scales` equals [1.0, 1.5], then the keys would
      include 'merged_logits', 'logits_1.00' and 'logits_1.50'.

  Raises:
    ValueError: If model_options doesn't specify crop_size and its
      add_image_level_feature = True, since add_image_level_feature requires
      crop_size information.
  """
  # Setup default values.
  if not image_pyramid:
    image_pyramid = [1.0]
  crop_height = (
      model_options.crop_size[0]
      if model_options.crop_size else tf.shape(images)[1])
  crop_width = (
      model_options.crop_size[1]
      if model_options.crop_size else tf.shape(images)[2])
  if model_options.image_pooling_crop_size:
    image_pooling_crop_height = model_options.image_pooling_crop_size[0]
    image_pooling_crop_width = model_options.image_pooling_crop_size[1]

  # Compute the height, width for the output logits.
  if model_options.decoder_output_stride:
    logits_output_stride = min(model_options.decoder_output_stride)
  else:
    logits_output_stride = model_options.output_stride

  logits_height = scale_dimension(
      crop_height,
      max(1.0, max(image_pyramid)) / logits_output_stride)
  logits_width = scale_dimension(
      crop_width,
      max(1.0, max(image_pyramid)) / logits_output_stride)

  # Compute the logits for each scale in the image pyramid.
  outputs_to_scales_to_logits = {
      k: {}
      for k in model_options.outputs_to_num_classes
  }

  num_channels = images.get_shape().as_list()[-1]

  for image_scale in image_pyramid:
    if image_scale != 1.0:
      scaled_height = scale_dimension(crop_height, image_scale)
      scaled_width = scale_dimension(crop_width, image_scale)
      scaled_crop_size = [scaled_height, scaled_width]
      scaled_images = _resize_bilinear(images, scaled_crop_size, images.dtype)
      if model_options.crop_size:
        scaled_images.set_shape(
            [None, scaled_height, scaled_width, num_channels])
      # Adjust image_pooling_crop_size accordingly.
      scaled_image_pooling_crop_size = None
      if model_options.image_pooling_crop_size:
        scaled_image_pooling_crop_size = [
            scale_dimension(image_pooling_crop_height, image_scale),
            scale_dimension(image_pooling_crop_width, image_scale)]
    else:
      scaled_crop_size = model_options.crop_size
      scaled_images = images
      scaled_image_pooling_crop_size = model_options.image_pooling_crop_size

    updated_options = model_options._replace(
        crop_size=scaled_crop_size,
        image_pooling_crop_size=scaled_image_pooling_crop_size)
    outputs_to_logits = _get_logits(
        scaled_images,
        updated_options,
        weight_decay=weight_decay,
        reuse=tf.AUTO_REUSE,
        is_training=is_training,
        fine_tune_batch_norm=fine_tune_batch_norm,
        nas_training_hyper_parameters=nas_training_hyper_parameters)

    # Resize the logits to have the same dimension before merging.
    for output in sorted(outputs_to_logits):
      outputs_to_logits[output] = _resize_bilinear(
          outputs_to_logits[output], [logits_height, logits_width],
          outputs_to_logits[output].dtype)

    # Return when only one input scale.
    if len(image_pyramid) == 1:
      for output in sorted(model_options.outputs_to_num_classes):
        outputs_to_scales_to_logits[output][
            MERGED_LOGITS_SCOPE] = outputs_to_logits[output]
      return outputs_to_scales_to_logits

    # Save logits to the output map.
    for output in sorted(model_options.outputs_to_num_classes):
      outputs_to_scales_to_logits[output][
          'logits_%.2f' % image_scale] = outputs_to_logits[output]

  # Merge the logits from all the multi-scale inputs.
  for output in sorted(model_options.outputs_to_num_classes):
    # Concatenate the multi-scale logits for each output type.
    all_logits = [
        tf.expand_dims(logits, axis=4)
        for logits in outputs_to_scales_to_logits[output].values()
    ]
    all_logits = tf.concat(all_logits, 4)
    merge_fn = (
        tf.reduce_max
        if model_options.merge_method == 'max' else tf.reduce_mean)
    outputs_to_scales_to_logits[output][MERGED_LOGITS_SCOPE] = merge_fn(
        all_logits, axis=4)

  return outputs_to_scales_to_logits


def extract_features(images,
                     model_options,
                     weight_decay=0.0001,
                     reuse=None,
                     is_training=False,
                     fine_tune_batch_norm=False,
                     nas_training_hyper_parameters=None):
  """Extracts features by the particular model_variant.

  Args:
    images: A tensor of size [batch, height, width, channels].
    model_options: A ModelOptions instance to configure models.
    weight_decay: The weight decay for model variables.
    reuse: Reuse the model variables or not.
    is_training: Is training or not.
    fine_tune_batch_norm: Fine-tune the batch norm parameters or not.
    nas_training_hyper_parameters: A dictionary storing hyper-parameters for
      training nas models. Its keys are:
      - `drop_path_keep_prob`: Probability to keep each path in the cell when
        training.
      - `total_training_steps`: Total training steps to help drop path
        probability calculation.

  Returns:
    concat_logits: A tensor of size [batch, feature_height, feature_width,
      feature_channels], where feature_height/feature_width are determined by
      the images height/width and output_stride.
    end_points: A dictionary from components of the network to the corresponding
      activation.
  """
  features, end_points = feature_extractor.extract_features(
      images,
      output_stride=model_options.output_stride,
      multi_grid=model_options.multi_grid,
      model_variant=model_options.model_variant,
      depth_multiplier=model_options.depth_multiplier,
      divisible_by=model_options.divisible_by,
      weight_decay=weight_decay,
      reuse=reuse,
      is_training=is_training,
      preprocessed_images_dtype=model_options.preprocessed_images_dtype,
      fine_tune_batch_norm=fine_tune_batch_norm,
      nas_architecture_options=model_options.nas_architecture_options,
      nas_training_hyper_parameters=nas_training_hyper_parameters,
      use_bounded_activation=model_options.use_bounded_activation)

  if not model_options.aspp_with_batch_norm:
    return features, end_points
  else:
    if model_options.dense_prediction_cell_config is not None:
      tf.logging.info('Using dense prediction cell config.')
      dense_prediction_layer = dense_prediction_cell.DensePredictionCell(
          config=model_options.dense_prediction_cell_config,
          hparams={
              'conv_rate_multiplier': 16 // model_options.output_stride,
          })
      concat_logits = dense_prediction_layer.build_cell(
          features,
          output_stride=model_options.output_stride,
          crop_size=model_options.crop_size,
          image_pooling_crop_size=model_options.image_pooling_crop_size,
          weight_decay=weight_decay,
          reuse=reuse,
          is_training=is_training,
          fine_tune_batch_norm=fine_tune_batch_norm)
      return concat_logits, end_points
    else:
      # The following codes employ the DeepLabv3 ASPP module. Note that we
      # could express the ASPP module as one particular dense prediction
      # cell architecture. We do not do so but leave the following codes
      # for backward compatibility.
      batch_norm_params = utils.get_batch_norm_params(
          decay=0.9997,
          epsilon=1e-5,
          scale=True,
          is_training=(is_training and fine_tune_batch_norm),
          sync_batch_norm_method=model_options.sync_batch_norm_method)
      batch_norm = utils.get_batch_norm_fn(
          model_options.sync_batch_norm_method)
      activation_fn = (
          tf.nn.relu6 if model_options.use_bounded_activation else tf.nn.relu)
      with slim.arg_scope(
          [slim.conv2d, slim.separable_conv2d],
          weights_regularizer=slim.l2_regularizer(weight_decay),
          activation_fn=activation_fn,
          normalizer_fn=batch_norm,
          padding='SAME',
          stride=1,
          reuse=reuse):
        with slim.arg_scope([batch_norm], **batch_norm_params):
          depth = model_options.aspp_convs_filters
          branch_logits = []

          if model_options.add_image_level_feature:
            if model_options.crop_size is not None:
              image_pooling_crop_size = model_options.image_pooling_crop_size
              # If image_pooling_crop_size is not specified, use crop_size.
              if image_pooling_crop_size is None:
                image_pooling_crop_size = model_options.crop_size
              pool_height = scale_dimension(
                  image_pooling_crop_size[0],
                  1. / model_options.output_stride)
              pool_width = scale_dimension(
                  image_pooling_crop_size[1],
                  1. / model_options.output_stride)
              image_feature = slim.avg_pool2d(
                  features, [pool_height, pool_width],
                  model_options.image_pooling_stride, padding='VALID')
              resize_height = scale_dimension(
                  model_options.crop_size[0],
                  1. / model_options.output_stride)
              resize_width = scale_dimension(
                  model_options.crop_size[1],
                  1. / model_options.output_stride)
            else:
              # If crop_size is None, we simply do global pooling.
              pool_height = tf.shape(features)[1]
              pool_width = tf.shape(features)[2]
              image_feature = tf.reduce_mean(
                  features, axis=[1, 2], keepdims=True)
              resize_height = pool_height
              resize_width = pool_width
            image_feature_activation_fn = tf.nn.relu
            image_feature_normalizer_fn = batch_norm
            if model_options.aspp_with_squeeze_and_excitation:
              image_feature_activation_fn = tf.nn.sigmoid
              if model_options.image_se_uses_qsigmoid:
                image_feature_activation_fn = utils.q_sigmoid
              image_feature_normalizer_fn = None
            image_feature = slim.conv2d(
                image_feature, depth, 1,
                activation_fn=image_feature_activation_fn,
                normalizer_fn=image_feature_normalizer_fn,
                scope=IMAGE_POOLING_SCOPE)
            image_feature = _resize_bilinear(
                image_feature,
                [resize_height, resize_width],
                image_feature.dtype)
            # Set shape for resize_height/resize_width if they are not Tensor.
            if isinstance(resize_height, tf.Tensor):
              resize_height = None
            if isinstance(resize_width, tf.Tensor):
              resize_width = None
            image_feature.set_shape([None, resize_height, resize_width, depth])
            if not model_options.aspp_with_squeeze_and_excitation:
              branch_logits.append(image_feature)

          # Employ a 1x1 convolution.
          branch_logits.append(slim.conv2d(features, depth, 1,
                                           scope=ASPP_SCOPE + str(0)))

          if model_options.atrous_rates:
            # Employ 3x3 convolutions with different atrous rates.
            for i, rate in enumerate(model_options.atrous_rates, 1):
              scope = ASPP_SCOPE + str(i)
              if model_options.aspp_with_separable_conv:
                aspp_features = split_separable_conv2d(
                    features,
                    filters=depth,
                    rate=rate,
                    weight_decay=weight_decay,
                    scope=scope)
              else:
                aspp_features = slim.conv2d(
                    features, depth, 3, rate=rate, scope=scope)
              branch_logits.append(aspp_features)

          # Merge branch logits.
          concat_logits = tf.concat(branch_logits, 3)
          if model_options.aspp_with_concat_projection:
            concat_logits = slim.conv2d(
                concat_logits, depth, 1, scope=CONCAT_PROJECTION_SCOPE)
            concat_logits = slim.dropout(
                concat_logits,
                keep_prob=0.9,
                is_training=is_training,
                scope=CONCAT_PROJECTION_SCOPE + '_dropout')
          if (model_options.add_image_level_feature and
              model_options.aspp_with_squeeze_and_excitation):
            concat_logits *= image_feature

          return concat_logits, end_points


def _get_logits(images,
                model_options,
                weight_decay=0.0001,
                reuse=None,
                is_training=False,
                fine_tune_batch_norm=False,
                nas_training_hyper_parameters=None):
  """Gets the logits by atrous/image spatial pyramid pooling.

  Args:
    images: A tensor of size [batch, height, width, channels].
    model_options: A ModelOptions instance to configure models.
    weight_decay: The weight decay for model variables.
    reuse: Reuse the model variables or not.
    is_training: Is training or not.
    fine_tune_batch_norm: Fine-tune the batch norm parameters or not.
    nas_training_hyper_parameters: A dictionary storing hyper-parameters for
      training nas models. Its keys are:
      - `drop_path_keep_prob`: Probability to keep each path in the cell when
        training.
      - `total_training_steps`: Total training steps to help drop path
        probability calculation.

  Returns:
    outputs_to_logits: A map from output_type to logits.
  """
  features, end_points = extract_features(
      images,
      model_options,
      weight_decay=weight_decay,
      reuse=reuse,
      is_training=is_training,
      fine_tune_batch_norm=fine_tune_batch_norm,
      nas_training_hyper_parameters=nas_training_hyper_parameters)

  if model_options.decoder_output_stride:
    crop_size = model_options.crop_size
    if crop_size is None:
      crop_size = [tf.shape(images)[1], tf.shape(images)[2]]
    features = refine_by_decoder(
        features,
        end_points,
        crop_size=crop_size,
        decoder_output_stride=model_options.decoder_output_stride,
        decoder_use_separable_conv=model_options.decoder_use_separable_conv,
        decoder_use_sum_merge=model_options.decoder_use_sum_merge,
        decoder_filters=model_options.decoder_filters,
        decoder_output_is_logits=model_options.decoder_output_is_logits,
        model_variant=model_options.model_variant,
        weight_decay=weight_decay,
        reuse=reuse,
        is_training=is_training,
        fine_tune_batch_norm=fine_tune_batch_norm,
        use_bounded_activation=model_options.use_bounded_activation)

  outputs_to_logits = {}
  for output in sorted(model_options.outputs_to_num_classes):
    if model_options.decoder_output_is_logits:
      outputs_to_logits[output] = tf.identity(features,
                                              name=output)
    else:
      outputs_to_logits[output] = get_branch_logits(
          features,
          model_options.outputs_to_num_classes[output],
          model_options.atrous_rates,
          aspp_with_batch_norm=model_options.aspp_with_batch_norm,
          kernel_size=model_options.logits_kernel_size,
          weight_decay=weight_decay,
          reuse=reuse,
          scope_suffix=output)

  return outputs_to_logits


def refine_by_decoder(features,
                      end_points,
                      crop_size=None,
                      decoder_output_stride=None,
                      decoder_use_separable_conv=False,
                      decoder_use_sum_merge=False,
                      decoder_filters=256,
                      decoder_output_is_logits=False,
                      model_variant=None,
                      weight_decay=0.0001,
                      reuse=None,
                      is_training=False,
                      fine_tune_batch_norm=False,
                      use_bounded_activation=False,
                      sync_batch_norm_method='None'):
  """Adds the decoder to obtain sharper segmentation results.

  Args:
    features: A tensor of size [batch, features_height, features_width,
      features_channels].
    end_points: A dictionary from components of the network to the corresponding
      activation.
    crop_size: A tuple [crop_height, crop_width] specifying whole patch crop
      size.
    decoder_output_stride: A list of integers specifying the output stride of
      low-level features used in the decoder module.
    decoder_use_separable_conv: Employ separable convolution for decoder or not.
    decoder_use_sum_merge: Boolean, decoder uses simple sum merge or not.
    decoder_filters: Integer, decoder filter size.
    decoder_output_is_logits: Boolean, using decoder output as logits or not.
    model_variant: Model variant for feature extraction.
    weight_decay: The weight decay for model variables.
    reuse: Reuse the model variables or not.
    is_training: Is training or not.
    fine_tune_batch_norm: Fine-tune the batch norm parameters or not.
    use_bounded_activation: Whether or not to use bounded activations. Bounded
      activations better lend themselves to quantized inference.
    sync_batch_norm_method: String, method used to sync batch norm. Currently
     only support `None` (no sync batch norm) and `tpu` (use tpu code to
     sync batch norm).

  Returns:
    Decoder output with size [batch, decoder_height, decoder_width,
      decoder_channels].

  Raises:
    ValueError: If crop_size is None.
  """
  if crop_size is None:
    raise ValueError('crop_size must be provided when using decoder.')
  batch_norm_params = utils.get_batch_norm_params(
      decay=0.9997,
      epsilon=1e-5,
      scale=True,
      is_training=(is_training and fine_tune_batch_norm),
      sync_batch_norm_method=sync_batch_norm_method)
  batch_norm = utils.get_batch_norm_fn(sync_batch_norm_method)
  decoder_depth = decoder_filters
  projected_filters = 48
  if decoder_use_sum_merge:
    # When using sum merge, the projected filters must be equal to decoder
    # filters.
    projected_filters = decoder_filters
  if decoder_output_is_logits:
    # Overwrite the setting when decoder output is logits.
    activation_fn = None
    normalizer_fn = None
    conv2d_kernel = 1
    # Use original conv instead of separable conv.
    decoder_use_separable_conv = False
  else:
    # Default setting when decoder output is not logits.
    activation_fn = tf.nn.relu6 if use_bounded_activation else tf.nn.relu
    normalizer_fn = batch_norm
    conv2d_kernel = 3
  with slim.arg_scope(
      [slim.conv2d, slim.separable_conv2d],
      weights_regularizer=slim.l2_regularizer(weight_decay),
      activation_fn=activation_fn,
      normalizer_fn=normalizer_fn,
      padding='SAME',
      stride=1,
      reuse=reuse):
    with slim.arg_scope([batch_norm], **batch_norm_params):
      with tf.variable_scope(DECODER_SCOPE, DECODER_SCOPE, [features]):
        decoder_features = features
        decoder_stage = 0
        scope_suffix = ''
        for output_stride in decoder_output_stride:
          feature_list = feature_extractor.networks_to_feature_maps[
              model_variant][
                  feature_extractor.DECODER_END_POINTS][output_stride]
          # If only one decoder stage, we do not change the scope name in
          # order for backward compactibility.
          if decoder_stage:
            scope_suffix = '_{}'.format(decoder_stage)
          for i, name in enumerate(feature_list):
            decoder_features_list = [decoder_features]
            # MobileNet and NAS variants use different naming convention.
            if ('mobilenet' in model_variant or
                model_variant.startswith('mnas') or
                model_variant.startswith('nas')):
              feature_name = name
            else:
              feature_name = '{}/{}'.format(
                  feature_extractor.name_scope[model_variant], name)
            decoder_features_list.append(
                slim.conv2d(
                    end_points[feature_name],
                    projected_filters,
                    1,
                    scope='feature_projection' + str(i) + scope_suffix))
            # Determine the output size.
            decoder_height = scale_dimension(crop_size[0], 1.0 / output_stride)
            decoder_width = scale_dimension(crop_size[1], 1.0 / output_stride)
            # Resize to decoder_height/decoder_width.
            for j, feature in enumerate(decoder_features_list):
              decoder_features_list[j] = _resize_bilinear(
                  feature, [decoder_height, decoder_width], feature.dtype)
              h = (None if isinstance(decoder_height, tf.Tensor)
                   else decoder_height)
              w = (None if isinstance(decoder_width, tf.Tensor)
                   else decoder_width)
              decoder_features_list[j].set_shape([None, h, w, None])
            if decoder_use_sum_merge:
              decoder_features = _decoder_with_sum_merge(
                  decoder_features_list,
                  decoder_depth,
                  conv2d_kernel=conv2d_kernel,
                  decoder_use_separable_conv=decoder_use_separable_conv,
                  weight_decay=weight_decay,
                  scope_suffix=scope_suffix)
            else:
              if not decoder_use_separable_conv:
                scope_suffix = str(i) + scope_suffix
              decoder_features = _decoder_with_concat_merge(
                  decoder_features_list,
                  decoder_depth,
                  decoder_use_separable_conv=decoder_use_separable_conv,
                  weight_decay=weight_decay,
                  scope_suffix=scope_suffix)
          decoder_stage += 1
        return decoder_features


def _decoder_with_sum_merge(decoder_features_list,
                            decoder_depth,
                            conv2d_kernel=3,
                            decoder_use_separable_conv=True,
                            weight_decay=0.0001,
                            scope_suffix=''):
  """Decoder with sum to merge features.

  Args:
    decoder_features_list: A list of decoder features.
    decoder_depth: Integer, the filters used in the convolution.
    conv2d_kernel: Integer, the convolution kernel size.
    decoder_use_separable_conv: Boolean, use separable conv or not.
    weight_decay: Weight decay for the model variables.
    scope_suffix: String, used in the scope suffix.

  Returns:
    decoder features merged with sum.

  Raises:
    RuntimeError: If decoder_features_list have length not equal to 2.
  """
  if len(decoder_features_list) != 2:
    raise RuntimeError('Expect decoder_features has length 2.')
  # Only apply one convolution when decoder use sum merge.
  if decoder_use_separable_conv:
    decoder_features = split_separable_conv2d(
        decoder_features_list[0],
        filters=decoder_depth,
        rate=1,
        weight_decay=weight_decay,
        scope='decoder_split_sep_conv0'+scope_suffix) + decoder_features_list[1]
  else:
    decoder_features = slim.conv2d(
        decoder_features_list[0],
        decoder_depth,
        conv2d_kernel,
        scope='decoder_conv0'+scope_suffix) + decoder_features_list[1]
  return decoder_features


def _decoder_with_concat_merge(decoder_features_list,
                               decoder_depth,
                               decoder_use_separable_conv=True,
                               weight_decay=0.0001,
                               scope_suffix=''):
  """Decoder with concatenation to merge features.

  This decoder method applies two convolutions to smooth the features obtained
  by concatenating the input decoder_features_list.

  This decoder module is proposed in the DeepLabv3+ paper.

  Args:
    decoder_features_list: A list of decoder features.
    decoder_depth: Integer, the filters used in the convolution.
    decoder_use_separable_conv: Boolean, use separable conv or not.
    weight_decay: Weight decay for the model variables.
    scope_suffix: String, used in the scope suffix.

  Returns:
    decoder features merged with concatenation.
  """
  if decoder_use_separable_conv:
    decoder_features = split_separable_conv2d(
        tf.concat(decoder_features_list, 3),
        filters=decoder_depth,
        rate=1,
        weight_decay=weight_decay,
        scope='decoder_conv0'+scope_suffix)
    decoder_features = split_separable_conv2d(
        decoder_features,
        filters=decoder_depth,
        rate=1,
        weight_decay=weight_decay,
        scope='decoder_conv1'+scope_suffix)
  else:
    num_convs = 2
    decoder_features = slim.repeat(
        tf.concat(decoder_features_list, 3),
        num_convs,
        slim.conv2d,
        decoder_depth,
        3,
        scope='decoder_conv'+scope_suffix)
  return decoder_features


def get_branch_logits(features,
                      num_classes,
                      atrous_rates=None,
                      aspp_with_batch_norm=False,
                      kernel_size=1,
                      weight_decay=0.0001,
                      reuse=None,
                      scope_suffix=''):
  """Gets the logits from each model's branch.

  The underlying model is branched out in the last layer when atrous
  spatial pyramid pooling is employed, and all branches are sum-merged
  to form the final logits.

  Args:
    features: A float tensor of shape [batch, height, width, channels].
    num_classes: Number of classes to predict.
    atrous_rates: A list of atrous convolution rates for last layer.
    aspp_with_batch_norm: Use batch normalization layers for ASPP.
    kernel_size: Kernel size for convolution.
    weight_decay: Weight decay for the model variables.
    reuse: Reuse model variables or not.
    scope_suffix: Scope suffix for the model variables.

  Returns:
    Merged logits with shape [batch, height, width, num_classes].

  Raises:
    ValueError: Upon invalid input kernel_size value.
  """
  # When using batch normalization with ASPP, ASPP has been applied before
  # in extract_features, and thus we simply apply 1x1 convolution here.
  if aspp_with_batch_norm or atrous_rates is None:
    if kernel_size != 1:
      raise ValueError('Kernel size must be 1 when atrous_rates is None or '
                       'using aspp_with_batch_norm. Gets %d.' % kernel_size)
    atrous_rates = [1]

  with slim.arg_scope(
      [slim.conv2d],
      weights_regularizer=slim.l2_regularizer(weight_decay),
      weights_initializer=tf.truncated_normal_initializer(stddev=0.01),
      reuse=reuse):
    with tf.variable_scope(LOGITS_SCOPE_NAME, LOGITS_SCOPE_NAME, [features]):
      branch_logits = []
      for i, rate in enumerate(atrous_rates):
        scope = scope_suffix
        if i:
          scope += '_%d' % i

        branch_logits.append(
            slim.conv2d(
                features,
                num_classes,
                kernel_size=kernel_size,
                rate=rate,
                activation_fn=None,
                normalizer_fn=None,
                scope=scope))

      return tf.add_n(branch_logits)
