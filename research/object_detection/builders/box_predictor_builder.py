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

"""Function to build box predictor from configuration."""

import collections
from absl import logging
import tensorflow as tf
from object_detection.predictors import convolutional_box_predictor
from object_detection.predictors import convolutional_keras_box_predictor
from object_detection.predictors import mask_rcnn_box_predictor
from object_detection.predictors import rfcn_box_predictor
from object_detection.predictors.heads import box_head
from object_detection.predictors.heads import class_head
from object_detection.predictors.heads import keras_box_head
from object_detection.predictors.heads import keras_class_head
from object_detection.predictors.heads import keras_mask_head
from object_detection.predictors.heads import mask_head
from object_detection.protos import box_predictor_pb2


def build_convolutional_box_predictor(is_training,
                                      num_classes,
                                      conv_hyperparams_fn,
                                      min_depth,
                                      max_depth,
                                      num_layers_before_predictor,
                                      use_dropout,
                                      dropout_keep_prob,
                                      kernel_size,
                                      box_code_size,
                                      apply_sigmoid_to_scores=False,
                                      add_background_class=True,
                                      class_prediction_bias_init=0.0,
                                      use_depthwise=False,
                                      mask_head_config=None):
  """Builds the ConvolutionalBoxPredictor from the arguments.

  Args:
    is_training: Indicates whether the BoxPredictor is in training mode.
    num_classes: number of classes.  Note that num_classes *does not*
      include the background category, so if groundtruth labels take values
      in {0, 1, .., K-1}, num_classes=K (and not K+1, even though the
      assigned classification targets can range from {0,... K}).
    conv_hyperparams_fn: A function to generate tf-slim arg_scope with
      hyperparameters for convolution ops.
    min_depth: Minimum feature depth prior to predicting box encodings
      and class predictions.
    max_depth: Maximum feature depth prior to predicting box encodings
      and class predictions. If max_depth is set to 0, no additional
      feature map will be inserted before location and class predictions.
    num_layers_before_predictor: Number of the additional conv layers before
      the predictor.
    use_dropout: Option to use dropout or not.  Note that a single dropout
      op is applied here prior to both box and class predictions, which stands
      in contrast to the ConvolutionalBoxPredictor below.
    dropout_keep_prob: Keep probability for dropout.
      This is only used if use_dropout is True.
    kernel_size: Size of final convolution kernel.  If the
      spatial resolution of the feature map is smaller than the kernel size,
      then the kernel size is automatically set to be
      min(feature_width, feature_height).
    box_code_size: Size of encoding for each box.
    apply_sigmoid_to_scores: If True, apply the sigmoid on the output
      class_predictions.
    add_background_class: Whether to add an implicit background class.
    class_prediction_bias_init: Constant value to initialize bias of the last
      conv2d layer before class prediction.
    use_depthwise: Whether to use depthwise convolutions for prediction
      steps. Default is False.
    mask_head_config: An optional MaskHead object containing configs for mask
      head construction.

  Returns:
    A ConvolutionalBoxPredictor class.
  """
  box_prediction_head = box_head.ConvolutionalBoxHead(
      is_training=is_training,
      box_code_size=box_code_size,
      kernel_size=kernel_size,
      use_depthwise=use_depthwise)
  class_prediction_head = class_head.ConvolutionalClassHead(
      is_training=is_training,
      num_class_slots=num_classes + 1 if add_background_class else num_classes,
      use_dropout=use_dropout,
      dropout_keep_prob=dropout_keep_prob,
      kernel_size=kernel_size,
      apply_sigmoid_to_scores=apply_sigmoid_to_scores,
      class_prediction_bias_init=class_prediction_bias_init,
      use_depthwise=use_depthwise)
  other_heads = {}
  if mask_head_config is not None:
    if not mask_head_config.masks_are_class_agnostic:
      logging.warning('Note that class specific mask prediction for SSD '
                      'models is memory consuming.')
    other_heads[convolutional_box_predictor.MASK_PREDICTIONS] = (
        mask_head.ConvolutionalMaskHead(
            is_training=is_training,
            num_classes=num_classes,
            use_dropout=use_dropout,
            dropout_keep_prob=dropout_keep_prob,
            kernel_size=kernel_size,
            use_depthwise=use_depthwise,
            mask_height=mask_head_config.mask_height,
            mask_width=mask_head_config.mask_width,
            masks_are_class_agnostic=mask_head_config.masks_are_class_agnostic))
  return convolutional_box_predictor.ConvolutionalBoxPredictor(
      is_training=is_training,
      num_classes=num_classes,
      box_prediction_head=box_prediction_head,
      class_prediction_head=class_prediction_head,
      other_heads=other_heads,
      conv_hyperparams_fn=conv_hyperparams_fn,
      num_layers_before_predictor=num_layers_before_predictor,
      min_depth=min_depth,
      max_depth=max_depth)


def build_convolutional_keras_box_predictor(is_training,
                                            num_classes,
                                            conv_hyperparams,
                                            freeze_batchnorm,
                                            inplace_batchnorm_update,
                                            num_predictions_per_location_list,
                                            min_depth,
                                            max_depth,
                                            num_layers_before_predictor,
                                            use_dropout,
                                            dropout_keep_prob,
                                            kernel_size,
                                            box_code_size,
                                            add_background_class=True,
                                            class_prediction_bias_init=0.0,
                                            use_depthwise=False,
                                            mask_head_config=None,
                                            name='BoxPredictor'):
  """Builds the Keras ConvolutionalBoxPredictor from the arguments.

  Args:
    is_training: Indicates whether the BoxPredictor is in training mode.
    num_classes: number of classes.  Note that num_classes *does not*
      include the background category, so if groundtruth labels take values
      in {0, 1, .., K-1}, num_classes=K (and not K+1, even though the
      assigned classification targets can range from {0,... K}).
    conv_hyperparams: A `hyperparams_builder.KerasLayerHyperparams` object
      containing hyperparameters for convolution ops.
    freeze_batchnorm: Whether to freeze batch norm parameters during
      training or not. When training with a small batch size (e.g. 1), it is
      desirable to freeze batch norm update and use pretrained batch norm
      params.
    inplace_batchnorm_update: Whether to update batch norm moving average
      values inplace. When this is false train op must add a control
      dependency on tf.graphkeys.UPDATE_OPS collection in order to update
      batch norm statistics.
    num_predictions_per_location_list: A list of integers representing the
      number of box predictions to be made per spatial location for each
      feature map.
    min_depth: Minimum feature depth prior to predicting box encodings
      and class predictions.
    max_depth: Maximum feature depth prior to predicting box encodings
      and class predictions. If max_depth is set to 0, no additional
      feature map will be inserted before location and class predictions.
    num_layers_before_predictor: Number of the additional conv layers before
      the predictor.
    use_dropout: Option to use dropout or not.  Note that a single dropout
      op is applied here prior to both box and class predictions, which stands
      in contrast to the ConvolutionalBoxPredictor below.
    dropout_keep_prob: Keep probability for dropout.
      This is only used if use_dropout is True.
    kernel_size: Size of final convolution kernel.  If the
      spatial resolution of the feature map is smaller than the kernel size,
      then the kernel size is automatically set to be
      min(feature_width, feature_height).
    box_code_size: Size of encoding for each box.
    add_background_class: Whether to add an implicit background class.
    class_prediction_bias_init: constant value to initialize bias of the last
      conv2d layer before class prediction.
    use_depthwise: Whether to use depthwise convolutions for prediction
      steps. Default is False.
    mask_head_config: An optional MaskHead object containing configs for mask
      head construction.
    name: A string name scope to assign to the box predictor. If `None`, Keras
      will auto-generate one from the class name.

  Returns:
    A Keras ConvolutionalBoxPredictor class.
  """
  box_prediction_heads = []
  class_prediction_heads = []
  mask_prediction_heads = []
  other_heads = {}
  if mask_head_config is not None:
    other_heads[convolutional_box_predictor.MASK_PREDICTIONS] = \
      mask_prediction_heads

  for stack_index, num_predictions_per_location in enumerate(
      num_predictions_per_location_list):
    box_prediction_heads.append(
        keras_box_head.ConvolutionalBoxHead(
            is_training=is_training,
            box_code_size=box_code_size,
            kernel_size=kernel_size,
            conv_hyperparams=conv_hyperparams,
            freeze_batchnorm=freeze_batchnorm,
            num_predictions_per_location=num_predictions_per_location,
            use_depthwise=use_depthwise,
            name='ConvolutionalBoxHead_%d' % stack_index))
    class_prediction_heads.append(
        keras_class_head.ConvolutionalClassHead(
            is_training=is_training,
            num_class_slots=(
                num_classes + 1 if add_background_class else num_classes),
            use_dropout=use_dropout,
            dropout_keep_prob=dropout_keep_prob,
            kernel_size=kernel_size,
            conv_hyperparams=conv_hyperparams,
            freeze_batchnorm=freeze_batchnorm,
            num_predictions_per_location=num_predictions_per_location,
            class_prediction_bias_init=class_prediction_bias_init,
            use_depthwise=use_depthwise,
            name='ConvolutionalClassHead_%d' % stack_index))
    if mask_head_config is not None:
      if not mask_head_config.masks_are_class_agnostic:
        logging.warning('Note that class specific mask prediction for SSD '
                        'models is memory consuming.')
      mask_prediction_heads.append(
          keras_mask_head.ConvolutionalMaskHead(
              is_training=is_training,
              num_classes=num_classes,
              use_dropout=use_dropout,
              dropout_keep_prob=dropout_keep_prob,
              kernel_size=kernel_size,
              conv_hyperparams=conv_hyperparams,
              freeze_batchnorm=freeze_batchnorm,
              num_predictions_per_location=num_predictions_per_location,
              use_depthwise=use_depthwise,
              mask_height=mask_head_config.mask_height,
              mask_width=mask_head_config.mask_width,
              masks_are_class_agnostic=mask_head_config.
              masks_are_class_agnostic,
              name='ConvolutionalMaskHead_%d' % stack_index))

  return convolutional_keras_box_predictor.ConvolutionalBoxPredictor(
      is_training=is_training,
      num_classes=num_classes,
      box_prediction_heads=box_prediction_heads,
      class_prediction_heads=class_prediction_heads,
      other_heads=other_heads,
      conv_hyperparams=conv_hyperparams,
      num_layers_before_predictor=num_layers_before_predictor,
      min_depth=min_depth,
      max_depth=max_depth,
      freeze_batchnorm=freeze_batchnorm,
      inplace_batchnorm_update=inplace_batchnorm_update,
      name=name)


def build_weight_shared_convolutional_box_predictor(
    is_training,
    num_classes,
    conv_hyperparams_fn,
    depth,
    num_layers_before_predictor,
    box_code_size,
    kernel_size=3,
    add_background_class=True,
    class_prediction_bias_init=0.0,
    use_dropout=False,
    dropout_keep_prob=0.8,
    share_prediction_tower=False,
    apply_batch_norm=True,
    use_depthwise=False,
    mask_head_config=None,
    score_converter_fn=tf.identity,
    box_encodings_clip_range=None):
  """Builds and returns a WeightSharedConvolutionalBoxPredictor class.

  Args:
    is_training: Indicates whether the BoxPredictor is in training mode.
    num_classes: number of classes.  Note that num_classes *does not*
      include the background category, so if groundtruth labels take values
      in {0, 1, .., K-1}, num_classes=K (and not K+1, even though the
      assigned classification targets can range from {0,... K}).
    conv_hyperparams_fn: A function to generate tf-slim arg_scope with
      hyperparameters for convolution ops.
    depth: depth of conv layers.
    num_layers_before_predictor: Number of the additional conv layers before
      the predictor.
    box_code_size: Size of encoding for each box.
    kernel_size: Size of final convolution kernel.
    add_background_class: Whether to add an implicit background class.
    class_prediction_bias_init: constant value to initialize bias of the last
      conv2d layer before class prediction.
    use_dropout: Whether to apply dropout to class prediction head.
    dropout_keep_prob: Probability of keeping activiations.
    share_prediction_tower: Whether to share the multi-layer tower between box
      prediction and class prediction heads.
    apply_batch_norm: Whether to apply batch normalization to conv layers in
      this predictor.
    use_depthwise: Whether to use depthwise separable conv2d instead of conv2d.
    mask_head_config: An optional MaskHead object containing configs for mask
      head construction.
    score_converter_fn: Callable score converter to perform elementwise op on
      class scores.
    box_encodings_clip_range: Min and max values for clipping the box_encodings.

  Returns:
    A WeightSharedConvolutionalBoxPredictor class.
  """
  box_prediction_head = box_head.WeightSharedConvolutionalBoxHead(
      box_code_size=box_code_size,
      kernel_size=kernel_size,
      use_depthwise=use_depthwise,
      box_encodings_clip_range=box_encodings_clip_range)
  class_prediction_head = (
      class_head.WeightSharedConvolutionalClassHead(
          num_class_slots=(
              num_classes + 1 if add_background_class else num_classes),
          kernel_size=kernel_size,
          class_prediction_bias_init=class_prediction_bias_init,
          use_dropout=use_dropout,
          dropout_keep_prob=dropout_keep_prob,
          use_depthwise=use_depthwise,
          score_converter_fn=score_converter_fn))
  other_heads = {}
  if mask_head_config is not None:
    if not mask_head_config.masks_are_class_agnostic:
      logging.warning('Note that class specific mask prediction for SSD '
                      'models is memory consuming.')
    other_heads[convolutional_box_predictor.MASK_PREDICTIONS] = (
        mask_head.WeightSharedConvolutionalMaskHead(
            num_classes=num_classes,
            kernel_size=kernel_size,
            use_dropout=use_dropout,
            dropout_keep_prob=dropout_keep_prob,
            mask_height=mask_head_config.mask_height,
            mask_width=mask_head_config.mask_width,
            masks_are_class_agnostic=mask_head_config.masks_are_class_agnostic))
  return convolutional_box_predictor.WeightSharedConvolutionalBoxPredictor(
      is_training=is_training,
      num_classes=num_classes,
      box_prediction_head=box_prediction_head,
      class_prediction_head=class_prediction_head,
      other_heads=other_heads,
      conv_hyperparams_fn=conv_hyperparams_fn,
      depth=depth,
      num_layers_before_predictor=num_layers_before_predictor,
      kernel_size=kernel_size,
      apply_batch_norm=apply_batch_norm,
      share_prediction_tower=share_prediction_tower,
      use_depthwise=use_depthwise)


def build_mask_rcnn_box_predictor(is_training,
                                  num_classes,
                                  fc_hyperparams_fn,
                                  use_dropout,
                                  dropout_keep_prob,
                                  box_code_size,
                                  add_background_class=True,
                                  share_box_across_classes=False,
                                  predict_instance_masks=False,
                                  conv_hyperparams_fn=None,
                                  mask_height=14,
                                  mask_width=14,
                                  mask_prediction_num_conv_layers=2,
                                  mask_prediction_conv_depth=256,
                                  masks_are_class_agnostic=False,
                                  convolve_then_upsample_masks=False):
  """Builds and returns a MaskRCNNBoxPredictor class.

  Args:
    is_training: Indicates whether the BoxPredictor is in training mode.
    num_classes: number of classes.  Note that num_classes *does not*
      include the background category, so if groundtruth labels take values
      in {0, 1, .., K-1}, num_classes=K (and not K+1, even though the
      assigned classification targets can range from {0,... K}).
    fc_hyperparams_fn: A function to generate tf-slim arg_scope with
      hyperparameters for fully connected ops.
    use_dropout: Option to use dropout or not.  Note that a single dropout
      op is applied here prior to both box and class predictions, which stands
      in contrast to the ConvolutionalBoxPredictor below.
    dropout_keep_prob: Keep probability for dropout.
      This is only used if use_dropout is True.
    box_code_size: Size of encoding for each box.
    add_background_class: Whether to add an implicit background class.
    share_box_across_classes: Whether to share boxes across classes rather
      than use a different box for each class.
    predict_instance_masks: If True, will add a third stage mask prediction
      to the returned class.
    conv_hyperparams_fn: A function to generate tf-slim arg_scope with
      hyperparameters for convolution ops.
    mask_height: Desired output mask height. The default value is 14.
    mask_width: Desired output mask width. The default value is 14.
    mask_prediction_num_conv_layers: Number of convolution layers applied to
      the image_features in mask prediction branch.
    mask_prediction_conv_depth: The depth for the first conv2d_transpose op
      applied to the image_features in the mask prediction branch. If set
      to 0, the depth of the convolution layers will be automatically chosen
      based on the number of object classes and the number of channels in the
      image features.
    masks_are_class_agnostic: Boolean determining if the mask-head is
      class-agnostic or not.
    convolve_then_upsample_masks: Whether to apply convolutions on mask
      features before upsampling using nearest neighbor resizing. Otherwise,
      mask features are resized to [`mask_height`, `mask_width`] using
      bilinear resizing before applying convolutions.

  Returns:
    A MaskRCNNBoxPredictor class.
  """
  box_prediction_head = box_head.MaskRCNNBoxHead(
      is_training=is_training,
      num_classes=num_classes,
      fc_hyperparams_fn=fc_hyperparams_fn,
      use_dropout=use_dropout,
      dropout_keep_prob=dropout_keep_prob,
      box_code_size=box_code_size,
      share_box_across_classes=share_box_across_classes)
  class_prediction_head = class_head.MaskRCNNClassHead(
      is_training=is_training,
      num_class_slots=num_classes + 1 if add_background_class else num_classes,
      fc_hyperparams_fn=fc_hyperparams_fn,
      use_dropout=use_dropout,
      dropout_keep_prob=dropout_keep_prob)
  third_stage_heads = {}
  if predict_instance_masks:
    third_stage_heads[
        mask_rcnn_box_predictor.
        MASK_PREDICTIONS] = mask_head.MaskRCNNMaskHead(
            num_classes=num_classes,
            conv_hyperparams_fn=conv_hyperparams_fn,
            mask_height=mask_height,
            mask_width=mask_width,
            mask_prediction_num_conv_layers=mask_prediction_num_conv_layers,
            mask_prediction_conv_depth=mask_prediction_conv_depth,
            masks_are_class_agnostic=masks_are_class_agnostic,
            convolve_then_upsample=convolve_then_upsample_masks)
  return mask_rcnn_box_predictor.MaskRCNNBoxPredictor(
      is_training=is_training,
      num_classes=num_classes,
      box_prediction_head=box_prediction_head,
      class_prediction_head=class_prediction_head,
      third_stage_heads=third_stage_heads)


def build_score_converter(score_converter_config, is_training):
  """Builds score converter based on the config.

  Builds one of [tf.identity, tf.sigmoid] score converters based on the config
  and whether the BoxPredictor is for training or inference.

  Args:
    score_converter_config:
      box_predictor_pb2.WeightSharedConvolutionalBoxPredictor.score_converter.
    is_training: Indicates whether the BoxPredictor is in training mode.

  Returns:
    Callable score converter op.

  Raises:
    ValueError: On unknown score converter.
  """
  if score_converter_config == (
      box_predictor_pb2.WeightSharedConvolutionalBoxPredictor.IDENTITY):
    return tf.identity
  if score_converter_config == (
      box_predictor_pb2.WeightSharedConvolutionalBoxPredictor.SIGMOID):
    return tf.identity if is_training else tf.sigmoid
  raise ValueError('Unknown score converter.')


BoxEncodingsClipRange = collections.namedtuple('BoxEncodingsClipRange',
                                               ['min', 'max'])


def build(argscope_fn, box_predictor_config, is_training, num_classes,
          add_background_class=True):
  """Builds box predictor based on the configuration.

  Builds box predictor based on the configuration. See box_predictor.proto for
  configurable options. Also, see box_predictor.py for more details.

  Args:
    argscope_fn: A function that takes the following inputs:
        * hyperparams_pb2.Hyperparams proto
        * a boolean indicating if the model is in training mode.
      and returns a tf slim argscope for Conv and FC hyperparameters.
    box_predictor_config: box_predictor_pb2.BoxPredictor proto containing
      configuration.
    is_training: Whether the models is in training mode.
    num_classes: Number of classes to predict.
    add_background_class: Whether to add an implicit background class.

  Returns:
    box_predictor: box_predictor.BoxPredictor object.

  Raises:
    ValueError: On unknown box predictor.
  """
  if not isinstance(box_predictor_config, box_predictor_pb2.BoxPredictor):
    raise ValueError('box_predictor_config not of type '
                     'box_predictor_pb2.BoxPredictor.')

  box_predictor_oneof = box_predictor_config.WhichOneof('box_predictor_oneof')

  if  box_predictor_oneof == 'convolutional_box_predictor':
    config_box_predictor = box_predictor_config.convolutional_box_predictor
    conv_hyperparams_fn = argscope_fn(config_box_predictor.conv_hyperparams,
                                      is_training)
    mask_head_config = (
        config_box_predictor.mask_head
        if config_box_predictor.HasField('mask_head') else None)
    return build_convolutional_box_predictor(
        is_training=is_training,
        num_classes=num_classes,
        add_background_class=add_background_class,
        conv_hyperparams_fn=conv_hyperparams_fn,
        use_dropout=config_box_predictor.use_dropout,
        dropout_keep_prob=config_box_predictor.dropout_keep_probability,
        box_code_size=config_box_predictor.box_code_size,
        kernel_size=config_box_predictor.kernel_size,
        num_layers_before_predictor=(
            config_box_predictor.num_layers_before_predictor),
        min_depth=config_box_predictor.min_depth,
        max_depth=config_box_predictor.max_depth,
        apply_sigmoid_to_scores=config_box_predictor.apply_sigmoid_to_scores,
        class_prediction_bias_init=(
            config_box_predictor.class_prediction_bias_init),
        use_depthwise=config_box_predictor.use_depthwise,
        mask_head_config=mask_head_config)

  if  box_predictor_oneof == 'weight_shared_convolutional_box_predictor':
    config_box_predictor = (
        box_predictor_config.weight_shared_convolutional_box_predictor)
    conv_hyperparams_fn = argscope_fn(config_box_predictor.conv_hyperparams,
                                      is_training)
    apply_batch_norm = config_box_predictor.conv_hyperparams.HasField(
        'batch_norm')
    mask_head_config = (
        config_box_predictor.mask_head
        if config_box_predictor.HasField('mask_head') else None)
    # During training phase, logits are used to compute the loss. Only apply
    # sigmoid at inference to make the inference graph TPU friendly.
    score_converter_fn = build_score_converter(
        config_box_predictor.score_converter, is_training)
    # Optionally apply clipping to box encodings, when box_encodings_clip_range
    # is set.
    box_encodings_clip_range = (
        BoxEncodingsClipRange(
            min=config_box_predictor.box_encodings_clip_range.min,
            max=config_box_predictor.box_encodings_clip_range.max)
        if config_box_predictor.HasField('box_encodings_clip_range') else None)

    return build_weight_shared_convolutional_box_predictor(
        is_training=is_training,
        num_classes=num_classes,
        add_background_class=add_background_class,
        conv_hyperparams_fn=conv_hyperparams_fn,
        depth=config_box_predictor.depth,
        num_layers_before_predictor=(
            config_box_predictor.num_layers_before_predictor),
        box_code_size=config_box_predictor.box_code_size,
        kernel_size=config_box_predictor.kernel_size,
        class_prediction_bias_init=(
            config_box_predictor.class_prediction_bias_init),
        use_dropout=config_box_predictor.use_dropout,
        dropout_keep_prob=config_box_predictor.dropout_keep_probability,
        share_prediction_tower=config_box_predictor.share_prediction_tower,
        apply_batch_norm=apply_batch_norm,
        use_depthwise=config_box_predictor.use_depthwise,
        mask_head_config=mask_head_config,
        score_converter_fn=score_converter_fn,
        box_encodings_clip_range=box_encodings_clip_range)

  if box_predictor_oneof == 'mask_rcnn_box_predictor':
    config_box_predictor = box_predictor_config.mask_rcnn_box_predictor
    fc_hyperparams_fn = argscope_fn(config_box_predictor.fc_hyperparams,
                                    is_training)
    conv_hyperparams_fn = None
    if config_box_predictor.HasField('conv_hyperparams'):
      conv_hyperparams_fn = argscope_fn(
          config_box_predictor.conv_hyperparams, is_training)
    return build_mask_rcnn_box_predictor(
        is_training=is_training,
        num_classes=num_classes,
        add_background_class=add_background_class,
        fc_hyperparams_fn=fc_hyperparams_fn,
        use_dropout=config_box_predictor.use_dropout,
        dropout_keep_prob=config_box_predictor.dropout_keep_probability,
        box_code_size=config_box_predictor.box_code_size,
        share_box_across_classes=(
            config_box_predictor.share_box_across_classes),
        predict_instance_masks=config_box_predictor.predict_instance_masks,
        conv_hyperparams_fn=conv_hyperparams_fn,
        mask_height=config_box_predictor.mask_height,
        mask_width=config_box_predictor.mask_width,
        mask_prediction_num_conv_layers=(
            config_box_predictor.mask_prediction_num_conv_layers),
        mask_prediction_conv_depth=(
            config_box_predictor.mask_prediction_conv_depth),
        masks_are_class_agnostic=(
            config_box_predictor.masks_are_class_agnostic),
        convolve_then_upsample_masks=(
            config_box_predictor.convolve_then_upsample_masks))

  if box_predictor_oneof == 'rfcn_box_predictor':
    config_box_predictor = box_predictor_config.rfcn_box_predictor
    conv_hyperparams_fn = argscope_fn(config_box_predictor.conv_hyperparams,
                                      is_training)
    box_predictor_object = rfcn_box_predictor.RfcnBoxPredictor(
        is_training=is_training,
        num_classes=num_classes,
        conv_hyperparams_fn=conv_hyperparams_fn,
        crop_size=[config_box_predictor.crop_height,
                   config_box_predictor.crop_width],
        num_spatial_bins=[config_box_predictor.num_spatial_bins_height,
                          config_box_predictor.num_spatial_bins_width],
        depth=config_box_predictor.depth,
        box_code_size=config_box_predictor.box_code_size)
    return box_predictor_object
  raise ValueError('Unknown box predictor: {}'.format(box_predictor_oneof))


def build_keras(conv_hyperparams_fn, freeze_batchnorm, inplace_batchnorm_update,
                num_predictions_per_location_list, box_predictor_config,
                is_training, num_classes, add_background_class=True):
  """Builds a Keras-based box predictor based on the configuration.

  Builds Keras-based box predictor based on the configuration.
  See box_predictor.proto for configurable options. Also, see box_predictor.py
  for more details.

  Args:
    conv_hyperparams_fn: A function that takes a hyperparams_pb2.Hyperparams
      proto and returns a `hyperparams_builder.KerasLayerHyperparams`
      for Conv or FC hyperparameters.
    freeze_batchnorm: Whether to freeze batch norm parameters during
      training or not. When training with a small batch size (e.g. 1), it is
      desirable to freeze batch norm update and use pretrained batch norm
      params.
    inplace_batchnorm_update: Whether to update batch norm moving average
      values inplace. When this is false train op must add a control
      dependency on tf.graphkeys.UPDATE_OPS collection in order to update
      batch norm statistics.
    num_predictions_per_location_list: A list of integers representing the
      number of box predictions to be made per spatial location for each
      feature map.
    box_predictor_config: box_predictor_pb2.BoxPredictor proto containing
      configuration.
    is_training: Whether the models is in training mode.
    num_classes: Number of classes to predict.
    add_background_class: Whether to add an implicit background class.

  Returns:
    box_predictor: box_predictor.KerasBoxPredictor object.

  Raises:
    ValueError: On unknown box predictor, or one with no Keras box predictor.
  """
  if not isinstance(box_predictor_config, box_predictor_pb2.BoxPredictor):
    raise ValueError('box_predictor_config not of type '
                     'box_predictor_pb2.BoxPredictor.')

  box_predictor_oneof = box_predictor_config.WhichOneof('box_predictor_oneof')

  if box_predictor_oneof == 'convolutional_box_predictor':
    config_box_predictor = box_predictor_config.convolutional_box_predictor
    conv_hyperparams = conv_hyperparams_fn(
        config_box_predictor.conv_hyperparams)

    mask_head_config = (
        config_box_predictor.mask_head
        if config_box_predictor.HasField('mask_head') else None)
    return build_convolutional_keras_box_predictor(
        is_training=is_training,
        num_classes=num_classes,
        add_background_class=add_background_class,
        conv_hyperparams=conv_hyperparams,
        freeze_batchnorm=freeze_batchnorm,
        inplace_batchnorm_update=inplace_batchnorm_update,
        num_predictions_per_location_list=num_predictions_per_location_list,
        use_dropout=config_box_predictor.use_dropout,
        dropout_keep_prob=config_box_predictor.dropout_keep_probability,
        box_code_size=config_box_predictor.box_code_size,
        kernel_size=config_box_predictor.kernel_size,
        num_layers_before_predictor=(
            config_box_predictor.num_layers_before_predictor),
        min_depth=config_box_predictor.min_depth,
        max_depth=config_box_predictor.max_depth,
        class_prediction_bias_init=(
            config_box_predictor.class_prediction_bias_init),
        use_depthwise=config_box_predictor.use_depthwise,
        mask_head_config=mask_head_config)

  raise ValueError(
      'Unknown box predictor for Keras: {}'.format(box_predictor_oneof))
