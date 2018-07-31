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

from object_detection.predictors import convolutional_box_predictor
from object_detection.predictors import mask_rcnn_box_predictor
from object_detection.predictors import rfcn_box_predictor
from object_detection.predictors.mask_rcnn_heads import box_head
from object_detection.predictors.mask_rcnn_heads import class_head
from object_detection.predictors.mask_rcnn_heads import mask_head
from object_detection.protos import box_predictor_pb2


def build(argscope_fn, box_predictor_config, is_training, num_classes):
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
    box_predictor_object = (
        convolutional_box_predictor.ConvolutionalBoxPredictor(
            is_training=is_training,
            num_classes=num_classes,
            conv_hyperparams_fn=conv_hyperparams_fn,
            min_depth=config_box_predictor.min_depth,
            max_depth=config_box_predictor.max_depth,
            num_layers_before_predictor=(
                config_box_predictor.num_layers_before_predictor),
            use_dropout=config_box_predictor.use_dropout,
            dropout_keep_prob=config_box_predictor.dropout_keep_probability,
            kernel_size=config_box_predictor.kernel_size,
            box_code_size=config_box_predictor.box_code_size,
            apply_sigmoid_to_scores=config_box_predictor.
            apply_sigmoid_to_scores,
            class_prediction_bias_init=(
                config_box_predictor.class_prediction_bias_init),
            use_depthwise=config_box_predictor.use_depthwise))
    return box_predictor_object

  if  box_predictor_oneof == 'weight_shared_convolutional_box_predictor':
    config_box_predictor = (
        box_predictor_config.weight_shared_convolutional_box_predictor)
    conv_hyperparams_fn = argscope_fn(config_box_predictor.conv_hyperparams,
                                      is_training)
    apply_batch_norm = config_box_predictor.conv_hyperparams.HasField(
        'batch_norm')
    box_predictor_object = (
        convolutional_box_predictor.WeightSharedConvolutionalBoxPredictor(
            is_training=is_training,
            num_classes=num_classes,
            conv_hyperparams_fn=conv_hyperparams_fn,
            depth=config_box_predictor.depth,
            num_layers_before_predictor=(
                config_box_predictor.num_layers_before_predictor),
            kernel_size=config_box_predictor.kernel_size,
            box_code_size=config_box_predictor.box_code_size,
            class_prediction_bias_init=config_box_predictor.
            class_prediction_bias_init,
            use_dropout=config_box_predictor.use_dropout,
            dropout_keep_prob=config_box_predictor.dropout_keep_probability,
            share_prediction_tower=config_box_predictor.share_prediction_tower,
            apply_batch_norm=apply_batch_norm))
    return box_predictor_object

  if box_predictor_oneof == 'mask_rcnn_box_predictor':
    config_box_predictor = box_predictor_config.mask_rcnn_box_predictor
    fc_hyperparams_fn = argscope_fn(config_box_predictor.fc_hyperparams,
                                    is_training)
    conv_hyperparams_fn = None
    if config_box_predictor.HasField('conv_hyperparams'):
      conv_hyperparams_fn = argscope_fn(
          config_box_predictor.conv_hyperparams, is_training)
    box_prediction_head = box_head.BoxHead(
        is_training=is_training,
        num_classes=num_classes,
        fc_hyperparams_fn=fc_hyperparams_fn,
        use_dropout=config_box_predictor.use_dropout,
        dropout_keep_prob=config_box_predictor.dropout_keep_probability,
        box_code_size=config_box_predictor.box_code_size,
        share_box_across_classes=(
            config_box_predictor.share_box_across_classes))
    class_prediction_head = class_head.ClassHead(
        is_training=is_training,
        num_classes=num_classes,
        fc_hyperparams_fn=fc_hyperparams_fn,
        use_dropout=config_box_predictor.use_dropout,
        dropout_keep_prob=config_box_predictor.dropout_keep_probability)
    third_stage_heads = {}
    if config_box_predictor.predict_instance_masks:
      third_stage_heads[
          mask_rcnn_box_predictor.MASK_PREDICTIONS] = mask_head.MaskHead(
              num_classes=num_classes,
              conv_hyperparams_fn=conv_hyperparams_fn,
              mask_height=config_box_predictor.mask_height,
              mask_width=config_box_predictor.mask_width,
              mask_prediction_num_conv_layers=(
                  config_box_predictor.mask_prediction_num_conv_layers),
              mask_prediction_conv_depth=(
                  config_box_predictor.mask_prediction_conv_depth),
              masks_are_class_agnostic=(
                  config_box_predictor.masks_are_class_agnostic))
    box_predictor_object = mask_rcnn_box_predictor.MaskRCNNBoxPredictor(
        is_training=is_training,
        num_classes=num_classes,
        box_prediction_head=box_prediction_head,
        class_prediction_head=class_prediction_head,
        third_stage_heads=third_stage_heads)
    return box_predictor_object

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
