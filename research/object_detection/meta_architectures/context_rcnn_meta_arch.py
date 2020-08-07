# Copyright 2020 The TensorFlow Authors. All Rights Reserved.
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
"""Context R-CNN meta-architecture definition.

This adds the ability to use attention into contextual features within the
Faster R-CNN object detection framework to improve object detection performance.
See https://arxiv.org/abs/1912.03538 for more information.
"""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import functools

from object_detection.core import standard_fields as fields
from object_detection.meta_architectures import context_rcnn_lib
from object_detection.meta_architectures import context_rcnn_lib_tf2
from object_detection.meta_architectures import faster_rcnn_meta_arch
from object_detection.utils import tf_version


class ContextRCNNMetaArch(faster_rcnn_meta_arch.FasterRCNNMetaArch):
  """Context R-CNN Meta-architecture definition."""

  def __init__(self,
               is_training,
               num_classes,
               image_resizer_fn,
               feature_extractor,
               number_of_stages,
               first_stage_anchor_generator,
               first_stage_target_assigner,
               first_stage_atrous_rate,
               first_stage_box_predictor_arg_scope_fn,
               first_stage_box_predictor_kernel_size,
               first_stage_box_predictor_depth,
               first_stage_minibatch_size,
               first_stage_sampler,
               first_stage_non_max_suppression_fn,
               first_stage_max_proposals,
               first_stage_localization_loss_weight,
               first_stage_objectness_loss_weight,
               crop_and_resize_fn,
               initial_crop_size,
               maxpool_kernel_size,
               maxpool_stride,
               second_stage_target_assigner,
               second_stage_mask_rcnn_box_predictor,
               second_stage_batch_size,
               second_stage_sampler,
               second_stage_non_max_suppression_fn,
               second_stage_score_conversion_fn,
               second_stage_localization_loss_weight,
               second_stage_classification_loss_weight,
               second_stage_classification_loss,
               second_stage_mask_prediction_loss_weight=1.0,
               hard_example_miner=None,
               parallel_iterations=16,
               add_summaries=True,
               clip_anchors_to_image=False,
               use_static_shapes=False,
               resize_masks=True,
               freeze_batchnorm=False,
               return_raw_detections_during_predict=False,
               output_final_box_features=False,
               attention_bottleneck_dimension=None,
               attention_temperature=None):
    """ContextRCNNMetaArch Constructor.

    Args:
      is_training: A boolean indicating whether the training version of the
        computation graph should be constructed.
      num_classes: Number of classes.  Note that num_classes *does not*
        include the background category, so if groundtruth labels take values
        in {0, 1, .., K-1}, num_classes=K (and not K+1, even though the
        assigned classification targets can range from {0,... K}).
      image_resizer_fn: A callable for image resizing.  This callable
        takes a rank-3 image tensor of shape [height, width, channels]
        (corresponding to a single image), an optional rank-3 instance mask
        tensor of shape [num_masks, height, width] and returns a resized rank-3
        image tensor, a resized mask tensor if one was provided in the input. In
        addition this callable must also return a 1-D tensor of the form
        [height, width, channels] containing the size of the true image, as the
        image resizer can perform zero padding. See protos/image_resizer.proto.
      feature_extractor: A FasterRCNNFeatureExtractor object.
      number_of_stages:  An integer values taking values in {1, 2, 3}. If
        1, the function will construct only the Region Proposal Network (RPN)
        part of the model. If 2, the function will perform box refinement and
        other auxiliary predictions all in the second stage. If 3, it will
        extract features from refined boxes and perform the auxiliary
        predictions on the non-maximum suppressed refined boxes.
        If is_training is true and the value of number_of_stages is 3, it is
        reduced to 2 since all the model heads are trained in parallel in second
        stage during training.
      first_stage_anchor_generator: An anchor_generator.AnchorGenerator object
        (note that currently we only support
        grid_anchor_generator.GridAnchorGenerator objects)
      first_stage_target_assigner: Target assigner to use for first stage of
        Faster R-CNN (RPN).
      first_stage_atrous_rate: A single integer indicating the atrous rate for
        the single convolution op which is applied to the `rpn_features_to_crop`
        tensor to obtain a tensor to be used for box prediction. Some feature
        extractors optionally allow for producing feature maps computed at
        denser resolutions.  The atrous rate is used to compensate for the
        denser feature maps by using an effectively larger receptive field.
        (This should typically be set to 1).
      first_stage_box_predictor_arg_scope_fn: Either a
        Keras layer hyperparams object or a function to construct tf-slim
        arg_scope for conv2d, separable_conv2d and fully_connected ops. Used
        for the RPN box predictor. If it is a keras hyperparams object the
        RPN box predictor will be a Keras model. If it is a function to
        construct an arg scope it will be a tf-slim box predictor.
      first_stage_box_predictor_kernel_size: Kernel size to use for the
        convolution op just prior to RPN box predictions.
      first_stage_box_predictor_depth: Output depth for the convolution op
        just prior to RPN box predictions.
      first_stage_minibatch_size: The "batch size" to use for computing the
        objectness and location loss of the region proposal network. This
        "batch size" refers to the number of anchors selected as contributing
        to the loss function for any given image within the image batch and is
        only called "batch_size" due to terminology from the Faster R-CNN paper.
      first_stage_sampler: Sampler to use for first stage loss (RPN loss).
      first_stage_non_max_suppression_fn: batch_multiclass_non_max_suppression
        callable that takes `boxes`, `scores` and optional `clip_window`(with
        all other inputs already set) and returns a dictionary containing
        tensors with keys: `detection_boxes`, `detection_scores`,
        `detection_classes`, `num_detections`. This is used to perform non max
        suppression  on the boxes predicted by the Region Proposal Network
        (RPN).
        See `post_processing.batch_multiclass_non_max_suppression` for the type
        and shape of these tensors.
      first_stage_max_proposals: Maximum number of boxes to retain after
        performing Non-Max Suppression (NMS) on the boxes predicted by the
        Region Proposal Network (RPN).
      first_stage_localization_loss_weight: A float
      first_stage_objectness_loss_weight: A float
      crop_and_resize_fn: A differentiable resampler to use for cropping RPN
        proposal features.
      initial_crop_size: A single integer indicating the output size
        (width and height are set to be the same) of the initial bilinear
        interpolation based cropping during ROI pooling.
      maxpool_kernel_size: A single integer indicating the kernel size of the
        max pool op on the cropped feature map during ROI pooling.
      maxpool_stride: A single integer indicating the stride of the max pool
        op on the cropped feature map during ROI pooling.
      second_stage_target_assigner: Target assigner to use for second stage of
        Faster R-CNN. If the model is configured with multiple prediction heads,
        this target assigner is used to generate targets for all heads (with the
        correct `unmatched_class_label`).
      second_stage_mask_rcnn_box_predictor: Mask R-CNN box predictor to use for
        the second stage.
      second_stage_batch_size: The batch size used for computing the
        classification and refined location loss of the box classifier.  This
        "batch size" refers to the number of proposals selected as contributing
        to the loss function for any given image within the image batch and is
        only called "batch_size" due to terminology from the Faster R-CNN paper.
      second_stage_sampler:  Sampler to use for second stage loss (box
        classifier loss).
      second_stage_non_max_suppression_fn: batch_multiclass_non_max_suppression
        callable that takes `boxes`, `scores`, optional `clip_window` and
        optional (kwarg) `mask` inputs (with all other inputs already set)
        and returns a dictionary containing tensors with keys:
        `detection_boxes`, `detection_scores`, `detection_classes`,
        `num_detections`, and (optionally) `detection_masks`. See
        `post_processing.batch_multiclass_non_max_suppression` for the type and
        shape of these tensors.
      second_stage_score_conversion_fn: Callable elementwise nonlinearity
        (that takes tensors as inputs and returns tensors).  This is usually
        used to convert logits to probabilities.
      second_stage_localization_loss_weight: A float indicating the scale factor
        for second stage localization loss.
      second_stage_classification_loss_weight: A float indicating the scale
        factor for second stage classification loss.
      second_stage_classification_loss: Classification loss used by the second
        stage classifier. Either losses.WeightedSigmoidClassificationLoss or
        losses.WeightedSoftmaxClassificationLoss.
      second_stage_mask_prediction_loss_weight: A float indicating the scale
        factor for second stage mask prediction loss. This is applicable only if
        second stage box predictor is configured to predict masks.
      hard_example_miner:  A losses.HardExampleMiner object (can be None).
      parallel_iterations: (Optional) The number of iterations allowed to run
        in parallel for calls to tf.map_fn.
      add_summaries: boolean (default: True) controlling whether summary ops
        should be added to tensorflow graph.
      clip_anchors_to_image: Normally, anchors generated for a given image size
        are pruned during training if they lie outside the image window. This
        option clips the anchors to be within the image instead of pruning.
      use_static_shapes: If True, uses implementation of ops with static shape
        guarantees.
      resize_masks: Indicates whether the masks presend in the groundtruth
        should be resized in the model with `image_resizer_fn`
      freeze_batchnorm: Whether to freeze batch norm parameters in the first
        stage box predictor during training or not. When training with a small
        batch size (e.g. 1), it is desirable to freeze batch norm update and
        use pretrained batch norm params.
      return_raw_detections_during_predict: Whether to return raw detection
        boxes in the predict() method. These are decoded boxes that have not
        been through postprocessing (i.e. NMS). Default False.
      output_final_box_features: Whether to output final box features. If true,
        it crops the feauture map based on the final box prediction and returns
        in the dict as detection_features.
      attention_bottleneck_dimension: A single integer. The bottleneck feature
        dimension of the attention block.
      attention_temperature: A single float. The attention temperature.

    Raises:
      ValueError: If `second_stage_batch_size` > `first_stage_max_proposals` at
        training time.
      ValueError: If first_stage_anchor_generator is not of type
        grid_anchor_generator.GridAnchorGenerator.
    """
    super(ContextRCNNMetaArch, self).__init__(
        is_training,
        num_classes,
        image_resizer_fn,
        feature_extractor,
        number_of_stages,
        first_stage_anchor_generator,
        first_stage_target_assigner,
        first_stage_atrous_rate,
        first_stage_box_predictor_arg_scope_fn,
        first_stage_box_predictor_kernel_size,
        first_stage_box_predictor_depth,
        first_stage_minibatch_size,
        first_stage_sampler,
        first_stage_non_max_suppression_fn,
        first_stage_max_proposals,
        first_stage_localization_loss_weight,
        first_stage_objectness_loss_weight,
        crop_and_resize_fn,
        initial_crop_size,
        maxpool_kernel_size,
        maxpool_stride,
        second_stage_target_assigner,
        second_stage_mask_rcnn_box_predictor,
        second_stage_batch_size,
        second_stage_sampler,
        second_stage_non_max_suppression_fn,
        second_stage_score_conversion_fn,
        second_stage_localization_loss_weight,
        second_stage_classification_loss_weight,
        second_stage_classification_loss,
        second_stage_mask_prediction_loss_weight=(
            second_stage_mask_prediction_loss_weight),
        hard_example_miner=hard_example_miner,
        parallel_iterations=parallel_iterations,
        add_summaries=add_summaries,
        clip_anchors_to_image=clip_anchors_to_image,
        use_static_shapes=use_static_shapes,
        resize_masks=resize_masks,
        freeze_batchnorm=freeze_batchnorm,
        return_raw_detections_during_predict=(
            return_raw_detections_during_predict),
        output_final_box_features=output_final_box_features)

    if tf_version.is_tf1():
      self._context_feature_extract_fn = functools.partial(
          context_rcnn_lib.compute_box_context_attention,
          bottleneck_dimension=attention_bottleneck_dimension,
          attention_temperature=attention_temperature,
          is_training=is_training)
    else:
      self._context_feature_extract_fn = context_rcnn_lib_tf2.AttentionBlock(
          bottleneck_dimension=attention_bottleneck_dimension,
          attention_temperature=attention_temperature,
          is_training=is_training)

  @staticmethod
  def get_side_inputs(features):
    """Overrides the get_side_inputs function in the base class.

    This function returns context_features and valid_context_size, which will be
    used in the _compute_second_stage_input_feature_maps function.

    Args:
      features: A dictionary of tensors.

    Returns:
      A dictionary of tensors contains context_features and valid_context_size.

    Raises:
      ValueError: If context_features or valid_context_size is not in the
        features.
    """
    if (fields.InputDataFields.context_features not in features or
        fields.InputDataFields.valid_context_size not in features):
      raise ValueError(
          "Please make sure context_features and valid_context_size are in the "
          "features")

    return {
        fields.InputDataFields.context_features:
            features[fields.InputDataFields.context_features],
        fields.InputDataFields.valid_context_size:
            features[fields.InputDataFields.valid_context_size]
    }

  def _compute_second_stage_input_feature_maps(self, features_to_crop,
                                               proposal_boxes_normalized,
                                               image_shape,
                                               context_features,
                                               valid_context_size):
    """Crops to a set of proposals from the feature map for a batch of images.

    This function overrides the one in the FasterRCNNMetaArch. Aside from
    cropping and resizing the feature maps, which is done in the parent class,
    it adds context attention features to the box features.

    Args:
      features_to_crop: A float32 Tensor with shape [batch_size, height, width,
        depth]
      proposal_boxes_normalized: A float32 Tensor with shape [batch_size,
        num_proposals, box_code_size] containing proposal boxes in normalized
        coordinates.
      image_shape: A 1D int32 tensors of size [4] containing the image shape.
      context_features: A float Tensor of shape [batch_size, context_size,
        num_context_features].
      valid_context_size: A int32 Tensor of shape [batch_size].

    Returns:
      A float32 Tensor with shape [K, new_height, new_width, depth].
    """
    del image_shape
    box_features = self._crop_and_resize_fn(
        features_to_crop, proposal_boxes_normalized, None,
        [self._initial_crop_size, self._initial_crop_size])

    attention_features = self._context_feature_extract_fn(
        box_features=box_features,
        context_features=context_features,
        valid_context_size=valid_context_size)

    # Adds box features with attention features.
    box_features += attention_features

    flattened_feature_maps = self._flatten_first_two_dimensions(box_features)

    return self._maxpool_layer(flattened_feature_maps)
