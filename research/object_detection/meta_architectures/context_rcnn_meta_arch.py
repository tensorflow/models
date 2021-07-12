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

import tensorflow.compat.v1 as tf

from object_detection.core import box_predictor
from object_detection.core import standard_fields as fields
from object_detection.meta_architectures import context_rcnn_lib
from object_detection.meta_architectures import context_rcnn_lib_tf2
from object_detection.meta_architectures import faster_rcnn_meta_arch
from object_detection.protos import faster_rcnn_pb2
from object_detection.utils import ops
from object_detection.utils import tf_version

_UNINITIALIZED_FEATURE_EXTRACTOR = '__uninitialized__'


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
               output_final_box_rpn_features=False,
               attention_bottleneck_dimension=None,
               attention_temperature=None,
               use_self_attention=False,
               use_long_term_attention=True,
               self_attention_in_sequence=False,
               num_attention_heads=1,
               num_attention_layers=1,
               attention_position=(
                   faster_rcnn_pb2.AttentionPosition.POST_BOX_CLASSIFIER)
               ):
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
        it crops the feature map based on the final box prediction and returns
        it in the output dict as detection_features.
      output_final_box_rpn_features: Whether to output rpn box features. If
        true, it crops the rpn feature map based on the final box prediction and
        returns it in the output dict as detection_features.
      attention_bottleneck_dimension: A single integer. The bottleneck feature
        dimension of the attention block.
      attention_temperature: A single float. The attention temperature.
      use_self_attention: Whether to use self-attention within the box features
        in the current frame.
      use_long_term_attention: Whether to use attention into the context
        features.
      self_attention_in_sequence: Whether self attention and long term attention
        are in sequence or parallel.
      num_attention_heads: The number of attention heads to use.
      num_attention_layers: The number of attention layers to use.
      attention_position: Whether attention should occur post rpn or post
      box classifier. Options are specified in the faster rcnn proto,
        default is post box classifier.

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
        output_final_box_features=output_final_box_features,
        output_final_box_rpn_features=output_final_box_rpn_features)

    self._attention_position = attention_position

    if tf_version.is_tf1():
      self._context_feature_extract_fn = functools.partial(
          context_rcnn_lib._compute_box_context_attention,
          bottleneck_dimension=attention_bottleneck_dimension,
          attention_temperature=attention_temperature,
          is_training=is_training,
          max_num_proposals=self.max_num_proposals,
          use_self_attention=use_self_attention,
          use_long_term_attention=use_long_term_attention,
          self_attention_in_sequence=self_attention_in_sequence,
          num_attention_heads=num_attention_heads,
          num_attention_layers=num_attention_layers)
    else:
      if use_self_attention:
        raise NotImplementedError
      if self_attention_in_sequence:
        raise NotImplementedError
      if not use_long_term_attention:
        raise NotImplementedError
      if num_attention_heads > 1:
        raise NotImplementedError
      if num_attention_layers > 1:
        raise NotImplementedError

      self._context_feature_extract_fn = context_rcnn_lib_tf2.AttentionBlock(
          bottleneck_dimension=attention_bottleneck_dimension,
          attention_temperature=attention_temperature,
          is_training=is_training,
          max_num_proposals=self.max_num_proposals)

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
          'Please make sure context_features and valid_context_size are in the '
          'features')

    return {
        fields.InputDataFields.context_features:
            features[fields.InputDataFields.context_features],
        fields.InputDataFields.valid_context_size:
            features[fields.InputDataFields.valid_context_size]
    }

  def _predict_second_stage(self, rpn_box_encodings,
                            rpn_objectness_predictions_with_background,
                            rpn_features_to_crop, anchors, image_shape,
                            true_image_shapes, **side_inputs):
    """Predicts the output tensors from second stage of Faster R-CNN.

    Args:
      rpn_box_encodings: 3-D float tensor of shape
        [batch_size, num_valid_anchors, self._box_coder.code_size] containing
        predicted boxes.
      rpn_objectness_predictions_with_background: 2-D float tensor of shape
        [batch_size, num_valid_anchors, 2] containing class
        predictions (logits) for each of the anchors.  Note that this
        tensor *includes* background class predictions (at class index 0).
      rpn_features_to_crop: A list of 4-D float32 or bfloat16 tensor with shape
        [batch_size, height_i, width_i, depth] representing image features to
        crop using the proposal boxes predicted by the RPN.
      anchors: 2-D float tensor of shape
        [num_anchors, self._box_coder.code_size].
      image_shape: A 1D int32 tensors of size [4] containing the image shape.
      true_image_shapes: int32 tensor of shape [batch, 3] where each row is
        of the form [height, width, channels] indicating the shapes
        of true images in the resized images, as resized images can be padded
        with zeros.
      **side_inputs: additional tensors that are required by the network.

    Returns:
      prediction_dict: a dictionary holding "raw" prediction tensors:
        1) refined_box_encodings: a 3-D float32 tensor with shape
          [total_num_proposals, num_classes, self._box_coder.code_size]
          representing predicted (final) refined box encodings, where
          total_num_proposals=batch_size*self._max_num_proposals. If using a
          shared box across classes the shape will instead be
          [total_num_proposals, 1, self._box_coder.code_size].
        2) class_predictions_with_background: a 3-D float32 tensor with shape
          [total_num_proposals, num_classes + 1] containing class
          predictions (logits) for each of the anchors, where
          total_num_proposals=batch_size*self._max_num_proposals.
          Note that this tensor *includes* background class predictions
          (at class index 0).
        3) num_proposals: An int32 tensor of shape [batch_size] representing the
          number of proposals generated by the RPN.  `num_proposals` allows us
          to keep track of which entries are to be treated as zero paddings and
          which are not since we always pad the number of proposals to be
          `self.max_num_proposals` for each image.
        4) proposal_boxes: A float32 tensor of shape
          [batch_size, self.max_num_proposals, 4] representing
          decoded proposal bounding boxes in absolute coordinates.
        5) proposal_boxes_normalized: A float32 tensor of shape
          [batch_size, self.max_num_proposals, 4] representing decoded proposal
          bounding boxes in normalized coordinates. Can be used to override the
          boxes proposed by the RPN, thus enabling one to extract features and
          get box classification and prediction for externally selected areas
          of the image.
        6) box_classifier_features: a 4-D float32/bfloat16 tensor
          representing the features for each proposal.
        If self._return_raw_detections_during_predict is True, the dictionary
        will also contain:
        7) raw_detection_boxes: a 4-D float32 tensor with shape
          [batch_size, self.max_num_proposals, num_classes, 4] in normalized
          coordinates.
        8) raw_detection_feature_map_indices: a 3-D int32 tensor with shape
          [batch_size, self.max_num_proposals, num_classes].
    """
    proposal_boxes_normalized, num_proposals = self._proposal_postprocess(
        rpn_box_encodings, rpn_objectness_predictions_with_background, anchors,
        image_shape, true_image_shapes)

    prediction_dict = self._box_prediction(rpn_features_to_crop,
                                           proposal_boxes_normalized,
                                           image_shape, true_image_shapes,
                                           num_proposals,
                                           **side_inputs)
    prediction_dict['num_proposals'] = num_proposals
    return prediction_dict

  def _box_prediction(self, rpn_features_to_crop, proposal_boxes_normalized,
                      image_shape, true_image_shapes, num_proposals,
                      **side_inputs):
    """Predicts the output tensors from second stage of Faster R-CNN.

    Args:
      rpn_features_to_crop: A list 4-D float32 or bfloat16 tensor with shape
        [batch_size, height_i, width_i, depth] representing image features to
        crop using the proposal boxes predicted by the RPN.
      proposal_boxes_normalized: A float tensor with shape [batch_size,
        max_num_proposals, 4] representing the (potentially zero padded)
        proposal boxes for all images in the batch.  These boxes are represented
        as normalized coordinates.
      image_shape: A 1D int32 tensors of size [4] containing the image shape.
      true_image_shapes: int32 tensor of shape [batch, 3] where each row is
        of the form [height, width, channels] indicating the shapes
        of true images in the resized images, as resized images can be padded
        with zeros.
      num_proposals: The number of valid box proposals.
      **side_inputs: additional tensors that are required by the network.

    Returns:
      prediction_dict: a dictionary holding "raw" prediction tensors:
        1) refined_box_encodings: a 3-D float32 tensor with shape
          [total_num_proposals, num_classes, self._box_coder.code_size]
          representing predicted (final) refined box encodings, where
          total_num_proposals=batch_size*self._max_num_proposals. If using a
          shared box across classes the shape will instead be
          [total_num_proposals, 1, self._box_coder.code_size].
        2) class_predictions_with_background: a 3-D float32 tensor with shape
          [total_num_proposals, num_classes + 1] containing class
          predictions (logits) for each of the anchors, where
          total_num_proposals=batch_size*self._max_num_proposals.
          Note that this tensor *includes* background class predictions
          (at class index 0).
        3) proposal_boxes: A float32 tensor of shape
          [batch_size, self.max_num_proposals, 4] representing
          decoded proposal bounding boxes in absolute coordinates.
        4) proposal_boxes_normalized: A float32 tensor of shape
          [batch_size, self.max_num_proposals, 4] representing decoded proposal
          bounding boxes in normalized coordinates. Can be used to override the
          boxes proposed by the RPN, thus enabling one to extract features and
          get box classification and prediction for externally selected areas
          of the image.
        5) box_classifier_features: a 4-D float32/bfloat16 tensor
          representing the features for each proposal.
        If self._return_raw_detections_during_predict is True, the dictionary
        will also contain:
        6) raw_detection_boxes: a 4-D float32 tensor with shape
          [batch_size, self.max_num_proposals, num_classes, 4] in normalized
          coordinates.
        7) raw_detection_feature_map_indices: a 3-D int32 tensor with shape
          [batch_size, self.max_num_proposals, num_classes].
        8) final_anchors: a 3-D float tensor of shape [batch_size,
          self.max_num_proposals, 4] containing the reference anchors for raw
          detection boxes in normalized coordinates.
    """
    flattened_proposal_feature_maps = (
        self._compute_second_stage_input_feature_maps(
            rpn_features_to_crop, proposal_boxes_normalized,
            image_shape, num_proposals, **side_inputs))

    box_classifier_features = self._extract_box_classifier_features(
        flattened_proposal_feature_maps, num_proposals, **side_inputs)

    if self._mask_rcnn_box_predictor.is_keras_model:
      box_predictions = self._mask_rcnn_box_predictor(
          [box_classifier_features],
          prediction_stage=2)
    else:
      box_predictions = self._mask_rcnn_box_predictor.predict(
          [box_classifier_features],
          num_predictions_per_location=[1],
          scope=self.second_stage_box_predictor_scope,
          prediction_stage=2)

    refined_box_encodings = tf.squeeze(
        box_predictions[box_predictor.BOX_ENCODINGS],
        axis=1, name='all_refined_box_encodings')
    class_predictions_with_background = tf.squeeze(
        box_predictions[box_predictor.CLASS_PREDICTIONS_WITH_BACKGROUND],
        axis=1, name='all_class_predictions_with_background')

    absolute_proposal_boxes = ops.normalized_to_image_coordinates(
        proposal_boxes_normalized, image_shape, self._parallel_iterations)

    prediction_dict = {
        'refined_box_encodings': tf.cast(refined_box_encodings,
                                         dtype=tf.float32),
        'class_predictions_with_background':
        tf.cast(class_predictions_with_background, dtype=tf.float32),
        'proposal_boxes': absolute_proposal_boxes,
        'box_classifier_features': box_classifier_features,
        'proposal_boxes_normalized': proposal_boxes_normalized,
        'final_anchors': proposal_boxes_normalized
    }

    if self._return_raw_detections_during_predict:
      prediction_dict.update(self._raw_detections_and_feature_map_inds(
          refined_box_encodings, absolute_proposal_boxes, true_image_shapes))

    return prediction_dict

  def _compute_second_stage_input_feature_maps(self, features_to_crop,
                                               proposal_boxes_normalized,
                                               image_shape,
                                               num_proposals,
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
      num_proposals: The number of valid box proposals.
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

    flattened_box_features = self._flatten_first_two_dimensions(box_features)

    flattened_box_features = self._maxpool_layer(flattened_box_features)

    if self._attention_position == (
        faster_rcnn_pb2.AttentionPosition.POST_RPN):
      attention_features = self._context_feature_extract_fn(
          box_features=flattened_box_features,
          num_proposals=num_proposals,
          context_features=context_features,
          valid_context_size=valid_context_size)

      # Adds box features with attention features.
      flattened_box_features += self._flatten_first_two_dimensions(
          attention_features)

    return flattened_box_features

  def _extract_box_classifier_features(
      self, flattened_box_features, num_proposals, context_features,
      valid_context_size,
      attention_position=(
          faster_rcnn_pb2.AttentionPosition.POST_BOX_CLASSIFIER)):
    if self._feature_extractor_for_box_classifier_features == (
        _UNINITIALIZED_FEATURE_EXTRACTOR):
      self._feature_extractor_for_box_classifier_features = (
          self._feature_extractor.get_box_classifier_feature_extractor_model(
              name=self.second_stage_feature_extractor_scope))

    if self._feature_extractor_for_box_classifier_features:
      box_classifier_features = (
          self._feature_extractor_for_box_classifier_features(
              flattened_box_features))
    else:
      box_classifier_features = (
          self._feature_extractor.extract_box_classifier_features(
              flattened_box_features,
              scope=self.second_stage_feature_extractor_scope))

    if self._attention_position == (
        faster_rcnn_pb2.AttentionPosition.POST_BOX_CLASSIFIER):
      attention_features = self._context_feature_extract_fn(
          box_features=box_classifier_features,
          num_proposals=num_proposals,
          context_features=context_features,
          valid_context_size=valid_context_size)

      # Adds box features with attention features.
      box_classifier_features += self._flatten_first_two_dimensions(
          attention_features)

    return box_classifier_features
