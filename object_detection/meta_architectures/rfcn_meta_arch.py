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
"""R-FCN meta-architecture definition.

R-FCN: Dai, Jifeng, et al. "R-FCN: Object Detection via Region-based
Fully Convolutional Networks." arXiv preprint arXiv:1605.06409 (2016).

The R-FCN meta architecture is similar to Faster R-CNN and only differs in the
second stage. Hence this class inherits FasterRCNNMetaArch and overrides only
the `_predict_second_stage` method.

Similar to Faster R-CNN we allow for two modes: first_stage_only=True and
first_stage_only=False.  In the former setting, all of the user facing methods
(e.g., predict, postprocess, loss) can be used as if the model consisted
only of the RPN, returning class agnostic proposals (these can be thought of as
approximate detections with no associated class information).  In the latter
setting, proposals are computed, then passed through a second stage
"box classifier" to yield (multi-class) detections.

Implementations of R-FCN models must define a new FasterRCNNFeatureExtractor and
override three methods: `preprocess`, `_extract_proposal_features` (the first
stage of the model), and `_extract_box_classifier_features` (the second stage of
the model). Optionally, the `restore_fn` method can be overridden.  See tests
for an example.

See notes in the documentation of Faster R-CNN meta-architecture as they all
apply here.
"""
import tensorflow as tf

from object_detection.core import box_predictor
from object_detection.meta_architectures import faster_rcnn_meta_arch
from object_detection.utils import ops


class RFCNMetaArch(faster_rcnn_meta_arch.FasterRCNNMetaArch):
  """R-FCN Meta-architecture definition."""

  def __init__(self,
               is_training,
               num_classes,
               image_resizer_fn,
               feature_extractor,
               first_stage_only,
               first_stage_anchor_generator,
               first_stage_atrous_rate,
               first_stage_box_predictor_arg_scope,
               first_stage_box_predictor_kernel_size,
               first_stage_box_predictor_depth,
               first_stage_minibatch_size,
               first_stage_positive_balance_fraction,
               first_stage_nms_score_threshold,
               first_stage_nms_iou_threshold,
               first_stage_max_proposals,
               first_stage_localization_loss_weight,
               first_stage_objectness_loss_weight,
               second_stage_rfcn_box_predictor,
               second_stage_batch_size,
               second_stage_balance_fraction,
               second_stage_non_max_suppression_fn,
               second_stage_score_conversion_fn,
               second_stage_localization_loss_weight,
               second_stage_classification_loss_weight,
               hard_example_miner,
               parallel_iterations=16):
    """RFCNMetaArch Constructor.

    Args:
      is_training: A boolean indicating whether the training version of the
        computation graph should be constructed.
      num_classes: Number of classes.  Note that num_classes *does not*
        include the background category, so if groundtruth labels take values
        in {0, 1, .., K-1}, num_classes=K (and not K+1, even though the
        assigned classification targets can range from {0,... K}).
      image_resizer_fn: A callable for image resizing.  This callable always
        takes a rank-3 image tensor (corresponding to a single image) and
        returns a rank-3 image tensor, possibly with new spatial dimensions.
        See builders/image_resizer_builder.py.
      feature_extractor: A FasterRCNNFeatureExtractor object.
      first_stage_only:  Whether to construct only the Region Proposal Network
        (RPN) part of the model.
      first_stage_anchor_generator: An anchor_generator.AnchorGenerator object
        (note that currently we only support
        grid_anchor_generator.GridAnchorGenerator objects)
      first_stage_atrous_rate: A single integer indicating the atrous rate for
        the single convolution op which is applied to the `rpn_features_to_crop`
        tensor to obtain a tensor to be used for box prediction. Some feature
        extractors optionally allow for producing feature maps computed at
        denser resolutions.  The atrous rate is used to compensate for the
        denser feature maps by using an effectively larger receptive field.
        (This should typically be set to 1).
      first_stage_box_predictor_arg_scope: Slim arg_scope for conv2d,
        separable_conv2d and fully_connected ops for the RPN box predictor.
      first_stage_box_predictor_kernel_size: Kernel size to use for the
        convolution op just prior to RPN box predictions.
      first_stage_box_predictor_depth: Output depth for the convolution op
        just prior to RPN box predictions.
      first_stage_minibatch_size: The "batch size" to use for computing the
        objectness and location loss of the region proposal network. This
        "batch size" refers to the number of anchors selected as contributing
        to the loss function for any given image within the image batch and is
        only called "batch_size" due to terminology from the Faster R-CNN paper.
      first_stage_positive_balance_fraction: Fraction of positive examples
        per image for the RPN. The recommended value for Faster RCNN is 0.5.
      first_stage_nms_score_threshold: Score threshold for non max suppression
        for the Region Proposal Network (RPN).  This value is expected to be in
        [0, 1] as it is applied directly after a softmax transformation.  The
        recommended value for Faster R-CNN is 0.
      first_stage_nms_iou_threshold: The Intersection Over Union (IOU) threshold
        for performing Non-Max Suppression (NMS) on the boxes predicted by the
        Region Proposal Network (RPN).
      first_stage_max_proposals: Maximum number of boxes to retain after
        performing Non-Max Suppression (NMS) on the boxes predicted by the
        Region Proposal Network (RPN).
      first_stage_localization_loss_weight: A float
      first_stage_objectness_loss_weight: A float
      second_stage_rfcn_box_predictor: RFCN box predictor to use for
        second stage.
      second_stage_batch_size: The batch size used for computing the
        classification and refined location loss of the box classifier.  This
        "batch size" refers to the number of proposals selected as contributing
        to the loss function for any given image within the image batch and is
        only called "batch_size" due to terminology from the Faster R-CNN paper.
      second_stage_balance_fraction: Fraction of positive examples to use
        per image for the box classifier. The recommended value for Faster RCNN
        is 0.25.
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
      second_stage_localization_loss_weight: A float
      second_stage_classification_loss_weight: A float
      hard_example_miner:  A losses.HardExampleMiner object (can be None).
      parallel_iterations: (Optional) The number of iterations allowed to run
        in parallel for calls to tf.map_fn.
    Raises:
      ValueError: If `second_stage_batch_size` > `first_stage_max_proposals`
      ValueError: If first_stage_anchor_generator is not of type
        grid_anchor_generator.GridAnchorGenerator.
    """
    super(RFCNMetaArch, self).__init__(
        is_training,
        num_classes,
        image_resizer_fn,
        feature_extractor,
        first_stage_only,
        first_stage_anchor_generator,
        first_stage_atrous_rate,
        first_stage_box_predictor_arg_scope,
        first_stage_box_predictor_kernel_size,
        first_stage_box_predictor_depth,
        first_stage_minibatch_size,
        first_stage_positive_balance_fraction,
        first_stage_nms_score_threshold,
        first_stage_nms_iou_threshold,
        first_stage_max_proposals,
        first_stage_localization_loss_weight,
        first_stage_objectness_loss_weight,
        None,  # initial_crop_size is not used in R-FCN
        None,  # maxpool_kernel_size is not use in R-FCN
        None,  # maxpool_stride is not use in R-FCN
        None,  # fully_connected_box_predictor is not used in R-FCN.
        second_stage_batch_size,
        second_stage_balance_fraction,
        second_stage_non_max_suppression_fn,
        second_stage_score_conversion_fn,
        second_stage_localization_loss_weight,
        second_stage_classification_loss_weight,
        hard_example_miner,
        parallel_iterations)

    self._rfcn_box_predictor = second_stage_rfcn_box_predictor

  def _predict_second_stage(self, rpn_box_encodings,
                            rpn_objectness_predictions_with_background,
                            rpn_features,
                            anchors,
                            image_shape):
    """Predicts the output tensors from 2nd stage of FasterRCNN.

    Args:
      rpn_box_encodings: 4-D float tensor of shape
        [batch_size, num_valid_anchors, self._box_coder.code_size] containing
        predicted boxes.
      rpn_objectness_predictions_with_background: 2-D float tensor of shape
        [batch_size, num_valid_anchors, 2] containing class
        predictions (logits) for each of the anchors.  Note that this
        tensor *includes* background class predictions (at class index 0).
      rpn_features: A 4-D float32 tensor with shape
        [batch_size, height, width, depth] representing image features from the
        RPN.
      anchors: 2-D float tensor of shape
        [num_anchors, self._box_coder.code_size].
      image_shape: A 1D int32 tensors of size [4] containing the image shape.

    Returns:
      prediction_dict: a dictionary holding "raw" prediction tensors:
        1) refined_box_encodings: a 3-D tensor with shape
          [total_num_proposals, num_classes, 4] representing predicted
          (final) refined box encodings, where
          total_num_proposals=batch_size*self._max_num_proposals
        2) class_predictions_with_background: a 3-D tensor with shape
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
          decoded proposal bounding boxes (in absolute coordinates).
    """
    proposal_boxes_normalized, _, num_proposals = self._postprocess_rpn(
        rpn_box_encodings, rpn_objectness_predictions_with_background,
        anchors, image_shape)

    box_classifier_features = (
        self._feature_extractor.extract_box_classifier_features(
            rpn_features,
            scope=self.second_stage_feature_extractor_scope))

    box_predictions = self._rfcn_box_predictor.predict(
        box_classifier_features,
        num_predictions_per_location=1,
        scope=self.second_stage_box_predictor_scope,
        proposal_boxes=proposal_boxes_normalized)
    refined_box_encodings = tf.squeeze(
        box_predictions[box_predictor.BOX_ENCODINGS], axis=1)
    class_predictions_with_background = tf.squeeze(
        box_predictions[box_predictor.CLASS_PREDICTIONS_WITH_BACKGROUND],
        axis=1)

    absolute_proposal_boxes = ops.normalized_to_image_coordinates(
        proposal_boxes_normalized, image_shape,
        parallel_iterations=self._parallel_iterations)

    prediction_dict = {
        'refined_box_encodings': refined_box_encodings,
        'class_predictions_with_background':
        class_predictions_with_background,
        'num_proposals': num_proposals,
        'proposal_boxes': absolute_proposal_boxes,
    }
    return prediction_dict
