import abc
import collections
import functools
import numpy as np
import tensorflow.compat.v1 as tf
import tensorflow.compat.v2 as tf2

from object_detection.core import box_list
from object_detection.core import box_list_ops
from object_detection.core import keypoint_ops
from object_detection.core import model
from object_detection.core import standard_fields as fields
from object_detection.core import target_assigner
from object_detection.core import box_predictor
from object_detection.utils import shape_utils
from object_detection.utils import ops
from object_detection.core import losses
from object_detection.utils import variables_helper

from object_detection.meta_architectures import detr_lib
from object_detection.matchers import hungarian_matcher
from object_detection.core import post_processing
import time


class DETRKerasFeatureExtractor(object):
  """Keras-based DETR Feature Extractor definition."""

  def __init__(self,
               is_training,
               features_stride,
               batch_norm_trainable=False,
               weight_decay=0.0):
    """Constructor.
    Args:
      is_training: A boolean indicating whether the training version of the
        computation graph should be constructed.
      features_stride: Output stride of first stage feature map.
      batch_norm_trainable: Whether to update batch norm parameters during
        training or not. When training with a relative large batch size
        (e.g. 8), it could be desirable to enable batch norm update.
      weight_decay: float weight decay for feature extractor (default: 0.0).
    """
    self._is_training = is_training
    self.features_stride = features_stride
    self._train_batch_norm = (batch_norm_trainable and is_training)
    self._weight_decay = weight_decay

  @abc.abstractmethod
  def preprocess(self, resized_inputs):
    """Feature-extractor specific preprocessing (minus image resizing)."""
    pass

  @abc.abstractmethod
  def get_proposal_feature_extractor_model(self, name):
    """Get model that extracts features."""
    pass


class DETRMetaArch(model.DetectionModel):
  def __init__(self,
                is_training,
                num_classes,
                image_resizer_fn,
                feature_extractor,
                giou_loss_weight,
                l1_loss_weight,
                cls_loss_weight,
                score_conversion_fn,
                num_queries,
                hidden_dimension,
                add_summaries):
    super(DETRMetaArch, self).__init__(num_classes=num_classes)
    self._image_resizer_fn = image_resizer_fn
    self.num_queries = num_queries
    self.hidden_dimension = hidden_dimension
    self.feature_extractor = feature_extractor
    self.first_stage = (
        feature_extractor.get_proposal_feature_extractor_model())
    self.target_assigner = target_assigner.DETRTargetAssigner()
    self.transformer_args = {"hidden_size": self.hidden_dimension,
                             "attention_dropout": 0.0,
                             "num_heads": 8,
                             "layer_postprocess_dropout": 0.1,
                             "dtype": tf.float32,
                             "num_hidden_layers": 6,
                             "filter_size": 2048,
                             "relu_dropout": 0.0}
    self.transformer = detr_lib.Transformer(**self.transformer_args)
    self.cls = tf.keras.layers.Dense(num_classes + 1)
    self.cls_activation = tf.keras.layers.Softmax()
    self.queries = tf.keras.backend.variable(
        value=tf.random_normal_initializer(stddev=1.0)(
            [self.num_queries, self.hidden_dimension]),
            name="object_queries",
            dtype=tf.float32)
    self._l1_localization_loss = losses.WeightedSmoothL1LocalizationLoss()
    self._giou_localization_loss = losses.WeightedGIOULocalizationLoss()
    self._classification_loss = losses.WeightedSoftmaxClassificationLoss()
    self._giou_loss_weight = giou_loss_weight
    self._l1_loss_weight = l1_loss_weight
    self._cls_loss_weight = cls_loss_weight
    self._post_filter = tf.keras.layers.Conv2D(
        self.hidden_dimension, 1)
    self._score_conversion_fn = score_conversion_fn
    self._box_ffn = tf.keras.Sequential(
        layers=[tf.keras.layers.Dense(
            self.hidden_dimension, activation="relu"),
                tf.keras.layers.Dense(4, activation="sigmoid")])
    self.is_training = is_training

  def predict(self, preprocessed_inputs, true_image_shapes, **side_inputs):
    """ Run inference for the Detection Transformer.

    Args:
      preprocessed_inputs: a [batch, height_out, width_out, channels] float
          tensor representing a batch of images.
      true_image_shapes: int32 tensor of shape [batch, 3] where each row is
          of the form [height, width, channels] indicating the shapes
          of true images in the resized images, as resized images can be
          padded with zeros.
      side_inputs: side inputs which are currently ignored by the model.

    Returns:
      a dictionary with the following entries:
        box_encodings: encoded predicted bounding boxes, with shape
            [batch_size, num_queries, 4]
        class_predictions_with_background: prediction logits pre-activation,
            with shape [batch_size, num_queries, num_classes + 1]
        num_proposals: tensor of shape [batch_size], with each element
            containing the value self.num_queries.
    """
    image_shape = tf.shape(preprocessed_inputs)
    
    x = self.first_stage(preprocessed_inputs, training=self.is_training)
    x = self._post_filter(x)
    x = tf.reshape(x, [x.shape[0], x.shape[1] * x.shape[2], x.shape[3]])
    x = self.transformer([x, tf.repeat(tf.expand_dims(self.queries, 0),
                                       x.shape[0],
                                       axis=0)], training=self.is_training)
    bboxes_encoded, logits = self._box_ffn(x), self.cls(x)

    batches_queries = tf.repeat(tf.expand_dims(self.num_queries,
        0), x.shape[0], axis=0)

    return {
      "box_encodings": bboxes_encoded,
      "class_predictions_with_background": logits,
      "num_proposals": batches_queries,
      "image_shape": image_shape
    }

  def preprocess(self, inputs):
    """Feature-extractor specific preprocessing.

    See base class.

    Args:
      inputs: a [batch, height_in, width_in, channels] float tensor
          representing a batch of images with values between 0 and 255.0.

    Returns:
      preprocessed_inputs: a [batch, height_out, width_out, channels] float
          tensor representing a batch of images.
      true_image_shapes: int32 tensor of shape [batch, 3] where each row is
          of the form [height, width, channels] indicating the shapes
          of true images in the resized images, as resized images can be padded
          with zeros.
    """
    (resized_inputs,
    true_image_shapes) = shape_utils.resize_images_and_return_shapes(
        inputs, self._image_resizer_fn)

    return (self.feature_extractor.preprocess(resized_inputs),
            true_image_shapes)

  def restore_from_objects(self, fine_tune_checkpoint_type='classification'):
    """Returns a map of Trackable objects to load from a foreign checkpoint.

    Returns a dictionary of Tensorflow 2 Trackable objects (e.g. tf.Module
    or Checkpoint). This enables the model to initialize based on weights from
    another task. For example, the feature extractor variables from a
    classification model can be used to bootstrap training of an object
    detector. When loading from an object detection model, the checkpoint model
    should have the same parameters as this detection model with exception of
    the num_classes parameter.

    Note that this function is intended to be used to restore Keras-based
    models when running Tensorflow 2, whereas restore_map (above) is intended
    to be used to restore Slim-based models when running Tensorflow 1.x.

    Args:
      fine_tune_checkpoint_type: whether to restore from a full detection
        checkpoint (with compatible variable names) or to restore from a
        classification checkpoint for initialization prior to training.
        Valid values: `detection`, `classification`. Default 'classification'.

    Returns:
      A dict mapping keys to Trackable objects (tf.Module or Checkpoint).
    """
    if fine_tune_checkpoint_type == 'classification':
      return {
          'feature_extractor':
              self.feature_extractor.classification_backbone
      }
    raise NotImplementedError(
        "Detection checkpoint type: " + fine_tune_checkpoint_type)

  def restore_map(self,
                  fine_tune_checkpoint_type='classification',
                  load_all_detection_checkpoint_vars=False):
    raise NotImplementedError("DETR is not supported in TF 1.x.")
    
  def loss(self, prediction_dict, true_image_shapes, scope=None):
    """Compute scalar loss tensors given prediction tensors.

    Args:
      prediction_dict: a dictionary holding prediction tensors (see the
        documentation for the predict method.
      true_image_shapes: int32 tensor of shape [batch, 3] where each row is
        of the form [height, width, channels] indicating the shapes
        of true images in the resized images, as resized images can be padded
        with zeros.
      scope: Optional scope name.

    Returns:
      a dictionary mapping loss keys ('localization_loss',
        'classification_loss') to scalar tensors representing
        corresponding loss values.
    """
    (groundtruth_boxes, groundtruth_classes_with_background_list,
      groundtruth_weights_list) = self._format_groundtruth_data(
        self._image_batch_shape_2d(prediction_dict['image_shape']))
    return self._loss_box_classifier(
          prediction_dict['box_encodings'],
          prediction_dict['class_predictions_with_background'],
          groundtruth_boxes,
          groundtruth_classes_with_background_list,
          groundtruth_weights_list)

  def _loss_box_classifier(self,
                           box_encodings,
                           class_predictions_with_background,
                           groundtruth_boxes,
                           groundtruth_classes_with_background_list,
                           groundtruth_weights_list):
    """Computes scalar box classifier loss tensors.

    Uses self._target_assigner to obtain regression and classification
    targets for the box classifier, and returns losses. All losses are computed
    independently for each image and then averaged across the batch.

    Args:
      box_encodings: a 3-D tensor with shape [batch_size, num_queries, 4]
        representing predicted (final) refined box encodings.
      class_predictions_with_background: a tensor with shape
        [batch_size, num_queries, num_classes + 1] containing class
        predictions (logits) for each of the boxes.  Note that this tensor
        *includes* background class predictions (at class index 0).
      groundtruth_boxes: a tensor with shape [batch_size, padded_gt_size, 4]
        containing groundtruth boxes in corner format.
      groundtruth_classes_with_background_list: one-hot
        tensors of shape [batch_size, num_queries, num_classes + 1]
        containing the class targets with the 0th index assumed to map to
        the background class.
      groundtruth_weights_list: A list of 1-D tf.float32 tensors of shape
        [num_boxes] containing weights for groundtruth boxes.

    Returns:
      a dictionary mapping loss keys ('localization_loss',
        'classification_loss') to scalar tensors representing
        corresponding loss values.
  """
    decoded_boxes = shape_utils.static_or_dynamic_map_fn(
        ops.center_to_corner_coordinate, box_encodings)
    batch_size = decoded_boxes.shape[0]

    (batch_cls_targets_with_background, batch_cls_weights, batch_reg_targets,
     batch_reg_weights) = self.target_assigner.batch_assign(
          pred_box_batch=decoded_boxes,
          gt_box_batch=groundtruth_boxes,
          pred_class_batch=class_predictions_with_background,
          gt_class_targets_batch=groundtruth_classes_with_background_list,
          gt_weights_batch=groundtruth_weights_list)

    print(batch_cls_targets_with_background)
    print(class_predictions_with_background)

    losses_mask = None
    if self.groundtruth_has_field(fields.InputDataFields.is_annotated):
      losses_mask = tf.stack(self.groundtruth_lists(
          fields.InputDataFields.is_annotated))

    l1_loc_loss = tf.reduce_sum(self._l1_localization_loss(
        box_encodings,
        batch_reg_targets,
        weights=batch_reg_weights,
        losses_mask=losses_mask)) * self._l1_loss_weight / batch_size

    giou_loc_loss = tf.reduce_sum(self._giou_localization_loss(
        tf.reshape(ops.center_to_corner_coordinate(box_encodings),
                   [batch_size, -1, 4]),
        tf.reshape(ops.center_to_corner_coordinate(batch_reg_targets),
                   [batch_size, -1, 4]),
        weights=batch_reg_weights,
        losses_mask=losses_mask)) * self._giou_loss_weight / batch_size

    # Down-weight background class loss by 10 to account for imbalance
    batch_cls_weights = tf.concat([tf.expand_dims(
        batch_cls_weights[:, :, 0] / 10, axis=2),
        batch_cls_weights[:, :, 1:]], axis=-1)

    cls_loss = tf.reduce_sum(self._classification_loss(
        class_predictions_with_background,
        batch_cls_targets_with_background,
        weights=batch_cls_weights,
        losses_mask=losses_mask)) * self._cls_loss_weight / batch_size

    loss_dict = {'Loss/localization_loss': l1_loc_loss + giou_loc_loss,
                 'Loss/classification_loss': cls_loss}

    return loss_dict

  def updates(self):
    raise RuntimeError('This model is intended to be used with model_lib_v2 '
                       'which does not support updates()')

  def regularization_losses(self):
    return []

  def postprocess(self, prediction_dict, true_image_shapes):
    """Convert prediction tensors to final detections.

    This function converts raw predictions tensors to final detection results.
    See base class for output format conventions. 

    Args:
      prediction_dict: a dictionary holding prediction tensors (see the
        documentation for the predict method.
      true_image_shapes: int32 tensor of shape [batch, 3] where each row is
        of the form [height, width, channels] indicating the shapes
        of true images in the resized images, as resized images can be padded
        with zeros.

    Returns:
      detections: a dictionary containing the following fields
        detection_boxes: [batch, num_queries, 4]
        detection_scores: [batch, num_queries]
        detection_multiclass_scores: [batch, num_queries, 2]
        detection_classes: [batch, num_queries]
        num_detections: [batch]
        raw_detection_boxes: [batch, num_queries, 4]
        raw_detection_scores: [batch, num_queries, num_classes + 1]
    """
    detections_dict = self._postprocess_box_classifier(
        prediction_dict['box_encodings'],
        prediction_dict['class_predictions_with_background'],
        true_image_shapes,
        prediction_dict['image_shape'])

    return detections_dict

  def _postprocess_box_classifier(self,
                                  refined_box_encodings,
                                  class_predictions_with_background,
                                  true_image_shapes,
                                  preprocessed_image_shapes):
    """Converts predictions from the box classifier to detections.

    Args:
      box_encodings: a 3-D tensor with shape [batch_size, num_queries, 4]
        representing predicted (final) refined box encodings.
      class_predictions_with_background: a tensor with shape
        [batch_size, num_queries, num_classes + 1] containing class
        predictions (logits) for each of the predictions.  Note that this
        tensor *includes* background class predictions (at class index 0).
      true_image_shapes: a 2-D int32 tensor containing shapes of input 
        image in the batch.
      preprocessed_image_shapes: a 1D int32 tensor containing shapes of
        the preprocessed input image in the batch, including the batch size.

    Returns:
      A dictionary containing:
        `detection_boxes`: [batch, num_queries, 4] in normalized co-ordinates.
        `detection_scores`: [batch, num_queries]
        `detection_multiclass_scores`: [batch, num_queries,
          num_classes_with_background] tensor with class score distribution for
          post-processed detection boxes including background class if any.
        `detection_classes`: [batch, num_queries]
        `num_detections`: [batch]
        `raw_detection_boxes`: [batch, num_queries, 4] tensor with decoded
          detection boxes in normalized coordinates.
        `raw_detection_scores`: [batch, num_queries,
          num_classes_with_background] tensor of multi-class scores for
          raw detection boxes.
    """
    clip_window = self._compute_clip_window(true_image_shapes)

    batch_size = shape_utils.combined_static_and_dynamic_shape(
        refined_box_encodings)[0]
    refined_decoded_boxes_batch = tf.reshape(ops.center_to_corner_coordinate(
        tf.reshape(refined_box_encodings, [-1, 4])),
        [batch_size, self.num_queries, 4])

    refined_decoded_boxes_batch = ops.normalized_to_image_coordinates(
        refined_decoded_boxes_batch, image_shape=preprocessed_image_shapes,
        temp=True)

    normalized_class_predictions_batch = self._score_conversion_fn(
        class_predictions_with_background)
    multiclass_scores = tf.slice(normalized_class_predictions_batch,
        [0, 0, 1], [-1, -1, -1])

    processed_boxes = refined_decoded_boxes_batch
    processed_classes = tf.argmax(normalized_class_predictions_batch,
                                  axis=2)
    processed_scores = tf.math.reduce_max(normalized_class_predictions_batch,
        axis=2)

    non_background_mask = tf.cast(tf.greater_equal(
        processed_classes, 1), tf.float32)

    processed_boxes = refined_decoded_boxes_batch * tf.repeat(
        tf.expand_dims(non_background_mask, axis=2), repeats=4, axis=2)

    processed_boxes = shape_utils.static_or_dynamic_map_fn(
        self._clip_window_prune_boxes, [processed_boxes, clip_window])
    processed_classes = tf.cast(processed_classes, dtype=tf.float32) - 1
    processed_scores = processed_scores * non_background_mask

    detections = {
      fields.DetectionResultFields.detection_boxes:
          processed_boxes,
      fields.DetectionResultFields.detection_scores:
          processed_scores,
      fields.DetectionResultFields.detection_classes:
          processed_classes,
      fields.DetectionResultFields.detection_multiclass_scores:
          multiclass_scores,
      fields.DetectionResultFields.num_detections:
        tf.cast(tf.count_nonzero(processed_scores, axis=-1),
                dtype=tf.float32),
      fields.DetectionResultFields.raw_detection_boxes:
          refined_decoded_boxes_batch,
      fields.DetectionResultFields.raw_detection_scores:
          normalized_class_predictions_batch
    }

    return detections

  def _format_groundtruth_data(self, image_shapes):
    """Helper function for preparing groundtruth data for target assignment.

    Converts groundtruth data into tensors and adds background class to
    the groundtruth labels.

    Args:
      image_shapes: a 2-D int32 tensor of shape [batch_size, 3] containing
        shapes of input image in the batch.

    Returns:
      groundtruth_boxes: A 3D tensor of shape [batch_size, padded_gt_size, 4].
      groundtruth_classes_with_background_list: A list of 2-D one-hot
        (or k-hot) tensors of shape [num_boxes, num_classes+1] containing the
        class targets with the 0th index assumed to map to the background
        class.
      groundtruth_weights_list: A list of shape [batch_size, padded_gt_size]
        with weights for the groundtruth boxes.
    """
    groundtruth_boxes = tf.stack(
        self.groundtruth_lists(fields.BoxListFields.boxes))
    groundtruth_classes_with_background_list = []
    for one_hot_encoding in self.groundtruth_lists(
        fields.BoxListFields.classes):
      groundtruth_classes_with_background_list.append(
          tf.cast(
              tf.pad(one_hot_encoding, [[0, 0], [1, 0]], mode='CONSTANT'),
              dtype=tf.float32))

    if self.groundtruth_has_field(fields.BoxListFields.weights):
      groundtruth_weights_list = self.groundtruth_lists(
          fields.BoxListFields.weights)
    else:
      # Set weights for all batch elements equally to 1.0
      groundtruth_weights_list = []
      for groundtruth_classes in groundtruth_classes_with_background_list:
        num_gt = tf.shape(groundtruth_classes)[0]
        groundtruth_weights = tf.ones(num_gt)
        groundtruth_weights_list.append(groundtruth_weights)

    return (groundtruth_boxes, groundtruth_classes_with_background_list,
            groundtruth_weights_list)

  def _image_batch_shape_2d(self, image_batch_shape_1d):
    """Takes a 1-D image batch shape tensor and converts it to a 2-D tensor.

    Example:
    If 1-D image batch shape tensor is [2, 300, 300, 3]. The corresponding 2-D
    image batch tensor would be [[300, 300, 3], [300, 300, 3]]

    Args:
      image_batch_shape_1d: 1-D tensor of the form [batch_size, height,
        width, channels].

    Returns:
      image_batch_shape_2d: 2-D tensor of shape [batch_size, 3] were each row
        is of the form [height, width, channels].
    """
    return tf.tile(tf.expand_dims(image_batch_shape_1d[1:], 0),
                    [image_batch_shape_1d[0], 1])

  def _compute_clip_window(self, image_shapes):
    """Computes clip window based on image shapes.

    This function assumes that the clip window's left top corner is at (0, 0).

    Args:
      image_shapes: A 2-D int32 tensor of shape [batch_size, 3] containing
      shapes of images in the batch. Each row represents [height, width,
      channels] of an image.

    Returns:
      A 2-D float32 tensor of shape [batch_size, 4] containing the clip window
      for each image in the form [ymin, xmin, ymax, xmax].
    """
    clip_heights = image_shapes[:, 0]
    clip_widths = image_shapes[:, 1]
    clip_window = tf.cast(
        tf.stack([
            tf.zeros_like(clip_heights),
            tf.zeros_like(clip_heights), clip_heights, clip_widths
        ],
                 axis=1),
        dtype=tf.float32)
    return clip_window

  def change_coordinate_frame(self, boxlist_window):
    """Change coordinate frame of the boxlist to be relative to window's frame.

    Given a window of the form [ymin, xmin, ymax, xmax],
    changes bounding box coordinates from boxlist to be relative to this window
    (e.g., the min corner maps to (0,0) and the max corner maps to (1,1)).

    An example use case is data augmentation: where we are given groundtruth
    boxes (boxlist) and would like to randomly crop the image to some
    window (window). In this case we need to change the coordinate frame of
    each groundtruth box to be relative to this new window.

    Args:
      boxlist: A BoxList object holding N boxes.
      window: A rank 1 tensor [4].

    Returns:
      Returns a BoxList object with N boxes.
    """
    boxlist = box_list.BoxList(boxlist_window[0])
    window = boxlist_window[1]
  
    win_height = window[2] - window[0]
    win_width = window[3] - window[1]
    boxlist_new = box_list_ops.scale(box_list.BoxList(
        boxlist.get() - [window[0], window[1], window[0], window[1]]),
                        1.0 / win_height, 1.0 / win_width)
    boxlist_new = box_list_ops._copy_extra_fields(boxlist_new, boxlist)
    return boxlist_new.get()

  def _clip_window_prune_boxes(self, sorted_boxes_and_clip_window,
                             pad_to_max_output_size=True,
                             change_coordinate_frame=True):
    """Prune boxes with zero area.

    Args:
      sorted_boxes_and_clip_window, a list containing:
        sorted_boxes: A BoxList containing k detections.
        clip_window: A float32 tensor of the form [y_min, x_min, y_max, x_max]
          representing the window to clip and normalize boxes to.
      pad_to_max_output_size: flag indicating whether to pad to max output size
        or not.
      change_coordinate_frame: Whether to normalize coordinates after clipping
        relative to clip_window (this can only be set to True if a clip_window is
        provided).

    Returns:
      sorted_boxes: A BoxList containing k detections after pruning.
    """
    sorted_boxes = box_list.BoxList(sorted_boxes_and_clip_window[0])
    clip_window = sorted_boxes_and_clip_window[1]
    sorted_boxes = box_list_ops.clip_to_window(
        sorted_boxes,
        clip_window,
        filter_nonoverlapping=not pad_to_max_output_size)

    if change_coordinate_frame:
      sorted_boxes = box_list_ops.change_coordinate_frame(sorted_boxes,
                                                          clip_window)
    return sorted_boxes.get()