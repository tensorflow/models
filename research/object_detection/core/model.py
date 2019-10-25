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
"""Abstract detection model.

This file defines a generic base class for detection models.  Programs that are
designed to work with arbitrary detection models should only depend on this
class.  We intend for the functions in this class to follow tensor-in/tensor-out
design, thus all functions have tensors or lists/dictionaries holding tensors as
inputs and outputs.

Abstractly, detection models predict output tensors given input images
which can be passed to a loss function at training time or passed to a
postprocessing function at eval time.  The computation graphs at a high level
consequently look as follows:

Training time:
inputs (images tensor) -> preprocess -> predict -> loss -> outputs (loss tensor)

Evaluation time:
inputs (images tensor) -> preprocess -> predict -> postprocess
 -> outputs (boxes tensor, scores tensor, classes tensor, num_detections tensor)

DetectionModels must thus implement four functions (1) preprocess, (2) predict,
(3) postprocess and (4) loss.  DetectionModels should make no assumptions about
the input size or aspect ratio --- they are responsible for doing any
resize/reshaping necessary (see docstring for the preprocess function).
Output classes are always integers in the range [0, num_classes).  Any mapping
of these integers to semantic labels is to be handled outside of this class.

Images are resized in the `preprocess` method. All of `preprocess`, `predict`,
and `postprocess` should be reentrant.

The `preprocess` method runs `image_resizer_fn` that returns resized_images and
`true_image_shapes`. Since `image_resizer_fn` can pad the images with zeros,
true_image_shapes indicate the slices that contain the image without padding.
This is useful for padding images to be a fixed size for batching.

The `postprocess` method uses the true image shapes to clip predictions that lie
outside of images.

By default, DetectionModels produce bounding box detections; However, we support
a handful of auxiliary annotations associated with each bounding box, namely,
instance masks and keypoints.
"""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import abc
import six
import tensorflow as tf

from object_detection.core import standard_fields as fields


# If using a new enough version of TensorFlow, detection models should be a
# tf module or keras model for tracking.
try:
  _BaseClass = tf.Module
except AttributeError:
  _BaseClass = object


class DetectionModel(six.with_metaclass(abc.ABCMeta, _BaseClass)):
  """Abstract base class for detection models.

  Extends tf.Module to guarantee variable tracking.
  """

  def __init__(self, num_classes):
    """Constructor.

    Args:
      num_classes: number of classes.  Note that num_classes *does not* include
      background categories that might be implicitly predicted in various
      implementations.
    """
    self._num_classes = num_classes
    self._groundtruth_lists = {}

  @property
  def num_classes(self):
    return self._num_classes

  def groundtruth_lists(self, field):
    """Access list of groundtruth tensors.

    Args:
      field: a string key, options are
        fields.BoxListFields.{boxes,classes,masks,keypoints} or
        fields.InputDataFields.is_annotated.

    Returns:
      a list of tensors holding groundtruth information (see also
      provide_groundtruth function below), with one entry for each image in the
      batch.
    Raises:
      RuntimeError: if the field has not been provided via provide_groundtruth.
    """
    if field not in self._groundtruth_lists:
      raise RuntimeError('Groundtruth tensor {} has not been provided'.format(
          field))
    return self._groundtruth_lists[field]

  def groundtruth_has_field(self, field):
    """Determines whether the groundtruth includes the given field.

    Args:
      field: a string key, options are
        fields.BoxListFields.{boxes,classes,masks,keypoints} or
        fields.InputDataFields.is_annotated.

    Returns:
      True if the groundtruth includes the given field, False otherwise.
    """
    return field in self._groundtruth_lists

  @abc.abstractmethod
  def preprocess(self, inputs):
    """Input preprocessing.

    To be overridden by implementations.

    This function is responsible for any scaling/shifting of input values that
    is necessary prior to running the detector on an input image.
    It is also responsible for any resizing, padding that might be necessary
    as images are assumed to arrive in arbitrary sizes.  While this function
    could conceivably be part of the predict method (below), it is often
    convenient to keep these separate --- for example, we may want to preprocess
    on one device, place onto a queue, and let another device (e.g., the GPU)
    handle prediction.

    A few important notes about the preprocess function:
    + We assume that this operation does not have any trainable variables nor
    does it affect the groundtruth annotations in any way (thus data
    augmentation operations such as random cropping should be performed
    externally).
    + There is no assumption that the batchsize in this function is the same as
    the batch size in the predict function.  In fact, we recommend calling the
    preprocess function prior to calling any batching operations (which should
    happen outside of the model) and thus assuming that batch sizes are equal
    to 1 in the preprocess function.
    + There is also no explicit assumption that the output resolutions
    must be fixed across inputs --- this is to support "fully convolutional"
    settings in which input images can have different shapes/resolutions.

    Args:
      inputs: a [batch, height_in, width_in, channels] float32 tensor
        representing a batch of images with values between 0 and 255.0.

    Returns:
      preprocessed_inputs: a [batch, height_out, width_out, channels] float32
        tensor representing a batch of images.
      true_image_shapes: int32 tensor of shape [batch, 3] where each row is
        of the form [height, width, channels] indicating the shapes
        of true images in the resized images, as resized images can be padded
        with zeros.
    """
    pass

  @abc.abstractmethod
  def predict(self, preprocessed_inputs, true_image_shapes):
    """Predict prediction tensors from inputs tensor.

    Outputs of this function can be passed to loss or postprocess functions.

    Args:
      preprocessed_inputs: a [batch, height, width, channels] float32 tensor
        representing a batch of images.
      true_image_shapes: int32 tensor of shape [batch, 3] where each row is
        of the form [height, width, channels] indicating the shapes
        of true images in the resized images, as resized images can be padded
        with zeros.

    Returns:
      prediction_dict: a dictionary holding prediction tensors to be
        passed to the Loss or Postprocess functions.
    """
    pass

  @abc.abstractmethod
  def postprocess(self, prediction_dict, true_image_shapes, **params):
    """Convert predicted output tensors to final detections.

    This stage typically performs a few things such as
    * Non-Max Suppression to remove overlapping detection boxes.
    * Score conversion and background class removal.

    Outputs adhere to the following conventions:
    * Classes are integers in [0, num_classes); background classes are removed
      and the first non-background class is mapped to 0. If the model produces
      class-agnostic detections, then no output is produced for classes.
    * Boxes are to be interpreted as being in [y_min, x_min, y_max, x_max]
      format and normalized relative to the image window.
    * `num_detections` is provided for settings where detections are padded to a
      fixed number of boxes.
    * We do not specifically assume any kind of probabilistic interpretation
      of the scores --- the only important thing is their relative ordering.
      Thus implementations of the postprocess function are free to output
      logits, probabilities, calibrated probabilities, or anything else.

    Args:
      prediction_dict: a dictionary holding prediction tensors.
      true_image_shapes: int32 tensor of shape [batch, 3] where each row is
        of the form [height, width, channels] indicating the shapes
        of true images in the resized images, as resized images can be padded
        with zeros.
      **params: Additional keyword arguments for specific implementations of
        DetectionModel.

    Returns:
      detections: a dictionary containing the following fields
        detection_boxes: [batch, max_detections, 4]
        detection_scores: [batch, max_detections]
        detection_classes: [batch, max_detections]
          (If a model is producing class-agnostic detections, this field may be
          missing)
        instance_masks: [batch, max_detections, image_height, image_width]
          (optional)
        keypoints: [batch, max_detections, num_keypoints, 2] (optional)
        num_detections: [batch]

        In addition to the above fields this stage also outputs the following
        raw tensors:

        raw_detection_boxes: [batch, total_detections, 4] tensor containing
          all detection boxes from `prediction_dict` in the format
          [ymin, xmin, ymax, xmax] and normalized co-ordinates.
        raw_detection_scores: [batch, total_detections,
          num_classes_with_background] tensor of class score logits for
          raw detection boxes.
    """
    pass

  @abc.abstractmethod
  def loss(self, prediction_dict, true_image_shapes):
    """Compute scalar loss tensors with respect to provided groundtruth.

    Calling this function requires that groundtruth tensors have been
    provided via the provide_groundtruth function.

    Args:
      prediction_dict: a dictionary holding predicted tensors
      true_image_shapes: int32 tensor of shape [batch, 3] where each row is
        of the form [height, width, channels] indicating the shapes
        of true images in the resized images, as resized images can be padded
        with zeros.

    Returns:
      a dictionary mapping strings (loss names) to scalar tensors representing
        loss values.
    """
    pass

  def provide_groundtruth(self,
                          groundtruth_boxes_list,
                          groundtruth_classes_list,
                          groundtruth_masks_list=None,
                          groundtruth_keypoints_list=None,
                          groundtruth_weights_list=None,
                          groundtruth_confidences_list=None,
                          groundtruth_is_crowd_list=None,
                          is_annotated_list=None):
    """Provide groundtruth tensors.

    Args:
      groundtruth_boxes_list: a list of 2-D tf.float32 tensors of shape
        [num_boxes, 4] containing coordinates of the groundtruth boxes.
          Groundtruth boxes are provided in [y_min, x_min, y_max, x_max]
          format and assumed to be normalized and clipped
          relative to the image window with y_min <= y_max and x_min <= x_max.
      groundtruth_classes_list: a list of 2-D tf.float32 one-hot (or k-hot)
        tensors of shape [num_boxes, num_classes] containing the class targets
        with the 0th index assumed to map to the first non-background class.
      groundtruth_masks_list: a list of 3-D tf.float32 tensors of
        shape [num_boxes, height_in, width_in] containing instance
        masks with values in {0, 1}.  If None, no masks are provided.
        Mask resolution `height_in`x`width_in` must agree with the resolution
        of the input image tensor provided to the `preprocess` function.
      groundtruth_keypoints_list: a list of 3-D tf.float32 tensors of
        shape [num_boxes, num_keypoints, 2] containing keypoints.
        Keypoints are assumed to be provided in normalized coordinates and
        missing keypoints should be encoded as NaN.
      groundtruth_weights_list: A list of 1-D tf.float32 tensors of shape
        [num_boxes] containing weights for groundtruth boxes.
      groundtruth_confidences_list: A list of 2-D tf.float32 tensors of shape
        [num_boxes, num_classes] containing class confidences for groundtruth
        boxes.
      groundtruth_is_crowd_list: A list of 1-D tf.bool tensors of shape
        [num_boxes] containing is_crowd annotations
      is_annotated_list: A list of scalar tf.bool tensors indicating whether
        images have been labeled or not.
    """
    self._groundtruth_lists[fields.BoxListFields.boxes] = groundtruth_boxes_list
    self._groundtruth_lists[
        fields.BoxListFields.classes] = groundtruth_classes_list
    if groundtruth_weights_list:
      self._groundtruth_lists[fields.BoxListFields.
                              weights] = groundtruth_weights_list
    if groundtruth_confidences_list:
      self._groundtruth_lists[fields.BoxListFields.
                              confidences] = groundtruth_confidences_list
    if groundtruth_masks_list:
      self._groundtruth_lists[
          fields.BoxListFields.masks] = groundtruth_masks_list
    if groundtruth_keypoints_list:
      self._groundtruth_lists[
          fields.BoxListFields.keypoints] = groundtruth_keypoints_list
    if groundtruth_is_crowd_list:
      self._groundtruth_lists[
          fields.BoxListFields.is_crowd] = groundtruth_is_crowd_list
    if is_annotated_list:
      self._groundtruth_lists[
          fields.InputDataFields.is_annotated] = is_annotated_list

  @abc.abstractmethod
  def regularization_losses(self):
    """Returns a list of regularization losses for this model.

    Returns a list of regularization losses for this model that the estimator
    needs to use during training/optimization.

    Returns:
      A list of regularization loss tensors.
    """
    pass

  @abc.abstractmethod
  def restore_map(self, fine_tune_checkpoint_type='detection'):
    """Returns a map of variables to load from a foreign checkpoint.

    Returns a map of variable names to load from a checkpoint to variables in
    the model graph. This enables the model to initialize based on weights from
    another task. For example, the feature extractor variables from a
    classification model can be used to bootstrap training of an object
    detector. When loading from an object detection model, the checkpoint model
    should have the same parameters as this detection model with exception of
    the num_classes parameter.

    Args:
      fine_tune_checkpoint_type: whether to restore from a full detection
        checkpoint (with compatible variable names) or to restore from a
        classification checkpoint for initialization prior to training.
        Valid values: `detection`, `classification`. Default 'detection'.

    Returns:
      A dict mapping variable names (to load from a checkpoint) to variables in
      the model graph.
    """
    pass

  @abc.abstractmethod
  def updates(self):
    """Returns a list of update operators for this model.

    Returns a list of update operators for this model that must be executed at
    each training step. The estimator's train op needs to have a control
    dependency on these updates.

    Returns:
      A list of update operators.
    """
    pass
