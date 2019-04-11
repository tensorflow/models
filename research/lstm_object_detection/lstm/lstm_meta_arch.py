# Copyright 2018 The TensorFlow Authors. All Rights Reserved.
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

"""LSTM Meta-architecture definition.

General tensorflow implementation of convolutional Multibox/SSD detection
models with LSTM states, for use on video data.

See https://arxiv.org/abs/1711.06368 for details.
"""
import re
import tensorflow as tf

from object_detection.core import box_list_ops
from object_detection.core import standard_fields as fields
from object_detection.meta_architectures import ssd_meta_arch
from object_detection.utils import ops
from object_detection.utils import shape_utils

slim = tf.contrib.slim


class LSTMMetaArch(ssd_meta_arch.SSDMetaArch):
  """LSTM Meta-architecture definition."""

  def __init__(self,
               is_training,
               anchor_generator,
               box_predictor,
               box_coder,
               feature_extractor,
               encode_background_as_zeros,
               image_resizer_fn,
               non_max_suppression_fn,
               score_conversion_fn,
               classification_loss,
               localization_loss,
               classification_loss_weight,
               localization_loss_weight,
               normalize_loss_by_num_matches,
               hard_example_miner,
               unroll_length,
               target_assigner_instance,
               add_summaries=True):
    super(LSTMMetaArch, self).__init__(
        is_training=is_training,
        anchor_generator=anchor_generator,
        box_predictor=box_predictor,
        box_coder=box_coder,
        feature_extractor=feature_extractor,
        encode_background_as_zeros=encode_background_as_zeros,
        image_resizer_fn=image_resizer_fn,
        non_max_suppression_fn=non_max_suppression_fn,
        score_conversion_fn=score_conversion_fn,
        classification_loss=classification_loss,
        localization_loss=localization_loss,
        classification_loss_weight=classification_loss_weight,
        localization_loss_weight=localization_loss_weight,
        normalize_loss_by_num_matches=normalize_loss_by_num_matches,
        hard_example_miner=hard_example_miner,
        target_assigner_instance=target_assigner_instance,
        add_summaries=add_summaries)
    self._unroll_length = unroll_length

  @property
  def unroll_length(self):
    return self._unroll_length

  @unroll_length.setter
  def unroll_length(self, unroll_length):
    self._unroll_length = unroll_length

  def predict(self, preprocessed_inputs, true_image_shapes, states=None,
              state_name='lstm_state', feature_scope=None):
    with tf.variable_scope(self._extract_features_scope,
                           values=[preprocessed_inputs], reuse=tf.AUTO_REUSE):
      feature_maps = self._feature_extractor.extract_features(
          preprocessed_inputs, states, state_name,
          unroll_length=self._unroll_length, scope=feature_scope)
    feature_map_spatial_dims = self._get_feature_map_spatial_dims(feature_maps)
    image_shape = shape_utils.combined_static_and_dynamic_shape(
        preprocessed_inputs)
    self._batch_size = preprocessed_inputs.shape[0].value / self._unroll_length
    self._states = states
    self._anchors = box_list_ops.concatenate(
        self._anchor_generator.generate(
            feature_map_spatial_dims,
            im_height=image_shape[1],
            im_width=image_shape[2]))
    prediction_dict = self._box_predictor.predict(
        feature_maps, self._anchor_generator.num_anchors_per_location())

    # Multiscale_anchor_generator currently has a different dim compared to
    # ssd_anchor_generator. Current fix is to check the dim of the box_encodings
    # tensor. If dim is not 3(multiscale_anchor_generator), squeeze the 3rd dim.
    # TODO(yinxiao): Remove this check once the anchor generator has unified
    # dimension.
    if len(prediction_dict['box_encodings'][0].get_shape().as_list()) == 3:
      box_encodings = tf.concat(prediction_dict['box_encodings'], axis=1)
    else:
      box_encodings = tf.squeeze(
          tf.concat(prediction_dict['box_encodings'], axis=1), axis=2)
    class_predictions_with_background = tf.concat(
        prediction_dict['class_predictions_with_background'], axis=1)
    predictions_dict = {
        'preprocessed_inputs': preprocessed_inputs,
        'box_encodings': box_encodings,
        'class_predictions_with_background': class_predictions_with_background,
        'feature_maps': feature_maps,
        'anchors': self._anchors.get(),
        'states_and_outputs': self._feature_extractor.states_and_outputs,
    }
    # In cases such as exporting the model, the states is always zero. Thus the
    # step should be ignored.
    if states is not None:
      predictions_dict['step'] = self._feature_extractor.step
    return predictions_dict

  def loss(self, prediction_dict, true_image_shapes, scope=None):
    """Computes scalar loss tensors with respect to provided groundtruth.

    Calling this function requires that groundtruth tensors have been
    provided via the provide_groundtruth function.

    Args:
      prediction_dict: a dictionary holding prediction tensors with
        1) box_encodings: 3-D float tensor of shape [batch_size, num_anchors,
          box_code_dimension] containing predicted boxes.
        2) class_predictions_with_background: 3-D float tensor of shape
          [batch_size, num_anchors, num_classes+1] containing class predictions
          (logits) for each of the anchors. Note that this tensor *includes*
          background class predictions.
      true_image_shapes: int32 tensor of shape [batch, 3] where each row is
        of the form [height, width, channels] indicating the shapes
        of true images in the resized images, as resized images can be padded
        with zeros.
      scope: Optional scope name.

    Returns:
      a dictionary mapping loss keys (`localization_loss` and
        `classification_loss`) to scalar tensors representing corresponding loss
        values.
    """
    with tf.name_scope(scope, 'Loss', prediction_dict.values()):
      keypoints = None
      if self.groundtruth_has_field(fields.BoxListFields.keypoints):
        keypoints = self.groundtruth_lists(fields.BoxListFields.keypoints)
      weights = None
      if self.groundtruth_has_field(fields.BoxListFields.weights):
        weights = self.groundtruth_lists(fields.BoxListFields.weights)
      (batch_cls_targets, batch_cls_weights, batch_reg_targets,
       batch_reg_weights, match_list) = self._assign_targets(
           self.groundtruth_lists(fields.BoxListFields.boxes),
           self.groundtruth_lists(fields.BoxListFields.classes),
           keypoints, weights)
      if self._add_summaries:
        self._summarize_target_assignment(
            self.groundtruth_lists(fields.BoxListFields.boxes), match_list)
      location_losses = self._localization_loss(
          prediction_dict['box_encodings'],
          batch_reg_targets,
          ignore_nan_targets=True,
          weights=batch_reg_weights)
      cls_losses = ops.reduce_sum_trailing_dimensions(
          self._classification_loss(
              prediction_dict['class_predictions_with_background'],
              batch_cls_targets,
              weights=batch_cls_weights),
          ndims=2)

      if self._hard_example_miner:
        (loc_loss_list, cls_loss_list) = self._apply_hard_mining(
            location_losses, cls_losses, prediction_dict, match_list)
        localization_loss = tf.reduce_sum(tf.stack(loc_loss_list))
        classification_loss = tf.reduce_sum(tf.stack(cls_loss_list))

        if self._add_summaries:
          self._hard_example_miner.summarize()
      else:
        if self._add_summaries:
          class_ids = tf.argmax(batch_cls_targets, axis=2)
          flattened_class_ids = tf.reshape(class_ids, [-1])
          flattened_classification_losses = tf.reshape(cls_losses, [-1])
          self._summarize_anchor_classification_loss(
              flattened_class_ids, flattened_classification_losses)
        localization_loss = tf.reduce_sum(location_losses)
        classification_loss = tf.reduce_sum(cls_losses)

      # Optionally normalize by number of positive matches
      normalizer = tf.constant(1.0, dtype=tf.float32)
      if self._normalize_loss_by_num_matches:
        normalizer = tf.maximum(tf.to_float(tf.reduce_sum(batch_reg_weights)),
                                1.0)

      with tf.name_scope('localization_loss'):
        localization_loss_normalizer = normalizer
        if self._normalize_loc_loss_by_codesize:
          localization_loss_normalizer *= self._box_coder.code_size
        localization_loss = ((self._localization_loss_weight / (
            localization_loss_normalizer)) * localization_loss)
      with tf.name_scope('classification_loss'):
        classification_loss = ((self._classification_loss_weight / normalizer) *
                               classification_loss)

      loss_dict = {
          'localization_loss': localization_loss,
          'classification_loss': classification_loss
      }
    return loss_dict

  def restore_map(self, fine_tune_checkpoint_type='lstm'):
    """Returns a map of variables to load from a foreign checkpoint.

    See parent class for details.

    Args:
      fine_tune_checkpoint_type: the type of checkpoint to restore from, either
        SSD/LSTM detection checkpoint (with compatible variable names)
        classification checkpoint for initialization prior to training.
        Available options: `classification`, `detection`, `interleaved`,
        and `lstm`.

    Returns:
      A dict mapping variable names (to load from a checkpoint) to variables in
      the model graph.
    Raises:
      ValueError: if fine_tune_checkpoint_type is not among
      `classification`/`detection`/`interleaved`/`lstm`.
    """
    if fine_tune_checkpoint_type not in [
        'classification', 'detection', 'lstm'
    ]:
      raise ValueError('Not supported fine_tune_checkpoint_type: {}'.format(
          fine_tune_checkpoint_type))

    variables_to_restore = {}
    for variable in tf.global_variables():
      var_name = variable.op.name
      if 'global_step' in var_name:
        continue

      # Remove FeatureExtractor prefix for classification checkpoints.
      if fine_tune_checkpoint_type == 'classification':
        var_name = (
            re.split('^' + self._extract_features_scope + '/', var_name)[-1])

      # When loading from single frame detection checkpoints, we need to
      # remap FeatureMaps variable names.
      if ('FeatureMaps' in var_name and
          fine_tune_checkpoint_type == 'detection'):
        var_name = var_name.replace('FeatureMaps',
                                    self.get_base_network_scope())
      variables_to_restore[var_name] = variable

    return variables_to_restore

  def get_base_network_scope(self):
    """Returns the variable scope of the base network.

    Returns:
      The variable scope of the feature extractor base network, e.g. MobilenetV1
    """
    return self._feature_extractor.get_base_network_scope()


class LSTMFeatureExtractor(ssd_meta_arch.SSDFeatureExtractor):
  """LSTM Meta-architecture  Feature Extractor definition."""

  @property
  def depth_multipliers(self):
    return self._depth_multipliers

  @depth_multipliers.setter
  def depth_multipliers(self, depth_multipliers):
    self._depth_multipliers = depth_multipliers

  @property
  def lstm_state_depth(self):
    return self._lstm_state_depth

  @lstm_state_depth.setter
  def lstm_state_depth(self, lstm_state_depth):
    self._lstm_state_depth = lstm_state_depth

  @property
  def states_and_outputs(self):
    """LSTM states and outputs.

    This variable includes both LSTM states {C_t} and outputs {h_t}.

    Returns:
      states_and_outputs: A list of 4-D float tensors, including the lstm state
        and output at each timestep.
    """
    return self._states_out

  @property
  def step(self):
    return self._step

  def preprocess(self, resized_inputs):
    """SSD preprocessing.

    Maps pixel values to the range [-1, 1].

    Args:
      resized_inputs: a [batch, height, width, channels] float tensor
        representing a batch of images.

    Returns:
      preprocessed_inputs: a [batch, height, width, channels] float tensor
        representing a batch of images.
    """
    return (2.0 / 255.0) * resized_inputs - 1.0

  def get_base_network_scope(self):
    """Returns the variable scope of the base network.

    Returns:
      The variable scope of the base network, e.g. MobilenetV1
    """
    return self._base_network_scope
