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

"""A function to build a DetectionModel from configuration."""
from lstm_object_detection.meta_architectures import lstm_ssd_meta_arch
from lstm_object_detection.models import lstm_ssd_interleaved_mobilenet_v2_feature_extractor
from lstm_object_detection.models import lstm_ssd_mobilenet_v1_feature_extractor
from object_detection.builders import anchor_generator_builder
from object_detection.builders import box_coder_builder
from object_detection.builders import box_predictor_builder
from object_detection.builders import hyperparams_builder
from object_detection.builders import image_resizer_builder
from object_detection.builders import losses_builder
from object_detection.builders import matcher_builder
from object_detection.builders import model_builder
from object_detection.builders import post_processing_builder
from object_detection.builders import region_similarity_calculator_builder as sim_calc
from object_detection.core import target_assigner

model_builder.SSD_FEATURE_EXTRACTOR_CLASS_MAP.update({
    'lstm_ssd_mobilenet_v1':
        lstm_ssd_mobilenet_v1_feature_extractor
        .LSTMSSDMobileNetV1FeatureExtractor,
    'lstm_ssd_interleaved_mobilenet_v2':
        lstm_ssd_interleaved_mobilenet_v2_feature_extractor
        .LSTMSSDInterleavedMobilenetV2FeatureExtractor,
})
SSD_FEATURE_EXTRACTOR_CLASS_MAP = model_builder.SSD_FEATURE_EXTRACTOR_CLASS_MAP


def build(model_config, lstm_config, is_training):
  """Builds a DetectionModel based on the model config.

  Args:
    model_config: A model.proto object containing the config for the desired
      DetectionModel.
    lstm_config: LstmModel config proto that specifies LSTM train/eval configs.
    is_training: True if this model is being built for training purposes.

  Returns:
    DetectionModel based on the config.

  Raises:
    ValueError: On invalid meta architecture or model.
  """
  return _build_lstm_model(model_config.ssd, lstm_config, is_training)


def _build_lstm_feature_extractor(feature_extractor_config,
                                  is_training,
                                  lstm_config,
                                  reuse_weights=None):
  """Builds a ssd_meta_arch.SSDFeatureExtractor based on config.

  Args:
    feature_extractor_config: A SSDFeatureExtractor proto config from ssd.proto.
    is_training: True if this feature extractor is being built for training.
    lstm_config: LSTM-SSD specific configs.
    reuse_weights: If the feature extractor should reuse weights.

  Returns:
    ssd_meta_arch.SSDFeatureExtractor based on config.

  Raises:
    ValueError: On invalid feature extractor type.
  """

  feature_type = feature_extractor_config.type
  depth_multiplier = feature_extractor_config.depth_multiplier
  min_depth = feature_extractor_config.min_depth
  pad_to_multiple = feature_extractor_config.pad_to_multiple
  use_explicit_padding = feature_extractor_config.use_explicit_padding
  use_depthwise = feature_extractor_config.use_depthwise
  conv_hyperparams = hyperparams_builder.build(
      feature_extractor_config.conv_hyperparams, is_training)
  override_base_feature_extractor_hyperparams = (
      feature_extractor_config.override_base_feature_extractor_hyperparams)

  if feature_type not in SSD_FEATURE_EXTRACTOR_CLASS_MAP:
    raise ValueError('Unknown ssd feature_extractor: {}'.format(feature_type))

  feature_extractor_class = SSD_FEATURE_EXTRACTOR_CLASS_MAP[feature_type]
  feature_extractor = feature_extractor_class(
      is_training, depth_multiplier, min_depth, pad_to_multiple,
      conv_hyperparams, reuse_weights, use_explicit_padding, use_depthwise,
      override_base_feature_extractor_hyperparams)

  # Extra configs for LSTM-SSD.
  feature_extractor.lstm_state_depth = lstm_config.lstm_state_depth
  feature_extractor.flatten_state = lstm_config.flatten_state
  feature_extractor.clip_state = lstm_config.clip_state
  feature_extractor.scale_state = lstm_config.scale_state
  feature_extractor.is_quantized = lstm_config.is_quantized
  feature_extractor.low_res = lstm_config.low_res
  # Extra configs for interleaved LSTM-SSD.
  if 'interleaved' in feature_extractor_config.type:
    feature_extractor.pre_bottleneck = lstm_config.pre_bottleneck
    feature_extractor.depth_multipliers = lstm_config.depth_multipliers
    if is_training:
      feature_extractor.interleave_method = lstm_config.train_interleave_method
    else:
      feature_extractor.interleave_method = lstm_config.eval_interleave_method
  return feature_extractor


def _build_lstm_model(ssd_config, lstm_config, is_training):
  """Builds an LSTM detection model based on the model config.

  Args:
    ssd_config: A ssd.proto object containing the config for the desired
      LSTMSSDMetaArch.
    lstm_config: LstmModel config proto that specifies LSTM train/eval configs.
    is_training: True if this model is being built for training purposes.

  Returns:
    LSTMSSDMetaArch based on the config.
  Raises:
    ValueError: If ssd_config.type is not recognized (i.e. not registered in
      model_class_map), or if lstm_config.interleave_strategy is not recognized.
    ValueError: If unroll_length is not specified in the config file.
  """
  feature_extractor = _build_lstm_feature_extractor(
      ssd_config.feature_extractor, is_training, lstm_config)

  box_coder = box_coder_builder.build(ssd_config.box_coder)
  matcher = matcher_builder.build(ssd_config.matcher)
  region_similarity_calculator = sim_calc.build(
      ssd_config.similarity_calculator)

  num_classes = ssd_config.num_classes
  ssd_box_predictor = box_predictor_builder.build(hyperparams_builder.build,
                                                  ssd_config.box_predictor,
                                                  is_training, num_classes)
  anchor_generator = anchor_generator_builder.build(ssd_config.anchor_generator)
  image_resizer_fn = image_resizer_builder.build(ssd_config.image_resizer)
  non_max_suppression_fn, score_conversion_fn = post_processing_builder.build(
      ssd_config.post_processing)
  (classification_loss, localization_loss, classification_weight,
   localization_weight, miner, _, _) = losses_builder.build(ssd_config.loss)

  normalize_loss_by_num_matches = ssd_config.normalize_loss_by_num_matches
  encode_background_as_zeros = ssd_config.encode_background_as_zeros
  negative_class_weight = ssd_config.negative_class_weight

  # Extra configs for lstm unroll length.
  unroll_length = None
  if 'lstm' in ssd_config.feature_extractor.type:
    if is_training:
      unroll_length = lstm_config.train_unroll_length
    else:
      unroll_length = lstm_config.eval_unroll_length
  if unroll_length is None:
    raise ValueError('No unroll length found in the config file')

  target_assigner_instance = target_assigner.TargetAssigner(
      region_similarity_calculator,
      matcher,
      box_coder,
      negative_class_weight=negative_class_weight)

  lstm_model = lstm_ssd_meta_arch.LSTMSSDMetaArch(
      is_training=is_training,
      anchor_generator=anchor_generator,
      box_predictor=ssd_box_predictor,
      box_coder=box_coder,
      feature_extractor=feature_extractor,
      encode_background_as_zeros=encode_background_as_zeros,
      image_resizer_fn=image_resizer_fn,
      non_max_suppression_fn=non_max_suppression_fn,
      score_conversion_fn=score_conversion_fn,
      classification_loss=classification_loss,
      localization_loss=localization_loss,
      classification_loss_weight=classification_weight,
      localization_loss_weight=localization_weight,
      normalize_loss_by_num_matches=normalize_loss_by_num_matches,
      hard_example_miner=miner,
      unroll_length=unroll_length,
      target_assigner_instance=target_assigner_instance)

  return lstm_model
