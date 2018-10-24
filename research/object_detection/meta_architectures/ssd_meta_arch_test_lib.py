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
"""Helper functions for SSD models meta architecture tests."""

import functools
import tensorflow as tf

from object_detection.core import anchor_generator
from object_detection.core import balanced_positive_negative_sampler as sampler
from object_detection.core import box_list
from object_detection.core import losses
from object_detection.core import post_processing
from object_detection.core import region_similarity_calculator as sim_calc
from object_detection.core import target_assigner
from object_detection.meta_architectures import ssd_meta_arch
from object_detection.utils import ops
from object_detection.utils import test_case
from object_detection.utils import test_utils

slim = tf.contrib.slim
keras = tf.keras.layers


class FakeSSDFeatureExtractor(ssd_meta_arch.SSDFeatureExtractor):
  """Fake ssd feature extracture for ssd meta arch tests."""

  def __init__(self):
    super(FakeSSDFeatureExtractor, self).__init__(
        is_training=True,
        depth_multiplier=0,
        min_depth=0,
        pad_to_multiple=1,
        conv_hyperparams_fn=None)

  def preprocess(self, resized_inputs):
    return tf.identity(resized_inputs)

  def extract_features(self, preprocessed_inputs):
    with tf.variable_scope('mock_model'):
      features = slim.conv2d(
          inputs=preprocessed_inputs,
          num_outputs=32,
          kernel_size=1,
          scope='layer1')
      return [features]


class FakeSSDKerasFeatureExtractor(ssd_meta_arch.SSDKerasFeatureExtractor):
  """Fake keras based ssd feature extracture for ssd meta arch tests."""

  def __init__(self):
    with tf.name_scope('mock_model'):
      super(FakeSSDKerasFeatureExtractor, self).__init__(
          is_training=True,
          depth_multiplier=0,
          min_depth=0,
          pad_to_multiple=1,
          conv_hyperparams=None,
          freeze_batchnorm=False,
          inplace_batchnorm_update=False,
      )

      self._conv = keras.Conv2D(filters=32, kernel_size=1, name='layer1')

  def preprocess(self, resized_inputs):
    return tf.identity(resized_inputs)

  def _extract_features(self, preprocessed_inputs, **kwargs):
    with tf.name_scope('mock_model'):
      return [self._conv(preprocessed_inputs)]


class MockAnchorGenerator2x2(anchor_generator.AnchorGenerator):
  """A simple 2x2 anchor grid on the unit square used for test only."""

  def name_scope(self):
    return 'MockAnchorGenerator'

  def num_anchors_per_location(self):
    return [1]

  def _generate(self, feature_map_shape_list, im_height, im_width):
    return [
        box_list.BoxList(
            tf.constant(
                [
                    [0, 0, .5, .5],
                    [0, .5, .5, 1],
                    [.5, 0, 1, .5],
                    [1., 1., 1.5, 1.5]  # Anchor that is outside clip_window.
                ],
                tf.float32))
    ]

  def num_anchors(self):
    return 4


class SSDMetaArchTestBase(test_case.TestCase):
  """Base class to test SSD based meta architectures."""

  def _create_model(self,
                    model_fn=ssd_meta_arch.SSDMetaArch,
                    apply_hard_mining=True,
                    normalize_loc_loss_by_codesize=False,
                    add_background_class=True,
                    random_example_sampling=False,
                    weight_regression_loss_by_score=False,
                    use_expected_classification_loss_under_sampling=False,
                    minimum_negative_sampling=1,
                    desired_negative_sampling_ratio=3,
                    use_keras=False,
                    predict_mask=False,
                    use_static_shapes=False,
                    nms_max_size_per_class=5):
    is_training = False
    num_classes = 1
    mock_anchor_generator = MockAnchorGenerator2x2()
    if use_keras:
      mock_box_predictor = test_utils.MockKerasBoxPredictor(
          is_training, num_classes, predict_mask=predict_mask)
    else:
      mock_box_predictor = test_utils.MockBoxPredictor(
          is_training, num_classes, predict_mask=predict_mask)
    mock_box_coder = test_utils.MockBoxCoder()
    if use_keras:
      fake_feature_extractor = FakeSSDKerasFeatureExtractor()
    else:
      fake_feature_extractor = FakeSSDFeatureExtractor()
    mock_matcher = test_utils.MockMatcher()
    region_similarity_calculator = sim_calc.IouSimilarity()
    encode_background_as_zeros = False

    def image_resizer_fn(image):
      return [tf.identity(image), tf.shape(image)]

    classification_loss = losses.WeightedSigmoidClassificationLoss()
    localization_loss = losses.WeightedSmoothL1LocalizationLoss()
    non_max_suppression_fn = functools.partial(
        post_processing.batch_multiclass_non_max_suppression,
        score_thresh=-20.0,
        iou_thresh=1.0,
        max_size_per_class=nms_max_size_per_class,
        max_total_size=nms_max_size_per_class,
        use_static_shapes=use_static_shapes)
    classification_loss_weight = 1.0
    localization_loss_weight = 1.0
    negative_class_weight = 1.0
    normalize_loss_by_num_matches = False

    hard_example_miner = None
    if apply_hard_mining:
      # This hard example miner is expected to be a no-op.
      hard_example_miner = losses.HardExampleMiner(
          num_hard_examples=None, iou_threshold=1.0)

    random_example_sampler = None
    if random_example_sampling:
      random_example_sampler = sampler.BalancedPositiveNegativeSampler(
          positive_fraction=0.5)

    target_assigner_instance = target_assigner.TargetAssigner(
        region_similarity_calculator,
        mock_matcher,
        mock_box_coder,
        negative_class_weight=negative_class_weight,
        weight_regression_loss_by_score=weight_regression_loss_by_score)

    expected_classification_loss_under_sampling = None
    if use_expected_classification_loss_under_sampling:
      expected_classification_loss_under_sampling = functools.partial(
          ops.expected_classification_loss_under_sampling,
          minimum_negative_sampling=minimum_negative_sampling,
          desired_negative_sampling_ratio=desired_negative_sampling_ratio)

    code_size = 4
    model = model_fn(
        is_training=is_training,
        anchor_generator=mock_anchor_generator,
        box_predictor=mock_box_predictor,
        box_coder=mock_box_coder,
        feature_extractor=fake_feature_extractor,
        encode_background_as_zeros=encode_background_as_zeros,
        image_resizer_fn=image_resizer_fn,
        non_max_suppression_fn=non_max_suppression_fn,
        score_conversion_fn=tf.identity,
        classification_loss=classification_loss,
        localization_loss=localization_loss,
        classification_loss_weight=classification_loss_weight,
        localization_loss_weight=localization_loss_weight,
        normalize_loss_by_num_matches=normalize_loss_by_num_matches,
        hard_example_miner=hard_example_miner,
        target_assigner_instance=target_assigner_instance,
        add_summaries=False,
        normalize_loc_loss_by_codesize=normalize_loc_loss_by_codesize,
        freeze_batchnorm=False,
        inplace_batchnorm_update=False,
        add_background_class=add_background_class,
        random_example_sampler=random_example_sampler,
        expected_classification_loss_under_sampling=
        expected_classification_loss_under_sampling)
    return model, num_classes, mock_anchor_generator.num_anchors(), code_size

  def _get_value_for_matching_key(self, dictionary, suffix):
    for key in dictionary.keys():
      if key.endswith(suffix):
        return dictionary[key]
    raise ValueError('key not found {}'.format(suffix))


if __name__ == '__main__':
  tf.test.main()
