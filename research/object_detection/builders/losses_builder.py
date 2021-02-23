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

"""A function to build localization and classification losses from config."""

import functools
from object_detection.core import balanced_positive_negative_sampler as sampler
from object_detection.core import losses
from object_detection.protos import losses_pb2
from object_detection.utils import ops


def build(loss_config):
  """Build losses based on the config.

  Builds classification, localization losses and optionally a hard example miner
  based on the config.

  Args:
    loss_config: A losses_pb2.Loss object.

  Returns:
    classification_loss: Classification loss object.
    localization_loss: Localization loss object.
    classification_weight: Classification loss weight.
    localization_weight: Localization loss weight.
    hard_example_miner: Hard example miner object.
    random_example_sampler: BalancedPositiveNegativeSampler object.

  Raises:
    ValueError: If hard_example_miner is used with sigmoid_focal_loss.
    ValueError: If random_example_sampler is getting non-positive value as
      desired positive example fraction.
  """
  classification_loss = _build_classification_loss(
      loss_config.classification_loss)
  localization_loss = _build_localization_loss(
      loss_config.localization_loss)
  classification_weight = loss_config.classification_weight
  localization_weight = loss_config.localization_weight
  hard_example_miner = None
  if loss_config.HasField('hard_example_miner'):
    if (loss_config.classification_loss.WhichOneof('classification_loss') ==
        'weighted_sigmoid_focal'):
      raise ValueError('HardExampleMiner should not be used with sigmoid focal '
                       'loss')
    hard_example_miner = build_hard_example_miner(
        loss_config.hard_example_miner,
        classification_weight,
        localization_weight)
  random_example_sampler = None
  if loss_config.HasField('random_example_sampler'):
    if loss_config.random_example_sampler.positive_sample_fraction <= 0:
      raise ValueError('RandomExampleSampler should not use non-positive'
                       'value as positive sample fraction.')
    random_example_sampler = sampler.BalancedPositiveNegativeSampler(
        positive_fraction=loss_config.random_example_sampler.
        positive_sample_fraction)

  if loss_config.expected_loss_weights == loss_config.NONE:
    expected_loss_weights_fn = None
  elif loss_config.expected_loss_weights == loss_config.EXPECTED_SAMPLING:
    expected_loss_weights_fn = functools.partial(
        ops.expected_classification_loss_by_expected_sampling,
        min_num_negative_samples=loss_config.min_num_negative_samples,
        desired_negative_sampling_ratio=loss_config
        .desired_negative_sampling_ratio)
  elif (loss_config.expected_loss_weights == loss_config
        .REWEIGHTING_UNMATCHED_ANCHORS):
    expected_loss_weights_fn = functools.partial(
        ops.expected_classification_loss_by_reweighting_unmatched_anchors,
        min_num_negative_samples=loss_config.min_num_negative_samples,
        desired_negative_sampling_ratio=loss_config
        .desired_negative_sampling_ratio)
  else:
    raise ValueError('Not a valid value for expected_classification_loss.')

  return (classification_loss, localization_loss, classification_weight,
          localization_weight, hard_example_miner, random_example_sampler,
          expected_loss_weights_fn)


def build_hard_example_miner(config,
                             classification_weight,
                             localization_weight):
  """Builds hard example miner based on the config.

  Args:
    config: A losses_pb2.HardExampleMiner object.
    classification_weight: Classification loss weight.
    localization_weight: Localization loss weight.

  Returns:
    Hard example miner.

  """
  loss_type = None
  if config.loss_type == losses_pb2.HardExampleMiner.BOTH:
    loss_type = 'both'
  if config.loss_type == losses_pb2.HardExampleMiner.CLASSIFICATION:
    loss_type = 'cls'
  if config.loss_type == losses_pb2.HardExampleMiner.LOCALIZATION:
    loss_type = 'loc'

  max_negatives_per_positive = None
  num_hard_examples = None
  if config.max_negatives_per_positive > 0:
    max_negatives_per_positive = config.max_negatives_per_positive
  if config.num_hard_examples > 0:
    num_hard_examples = config.num_hard_examples
  hard_example_miner = losses.HardExampleMiner(
      num_hard_examples=num_hard_examples,
      iou_threshold=config.iou_threshold,
      loss_type=loss_type,
      cls_loss_weight=classification_weight,
      loc_loss_weight=localization_weight,
      max_negatives_per_positive=max_negatives_per_positive,
      min_negatives_per_image=config.min_negatives_per_image)
  return hard_example_miner


def build_faster_rcnn_classification_loss(loss_config):
  """Builds a classification loss for Faster RCNN based on the loss config.

  Args:
    loss_config: A losses_pb2.ClassificationLoss object.

  Returns:
    Loss based on the config.

  Raises:
    ValueError: On invalid loss_config.
  """
  if not isinstance(loss_config, losses_pb2.ClassificationLoss):
    raise ValueError('loss_config not of type losses_pb2.ClassificationLoss.')

  loss_type = loss_config.WhichOneof('classification_loss')

  if loss_type == 'weighted_sigmoid':
    return losses.WeightedSigmoidClassificationLoss()
  if loss_type == 'weighted_softmax':
    config = loss_config.weighted_softmax
    return losses.WeightedSoftmaxClassificationLoss(
        logit_scale=config.logit_scale)
  if loss_type == 'weighted_logits_softmax':
    config = loss_config.weighted_logits_softmax
    return losses.WeightedSoftmaxClassificationAgainstLogitsLoss(
        logit_scale=config.logit_scale)
  if loss_type == 'weighted_sigmoid_focal':
    config = loss_config.weighted_sigmoid_focal
    alpha = None
    if config.HasField('alpha'):
      alpha = config.alpha
    return losses.SigmoidFocalClassificationLoss(
        gamma=config.gamma,
        alpha=alpha)

  # By default, Faster RCNN second stage classifier uses Softmax loss
  # with anchor-wise outputs.
  config = loss_config.weighted_softmax
  return losses.WeightedSoftmaxClassificationLoss(
      logit_scale=config.logit_scale)


def _build_localization_loss(loss_config):
  """Builds a localization loss based on the loss config.

  Args:
    loss_config: A losses_pb2.LocalizationLoss object.

  Returns:
    Loss based on the config.

  Raises:
    ValueError: On invalid loss_config.
  """
  if not isinstance(loss_config, losses_pb2.LocalizationLoss):
    raise ValueError('loss_config not of type losses_pb2.LocalizationLoss.')

  loss_type = loss_config.WhichOneof('localization_loss')

  if loss_type == 'weighted_l2':
    return losses.WeightedL2LocalizationLoss()

  if loss_type == 'weighted_smooth_l1':
    return losses.WeightedSmoothL1LocalizationLoss(
        loss_config.weighted_smooth_l1.delta)

  if loss_type == 'weighted_iou':
    return losses.WeightedIOULocalizationLoss()

  if loss_type == 'l1_localization_loss':
    return losses.L1LocalizationLoss()

  raise ValueError('Empty loss config.')


def _build_classification_loss(loss_config):
  """Builds a classification loss based on the loss config.

  Args:
    loss_config: A losses_pb2.ClassificationLoss object.

  Returns:
    Loss based on the config.

  Raises:
    ValueError: On invalid loss_config.
  """
  if not isinstance(loss_config, losses_pb2.ClassificationLoss):
    raise ValueError('loss_config not of type losses_pb2.ClassificationLoss.')

  loss_type = loss_config.WhichOneof('classification_loss')

  if loss_type == 'weighted_sigmoid':
    return losses.WeightedSigmoidClassificationLoss()

  elif loss_type == 'weighted_sigmoid_focal':
    config = loss_config.weighted_sigmoid_focal
    alpha = None
    if config.HasField('alpha'):
      alpha = config.alpha
    return losses.SigmoidFocalClassificationLoss(
        gamma=config.gamma,
        alpha=alpha)

  elif loss_type == 'weighted_softmax':
    config = loss_config.weighted_softmax
    return losses.WeightedSoftmaxClassificationLoss(
        logit_scale=config.logit_scale)

  elif loss_type == 'weighted_logits_softmax':
    config = loss_config.weighted_logits_softmax
    return losses.WeightedSoftmaxClassificationAgainstLogitsLoss(
        logit_scale=config.logit_scale)

  elif loss_type == 'bootstrapped_sigmoid':
    config = loss_config.bootstrapped_sigmoid
    return losses.BootstrappedSigmoidClassificationLoss(
        alpha=config.alpha,
        bootstrap_type=('hard' if config.hard_bootstrap else 'soft'))

  elif loss_type == 'penalty_reduced_logistic_focal_loss':
    config = loss_config.penalty_reduced_logistic_focal_loss
    return losses.PenaltyReducedLogisticFocalLoss(
        alpha=config.alpha, beta=config.beta)

  elif loss_type == 'weighted_dice_classification_loss':
    config = loss_config.weighted_dice_classification_loss
    return losses.WeightedDiceClassificationLoss(
        squared_normalization=config.squared_normalization)

  else:
    raise ValueError('Empty loss config.')
