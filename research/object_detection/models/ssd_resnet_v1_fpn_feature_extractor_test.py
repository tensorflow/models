"""Tests for ssd resnet v1 FPN feature extractors."""
import tensorflow as tf

from object_detection.models import ssd_resnet_v1_fpn_feature_extractor
from object_detection.models import ssd_resnet_v1_fpn_feature_extractor_testbase


class SSDResnet50V1FeatureExtractorTest(
    ssd_resnet_v1_fpn_feature_extractor_testbase.
    SSDResnetFPNFeatureExtractorTestBase):
  """SSDResnet50v1Fpn feature extractor test."""

  def _create_feature_extractor(self, depth_multiplier, pad_to_multiple):
    min_depth = 32
    conv_hyperparams = {}
    batch_norm_trainable = True
    is_training = True
    return ssd_resnet_v1_fpn_feature_extractor.SSDResnet50V1FpnFeatureExtractor(
        is_training, depth_multiplier, min_depth, pad_to_multiple,
        conv_hyperparams, batch_norm_trainable)

  def _resnet_scope_name(self):
    return 'resnet_v1_50'


class SSDResnet101V1FeatureExtractorTest(
    ssd_resnet_v1_fpn_feature_extractor_testbase.
    SSDResnetFPNFeatureExtractorTestBase):
  """SSDResnet101v1Fpn feature extractor test."""

  def _create_feature_extractor(self, depth_multiplier, pad_to_multiple):
    min_depth = 32
    conv_hyperparams = {}
    batch_norm_trainable = True
    is_training = True
    return (
        ssd_resnet_v1_fpn_feature_extractor.SSDResnet101V1FpnFeatureExtractor(
            is_training, depth_multiplier, min_depth, pad_to_multiple,
            conv_hyperparams, batch_norm_trainable))

  def _resnet_scope_name(self):
    return 'resnet_v1_101'


class SSDResnet152V1FeatureExtractorTest(
    ssd_resnet_v1_fpn_feature_extractor_testbase.
    SSDResnetFPNFeatureExtractorTestBase):
  """SSDResnet152v1Fpn feature extractor test."""

  def _create_feature_extractor(self, depth_multiplier, pad_to_multiple):
    min_depth = 32
    conv_hyperparams = {}
    batch_norm_trainable = True
    is_training = True
    return (
        ssd_resnet_v1_fpn_feature_extractor.SSDResnet152V1FpnFeatureExtractor(
            is_training, depth_multiplier, min_depth, pad_to_multiple,
            conv_hyperparams, batch_norm_trainable))

  def _resnet_scope_name(self):
    return 'resnet_v1_152'


if __name__ == '__main__':
  tf.test.main()
