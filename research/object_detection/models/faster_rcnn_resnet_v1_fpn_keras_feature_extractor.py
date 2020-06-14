"""Faster RCNN Keras-based Resnet v1 FPN Feature Extractor."""

from object_detection.meta_architectures import faster_rcnn_meta_arch
from object_detection.models import feature_map_generators
from object_detection.models.keras_models import resnet_v1
from object_detection.models.keras_models import model_utils
from object_detection.utils import ops


class FasterRCNNFPNKerasFeatureExtractor(
    faster_rcnn_meta_arch.FasterRCNNKerasFeatureExtractor):
  """Faster RCNN Feature Extractor using Keras-based Resnet v1  FPN features."""

  def __init__(self, ...):
    # TODO: constructor
    pass

  def build(self, ...):
    # TODO: Build the structure, should be very similar as ssd_*_fpn_keras_feature_extractor.py
    # ResNet-101 (object_detection.models.keras_models)
    # object_detection.models.feature_map_generators
    pass
  
  def preprocess(self, ...):
    # TODO: should be the same as others
    pass
  
  def _extract_proposal_features(self, ...):
    # TODO: Extracts first stage RPN features
    # Fpn_feature_levels 
    pass
  
  def _extract_box_classifier_features(self, ...):
    # TODO: Extracts second stage box classifier features.
    pass
  
  def restore_from_classification_checkpoint_fn(self, ...):
    # follow the none fpn version
    pass
