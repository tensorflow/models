# Copyright 2024 The TensorFlow Authors. All Rights Reserved.
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

"""Data classes for tf.Example proto feature keys in vision tasks.

Feature keys are grouped by feature types. Key names follow conventions in
go/tf-example.
"""
import dataclasses
import functools

from official.core import tf_example_feature_key

# Disable init function to use the one defined in base class.
dataclass = functools.partial(dataclasses.dataclass(init=False))


@dataclass
class EncodedImageFeatureKey(tf_example_feature_key.TfExampleFeatureKeyBase):
  """Feature keys for a single encoded image.

  The image matrix is expected to be in the shape of (height, width,
  num_channels).

  Attributes:
      encoded: encoded image bytes.
      format: format string, e.g. 'PNG'.
      height: number of rows.
      width: number of columns.
      num_channels: number of channels.
      source_id: Unique string ID to identify the image.
      label: the label or a list of labels for the image.
  """
  encoded: str = 'image/encoded'
  format: str = 'image/format'
  height: str = 'image/height'
  width: str = 'image/width'
  num_channels: str = 'image/channels'
  source_id: str = 'image/source_id'
  label: str = 'image/class/label'


@dataclass
class BoxFeatureKey(tf_example_feature_key.TfExampleFeatureKeyBase):
  """Feature keys for normalized boxes representing objects in an image.

  Each box is defined by ((ymin, xmin), (ymax, xmax)).

  The origin point of an image matrix is top left.

  Note: The coordinate values are normalized to [0, 1], this is commonly adopted
  by most model implementations.

  Attributes:
      xmin: The x coordinate (column) of top-left corner.
      xmax: The x coordinate (column) of bottom-right corner.
      ymin: The y coordinate (row) of top-left corner.
      ymax: The y coordinate (row) of bottom-right corner.
      label: The class id.
      confidence: The confidence score of the box, could be prior score (for
        training) or predicted score (for prediction).
  """
  xmin: str = 'image/object/bbox/xmin'
  xmax: str = 'image/object/bbox/xmax'
  ymin: str = 'image/object/bbox/ymin'
  ymax: str = 'image/object/bbox/ymax'
  label: str = 'image/object/class/label'
  confidence: str = 'image/object/bbox/confidence'


@dataclass
class BoxPixelFeatureKey(tf_example_feature_key.TfExampleFeatureKeyBase):
  """Feature keys for boxes in pixel values representing objects in an image.

  Each box is defined by ((ymin, xmin), (ymax, xmax)).

  Note: The coordinate values are in the scale of the context image. The image
  size is usually stored in `EncodedImageFeatureKey`.

  Attributes:
      xmin: The x coordinate (column) of top-left corner.
      xmax: The x coordinate (column) of bottom-right corner.
      ymin: The y coordinate (row) of top-left corner.
      ymax: The y coordinate (row) of bottom-right corner.
      label: The class id.
      confidence: The confidence score of the box, could be prior score (for
        training) or predicted score (for prediction).
  """
  xmin: str = 'image/object/bbox/xmin_pixels'
  xmax: str = 'image/object/bbox/xmax_pixels'
  ymin: str = 'image/object/bbox/ymin_pixels'
  ymax: str = 'image/object/bbox/ymax_pixels'
  label: str = 'image/object/class/label'
  confidence: str = 'image/object/bbox/confidence'


@dataclass
class EncodedInstanceMaskFeatureKey(
    tf_example_feature_key.TfExampleFeatureKeyBase):
  """Feature keys for a single encoded instance mask.

  The instance mask matrices are expected to be in the shape of (num_instances,
  height, width, 1) or (num_instance, height, width). The height and width
  correspond to the image height and width. For each instance mask, the pixel
  value is either 0, representing a background, or 1, representing the object.

  TODO(b/223653024): Add keys for visualization mask as well.

  Attributes:
      mask: Encoded instance mask bytes.
      area: Total number of pixels that are marked as objects.
  """
  mask: str = 'image/object/mask'
  area: str = 'image/object/area'


@dataclass
class EncodedSemanticMaskFeatureKey(
    tf_example_feature_key.TfExampleFeatureKeyBase):
  """Feature keys for a encoded semantic mask and its associated images.

  The semantic mask matrix is expected to be in the shape of (height, width, 1)
  or (height, width). The visualization mask matrix is expected to be in the
  shape of (height, width, 3). The height and width correspond to the image
  height and width. Each pixel in the semantic mask respresents a class.

  Attributes:
      mask: Encoded semantic mask bytes.
      mask_format: Format string for semantic mask, e.g. 'PNG'.
      visualization_mask: Encoded visualization mask bytes.
      visualization_mask_format: Format string for visualization mask, e.g.
        'PNG'.
  """
  mask: str = 'image/segmentation/class/encoded'
  mask_format: str = 'image/segmentation/class/format'
  visualization_mask: str = 'image/segmentation/class/visualization/encoded'
  visualization_mask_format: str = 'image/segmentation/class/visualization/format'


@dataclass
class EncodedPanopticMaskFeatureKey(
    tf_example_feature_key.TfExampleFeatureKeyBase):
  """Feature keys for encoded panoptic category and instance masks.

  Both panoptic mask matrices are expected to be in the shape of (height, width,
  1) or (height, width). The height and width correspond to the image height and
  width. For category mask, each pixel represents a class ID, and for instance
  mask, each pixel represents an instance ID.

  TODO(b/223653024): Add keys for visualization mask as well.

  Attributes:
      category_mask: Encoded panoptic category mask bytes.
      category_mask_format: Format string for panoptic category mask, e.g.
        'PNG'.
      instance_mask: Encoded panoptic instance mask bytes.
      instance_mask_format: Format string for panoptic instance mask, e.g.
        'PNG'.
  """
  category_mask: str = 'image/panoptic/category/encoded'
  category_mask_format: str = 'image/panoptic/category/format'
  instance_mask: str = 'image/panoptic/instance/encoded'
  instance_mask_format: str = 'image/panoptic/instance/format'
