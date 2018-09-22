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

"""Contains classes specifying naming conventions used for object detection.


Specifies:
  InputDataFields: standard fields used by reader/preprocessor/batcher.
  DetectionResultFields: standard fields returned by object detector.
  BoxListFields: standard field used by BoxList
  TfExampleFields: standard fields for tf-example data format (go/tf-example).
"""


class InputDataFields(object):
  """Names for the input tensors.

  Holds the standard data field names to use for identifying input tensors. This
  should be used by the decoder to identify keys for the returned tensor_dict
  containing input tensors. And it should be used by the model to identify the
  tensors it needs.

  Attributes:
    image: image.
    image_additional_channels: additional channels.
    original_image: image in the original input size.
    original_image_spatial_shape: image in the original input size.
    key: unique key corresponding to image.
    source_id: source of the original image.
    filename: original filename of the dataset (without common path).
    groundtruth_image_classes: image-level class labels.
    groundtruth_image_confidences: image-level class confidences.
    groundtruth_boxes: coordinates of the ground truth boxes in the image.
    groundtruth_classes: box-level class labels.
    groundtruth_label_types: box-level label types (e.g. explicit negative).
    groundtruth_is_crowd: [DEPRECATED, use groundtruth_group_of instead]
      is the groundtruth a single object or a crowd.
    groundtruth_area: area of a groundtruth segment.
    groundtruth_difficult: is a `difficult` object
    groundtruth_group_of: is a `group_of` objects, e.g. multiple objects of the
      same class, forming a connected group, where instances are heavily
      occluding each other.
    proposal_boxes: coordinates of object proposal boxes.
    proposal_objectness: objectness score of each proposal.
    groundtruth_instance_masks: ground truth instance masks.
    groundtruth_instance_boundaries: ground truth instance boundaries.
    groundtruth_instance_classes: instance mask-level class labels.
    groundtruth_keypoints: ground truth keypoints.
    groundtruth_keypoint_visibilities: ground truth keypoint visibilities.
    groundtruth_label_scores: groundtruth label scores.
    groundtruth_weights: groundtruth weight factor for bounding boxes.
    num_groundtruth_boxes: number of groundtruth boxes.
    is_annotated: whether an image has been labeled or not.
    true_image_shapes: true shapes of images in the resized images, as resized
      images can be padded with zeros.
    multiclass_scores: the label score per class for each box.
  """
  image = 'image'
  image_additional_channels = 'image_additional_channels'
  original_image = 'original_image'
  original_image_spatial_shape = 'original_image_spatial_shape'
  key = 'key'
  source_id = 'source_id'
  filename = 'filename'
  groundtruth_image_classes = 'groundtruth_image_classes'
  groundtruth_image_confidences = 'groundtruth_image_confidences'
  groundtruth_boxes = 'groundtruth_boxes'
  groundtruth_classes = 'groundtruth_classes'
  groundtruth_confidences = 'groundtruth_confidences'
  groundtruth_label_types = 'groundtruth_label_types'
  groundtruth_is_crowd = 'groundtruth_is_crowd'
  groundtruth_area = 'groundtruth_area'
  groundtruth_difficult = 'groundtruth_difficult'
  groundtruth_group_of = 'groundtruth_group_of'
  proposal_boxes = 'proposal_boxes'
  proposal_objectness = 'proposal_objectness'
  groundtruth_instance_masks = 'groundtruth_instance_masks'
  groundtruth_instance_boundaries = 'groundtruth_instance_boundaries'
  groundtruth_instance_classes = 'groundtruth_instance_classes'
  groundtruth_keypoints = 'groundtruth_keypoints'
  groundtruth_keypoint_visibilities = 'groundtruth_keypoint_visibilities'
  groundtruth_label_scores = 'groundtruth_label_scores'
  groundtruth_weights = 'groundtruth_weights'
  num_groundtruth_boxes = 'num_groundtruth_boxes'
  is_annotated = 'is_annotated'
  true_image_shape = 'true_image_shape'
  multiclass_scores = 'multiclass_scores'


class DetectionResultFields(object):
  """Naming conventions for storing the output of the detector.

  Attributes:
    source_id: source of the original image.
    key: unique key corresponding to image.
    detection_boxes: coordinates of the detection boxes in the image.
    detection_scores: detection scores for the detection boxes in the image.
    detection_classes: detection-level class labels.
    detection_masks: contains a segmentation mask for each detection box.
    detection_boundaries: contains an object boundary for each detection box.
    detection_keypoints: contains detection keypoints for each detection box.
    num_detections: number of detections in the batch.
  """

  source_id = 'source_id'
  key = 'key'
  detection_boxes = 'detection_boxes'
  detection_scores = 'detection_scores'
  detection_classes = 'detection_classes'
  detection_masks = 'detection_masks'
  detection_boundaries = 'detection_boundaries'
  detection_keypoints = 'detection_keypoints'
  num_detections = 'num_detections'


class BoxListFields(object):
  """Naming conventions for BoxLists.

  Attributes:
    boxes: bounding box coordinates.
    classes: classes per bounding box.
    scores: scores per bounding box.
    weights: sample weights per bounding box.
    objectness: objectness score per bounding box.
    masks: masks per bounding box.
    boundaries: boundaries per bounding box.
    keypoints: keypoints per bounding box.
    keypoint_heatmaps: keypoint heatmaps per bounding box.
    is_crowd: is_crowd annotation per bounding box.
  """
  boxes = 'boxes'
  classes = 'classes'
  scores = 'scores'
  weights = 'weights'
  objectness = 'objectness'
  masks = 'masks'
  boundaries = 'boundaries'
  keypoints = 'keypoints'
  keypoint_heatmaps = 'keypoint_heatmaps'
  is_crowd = 'is_crowd'


class TfExampleFields(object):
  """TF-example proto feature names for object detection.

  Holds the standard feature names to load from an Example proto for object
  detection.

  Attributes:
    image_encoded: JPEG encoded string
    image_format: image format, e.g. "JPEG"
    filename: filename
    channels: number of channels of image
    colorspace: colorspace, e.g. "RGB"
    height: height of image in pixels, e.g. 462
    width: width of image in pixels, e.g. 581
    source_id: original source of the image
    image_class_text: image-level label in text format
    image_class_label: image-level label in numerical format
    object_class_text: labels in text format, e.g. ["person", "cat"]
    object_class_label: labels in numbers, e.g. [16, 8]
    object_bbox_xmin: xmin coordinates of groundtruth box, e.g. 10, 30
    object_bbox_xmax: xmax coordinates of groundtruth box, e.g. 50, 40
    object_bbox_ymin: ymin coordinates of groundtruth box, e.g. 40, 50
    object_bbox_ymax: ymax coordinates of groundtruth box, e.g. 80, 70
    object_view: viewpoint of object, e.g. ["frontal", "left"]
    object_truncated: is object truncated, e.g. [true, false]
    object_occluded: is object occluded, e.g. [true, false]
    object_difficult: is object difficult, e.g. [true, false]
    object_group_of: is object a single object or a group of objects
    object_depiction: is object a depiction
    object_is_crowd: [DEPRECATED, use object_group_of instead]
      is the object a single object or a crowd
    object_segment_area: the area of the segment.
    object_weight: a weight factor for the object's bounding box.
    instance_masks: instance segmentation masks.
    instance_boundaries: instance boundaries.
    instance_classes: Classes for each instance segmentation mask.
    detection_class_label: class label in numbers.
    detection_bbox_ymin: ymin coordinates of a detection box.
    detection_bbox_xmin: xmin coordinates of a detection box.
    detection_bbox_ymax: ymax coordinates of a detection box.
    detection_bbox_xmax: xmax coordinates of a detection box.
    detection_score: detection score for the class label and box.
  """
  image_encoded = 'image/encoded'
  image_format = 'image/format'  # format is reserved keyword
  filename = 'image/filename'
  channels = 'image/channels'
  colorspace = 'image/colorspace'
  height = 'image/height'
  width = 'image/width'
  source_id = 'image/source_id'
  image_class_text = 'image/class/text'
  image_class_label = 'image/class/label'
  object_class_text = 'image/object/class/text'
  object_class_label = 'image/object/class/label'
  object_bbox_ymin = 'image/object/bbox/ymin'
  object_bbox_xmin = 'image/object/bbox/xmin'
  object_bbox_ymax = 'image/object/bbox/ymax'
  object_bbox_xmax = 'image/object/bbox/xmax'
  object_view = 'image/object/view'
  object_truncated = 'image/object/truncated'
  object_occluded = 'image/object/occluded'
  object_difficult = 'image/object/difficult'
  object_group_of = 'image/object/group_of'
  object_depiction = 'image/object/depiction'
  object_is_crowd = 'image/object/is_crowd'
  object_segment_area = 'image/object/segment/area'
  object_weight = 'image/object/weight'
  instance_masks = 'image/segmentation/object'
  instance_boundaries = 'image/boundaries/object'
  instance_classes = 'image/segmentation/object/class'
  detection_class_label = 'image/detection/label'
  detection_bbox_ymin = 'image/detection/bbox/ymin'
  detection_bbox_xmin = 'image/detection/bbox/xmin'
  detection_bbox_ymax = 'image/detection/bbox/ymax'
  detection_bbox_xmax = 'image/detection/bbox/xmax'
  detection_score = 'image/detection/score'
