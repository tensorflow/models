# Copyright 2021 The TensorFlow Authors. All Rights Reserved.
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

"""Util functions related to pycocotools and COCO eval."""

import copy
import json

# Import libraries
from absl import logging
import numpy as np
from PIL import Image
from pycocotools import coco
from pycocotools import mask as mask_api
import six
import tensorflow as tf

from official.vision.beta.dataloaders import tf_example_decoder
from official.vision.beta.ops import box_ops
from official.vision.beta.ops import mask_ops


class COCOWrapper(coco.COCO):
  """COCO wrapper class.

  This class wraps COCO API object, which provides the following additional
  functionalities:
    1. Support string type image id.
    2. Support loading the groundtruth dataset using the external annotation
       dictionary.
    3. Support loading the prediction results using the external annotation
       dictionary.
  """

  def __init__(self, eval_type='box', annotation_file=None, gt_dataset=None):
    """Instantiates a COCO-style API object.

    Args:
      eval_type: either 'box' or 'mask'.
      annotation_file: a JSON file that stores annotations of the eval dataset.
        This is required if `gt_dataset` is not provided.
      gt_dataset: the groundtruth eval datatset in COCO API format.
    """
    if ((annotation_file and gt_dataset) or
        ((not annotation_file) and (not gt_dataset))):
      raise ValueError('One and only one of `annotation_file` and `gt_dataset` '
                       'needs to be specified.')

    if eval_type not in ['box', 'mask']:
      raise ValueError('The `eval_type` can only be either `box` or `mask`.')

    coco.COCO.__init__(self, annotation_file=annotation_file)
    self._eval_type = eval_type
    if gt_dataset:
      self.dataset = gt_dataset
      self.createIndex()

  def loadRes(self, predictions):
    """Loads result file and return a result api object.

    Args:
      predictions: a list of dictionary each representing an annotation in COCO
        format. The required fields are `image_id`, `category_id`, `score`,
        `bbox`, `segmentation`.

    Returns:
      res: result COCO api object.

    Raises:
      ValueError: if the set of image id from predctions is not the subset of
        the set of image id of the groundtruth dataset.
    """
    res = coco.COCO()
    res.dataset['images'] = copy.deepcopy(self.dataset['images'])
    res.dataset['categories'] = copy.deepcopy(self.dataset['categories'])

    image_ids = [ann['image_id'] for ann in predictions]
    if set(image_ids) != (set(image_ids) & set(self.getImgIds())):
      raise ValueError('Results do not correspond to the current dataset!')
    for ann in predictions:
      x1, x2, y1, y2 = [ann['bbox'][0], ann['bbox'][0] + ann['bbox'][2],
                        ann['bbox'][1], ann['bbox'][1] + ann['bbox'][3]]
      if self._eval_type == 'box':
        ann['area'] = ann['bbox'][2] * ann['bbox'][3]
        ann['segmentation'] = [
            [x1, y1, x1, y2, x2, y2, x2, y1]]
      elif self._eval_type == 'mask':
        ann['area'] = mask_api.area(ann['segmentation'])

    res.dataset['annotations'] = copy.deepcopy(predictions)
    res.createIndex()
    return res


def convert_predictions_to_coco_annotations(predictions):
  """Converts a batch of predictions to annotations in COCO format.

  Args:
    predictions: a dictionary of lists of numpy arrays including the following
      fields. K below denotes the maximum number of instances per image.
      Required fields:
        - source_id: a list of numpy arrays of int or string of shape
            [batch_size].
        - num_detections: a list of numpy arrays of int of shape [batch_size].
        - detection_boxes: a list of numpy arrays of float of shape
            [batch_size, K, 4], where coordinates are in the original image
            space (not the scaled image space).
        - detection_classes: a list of numpy arrays of int of shape
            [batch_size, K].
        - detection_scores: a list of numpy arrays of float of shape
            [batch_size, K].
      Optional fields:
        - detection_masks: a list of numpy arrays of float of shape
            [batch_size, K, mask_height, mask_width].

  Returns:
    coco_predictions: prediction in COCO annotation format.
  """
  coco_predictions = []
  num_batches = len(predictions['source_id'])
  batch_size = predictions['source_id'][0].shape[0]
  max_num_detections = predictions['detection_classes'][0].shape[1]
  use_outer_box = 'detection_outer_boxes' in predictions
  for i in range(num_batches):
    predictions['detection_boxes'][i] = box_ops.yxyx_to_xywh(
        predictions['detection_boxes'][i])
    if use_outer_box:
      predictions['detection_outer_boxes'][i] = box_ops.yxyx_to_xywh(
          predictions['detection_outer_boxes'][i])
      mask_boxes = predictions['detection_outer_boxes']
    else:
      mask_boxes = predictions['detection_boxes']

    for j in range(batch_size):
      if 'detection_masks' in predictions:
        image_masks = mask_ops.paste_instance_masks(
            predictions['detection_masks'][i][j],
            mask_boxes[i][j],
            int(predictions['image_info'][i][j, 0, 0]),
            int(predictions['image_info'][i][j, 0, 1]))
        binary_masks = (image_masks > 0.0).astype(np.uint8)
        encoded_masks = [
            mask_api.encode(np.asfortranarray(binary_mask))
            for binary_mask in list(binary_masks)]
      for k in range(max_num_detections):
        ann = {}
        ann['image_id'] = predictions['source_id'][i][j]
        ann['category_id'] = predictions['detection_classes'][i][j, k]
        ann['bbox'] = predictions['detection_boxes'][i][j, k]
        ann['score'] = predictions['detection_scores'][i][j, k]
        if 'detection_masks' in predictions:
          ann['segmentation'] = encoded_masks[k]
        coco_predictions.append(ann)

  for i, ann in enumerate(coco_predictions):
    ann['id'] = i + 1

  return coco_predictions


def convert_groundtruths_to_coco_dataset(groundtruths, label_map=None):
  """Converts groundtruths to the dataset in COCO format.

  Args:
    groundtruths: a dictionary of numpy arrays including the fields below.
      Note that each element in the list represent the number for a single
      example without batch dimension. K below denotes the actual number of
      instances for each image.
      Required fields:
        - source_id: a list of numpy arrays of int or string of shape
          [batch_size].
        - height: a list of numpy arrays of int of shape [batch_size].
        - width: a list of numpy arrays of int of shape [batch_size].
        - num_detections: a list of numpy arrays of int of shape [batch_size].
        - boxes: a list of numpy arrays of float of shape [batch_size, K, 4],
            where coordinates are in the original image space (not the
            normalized coordinates).
        - classes: a list of numpy arrays of int of shape [batch_size, K].
      Optional fields:
        - is_crowds: a list of numpy arrays of int of shape [batch_size, K]. If
            th field is absent, it is assumed that this instance is not crowd.
        - areas: a list of numy arrays of float of shape [batch_size, K]. If the
            field is absent, the area is calculated using either boxes or
            masks depending on which one is available.
        - masks: a list of numpy arrays of string of shape [batch_size, K],
    label_map: (optional) a dictionary that defines items from the category id
      to the category name. If `None`, collect the category mappping from the
      `groundtruths`.

  Returns:
    coco_groundtruths: the groundtruth dataset in COCO format.
  """
  source_ids = np.concatenate(groundtruths['source_id'], axis=0)
  heights = np.concatenate(groundtruths['height'], axis=0)
  widths = np.concatenate(groundtruths['width'], axis=0)
  gt_images = [{'id': int(i), 'height': int(h), 'width': int(w)} for i, h, w
               in zip(source_ids, heights, widths)]

  gt_annotations = []
  num_batches = len(groundtruths['source_id'])
  batch_size = groundtruths['source_id'][0].shape[0]
  for i in range(num_batches):
    max_num_instances = groundtruths['classes'][i].shape[1]
    for j in range(batch_size):
      num_instances = groundtruths['num_detections'][i][j]
      if num_instances > max_num_instances:
        logging.warning(
            'num_groundtruths is larger than max_num_instances, %d v.s. %d',
            num_instances, max_num_instances)
        num_instances = max_num_instances
      for k in range(int(num_instances)):
        ann = {}
        ann['image_id'] = int(groundtruths['source_id'][i][j])
        if 'is_crowds' in groundtruths:
          ann['iscrowd'] = int(groundtruths['is_crowds'][i][j, k])
        else:
          ann['iscrowd'] = 0
        ann['category_id'] = int(groundtruths['classes'][i][j, k])
        boxes = groundtruths['boxes'][i]
        ann['bbox'] = [
            float(boxes[j, k, 1]),
            float(boxes[j, k, 0]),
            float(boxes[j, k, 3] - boxes[j, k, 1]),
            float(boxes[j, k, 2] - boxes[j, k, 0])]
        if 'areas' in groundtruths:
          ann['area'] = float(groundtruths['areas'][i][j, k])
        else:
          ann['area'] = float(
              (boxes[j, k, 3] - boxes[j, k, 1]) *
              (boxes[j, k, 2] - boxes[j, k, 0]))
        if 'masks' in groundtruths:
          mask = Image.open(six.BytesIO(groundtruths['masks'][i][j, k]))
          width, height = mask.size
          np_mask = (
              np.array(mask.getdata()).reshape(height, width).astype(np.uint8))
          np_mask[np_mask > 0] = 255
          encoded_mask = mask_api.encode(np.asfortranarray(np_mask))
          ann['segmentation'] = encoded_mask
          if 'areas' not in groundtruths:
            ann['area'] = mask_api.area(encoded_mask)
        gt_annotations.append(ann)

  for i, ann in enumerate(gt_annotations):
    ann['id'] = i + 1

  if label_map:
    gt_categories = [{'id': i, 'name': label_map[i]} for i in label_map]
  else:
    category_ids = [gt['category_id'] for gt in gt_annotations]
    gt_categories = [{'id': i} for i in set(category_ids)]

  gt_dataset = {
      'images': gt_images,
      'categories': gt_categories,
      'annotations': copy.deepcopy(gt_annotations),
  }
  return gt_dataset


class COCOGroundtruthGenerator:
  """Generates the groundtruth annotations from a single example."""

  def __init__(self, file_pattern, num_examples, include_mask):
    self._file_pattern = file_pattern
    self._num_examples = num_examples
    self._include_mask = include_mask
    self._dataset_fn = tf.data.TFRecordDataset

  def _parse_single_example(self, example):
    """Parses a single serialized tf.Example proto.

    Args:
      example: a serialized tf.Example proto string.

    Returns:
      A dictionary of groundtruth with the following fields:
        source_id: a scalar tensor of int64 representing the image source_id.
        height: a scalar tensor of int64 representing the image height.
        width: a scalar tensor of int64 representing the image width.
        boxes: a float tensor of shape [K, 4], representing the groundtruth
          boxes in absolute coordinates with respect to the original image size.
        classes: a int64 tensor of shape [K], representing the class labels of
          each instances.
        is_crowds: a bool tensor of shape [K], indicating whether the instance
          is crowd.
        areas: a float tensor of shape [K], indicating the area of each
          instance.
        masks: a string tensor of shape [K], containing the bytes of the png
          mask of each instance.
    """
    decoder = tf_example_decoder.TfExampleDecoder(
        include_mask=self._include_mask)
    decoded_tensors = decoder.decode(example)

    image = decoded_tensors['image']
    image_size = tf.shape(image)[0:2]
    boxes = box_ops.denormalize_boxes(
        decoded_tensors['groundtruth_boxes'], image_size)
    groundtruths = {
        'source_id': tf.string_to_number(
            decoded_tensors['source_id'], out_type=tf.int64),
        'height': decoded_tensors['height'],
        'width': decoded_tensors['width'],
        'num_detections': tf.shape(decoded_tensors['groundtruth_classes'])[0],
        'boxes': boxes,
        'classes': decoded_tensors['groundtruth_classes'],
        'is_crowds': decoded_tensors['groundtruth_is_crowd'],
        'areas': decoded_tensors['groundtruth_area'],
    }
    if self._include_mask:
      groundtruths.update({
          'masks': decoded_tensors['groundtruth_instance_masks_png'],
      })
    return groundtruths

  def _build_pipeline(self):
    """Builds data pipeline to generate groundtruth annotations."""
    dataset = tf.data.Dataset.list_files(self._file_pattern, shuffle=False)
    dataset = dataset.interleave(
        map_func=lambda filename: self._dataset_fn(filename).prefetch(1),
        cycle_length=12,
        num_parallel_calls=tf.data.experimental.AUTOTUNE)

    dataset = dataset.map(self._parse_single_example,
                          num_parallel_calls=tf.data.experimental.AUTOTUNE)
    dataset = dataset.batch(1, drop_remainder=False)
    dataset = dataset.prefetch(tf.data.experimental.AUTOTUNE)
    return dataset

  def __call__(self):
    for groundtruth_result in self._build_pipeline():
      yield groundtruth_result


def scan_and_generator_annotation_file(file_pattern: str,
                                       num_samples: int,
                                       include_mask: bool,
                                       annotation_file: str):
  """Scans and generate the COCO-style annotation JSON file given a dataset."""
  groundtruth_generator = COCOGroundtruthGenerator(
      file_pattern, num_samples, include_mask)
  generate_annotation_file(groundtruth_generator, annotation_file)


def generate_annotation_file(groundtruth_generator,
                             annotation_file):
  """Generates COCO-style annotation JSON file given a groundtruth generator."""
  groundtruths = {}
  logging.info('Loading groundtruth annotations from dataset to memory...')
  for groundtruth in groundtruth_generator():
    for k, v in six.iteritems(groundtruth):
      if k not in groundtruths:
        groundtruths[k] = [v]
      else:
        groundtruths[k].append(v)
  gt_dataset = convert_groundtruths_to_coco_dataset(groundtruths)

  logging.info('Saving groundtruth annotations to the JSON file...')
  with tf.io.gfile.GFile(annotation_file, 'w') as f:
    f.write(json.dumps(gt_dataset))
  logging.info('Done saving the JSON file...')
