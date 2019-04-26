# Copyright 2019 The TensorFlow Authors. All Rights Reserved.
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
r"""Generate Visual Wakewords Dataset.

    Helper functions to generate the Visual WakeWords dataset. It filters raw
    COCO annotations file to Visual WakeWords Dataset annotations.
    The resulting annotations and COCO images are then
    converted to TF records.
    See build_visualwakewords_data.py for the sample usage.
"""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import hashlib
import io
import json
import os
import contextlib2

import PIL.Image

import tensorflow as tf

from datasets import dataset_utils

tf.logging.set_verbosity(tf.logging.INFO)


def create_visual_wakeword_annotations(annotations_file,
                                       visualwakewords_annotations_path,
                                       small_object_area_threshold,
                                       foreground_class_of_interest,
                                       visualwakewords_labels_filename):
  """Generate visual wakewords annotations file.

  Loads COCO annotation json files and filters to person/not-person
  class (or user-specified class) to generate visual wakewords annotations file.
  Each image is assigned a label 1 or 0. The label 1 is assigned as long
  as it has at least one foreground_class_of_interest (e.g. person)
  bounding box greater than 5% of the image area.

  Args:
    annotations_file: JSON file containing COCO bounding box annotations
    visualwakewords_annotations_path: output path to annotations file
    small_object_area_threshold: threshold on fraction of image area below which
      small object bounding boxes are filtered
    foreground_class_of_interest: category from COCO dataset that is filtered by
      the visual wakewords dataset
    visualwakewords_labels_filename: The filename to write the visual wakewords
      label file
  """
  # default object of interest is person
  foreground_class_of_interest_id = 1
  with tf.gfile.GFile(annotations_file, 'r') as fid:
    groundtruth_data = json.load(fid)
    images = groundtruth_data['images']
    # Create category index
    category_index = {}
    for category in groundtruth_data['categories']:
      if category['name'] == foreground_class_of_interest:
        foreground_class_of_interest_id = category['id']
        category_index[category['id']] = category

    # Create annotations index
    annotations_index = {}
    annotations_index_filtered = {}
    if 'annotations' in groundtruth_data:
      tf.logging.info(
          'Found groundtruth annotations. Building annotations index.')
      for annotation in groundtruth_data['annotations']:
        image_id = annotation['image_id']
        if image_id not in annotations_index:
          annotations_index[image_id] = []
          annotations_index_filtered[image_id] = []
        annotations_index[image_id].append(annotation)
      missing_annotation_count = 0
      for image in images:
        image_id = image['id']
        if image_id not in annotations_index:
          missing_annotation_count += 1
          annotations_index[image_id] = []
          annotations_index_filtered[image_id] = []
      tf.logging.info('%d images are missing annotations.',
                      missing_annotation_count)
    # Create filtered annotations index
    for idx, image in enumerate(images):
      if idx % 100 == 0:
        tf.logging.info('On image %d of %d', idx, len(images))
      annotations_list = annotations_index[image['id']]
      annotations_list_filtered = _filter_annotations_list(
          annotations_list, image, small_object_area_threshold,
          foreground_class_of_interest_id)
      annotations_index_filtered[image['id']].append(annotations_list_filtered)
    # Output Visual WakeWords annotations and labels
    labels_to_class_names = {0: 'background', 1: foreground_class_of_interest}
    with open(visualwakewords_labels_filename, 'w') as fp:
      for label in labels_to_class_names:
        fp.write(str(label) + ':' + str(labels_to_class_names[label]) + '\n')
    with open(visualwakewords_annotations_path, 'w') as fp:
      json.dump(
          {
              'images': images,
              'annotations': annotations_index_filtered,
              'categories': category_index
          }, fp)


def _filter_annotations_list(annotations_list, image,
                             small_object_area_threshold,
                             foreground_class_of_interest_id):
  """Filters COCO annotations_list to visual wakewords annotations_list.

  Each image is assigned a label 1 or 0. The label 1 is assigned as long
  as it has at least one foreground_class_of_interest (e.g. person)
  bounding box greater than 5% of the image area.

  Args:
    annotations_list: list of dicts with keys: [ u'id', u'image_id',
    u'category_id', u'segmentation', u'area', u'bbox' : [x,y,width,height],
      u'iscrowd']. Notice that bounding box coordinates in the official COCO
      dataset are given as [x, y, width, height] tuples using absolute
      coordinates where x, y represent the top-left (0-indexed) corner.
    image: dict with keys: [u'license', u'file_name', u'coco_url', u'height',
      u'width', u'date_captured', u'flickr_url', u'id']
    small_object_area_threshold: threshold on fraction of image area below which
      small objects are filtered
    foreground_class_of_interest_id: category of COCO dataset which visual
      wakewords filters

  Returns:
    filtered_annotations_list: list of dicts with keys: [ u'image_id',
    u'label', u'category_id', u'count',
    u'object':[{"category_id", "area", "bbox" : [x,y,width,height],}]
  """
  category_ids = []
  area = []
  flag_small_object = []
  num_ann = 0
  image_height = image['height']
  image_width = image['width']
  image_area = image_height * image_width
  bbox = []
  # count of filtered object
  count = 0
  for object_annotations in annotations_list:
    (x, y, width, height) = tuple(object_annotations['bbox'])
    category_id = int(object_annotations['category_id'])
    category_ids.append(category_id)
    obj_area = object_annotations['area']
    normalized_object_area = obj_area / image_area
    # Filter small object bounding boxes
    if category_id == foreground_class_of_interest_id:
      if normalized_object_area < small_object_area_threshold:
        flag_small_object.append(True)
      else:
        flag_small_object.append(False)
        bbox.append({
            u'bbox': [x, y, width, height],
            u'area': obj_area,
            u'category_id': category_id
        })
        count = count + 1
    area.append(obj_area)
    num_ann = num_ann + 1
  # Filtered annotations_list with two classes corresponding to
  # foreground_class_of_interest_id (e.g. person) and
  # background (e.g. not-person)
  if (foreground_class_of_interest_id in category_ids) and (
      False in flag_small_object):
    return {
        u'image_id': image['id'],
        u'label': 1,
        u'object': bbox,
        u'count': count
    }
  else:
    return {u'image_id': image['id'], u'label': 0, u'object': [], u'count': 0}


def create_tf_record_for_visualwakewords_dataset(annotations_file, image_dir,
                                                 output_path, num_shards):
  """Loads Visual WakeWords annotations/images and converts to tf.Record format.

  Args:
    annotations_file: JSON file containing bounding box annotations.
    image_dir: Directory containing the image files.
    output_path: Path to output tf.Record file.
    num_shards: number of output file shards.
  """
  with contextlib2.ExitStack() as tf_record_close_stack, \
      tf.gfile.GFile(annotations_file, 'r') as fid:
    output_tfrecords = dataset_utils.open_sharded_output_tfrecords(
        tf_record_close_stack, output_path, num_shards)
    groundtruth_data = json.load(fid)
    images = groundtruth_data['images']

    category_index = {}
    for category in groundtruth_data['categories'].values():
      # if not background class
      if category['id'] != 0:
        category_index[category['id']] = category

    annotations_index = {}
    if 'annotations' in groundtruth_data:
      tf.logging.info(
          'Found groundtruth annotations. Building annotations index.')
      for annotation in groundtruth_data['annotations'].values():
        image_id = annotation[0]['image_id']
        if image_id not in annotations_index:
          annotations_index[image_id] = []
        annotations_index[image_id].append(annotation[0])
    missing_annotation_count = 0
    for image in images:
      image_id = image['id']
      if image_id not in annotations_index:
        missing_annotation_count += 1
        annotations_index[image_id] = []
    tf.logging.info('%d images are missing annotations.',
                    missing_annotation_count)

    total_num_annotations_skipped = 0
    for idx, image in enumerate(images):
      if idx % 100 == 0:
        tf.logging.info('On image %d of %d', idx, len(images))
      annotations_list = annotations_index[image['id']]
      _, tf_example, num_annotations_skipped = _create_tf_example(
          image, annotations_list[0], image_dir)
      total_num_annotations_skipped += num_annotations_skipped
      shard_idx = idx % num_shards
      output_tfrecords[shard_idx].write(tf_example.SerializeToString())
    tf.logging.info('Finished writing, skipped %d annotations.',
                    total_num_annotations_skipped)


def _create_tf_example(image, annotations_list, image_dir):
  """Converts image and annotations to a tf.Example proto.

  Args:
    image: dict with keys: [u'license', u'file_name', u'coco_url', u'height',
      u'width', u'date_captured', u'flickr_url', u'id']
    annotations_list:
      list of dicts with keys: [u'image_id', u'bbox', u'label',
      object[{"category_id", "area", "bbox" : [x,y,width,height],}]]. Notice
        that bounding box coordinates in the COCO dataset are given as [x, y,
        width, height] tuples using absolute coordinates where x, y represent
        the top-left (0-indexed) corner. This function converts to the format
        that can be used by the Tensorflow Object Detection API (which is [ymin,
        xmin, ymax, xmax] with coordinates normalized relative to image size).
    image_dir: directory containing the image files.

  Returns:
    example: The converted tf.Example
    num_annotations_skipped: Number of (invalid) annotations that were ignored.

  Raises:
    ValueError: if the image pointed to by data['filename'] is not a valid JPEG
  """
  image_height = image['height']
  image_width = image['width']
  filename = image['file_name']
  image_id = image['id']

  full_path = os.path.join(image_dir, filename)
  with tf.gfile.GFile(full_path, 'rb') as fid:
    encoded_jpg = fid.read()
  encoded_jpg_io = io.BytesIO(encoded_jpg)
  image = PIL.Image.open(encoded_jpg_io)
  key = hashlib.sha256(encoded_jpg).hexdigest()

  xmin = []
  xmax = []
  ymin = []
  ymax = []
  category_ids = []
  area = []
  num_annotations_skipped = 0
  label = annotations_list['label']
  for object_annotations in annotations_list['object']:
    (x, y, width, height) = tuple(object_annotations['bbox'])
    if width <= 0 or height <= 0:
      num_annotations_skipped += 1
      continue
    if x + width > image_width or y + height > image_height:
      num_annotations_skipped += 1
      continue
    xmin.append(float(x) / image_width)
    xmax.append(float(x + width) / image_width)
    ymin.append(float(y) / image_height)
    ymax.append(float(y + height) / image_height)
    category_id = int(object_annotations['category_id'])
    category_ids.append(category_id)
    area.append(object_annotations['area'])

  feature_dict = {
      'image/height':
          dataset_utils.int64_feature(image_height),
      'image/width':
          dataset_utils.int64_feature(image_width),
      'image/filename':
          dataset_utils.bytes_feature(filename.encode('utf8')),
      'image/source_id':
          dataset_utils.bytes_feature(str(image_id).encode('utf8')),
      'image/key/sha256':
          dataset_utils.bytes_feature(key.encode('utf8')),
      'image/encoded':
          dataset_utils.bytes_feature(encoded_jpg),
      'image/format':
          dataset_utils.bytes_feature('jpeg'.encode('utf8')),
      'image/class/label':
          dataset_utils.int64_feature(label),
      'image/object/bbox/xmin':
          dataset_utils.float_list_feature(xmin),
      'image/object/bbox/xmax':
          dataset_utils.float_list_feature(xmax),
      'image/object/bbox/ymin':
          dataset_utils.float_list_feature(ymin),
      'image/object/bbox/ymax':
          dataset_utils.float_list_feature(ymax),
      'image/object/class/label':
          dataset_utils.int64_feature(label),
      'image/object/area':
          dataset_utils.float_list_feature(area),
  }
  example = tf.train.Example(features=tf.train.Features(feature=feature_dict))
  return key, example, num_annotations_skipped
