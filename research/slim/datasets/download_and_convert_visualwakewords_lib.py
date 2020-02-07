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
r"""Helper functions to generate the Visual WakeWords dataset.

    It filters raw COCO annotations file to Visual WakeWords Dataset
    annotations. The resulting annotations and COCO images are then converted
    to TF records.
    See download_and_convert_visualwakewords.py for the sample usage.
"""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import collections
import hashlib
import io
import json
import os
import contextlib2

import PIL.Image

import tensorflow as tf

from datasets import dataset_utils

tf.compat.v1.logging.set_verbosity(tf.compat.v1.logging.INFO)

tf.compat.v1.app.flags.DEFINE_string(
    'coco_train_url',
    'http://images.cocodataset.org/zips/train2014.zip',
    'Link to zip file containing coco training data')
tf.compat.v1.app.flags.DEFINE_string(
    'coco_validation_url',
    'http://images.cocodataset.org/zips/val2014.zip',
    'Link to zip file containing coco validation data')
tf.compat.v1.app.flags.DEFINE_string(
    'coco_annotations_url',
    'http://images.cocodataset.org/annotations/annotations_trainval2014.zip',
    'Link to zip file containing coco annotation data')

FLAGS = tf.compat.v1.app.flags.FLAGS


def download_coco_dataset(dataset_dir):
  """Download the coco dataset.

  Args:
    dataset_dir: Path where coco dataset should be downloaded.
  """
  dataset_utils.download_and_uncompress_zipfile(FLAGS.coco_train_url,
                                                dataset_dir)
  dataset_utils.download_and_uncompress_zipfile(FLAGS.coco_validation_url,
                                                dataset_dir)
  dataset_utils.download_and_uncompress_zipfile(FLAGS.coco_annotations_url,
                                                dataset_dir)


def create_labels_file(foreground_class_of_interest,
                       visualwakewords_labels_file):
  """Generate visualwakewords labels file.

  Args:
    foreground_class_of_interest: category from COCO dataset that is filtered by
      the visualwakewords dataset
    visualwakewords_labels_file: output visualwakewords label file
  """
  labels_to_class_names = {0: 'background', 1: foreground_class_of_interest}
  with open(visualwakewords_labels_file, 'w') as fp:
    for label in labels_to_class_names:
      fp.write(str(label) + ':' + str(labels_to_class_names[label]) + '\n')


def create_visual_wakeword_annotations(annotations_file,
                                       visualwakewords_annotations_file,
                                       small_object_area_threshold,
                                       foreground_class_of_interest):
  """Generate visual wakewords annotations file.

  Loads COCO annotation json files to generate visualwakewords annotations file.

  Args:
    annotations_file: JSON file containing COCO bounding box annotations
    visualwakewords_annotations_file: path to output annotations file
    small_object_area_threshold: threshold on fraction of image area below which
      small object bounding boxes are filtered
    foreground_class_of_interest: category from COCO dataset that is filtered by
      the visual wakewords dataset
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
    # Create annotations index, a map of image_id to it's annotations
    tf.logging.info('Building annotations index...')
    annotations_index = collections.defaultdict(
        lambda: collections.defaultdict(list))
    # structure is { "image_id": {"objects" : [list of the image annotations]}}
    for annotation in groundtruth_data['annotations']:
      annotations_index[annotation['image_id']]['objects'].append(annotation)
    missing_annotation_count = len(images) - len(annotations_index)
    tf.logging.info('%d images are missing annotations.',
                    missing_annotation_count)
    # Create filtered annotations index
    annotations_index_filtered = {}
    for idx, image in enumerate(images):
      if idx % 100 == 0:
        tf.logging.info('On image %d of %d', idx, len(images))
      annotations = annotations_index[image['id']]
      annotations_filtered = _filter_annotations(
          annotations, image, small_object_area_threshold,
          foreground_class_of_interest_id)
      annotations_index_filtered[image['id']] = annotations_filtered

    with open(visualwakewords_annotations_file, 'w') as fp:
      json.dump(
          {
              'images': images,
              'annotations': annotations_index_filtered,
              'categories': category_index
          }, fp)


def _filter_annotations(annotations, image, small_object_area_threshold,
                        foreground_class_of_interest_id):
  """Filters COCO annotations to visual wakewords annotations.

  Args:
    annotations: dicts with keys: {
      u'objects': [{u'id', u'image_id', u'category_id', u'segmentation',
                  u'area', u'bbox' : [x,y,width,height], u'iscrowd'}] } Notice
                    that bounding box coordinates in the official COCO dataset
                    are given as [x, y, width, height] tuples using absolute
                    coordinates where x, y represent the top-left (0-indexed)
                    corner.
    image: dict with keys: [u'license', u'file_name', u'coco_url', u'height',
      u'width', u'date_captured', u'flickr_url', u'id']
    small_object_area_threshold: threshold on fraction of image area below which
      small objects are filtered
    foreground_class_of_interest_id: category of COCO dataset which visual
      wakewords filters

  Returns:
    annotations_filtered: dict with keys: {
      u'objects': [{"area", "bbox" : [x,y,width,height]}],
      u'label',
      }
  """
  objects = []
  image_area = image['height'] * image['width']
  for annotation in annotations['objects']:
    normalized_object_area = annotation['area'] / image_area
    category_id = int(annotation['category_id'])
    # Filter valid bounding boxes
    if category_id == foreground_class_of_interest_id and \
        normalized_object_area > small_object_area_threshold:
      objects.append({
          u'area': annotation['area'],
          u'bbox': annotation['bbox'],
      })
  label = 1 if objects else 0
  return {
      'objects': objects,
      'label': label,
  }


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
    annotations_index = groundtruth_data['annotations']
    annotations_index = {int(k): v for k, v in annotations_index.iteritems()}
    # convert 'unicode' key to 'int' key after we parse the json file

    for idx, image in enumerate(images):
      if idx % 100 == 0:
        tf.logging.info('On image %d of %d', idx, len(images))
      annotations = annotations_index[image['id']]
      tf_example = _create_tf_example(image, annotations, image_dir)
      shard_idx = idx % num_shards
      output_tfrecords[shard_idx].write(tf_example.SerializeToString())


def _create_tf_example(image, annotations, image_dir):
  """Converts image and annotations to a tf.Example proto.

  Args:
    image: dict with keys: [u'license', u'file_name', u'coco_url', u'height',
      u'width', u'date_captured', u'flickr_url', u'id']
    annotations: dict with objects (a list of image annotations) and a label.
      {u'objects':[{"area", "bbox" : [x,y,width,height}], u'label'}. Notice
      that bounding box coordinates in the COCO dataset are given as[x, y,
      width, height] tuples using absolute coordinates where x, y represent
      the top-left (0-indexed) corner. This function also converts to the format
      that can be used by the Tensorflow Object Detection API (which is [ymin,
      xmin, ymax, xmax] with coordinates normalized relative to image size).
    image_dir: directory containing the image files.
  Returns:
    tf_example: The converted tf.Example

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

  xmin, xmax, ymin, ymax, area = [], [], [], [], []
  for obj in annotations['objects']:
    (x, y, width, height) = tuple(obj['bbox'])
    xmin.append(float(x) / image_width)
    xmax.append(float(x + width) / image_width)
    ymin.append(float(y) / image_height)
    ymax.append(float(y + height) / image_height)
    area.append(obj['area'])

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
          dataset_utils.int64_feature(annotations['label']),
      'image/object/bbox/xmin':
          dataset_utils.float_list_feature(xmin),
      'image/object/bbox/xmax':
          dataset_utils.float_list_feature(xmax),
      'image/object/bbox/ymin':
          dataset_utils.float_list_feature(ymin),
      'image/object/bbox/ymax':
          dataset_utils.float_list_feature(ymax),
      'image/object/area':
          dataset_utils.float_list_feature(area),
  }
  example = tf.train.Example(features=tf.train.Features(feature=feature_dict))
  return example
