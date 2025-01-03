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
r"""Convert raw COCO dataset to TFRecord for object_detection.

This tool supports data generation for object detection (boxes, masks),
keypoint detection, and DensePose.

Please note that this tool creates sharded output files.

Example usage:
    python create_coco_tf_record.py --logtostderr \
      --train_image_dir="${TRAIN_IMAGE_DIR}" \
      --val_image_dir="${VAL_IMAGE_DIR}" \
      --test_image_dir="${TEST_IMAGE_DIR}" \
      --train_annotations_file="${TRAIN_ANNOTATIONS_FILE}" \
      --val_annotations_file="${VAL_ANNOTATIONS_FILE}" \
      --testdev_annotations_file="${TESTDEV_ANNOTATIONS_FILE}" \
      --output_dir="${OUTPUT_DIR}"
"""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import hashlib
import io
import json
import logging
import os
import contextlib2
import numpy as np
import PIL.Image

from pycocotools import mask
import tensorflow.compat.v1 as tf

from object_detection.dataset_tools import tf_record_creation_util
from object_detection.utils import dataset_util
from object_detection.utils import label_map_util

flags = tf.app.flags
tf.flags.DEFINE_boolean(
    'include_masks', False, 'Whether to include instance segmentations masks '
    '(PNG encoded) in the result. default: False.')
tf.flags.DEFINE_string('train_image_dir', '', 'Training image directory.')
tf.flags.DEFINE_string('val_image_dir', '', 'Validation image directory.')
tf.flags.DEFINE_string('test_image_dir', '', 'Test image directory.')
tf.flags.DEFINE_string('train_annotations_file', '',
                       'Training annotations JSON file.')
tf.flags.DEFINE_string('val_annotations_file', '',
                       'Validation annotations JSON file.')
tf.flags.DEFINE_string('testdev_annotations_file', '',
                       'Test-dev annotations JSON file.')
tf.flags.DEFINE_string('train_keypoint_annotations_file', '',
                       'Training annotations JSON file.')
tf.flags.DEFINE_string('val_keypoint_annotations_file', '',
                       'Validation annotations JSON file.')
# DensePose is only available for coco 2014.
tf.flags.DEFINE_string('train_densepose_annotations_file', '',
                       'Training annotations JSON file for DensePose.')
tf.flags.DEFINE_string('val_densepose_annotations_file', '',
                       'Validation annotations JSON file for DensePose.')
tf.flags.DEFINE_string('output_dir', '/tmp/', 'Output data directory.')
# Whether to only produce images/annotations on person class (for keypoint /
# densepose task).
tf.flags.DEFINE_boolean('remove_non_person_annotations', False, 'Whether to '
                        'remove all annotations for non-person objects.')
tf.flags.DEFINE_boolean('remove_non_person_images', False, 'Whether to '
                        'remove all examples that do not contain a person.')

FLAGS = flags.FLAGS

logger = tf.get_logger()
logger.setLevel(logging.INFO)

_COCO_KEYPOINT_NAMES = [
    b'nose', b'left_eye', b'right_eye', b'left_ear', b'right_ear',
    b'left_shoulder', b'right_shoulder', b'left_elbow', b'right_elbow',
    b'left_wrist', b'right_wrist', b'left_hip', b'right_hip',
    b'left_knee', b'right_knee', b'left_ankle', b'right_ankle'
]

_COCO_PART_NAMES = [
    b'torso_back', b'torso_front', b'right_hand', b'left_hand', b'left_foot',
    b'right_foot', b'right_upper_leg_back', b'left_upper_leg_back',
    b'right_upper_leg_front', b'left_upper_leg_front', b'right_lower_leg_back',
    b'left_lower_leg_back', b'right_lower_leg_front', b'left_lower_leg_front',
    b'left_upper_arm_back', b'right_upper_arm_back', b'left_upper_arm_front',
    b'right_upper_arm_front', b'left_lower_arm_back', b'right_lower_arm_back',
    b'left_lower_arm_front', b'right_lower_arm_front', b'right_face',
    b'left_face',
]

_DP_PART_ID_OFFSET = 1


def clip_to_unit(x):
  return min(max(x, 0.0), 1.0)


def create_tf_example(image,
                      annotations_list,
                      image_dir,
                      category_index,
                      include_masks=False,
                      keypoint_annotations_dict=None,
                      densepose_annotations_dict=None,
                      remove_non_person_annotations=False,
                      remove_non_person_images=False):
  """Converts image and annotations to a tf.Example proto.

  Args:
    image: dict with keys: [u'license', u'file_name', u'coco_url', u'height',
      u'width', u'date_captured', u'flickr_url', u'id']
    annotations_list:
      list of dicts with keys: [u'segmentation', u'area', u'iscrowd',
        u'image_id', u'bbox', u'category_id', u'id'] Notice that bounding box
        coordinates in the official COCO dataset are given as [x, y, width,
        height] tuples using absolute coordinates where x, y represent the
        top-left (0-indexed) corner.  This function converts to the format
        expected by the Tensorflow Object Detection API (which is which is
        [ymin, xmin, ymax, xmax] with coordinates normalized relative to image
        size).
    image_dir: directory containing the image files.
    category_index: a dict containing COCO category information keyed by the
      'id' field of each category.  See the label_map_util.create_category_index
      function.
    include_masks: Whether to include instance segmentations masks
      (PNG encoded) in the result. default: False.
    keypoint_annotations_dict: A dictionary that maps from annotation_id to a
      dictionary with keys: [u'keypoints', u'num_keypoints'] represeting the
      keypoint information for this person object annotation. If None, then
      no keypoint annotations will be populated.
    densepose_annotations_dict: A dictionary that maps from annotation_id to a
      dictionary with keys: [u'dp_I', u'dp_x', u'dp_y', 'dp_U', 'dp_V']
      representing part surface coordinates. For more information see
      http://densepose.org/.
    remove_non_person_annotations: Whether to remove any annotations that are
      not the "person" class.
    remove_non_person_images: Whether to remove any images that do not contain
      at least one "person" annotation.

  Returns:
    key: SHA256 hash of the image.
    example: The converted tf.Example
    num_annotations_skipped: Number of (invalid) annotations that were ignored.
    num_keypoint_annotation_skipped: Number of keypoint annotations that were
      skipped.
    num_densepose_annotation_skipped: Number of DensePose annotations that were
      skipped.

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
  is_crowd = []
  category_names = []
  category_ids = []
  area = []
  encoded_mask_png = []
  keypoints_x = []
  keypoints_y = []
  keypoints_visibility = []
  keypoints_name = []
  num_keypoints = []
  include_keypoint = keypoint_annotations_dict is not None
  num_annotations_skipped = 0
  num_keypoint_annotation_used = 0
  num_keypoint_annotation_skipped = 0
  dp_part_index = []
  dp_x = []
  dp_y = []
  dp_u = []
  dp_v = []
  dp_num_points = []
  densepose_keys = ['dp_I', 'dp_U', 'dp_V', 'dp_x', 'dp_y', 'bbox']
  include_densepose = densepose_annotations_dict is not None
  num_densepose_annotation_used = 0
  num_densepose_annotation_skipped = 0
  for object_annotations in annotations_list:
    (x, y, width, height) = tuple(object_annotations['bbox'])
    if width <= 0 or height <= 0:
      num_annotations_skipped += 1
      continue
    if x + width > image_width or y + height > image_height:
      num_annotations_skipped += 1
      continue
    category_id = int(object_annotations['category_id'])
    category_name = category_index[category_id]['name'].encode('utf8')
    if remove_non_person_annotations and category_name != b'person':
      num_annotations_skipped += 1
      continue
    xmin.append(float(x) / image_width)
    xmax.append(float(x + width) / image_width)
    ymin.append(float(y) / image_height)
    ymax.append(float(y + height) / image_height)
    is_crowd.append(object_annotations['iscrowd'])
    category_ids.append(category_id)
    category_names.append(category_name)
    area.append(object_annotations['area'])

    if include_masks:
      run_len_encoding = mask.frPyObjects(object_annotations['segmentation'],
                                          image_height, image_width)
      binary_mask = mask.decode(run_len_encoding)
      if not object_annotations['iscrowd']:
        binary_mask = np.amax(binary_mask, axis=2)
      pil_image = PIL.Image.fromarray(binary_mask)
      output_io = io.BytesIO()
      pil_image.save(output_io, format='PNG')
      encoded_mask_png.append(output_io.getvalue())

    if include_keypoint:
      annotation_id = object_annotations['id']
      if annotation_id in keypoint_annotations_dict:
        num_keypoint_annotation_used += 1
        keypoint_annotations = keypoint_annotations_dict[annotation_id]
        keypoints = keypoint_annotations['keypoints']
        num_kpts = keypoint_annotations['num_keypoints']
        keypoints_x_abs = keypoints[::3]
        keypoints_x.extend(
            [float(x_abs) / image_width for x_abs in keypoints_x_abs])
        keypoints_y_abs = keypoints[1::3]
        keypoints_y.extend(
            [float(y_abs) / image_height for y_abs in keypoints_y_abs])
        keypoints_visibility.extend(keypoints[2::3])
        keypoints_name.extend(_COCO_KEYPOINT_NAMES)
        num_keypoints.append(num_kpts)
      else:
        keypoints_x.extend([0.0] * len(_COCO_KEYPOINT_NAMES))
        keypoints_y.extend([0.0] * len(_COCO_KEYPOINT_NAMES))
        keypoints_visibility.extend([0] * len(_COCO_KEYPOINT_NAMES))
        keypoints_name.extend(_COCO_KEYPOINT_NAMES)
        num_keypoints.append(0)

    if include_densepose:
      annotation_id = object_annotations['id']
      if (annotation_id in densepose_annotations_dict and
          all(key in densepose_annotations_dict[annotation_id]
              for key in densepose_keys)):
        dp_annotations = densepose_annotations_dict[annotation_id]
        num_densepose_annotation_used += 1
        dp_num_points.append(len(dp_annotations['dp_I']))
        dp_part_index.extend([int(i - _DP_PART_ID_OFFSET)
                              for i in dp_annotations['dp_I']])
        # DensePose surface coordinates are defined on a [256, 256] grid
        # relative to each instance box (i.e. absolute coordinates in range
        # [0., 256.]). The following converts the coordinates
        # so that they are expressed in normalized image coordinates.
        dp_x_box_rel = [
            clip_to_unit(val / 256.) for val in dp_annotations['dp_x']]
        dp_x_norm = [(float(x) + x_box_rel * width) / image_width
                     for x_box_rel in dp_x_box_rel]
        dp_y_box_rel = [
            clip_to_unit(val / 256.) for val in dp_annotations['dp_y']]
        dp_y_norm = [(float(y) + y_box_rel * height) / image_height
                     for y_box_rel in dp_y_box_rel]
        dp_x.extend(dp_x_norm)
        dp_y.extend(dp_y_norm)
        dp_u.extend(dp_annotations['dp_U'])
        dp_v.extend(dp_annotations['dp_V'])
      else:
        dp_num_points.append(0)

  if (remove_non_person_images and
      not any(name == b'person' for name in category_names)):
    return (key, None, num_annotations_skipped,
            num_keypoint_annotation_skipped, num_densepose_annotation_skipped)
  feature_dict = {
      'image/height':
          dataset_util.int64_feature(image_height),
      'image/width':
          dataset_util.int64_feature(image_width),
      'image/filename':
          dataset_util.bytes_feature(filename.encode('utf8')),
      'image/source_id':
          dataset_util.bytes_feature(str(image_id).encode('utf8')),
      'image/key/sha256':
          dataset_util.bytes_feature(key.encode('utf8')),
      'image/encoded':
          dataset_util.bytes_feature(encoded_jpg),
      'image/format':
          dataset_util.bytes_feature('jpeg'.encode('utf8')),
      'image/object/bbox/xmin':
          dataset_util.float_list_feature(xmin),
      'image/object/bbox/xmax':
          dataset_util.float_list_feature(xmax),
      'image/object/bbox/ymin':
          dataset_util.float_list_feature(ymin),
      'image/object/bbox/ymax':
          dataset_util.float_list_feature(ymax),
      'image/object/class/text':
          dataset_util.bytes_list_feature(category_names),
      'image/object/is_crowd':
          dataset_util.int64_list_feature(is_crowd),
      'image/object/area':
          dataset_util.float_list_feature(area),
  }
  if include_masks:
    feature_dict['image/object/mask'] = (
        dataset_util.bytes_list_feature(encoded_mask_png))
  if include_keypoint:
    feature_dict['image/object/keypoint/x'] = (
        dataset_util.float_list_feature(keypoints_x))
    feature_dict['image/object/keypoint/y'] = (
        dataset_util.float_list_feature(keypoints_y))
    feature_dict['image/object/keypoint/num'] = (
        dataset_util.int64_list_feature(num_keypoints))
    feature_dict['image/object/keypoint/visibility'] = (
        dataset_util.int64_list_feature(keypoints_visibility))
    feature_dict['image/object/keypoint/text'] = (
        dataset_util.bytes_list_feature(keypoints_name))
    num_keypoint_annotation_skipped = (
        len(keypoint_annotations_dict) - num_keypoint_annotation_used)
  if include_densepose:
    feature_dict['image/object/densepose/num'] = (
        dataset_util.int64_list_feature(dp_num_points))
    feature_dict['image/object/densepose/part_index'] = (
        dataset_util.int64_list_feature(dp_part_index))
    feature_dict['image/object/densepose/x'] = (
        dataset_util.float_list_feature(dp_x))
    feature_dict['image/object/densepose/y'] = (
        dataset_util.float_list_feature(dp_y))
    feature_dict['image/object/densepose/u'] = (
        dataset_util.float_list_feature(dp_u))
    feature_dict['image/object/densepose/v'] = (
        dataset_util.float_list_feature(dp_v))
    num_densepose_annotation_skipped = (
        len(densepose_annotations_dict) - num_densepose_annotation_used)

  example = tf.train.Example(features=tf.train.Features(feature=feature_dict))
  return (key, example, num_annotations_skipped,
          num_keypoint_annotation_skipped, num_densepose_annotation_skipped)


def _create_tf_record_from_coco_annotations(annotations_file, image_dir,
                                            output_path, include_masks,
                                            num_shards,
                                            keypoint_annotations_file='',
                                            densepose_annotations_file='',
                                            remove_non_person_annotations=False,
                                            remove_non_person_images=False):
  """Loads COCO annotation json files and converts to tf.Record format.

  Args:
    annotations_file: JSON file containing bounding box annotations.
    image_dir: Directory containing the image files.
    output_path: Path to output tf.Record file.
    include_masks: Whether to include instance segmentations masks
      (PNG encoded) in the result. default: False.
    num_shards: number of output file shards.
    keypoint_annotations_file: JSON file containing the person keypoint
      annotations. If empty, then no person keypoint annotations will be
      generated.
    densepose_annotations_file: JSON file containing the DensePose annotations.
      If empty, then no DensePose annotations will be generated.
    remove_non_person_annotations: Whether to remove any annotations that are
      not the "person" class.
    remove_non_person_images: Whether to remove any images that do not contain
      at least one "person" annotation.
  """
  with contextlib2.ExitStack() as tf_record_close_stack, \
      tf.gfile.GFile(annotations_file, 'r') as fid:
    output_tfrecords = tf_record_creation_util.open_sharded_output_tfrecords(
        tf_record_close_stack, output_path, num_shards)
    groundtruth_data = json.load(fid)
    images = groundtruth_data['images']
    category_index = label_map_util.create_category_index(
        groundtruth_data['categories'])

    annotations_index = {}
    if 'annotations' in groundtruth_data:
      logging.info('Found groundtruth annotations. Building annotations index.')
      for annotation in groundtruth_data['annotations']:
        image_id = annotation['image_id']
        if image_id not in annotations_index:
          annotations_index[image_id] = []
        annotations_index[image_id].append(annotation)
    missing_annotation_count = 0
    for image in images:
      image_id = image['id']
      if image_id not in annotations_index:
        missing_annotation_count += 1
        annotations_index[image_id] = []
    logging.info('%d images are missing annotations.',
                 missing_annotation_count)

    keypoint_annotations_index = {}
    if keypoint_annotations_file:
      with tf.gfile.GFile(keypoint_annotations_file, 'r') as kid:
        keypoint_groundtruth_data = json.load(kid)
      if 'annotations' in keypoint_groundtruth_data:
        for annotation in keypoint_groundtruth_data['annotations']:
          image_id = annotation['image_id']
          if image_id not in keypoint_annotations_index:
            keypoint_annotations_index[image_id] = {}
          keypoint_annotations_index[image_id][annotation['id']] = annotation

    densepose_annotations_index = {}
    if densepose_annotations_file:
      with tf.gfile.GFile(densepose_annotations_file, 'r') as fid:
        densepose_groundtruth_data = json.load(fid)
      if 'annotations' in densepose_groundtruth_data:
        for annotation in densepose_groundtruth_data['annotations']:
          image_id = annotation['image_id']
          if image_id not in densepose_annotations_index:
            densepose_annotations_index[image_id] = {}
          densepose_annotations_index[image_id][annotation['id']] = annotation

    total_num_annotations_skipped = 0
    total_num_keypoint_annotations_skipped = 0
    total_num_densepose_annotations_skipped = 0
    for idx, image in enumerate(images):
      if idx % 100 == 0:
        logging.info('On image %d of %d', idx, len(images))
      annotations_list = annotations_index[image['id']]
      keypoint_annotations_dict = None
      if keypoint_annotations_file:
        keypoint_annotations_dict = {}
        if image['id'] in keypoint_annotations_index:
          keypoint_annotations_dict = keypoint_annotations_index[image['id']]
      densepose_annotations_dict = None
      if densepose_annotations_file:
        densepose_annotations_dict = {}
        if image['id'] in densepose_annotations_index:
          densepose_annotations_dict = densepose_annotations_index[image['id']]
      (_, tf_example, num_annotations_skipped, num_keypoint_annotations_skipped,
       num_densepose_annotations_skipped) = create_tf_example(
           image, annotations_list, image_dir, category_index, include_masks,
           keypoint_annotations_dict, densepose_annotations_dict,
           remove_non_person_annotations, remove_non_person_images)
      total_num_annotations_skipped += num_annotations_skipped
      total_num_keypoint_annotations_skipped += num_keypoint_annotations_skipped
      total_num_densepose_annotations_skipped += (
          num_densepose_annotations_skipped)
      shard_idx = idx % num_shards
      if tf_example:
        output_tfrecords[shard_idx].write(tf_example.SerializeToString())
    logging.info('Finished writing, skipped %d annotations.',
                 total_num_annotations_skipped)
    if keypoint_annotations_file:
      logging.info('Finished writing, skipped %d keypoint annotations.',
                   total_num_keypoint_annotations_skipped)
    if densepose_annotations_file:
      logging.info('Finished writing, skipped %d DensePose annotations.',
                   total_num_densepose_annotations_skipped)


def main(_):
  assert FLAGS.train_image_dir, '`train_image_dir` missing.'
  assert FLAGS.val_image_dir, '`val_image_dir` missing.'
  assert FLAGS.test_image_dir, '`test_image_dir` missing.'
  assert FLAGS.train_annotations_file, '`train_annotations_file` missing.'
  assert FLAGS.val_annotations_file, '`val_annotations_file` missing.'
  assert FLAGS.testdev_annotations_file, '`testdev_annotations_file` missing.'

  if not tf.gfile.IsDirectory(FLAGS.output_dir):
    tf.gfile.MakeDirs(FLAGS.output_dir)
  train_output_path = os.path.join(FLAGS.output_dir, 'coco_train.record')
  val_output_path = os.path.join(FLAGS.output_dir, 'coco_val.record')
  testdev_output_path = os.path.join(FLAGS.output_dir, 'coco_testdev.record')

  _create_tf_record_from_coco_annotations(
      FLAGS.train_annotations_file,
      FLAGS.train_image_dir,
      train_output_path,
      FLAGS.include_masks,
      num_shards=100,
      keypoint_annotations_file=FLAGS.train_keypoint_annotations_file,
      densepose_annotations_file=FLAGS.train_densepose_annotations_file,
      remove_non_person_annotations=FLAGS.remove_non_person_annotations,
      remove_non_person_images=FLAGS.remove_non_person_images)
  _create_tf_record_from_coco_annotations(
      FLAGS.val_annotations_file,
      FLAGS.val_image_dir,
      val_output_path,
      FLAGS.include_masks,
      num_shards=50,
      keypoint_annotations_file=FLAGS.val_keypoint_annotations_file,
      densepose_annotations_file=FLAGS.val_densepose_annotations_file,
      remove_non_person_annotations=FLAGS.remove_non_person_annotations,
      remove_non_person_images=FLAGS.remove_non_person_images)
  _create_tf_record_from_coco_annotations(
      FLAGS.testdev_annotations_file,
      FLAGS.test_image_dir,
      testdev_output_path,
      FLAGS.include_masks,
      num_shards=50)


if __name__ == '__main__':
  tf.app.run()
