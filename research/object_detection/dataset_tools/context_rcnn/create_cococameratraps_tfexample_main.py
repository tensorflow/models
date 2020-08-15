# Copyright 2020 The TensorFlow Authors. All Rights Reserved.
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
r"""Beam pipeline to create COCO Camera Traps Object Detection TFRecords.

Please note that this tool creates sharded output files.

This tool assumes the input annotations are in the COCO Camera Traps json
format, specified here:
https://github.com/Microsoft/CameraTraps/blob/master/data_management/README.md

Example usage:

    python create_cococameratraps_tfexample_main.py \
      --alsologtostderr \
      --output_tfrecord_prefix="/path/to/output/tfrecord/location/prefix" \
      --image_directory="/path/to/image/folder/" \
      --input_annotations_file="path/to/annotations.json"

"""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import argparse
import hashlib
import io
import json
import os
import numpy as np
import PIL.Image
import tensorflow as tf
from object_detection.utils import dataset_util

try:
  import apache_beam as beam  # pylint:disable=g-import-not-at-top
except ModuleNotFoundError:
  pass


class ParseImage(beam.DoFn):
  """A DoFn that parses a COCO-CameraTraps json and emits TFRecords."""

  def __init__(self, image_directory, images, annotations, categories,
               keep_bboxes):
    """Initialization function.

    Args:
      image_directory: Path to image directory
      images: list of COCO Camera Traps style image dictionaries
      annotations: list of COCO Camera Traps style annotation dictionaries
      categories: list of COCO Camera Traps style category dictionaries
      keep_bboxes: Whether to keep any bounding boxes that exist in the
        annotations
    """

    self._image_directory = image_directory
    self._image_dict = {im['id']: im for im in images}
    self._annotation_dict = {im['id']: [] for im in images}
    self._category_dict = {int(cat['id']): cat for cat in categories}
    for ann in annotations:
      self._annotation_dict[ann['image_id']].append(ann)
    self._images = images
    self._keep_bboxes = keep_bboxes

    self._num_examples_processed = beam.metrics.Metrics.counter(
        'cococameratraps_data_generation', 'num_tf_examples_processed')

  def process(self, image_id):
    """Builds a tf.Example given an image id.

    Args:
      image_id: the image id of the associated image

    Returns:
      List of tf.Examples.
    """

    image = self._image_dict[image_id]
    annotations = self._annotation_dict[image_id]
    image_height = image['height']
    image_width = image['width']
    filename = image['file_name']
    image_id = image['id']
    image_location_id = image['location']

    image_datetime = str(image['date_captured'])

    image_sequence_id = str(image['seq_id'])
    image_sequence_num_frames = int(image['seq_num_frames'])
    image_sequence_frame_num = int(image['frame_num'])

    full_path = os.path.join(self._image_directory, filename)

    try:
      # Ensure the image exists and is not corrupted
      with tf.io.gfile.GFile(full_path, 'rb') as fid:
        encoded_jpg = fid.read()
      encoded_jpg_io = io.BytesIO(encoded_jpg)
      image = PIL.Image.open(encoded_jpg_io)
      image = tf.io.decode_jpeg(encoded_jpg, channels=3)
    except Exception:  # pylint: disable=broad-except
      # The image file is missing or corrupt
      return []

    key = hashlib.sha256(encoded_jpg).hexdigest()
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
        'image/location':
            dataset_util.bytes_feature(str(image_location_id).encode('utf8')),
        'image/seq_num_frames':
            dataset_util.int64_feature(image_sequence_num_frames),
        'image/seq_frame_num':
            dataset_util.int64_feature(image_sequence_frame_num),
        'image/seq_id':
            dataset_util.bytes_feature(image_sequence_id.encode('utf8')),
        'image/date_captured':
            dataset_util.bytes_feature(image_datetime.encode('utf8'))
    }

    num_annotations_skipped = 0
    if annotations:
      xmin = []
      xmax = []
      ymin = []
      ymax = []
      category_names = []
      category_ids = []
      area = []

      for object_annotations in annotations:
        if 'bbox' in object_annotations and self._keep_bboxes:
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
          if 'area' in object_annotations:
            area.append(object_annotations['area'])
          else:
            # approximate area using l*w/2
            area.append(width*height/2.0)

        category_id = int(object_annotations['category_id'])
        category_ids.append(category_id)
        category_names.append(
            self._category_dict[category_id]['name'].encode('utf8'))

      feature_dict.update({
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
          'image/object/class/label':
              dataset_util.int64_list_feature(category_ids),
          'image/object/area':
              dataset_util.float_list_feature(area),
      })

      # For classification, add the first category to image/class/label and
      # image/class/text
      if not category_ids:
        feature_dict.update({
            'image/class/label':
                dataset_util.int64_list_feature([0]),
            'image/class/text':
                dataset_util.bytes_list_feature(['empty'.encode('utf8')]),
        })
      else:
        feature_dict.update({
            'image/class/label':
                dataset_util.int64_list_feature([category_ids[0]]),
            'image/class/text':
                dataset_util.bytes_list_feature([category_names[0]]),
        })

    else:
      # Add empty class if there are no annotations
      feature_dict.update({
          'image/class/label':
              dataset_util.int64_list_feature([0]),
          'image/class/text':
              dataset_util.bytes_list_feature(['empty'.encode('utf8')]),
      })

    example = tf.train.Example(features=tf.train.Features(feature=feature_dict))
    self._num_examples_processed.inc(1)

    return [(example)]


def load_json_data(data_file):
  with tf.io.gfile.GFile(data_file, 'r') as fid:
    data_dict = json.load(fid)
  return data_dict


def create_pipeline(pipeline,
                    image_directory,
                    input_annotations_file,
                    output_tfrecord_prefix=None,
                    num_images_per_shard=200,
                    keep_bboxes=True):
  """Creates a beam pipeline for producing a COCO-CameraTraps Image dataset.

  Args:
    pipeline: Initialized beam pipeline.
    image_directory: Path to image directory
    input_annotations_file: Path to a coco-cameratraps annotation file
    output_tfrecord_prefix: Absolute path for tfrecord outputs. Final files will
      be named {output_tfrecord_prefix}@N.
    num_images_per_shard: The number of images to store in each shard
    keep_bboxes: Whether to keep any bounding boxes that exist in the json file
  """

  data = load_json_data(input_annotations_file)

  num_shards = int(np.ceil(float(len(data['images']))/num_images_per_shard))

  image_examples = (
      pipeline | ('CreateCollections') >> beam.Create(
          [im['id'] for im in data['images']])
      | ('ParseImage') >> beam.ParDo(ParseImage(
          image_directory, data['images'], data['annotations'],
          data['categories'], keep_bboxes=keep_bboxes)))
  _ = (image_examples
       | ('Reshuffle') >> beam.Reshuffle()
       | ('WriteTfImageExample') >> beam.io.tfrecordio.WriteToTFRecord(
           output_tfrecord_prefix,
           num_shards=num_shards,
           coder=beam.coders.ProtoCoder(tf.train.Example)))


def parse_args(argv):
  """Command-line argument parser.

  Args:
    argv: command line arguments
  Returns:
    beam_args: Arguments for the beam pipeline.
    pipeline_args: Arguments for the pipeline options, such as runner type.
  """
  parser = argparse.ArgumentParser()
  parser.add_argument(
      '--image_directory',
      dest='image_directory',
      required=True,
      help='Path to the directory where the images are stored.')
  parser.add_argument(
      '--output_tfrecord_prefix',
      dest='output_tfrecord_prefix',
      required=True,
      help='Path and prefix to store TFRecords containing images in tf.Example'
      'format.')
  parser.add_argument(
      '--input_annotations_file',
      dest='input_annotations_file',
      required=True,
      help='Path to Coco-CameraTraps style annotations file.')
  parser.add_argument(
      '--num_images_per_shard',
      dest='num_images_per_shard',
      default=200,
      help='The number of  images to be stored in each outputshard.')
  beam_args, pipeline_args = parser.parse_known_args(argv)
  return beam_args, pipeline_args


def main(argv=None, save_main_session=True):
  """Runs the Beam pipeline that performs inference.

  Args:
    argv: Command line arguments.
    save_main_session: Whether to save the main session.
  """
  args, pipeline_args = parse_args(argv)

  pipeline_options = beam.options.pipeline_options.PipelineOptions(
            pipeline_args)
  pipeline_options.view_as(
      beam.options.pipeline_options.SetupOptions).save_main_session = (
          save_main_session)

  dirname = os.path.dirname(args.output_tfrecord_prefix)
  tf.io.gfile.makedirs(dirname)

  p = beam.Pipeline(options=pipeline_options)
  create_pipeline(
      pipeline=p,
      image_directory=args.image_directory,
      input_annotations_file=args.input_annotations_file,
      output_tfrecord_prefix=args.output_tfrecord_prefix,
      num_images_per_shard=args.num_images_per_shard)
  p.run()


if __name__ == '__main__':
  main()
