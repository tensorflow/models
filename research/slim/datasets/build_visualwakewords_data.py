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
r"""Build Visual WakeWords Dataset with images and labels for person/not-person.

This script generates the Visual WakeWords dataset annotations from
the raw COCO dataset and converts them to TFRecord.
Visual WakeWords Dataset derives from the COCO dataset to design tiny models
classifying two classes, such as person/not-person. The COCO annotations
are filtered to two classes: foreground_class_of_interest and background
( for e.g. person and not-person). Bounding boxes for small objects
with area less than 5% of the image area are filtered out.

The resulting annotations file has the following fields, where
the image and categories fields are same as COCO dataset, while the annotation
field corresponds to the foreground_class_of_interest/background class and
bounding boxes for the foreground_class_of_interest class.

  images{"id", "width", "height", "file_name", "license", "flickr_url",
  "coco_url", "date_captured",}

  annotations{
  "image_id", object[{"category_id", "area", "bbox" : [x,y,width,height],}]
  "count",
  "label"
  }

  categories[{
  "id", "name", "supercategory",
  }]


The TFRecord file contains the following features:
{ image/height, image/width, image/source_id, image/encoded,
  image/class/label_text, image/class/label,
  image/object/class/text,
  image/object/bbox/ymin, image/object/bbox/xmin, image/object/bbox/ymax,
  image/object/bbox/xmax, image/object/area
  image/filename, image/format, image/key/sha256}
For classification models, you need the image/encoded and image/class/label.
Please note that this tool creates sharded output files.

Example usage:
Add folder tensorflow/models/research/slim to your PYTHONPATH,
and from this folder, run the following commands:

    bash download_mscoco.sh path-to-mscoco-dataset
    TRAIN_IMAGE_DIR="path-to-mscoco-dataset/train2014"
    VAL_IMAGE_DIR="path-to-mscoco-dataset/val2014"

    TRAIN_ANNOTATIONS_FILE="path-to-mscoco-dataset/annotations/instances_train2014.json"
    VAL_ANNOTATIONS_FILE="path-to-mscoco-dataset/annotations/instances_val2014.json"

    python datasets/build_visualwakewords_data.py --logtostderr \
      --train_image_dir="${TRAIN_IMAGE_DIR}" \
      --val_image_dir="${VAL_IMAGE_DIR}" \
      --train_annotations_file="${TRAIN_ANNOTATIONS_FILE}" \
      --val_annotations_file="${VAL_ANNOTATIONS_FILE}" \
      --output_dir="${OUTPUT_DIR}" \
      --small_object_area_threshold=0.005 \
      --foreground_class_of_interest='person'
"""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os
import tensorflow as tf
from datasets import build_visualwakewords_data_lib

flags = tf.app.flags
tf.flags.DEFINE_string('train_image_dir', '', 'Training image directory.')
tf.flags.DEFINE_string('val_image_dir', '', 'Validation image directory.')
tf.flags.DEFINE_string('train_annotations_file', '',
                       'Training annotations JSON file.')
tf.flags.DEFINE_string('val_annotations_file', '',
                       'Validation annotations JSON file.')
tf.flags.DEFINE_string('output_dir', '/tmp/', 'Output data directory.')
tf.flags.DEFINE_float(
    'small_object_area_threshold', 0.005,
    'Threshold of fraction of image area below which small'
    'objects are filtered')
tf.flags.DEFINE_string(
    'foreground_class_of_interest', 'person',
    'Build a binary classifier based on the presence or absence'
    'of this object in the scene (default is person/not-person)')

FLAGS = flags.FLAGS

tf.logging.set_verbosity(tf.logging.INFO)


def main(unused_argv):
  # Path to COCO dataset images and annotations
  assert FLAGS.train_image_dir, '`train_image_dir` missing.'
  assert FLAGS.val_image_dir, '`val_image_dir` missing.'
  assert FLAGS.train_annotations_file, '`train_annotations_file` missing.'
  assert FLAGS.val_annotations_file, '`val_annotations_file` missing.'
  visualwakewords_annotations_train = os.path.join(
      FLAGS.output_dir, 'instances_visualwakewords_train2014.json')
  visualwakewords_annotations_val = os.path.join(
      FLAGS.output_dir, 'instances_visualwakewords_val2014.json')
  visualwakewords_labels_filename = os.path.join(FLAGS.output_dir,
                                                 'labels.txt')
  small_object_area_threshold = FLAGS.small_object_area_threshold
  foreground_class_of_interest = FLAGS.foreground_class_of_interest
  # Create the Visual WakeWords annotations from COCO annotations
  if not tf.gfile.IsDirectory(FLAGS.output_dir):
    tf.gfile.MakeDirs(FLAGS.output_dir)
  build_visualwakewords_data_lib.create_visual_wakeword_annotations(
      FLAGS.train_annotations_file, visualwakewords_annotations_train,
      small_object_area_threshold, foreground_class_of_interest,
      visualwakewords_labels_filename)
  build_visualwakewords_data_lib.create_visual_wakeword_annotations(
      FLAGS.val_annotations_file, visualwakewords_annotations_val,
      small_object_area_threshold, foreground_class_of_interest,
      visualwakewords_labels_filename)

  # Create the TF Records for Visual WakeWords Dataset
  if not tf.gfile.IsDirectory(FLAGS.output_dir):
    tf.gfile.MakeDirs(FLAGS.output_dir)
  train_output_path = os.path.join(FLAGS.output_dir, 'train.record')
  val_output_path = os.path.join(FLAGS.output_dir, 'val.record')
  build_visualwakewords_data_lib.create_tf_record_for_visualwakewords_dataset(
      visualwakewords_annotations_train,
      FLAGS.train_image_dir,
      train_output_path,
      num_shards=100)
  build_visualwakewords_data_lib.create_tf_record_for_visualwakewords_dataset(
      visualwakewords_annotations_val,
      FLAGS.val_image_dir,
      val_output_path,
      num_shards=10)


if __name__ == '__main__':
  tf.app.run()
