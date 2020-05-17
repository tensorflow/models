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
r"""Downloads and converts VisualWakewords data to TFRecords of TF-Example protos.

This module downloads the COCO dataset, uncompresses it, derives the
VisualWakeWords dataset to create two TFRecord datasets: one for
train and one for test. Each TFRecord dataset is comprised of a set of
TF-Example protocol buffers, each of which contain a single image and label.

The script should take several minutes to run.
Please note that this tool creates sharded output files.

VisualWakeWords dataset is used to design tiny models classifying two classes,
such as person/not-person. The two steps to generate the VisualWakeWords
dataset from the COCO dataset are given below:

1. Use COCO annotations to create VisualWakeWords annotations:

Note: A bounding box is 'valid' if it has the foreground_class_of_interest
(e.g. person) and it's area is greater than 0.5% of the image area.

The resulting annotations file has the following fields, where 'images' are
the same as COCO dataset. 'categories' only contains information about the
foreground_class_of_interest (e.g. person) and 'annotations' maps an image to
objects (a list of valid bounding boxes) and label (value is 1 if it has
atleast one valid bounding box, otherwise 0)

  images[{
  "id", "width", "height", "file_name", "flickr_url", "coco_url",
  "license", "date_captured",
  }]

  categories{
  "id": {"id", "name", "supercategory"}
  }

  annotations{
  "image_id": {"objects":[{"area", "bbox" : [x,y,width,height]}], "label"}
  }

2. Use VisualWakeWords annotations to create TFRecords:

The resulting TFRecord file contains the following features:
{ image/height, image/width, image/source_id, image/encoded,
  image/class/label_text, image/class/label,
  image/object/class/text,
  image/object/bbox/ymin, image/object/bbox/xmin, image/object/bbox/ymax,
  image/object/bbox/xmax, image/object/area
  image/filename, image/format, image/key/sha256}
For classification models, you need the image/encoded and image/class/label.

Example usage:
Run download_and_convert_data.py in the parent directory as follows:

    python download_and_convert_visualwakewords.py --logtostderr \
      --dataset_name=visualwakewords \
      --dataset_dir="${DATASET_DIR}" \
      --small_object_area_threshold=0.005 \
      --foreground_class_of_interest='person'

"""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os
import tensorflow.compat.v1 as tf
from datasets import download_and_convert_visualwakewords_lib

tf.logging.set_verbosity(tf.logging.INFO)

tf.app.flags.DEFINE_string(
    'coco_dirname', 'coco_dataset',
    'A subdirectory in visualwakewords dataset directory'
    'containing the coco dataset')

FLAGS = tf.app.flags.FLAGS


def run(dataset_dir, small_object_area_threshold, foreground_class_of_interest):
  """Runs the download and conversion operation.

  Args:
    dataset_dir: The dataset directory where the dataset is stored.
    small_object_area_threshold: Threshold of fraction of image area below which
      small objects are filtered
    foreground_class_of_interest: Build a binary classifier based on the
      presence or absence of this object in the image.
  """
  # 1. Download the coco dataset into a subdirectory under the visualwakewords
  #    dataset directory
  coco_dir = os.path.join(dataset_dir, FLAGS.coco_dirname)

  if not tf.gfile.IsDirectory(coco_dir):
    tf.gfile.MakeDirs(coco_dir)

  download_and_convert_visualwakewords_lib.download_coco_dataset(coco_dir)

  # Path to COCO annotations
  train_annotations_file = os.path.join(coco_dir, 'annotations',
                                        'instances_train2014.json')
  val_annotations_file = os.path.join(coco_dir, 'annotations',
                                      'instances_val2014.json')
  train_image_dir = os.path.join(coco_dir, 'train2014')
  val_image_dir = os.path.join(coco_dir, 'val2014')

  # Path to VisualWakeWords annotations
  visualwakewords_annotations_train = os.path.join(
      dataset_dir, 'instances_visualwakewords_train2014.json')
  visualwakewords_annotations_val = os.path.join(
      dataset_dir, 'instances_visualwakewords_val2014.json')
  visualwakewords_labels_filename = os.path.join(dataset_dir, 'labels.txt')
  train_output_path = os.path.join(dataset_dir, 'train.record')
  val_output_path = os.path.join(dataset_dir, 'val.record')

  # 2. Create a labels file
  tf.logging.info('Creating a labels file...')
  download_and_convert_visualwakewords_lib.create_labels_file(
      foreground_class_of_interest, visualwakewords_labels_filename)

  # 3. Use COCO annotations to create VisualWakeWords annotations
  tf.logging.info('Creating train VisualWakeWords annotations...')
  download_and_convert_visualwakewords_lib.create_visual_wakeword_annotations(
      train_annotations_file, visualwakewords_annotations_train,
      small_object_area_threshold, foreground_class_of_interest)
  tf.logging.info('Creating validation VisualWakeWords annotations...')
  download_and_convert_visualwakewords_lib.create_visual_wakeword_annotations(
      val_annotations_file, visualwakewords_annotations_val,
      small_object_area_threshold, foreground_class_of_interest)

  # 4. Use VisualWakeWords annotations to create the TFRecords
  tf.logging.info('Creating train TFRecords for VisualWakeWords dataset...')
  download_and_convert_visualwakewords_lib.create_tf_record_for_visualwakewords_dataset(
      visualwakewords_annotations_train,
      train_image_dir,
      train_output_path,
      num_shards=100)

  tf.logging.info(
      'Creating validation TFRecords for VisualWakeWords dataset...')
  download_and_convert_visualwakewords_lib.create_tf_record_for_visualwakewords_dataset(
      visualwakewords_annotations_val,
      val_image_dir,
      val_output_path,
      num_shards=10)
