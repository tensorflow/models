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

r"""Convert VOC format dataset to TFRecord for object_detection.

For example
Hollywood head dataset:
See: http://www.di.ens.fr/willow/research/headdetection/
     Context-aware CNNs for person head detection

HDA pedestrian dataset:
See: http://vislab.isr.ist.utl.pt/hda-dataset/

Example usage:
    ./create_head_tf_record_pascal_fmt --data_dir=/startdt_data/HollywoodHeads2 \
        --output_dir=models/head_detector
        --mode=train
"""

import hashlib
import io
import logging
import os, sys
import random
import re
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__))+"/..")
from lxml import etree
import PIL.Image
import tensorflow as tf

from object_detection.utils import dataset_util
from object_detection.utils import label_map_util

flags = tf.app.flags
flags.DEFINE_string('data_dir', '', 'Root directory to raw pet dataset, like /startdt_data/HDA_Dataset_V1.3/VOC_fmt_training_fisheye')
flags.DEFINE_string('output_dir', '', 'Path to directory to output TFRecords, like models/hda_cam_person_fisheye')
flags.DEFINE_string('label_map_path', '',
                    'Path to label map proto, like data/hda_person_label_map.pbtxt')
flags.DEFINE_string('mode', '', 'generate train or val output: train/val')
FLAGS = flags.FLAGS


def dict_to_tf_example(data,
                       label_map_dict,
                       image_subdirectory,
                       ignore_difficult_instances=False):
  """Convert XML derived dict to tf.Example proto.

  Notice that this function normalizes the bounding box coordinates provided
  by the raw data.

  Args:
    data: dict holding PASCAL XML fields for a single image (obtained by
      running dataset_util.recursive_parse_xml_to_dict)
    label_map_dict: A map from string label names to integers ids.
    image_subdirectory: String specifying subdirectory within the
      Pascal dataset (here only head available) directory holding the actual image data.
    ignore_difficult_instances: Whether to skip difficult instances in the
      dataset  (default: False).

  Returns:
    example: The converted tf.Example.

  Raises:
    ValueError: if the image pointed to by data['filename'] is not a valid JPEG
  """
  img_path = os.path.splitext(os.path.join(image_subdirectory, data['filename']))[0] + ".jpg"
  with tf.gfile.GFile(img_path, 'rb') as fid:
    encoded_jpg = fid.read()
  encoded_jpg_io = io.BytesIO(encoded_jpg)
  image = PIL.Image.open(encoded_jpg_io)
  if image.format != 'JPEG':
    raise ValueError('Image format not JPEG')
  # generate hash key for image
  key = hashlib.sha256(encoded_jpg).hexdigest()

  width = int(data['size']['width'])
  height = int(data['size']['height'])

  xmin = []
  ymin = []
  xmax = []
  ymax = []
  classes = []
  classes_text = []
  # truncated = []
  # poses = []
  difficult_obj = []
  for obj in data['object']:
    difficult = bool(int(obj['difficult']))
    if ignore_difficult_instances and difficult:
      continue

    difficult_obj.append(int(difficult))

    xmin.append(float(obj['bndbox']['xmin']) / width)
    ymin.append(float(obj['bndbox']['ymin']) / height)
    xmax.append(float(obj['bndbox']['xmax']) / width)
    ymax.append(float(obj['bndbox']['ymax']) / height)
    class_name = obj['name']
    classes_text.append(class_name.encode('utf8'))
    classes.append(label_map_dict[class_name])
    # truncated.append(int(obj['truncated']))
    # poses.append(obj['pose'].encode('utf8'))

  example = tf.train.Example(features=tf.train.Features(feature={
      'image/height': dataset_util.int64_feature(height),
      'image/width': dataset_util.int64_feature(width),
      'image/filename': dataset_util.bytes_feature(
          data['filename'].encode('utf8')),
      'image/source_id': dataset_util.bytes_feature(
          data['filename'].encode('utf8')),
      'image/key/sha256': dataset_util.bytes_feature(key.encode('utf8')),
      'image/encoded': dataset_util.bytes_feature(encoded_jpg),
      'image/format': dataset_util.bytes_feature('jpeg'.encode('utf8')),
      'image/object/bbox/xmin': dataset_util.float_list_feature(xmin),
      'image/object/bbox/xmax': dataset_util.float_list_feature(xmax),
      'image/object/bbox/ymin': dataset_util.float_list_feature(ymin),
      'image/object/bbox/ymax': dataset_util.float_list_feature(ymax),
      'image/object/class/text': dataset_util.bytes_list_feature(classes_text),
      'image/object/class/label': dataset_util.int64_list_feature(classes),
      'image/object/difficult': dataset_util.int64_list_feature(difficult_obj),
      # 'image/object/truncated': dataset_util.int64_list_feature(truncated),
      # 'image/object/view': dataset_util.bytes_list_feature(poses),
  }))
  return example


def create_tf_record(output_filename,
                     label_map_dict,
                     annotations_dir,
                     image_dir,
                     examples):
    """Creates a TFRecord file from examples.

    Args:
        output_filename: Path to where output file is saved.
        label_map_dict: The label map dictionary.
        annotations_dir: Directory where annotation files are stored.
        image_dir: Directory where image files are stored.
        examples: Examples to parse and save to tf record.
    """
    writer = tf.python_io.TFRecordWriter(output_filename)
    for idx, example in enumerate(examples):
        if idx % 100 == 0:
            logging.info('On image %d of %d', idx, len(examples))
        path = os.path.join(annotations_dir, example + '.xml')
        print "processing...", example
        if not os.path.exists(path):
            logging.warning('Could not find %s, ignoring example.', path)
            continue
        with tf.gfile.GFile(path, 'r') as fid:
            try:
                xml_str = fid.read()
                xml = etree.fromstring(xml_str)
                data = dataset_util.recursive_parse_xml_to_dict(xml)['annotation']
                tf_example = dict_to_tf_example(data, label_map_dict, image_dir)
                writer.write(tf_example.SerializeToString())
            except:
                print "Fail to open image: ", example
    writer.close()


# TODO: Add test for pet/PASCAL main files.
def main(_):
    data_dir = FLAGS.data_dir
    mode = FLAGS.mode
    assert mode in ["train", "val"]
    logging.info("Generate data for model {}!".format(mode))
    label_map_dict = label_map_util.get_label_map_dict(FLAGS.label_map_path)

    logging.info('Reading from Pet dataset.')
    image_dir = os.path.join(data_dir, 'JPEGImages')
    annotations_dir = os.path.join(data_dir, 'Annotations')

    # Test images are not included in the downloaded data set, so we shall perform
    # our own split.
    # random.seed(42)
    # random.shuffle(examples_list)
    # num_examples = len(examples_list)
    # num_train = int(num_examples)
    # train_examples = examples_list[:num_train]
    if not os.path.exists(FLAGS.output_dir):
        os.makedirs(FLAGS.output_dir)
    if mode == 'train':
        examples_path = os.path.join(data_dir, 'ImageSets/Main/trainval.txt')
        examples_list = dataset_util.read_examples_list(examples_path)
        logging.info('%d training examples.', len(examples_list))
        train_output_path = os.path.join(FLAGS.output_dir, 'train.record')
        create_tf_record(train_output_path, label_map_dict, annotations_dir,
                         image_dir, examples_list)
    elif mode == 'val':
        examples_path = os.path.join(data_dir, 'ImageSets/Main/val.txt')
        examples_list = dataset_util.read_examples_list(examples_path)
        val_output_path = os.path.join(FLAGS.output_dir, 'val.record')
        create_tf_record(val_output_path, label_map_dict, annotations_dir,
                       image_dir, examples_list)

if __name__ == '__main__':
  tf.app.run()
