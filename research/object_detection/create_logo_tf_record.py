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

r"""Convert the Oxford pet dataset to TFRecord for object_detection.

See: O. M. Parkhi, A. Vedaldi, A. Zisserman, C. V. Jawahar
     Cats and Dogs
     IEEE Conference on Computer Vision and Pattern Recognition, 2012
     http://www.robots.ox.ac.uk/~vgg/data/pets/

Example usage:
    ./create_pet_tf_record --data_dir=/home/user/pet \
        --output_dir=/home/user/pet/output
"""

import hashlib
import io
import logging
import os
import random
import re

from lxml import etree
import PIL.Image
import tensorflow as tf

from object_detection.utils import dataset_util
from object_detection.utils import label_map_util

flags = tf.app.flags
flags.DEFINE_string('data_dir', r'E:\data_mining\data\east_ic_logo', 'Root directory to raw pet dataset.')
flags.DEFINE_string('output_dir', r'E:\data_mining\data\east_ic_logo', 'Path to directory to output TFRecords.')
flags.DEFINE_string('label_map_path', r'D:\WorkSpace\models\research\object_detection\data\logo_label_map.pbtxt',
                    'Path to label map proto')
FLAGS = flags.FLAGS


def dict_to_tf_example(img_path,
                       label_map_dict,
                       class_name):
    """Convert XML derived dict to tf.Example proto.

    Notice that this function normalizes the bounding box coordinates provided
    by the raw data.

    Args:
      data: dict holding PASCAL XML fields for a single image (obtained by
        running dataset_util.recursive_parse_xml_to_dict)
      label_map_dict: A map from string label names to integers ids.
      image_subdirectory: String specifying subdirectory within the
        Pascal dataset directory holding the actual image data.
      ignore_difficult_instances: Whether to skip difficult instances in the
        dataset  (default: False).

    Returns:
      example: The converted tf.Example.

    Raises:
      ValueError: if the image pointed to by data['filename'] is not a valid JPEG
    """

    print("img_path = " + img_path + ",class_name = " + class_name)

    with tf.gfile.GFile(img_path, 'rb') as fid:
        encoded_jpg = fid.read()
    encoded_jpg_io = io.BytesIO(encoded_jpg)
    image = PIL.Image.open(encoded_jpg_io)
    if image.format != 'JPEG':
        raise ValueError('Image format not JPEG')
    key = hashlib.sha256(encoded_jpg).hexdigest()

    img_width = image.width
    img_height = image.height

    filename = os.path.basename(img_path)

    classes = []
    classes_text = []

    if class_name == "BigLogo":
        logo_width = 160
        logo_height = 76
        x_padding = 20
        y_padding = 16
        print("big logo = " + class_name)
    else:
        print("small logo = " + class_name)
        logo_width = 96
        logo_height = 44
        x_padding = 10
        y_padding = 10

    classes_text.append(class_name.encode('utf8'))
    classes.append(label_map_dict[class_name])
    xmin = [(img_width - logo_width - x_padding) / img_width]
    xmax = [(img_width - x_padding) / img_width]
    ymin = [y_padding / img_height]  # List of normalized top y coordinates in bounding box (1 per box)
    ymax = [(y_padding + logo_height) / img_height]  # List of normalized bottom y coordinates in bounding box

    example = tf.train.Example(features=tf.train.Features(feature={
        'image/height': dataset_util.int64_feature(img_height),
        'image/width': dataset_util.int64_feature(img_width),
        'image/filename': dataset_util.bytes_feature(filename.encode('utf8')),
        'image/source_id': dataset_util.bytes_feature(filename.encode('utf8')),
        'image/key/sha256': dataset_util.bytes_feature(key.encode('utf8')),
        'image/encoded': dataset_util.bytes_feature(encoded_jpg),
        'image/format': dataset_util.bytes_feature('jpeg'.encode('utf8')),
        'image/object/bbox/xmin': dataset_util.float_list_feature(xmin),
        'image/object/bbox/xmax': dataset_util.float_list_feature(xmax),
        'image/object/bbox/ymin': dataset_util.float_list_feature(ymin),
        'image/object/bbox/ymax': dataset_util.float_list_feature(ymax),
        'image/object/class/text': dataset_util.bytes_list_feature(classes_text),
        'image/object/class/label': dataset_util.int64_list_feature(classes),
    }))
    return example


# TODO: Add test for pet/PASCAL main files.
def main(_):
    data_dir = FLAGS.data_dir
    label_map_dict = label_map_util.get_label_map_dict(FLAGS.label_map_path)

    logging.info('Reading from Logo dataset.')


    examples_list = []

    for class_dir_name in os.listdir(FLAGS.data_dir):
        class_dir_path = os.path.join(FLAGS.data_dir, class_dir_name)
        if os.path.isdir(class_dir_path):
            print("class_dir_path = " + class_dir_path)
            for filename in os.listdir(class_dir_path):
                img_path = os.path.join(class_dir_path, filename)

                examples_list.append(img_path)

    random.seed(42)
    random.shuffle(examples_list)
    num_examples = len(examples_list)
    num_train = int(0.7 * num_examples)
    train_examples = examples_list[:num_train]
    val_examples = examples_list[num_train:]
    print('%d training and %d validation examples.', len(train_examples), len(val_examples))

    eval_output_path = os.path.join(FLAGS.output_dir, 'logo_train.record')
    writer = tf.python_io.TFRecordWriter(eval_output_path)
    for train_path in train_examples:
        head, tail = os.path.split(train_path)
        class_name = os.path.basename(head)
        tf_example = dict_to_tf_example(train_path, label_map_dict, class_name)
        writer.write(tf_example.SerializeToString())
    writer.close()

    eval_output_path = os.path.join(FLAGS.output_dir, 'logo_eval.record')
    writer = tf.python_io.TFRecordWriter(eval_output_path)
    for eval_path in val_examples:
        head, tail = os.path.split(eval_path)
        class_name = os.path.basename(head)
        tf_example = dict_to_tf_example(eval_path, label_map_dict, class_name)
        writer.write(tf_example.SerializeToString())

    writer.close()


if __name__ == '__main__':
    tf.app.run()
