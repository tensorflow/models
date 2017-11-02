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
flags.DEFINE_string('data_dir', r'E:\data_mining\data\east_ic_logo\train', 'Root directory to raw pet dataset.')
FLAGS = flags.FLAGS


def dict_to_tf_example(filename,
                       image_subdirectory,
                       ):
    img_path = os.path.join(image_subdirectory, filename)
    with tf.gfile.GFile(img_path, 'rb') as fid:
        encoded_jpg = fid.read()
    encoded_jpg_io = io.BytesIO(encoded_jpg)
    image = PIL.Image.open(encoded_jpg_io)
    if image.format != 'JPEG':
        raise ValueError('Image format not JPEG')
    key = hashlib.sha256(encoded_jpg).hexdigest()

    img_width = image.width
    img_height = image.height
    # print("width = ", img_width, ",height = ", img_height)
    if img_height > 1024 or img_height < 400:
        print("错误图片：height img_path = " + img_path)
        os.remove(img_path)

    if img_width > 1024 or img_width < 400:
        print("错误图片：img_width img_path = " + img_path)
        os.remove(img_path)


# TODO: Add test for pet/PASCAL main files.
def main(_):
    data_dir = FLAGS.data_dir

    logging.info('Reading from Logo dataset.')

    for class_dir_name in os.listdir(FLAGS.data_dir):
        class_dir_path = os.path.join(FLAGS.data_dir, class_dir_name)
        if os.path.isdir(class_dir_path):
            print("class_dir_path = " + class_dir_path)
            for filename in os.listdir(class_dir_path):
                dict_to_tf_example(filename,class_dir_path)
                # writer.write(tf_example.SerializeToString())

                # writer.close()


if __name__ == '__main__':
    tf.app.run()
