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
flags.DEFINE_string('data_dir', r'E:\data_mining\data\east_ic原始数据\RawBigLogo', 'Root directory to raw pet dataset.')
flags.DEFINE_string('output_dir', r'C:\Users\sunhongzhi\Desktop\random_big_1000', 'Path to directory to output TFRecords.')
FLAGS = flags.FLAGS


# TODO: Add test for pet/PASCAL main files.
def main(_):
    examples_list = []
    for filename in os.listdir(FLAGS.data_dir):
        img_path = os.path.join(FLAGS.data_dir, filename)
        examples_list.append(img_path)

    random.seed(42)
    random.shuffle(examples_list)
    train_examples = examples_list[:1000]
    print('count = ', len(train_examples))

    for img_path in train_examples:
        print("image_path = " + img_path)
        with PIL.Image.open(img_path) as image:
            filename = os.path.basename(img_path)
            new_img_path = os.path.join(FLAGS.output_dir, filename)
            image.save(new_img_path)


if __name__ == '__main__':
    tf.app.run()
