# Copyright 2016 Google Inc. All Rights Reserved.
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

r"""LSUN dataset formatting.

Download and format the LSUN dataset as follow:
git clone https://github.com/fyu/lsun.git
cd lsun
python2.7 download.py -c [CATEGORY]

Then unzip the downloaded .zip files before executing:
python2.7 data.py export [IMAGE_DB_PATH] --out_dir [LSUN_FOLDER] --flat

Then use the script as follow:
python lsun_formatting.py \
    --file_out [OUTPUT_FILE_PATH_PREFIX] \
    --fn_root [LSUN_FOLDER]

"""
from __future__ import print_function

import os
import os.path

import numpy
import skimage.transform
from PIL import Image
import tensorflow as tf


tf.flags.DEFINE_string("file_out", "",
                       "Filename of the output .tfrecords file.")
tf.flags.DEFINE_string("fn_root", "", "Name of root file path.")

FLAGS = tf.flags.FLAGS


def _int64_feature(value):
    return tf.train.Feature(int64_list=tf.train.Int64List(value=[value]))


def _bytes_feature(value):
    return tf.train.Feature(bytes_list=tf.train.BytesList(value=[value]))


def main():
    """Main converter function."""
    fn_root = FLAGS.fn_root
    img_fn_list = os.listdir(fn_root)
    img_fn_list = [img_fn for img_fn in img_fn_list
                   if img_fn.endswith('.webp')]
    num_examples = len(img_fn_list)

    n_examples_per_file = 10000
    for example_idx, img_fn in enumerate(img_fn_list):
        if example_idx % n_examples_per_file == 0:
            file_out = "%s_%05d.tfrecords"
            file_out = file_out % (FLAGS.file_out,
                                   example_idx // n_examples_per_file)
            print("Writing on:", file_out)
            writer = tf.python_io.TFRecordWriter(file_out)
        if example_idx % 1000 == 0:
            print(example_idx, "/", num_examples)
        image_raw = numpy.array(Image.open(os.path.join(fn_root, img_fn)))
        rows = image_raw.shape[0]
        cols = image_raw.shape[1]
        depth = image_raw.shape[2]
        downscale = min(rows / 96., cols / 96.)
        image_raw = skimage.transform.pyramid_reduce(image_raw, downscale)
        image_raw *= 255.
        image_raw = image_raw.astype("uint8")
        rows = image_raw.shape[0]
        cols = image_raw.shape[1]
        depth = image_raw.shape[2]
        image_raw = image_raw.tostring()
        example = tf.train.Example(
            features=tf.train.Features(
                feature={
                    "height": _int64_feature(rows),
                    "width": _int64_feature(cols),
                    "depth": _int64_feature(depth),
                    "image_raw": _bytes_feature(image_raw)
                }
            )
        )
        writer.write(example.SerializeToString())
        if example_idx % n_examples_per_file == (n_examples_per_file - 1):
            writer.close()
    writer.close()


if __name__ == "__main__":
    main()
