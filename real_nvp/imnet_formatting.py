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

Download and format the Imagenet dataset as follow:
mkdir [IMAGENET_PATH]
cd [IMAGENET_PATH]
for FILENAME in train_32x32.tar valid_32x32.tar train_64x64.tar valid_64x64.tar
do
    curl -O http://image-net.org/small/$FILENAME
    tar -xvf $FILENAME
done

Then use the script as follow:
for DIRNAME in train_32x32 valid_32x32 train_64x64 valid_64x64
do
    python imnet_formatting.py \
        --file_out $DIRNAME \
        --fn_root $DIRNAME
done

"""

import os
import os.path

import scipy.io
import scipy.io.wavfile
import scipy.ndimage
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
    # LSUN
    fn_root = FLAGS.fn_root
    img_fn_list = os.listdir(fn_root)
    img_fn_list = [img_fn for img_fn in img_fn_list
                   if img_fn.endswith('.png')]
    num_examples = len(img_fn_list)

    n_examples_per_file = 10000
    for example_idx, img_fn in enumerate(img_fn_list):
        if example_idx % n_examples_per_file == 0:
            file_out = "%s_%05d.tfrecords"
            file_out = file_out % (FLAGS.file_out,
                                   example_idx // n_examples_per_file)
            print "Writing on:", file_out
            writer = tf.python_io.TFRecordWriter(file_out)
        if example_idx % 1000 == 0:
            print example_idx, "/", num_examples
        image_raw = scipy.ndimage.imread(os.path.join(fn_root, img_fn))
        rows = image_raw.shape[0]
        cols = image_raw.shape[1]
        depth = image_raw.shape[2]
        image_raw = image_raw.astype("uint8")
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
