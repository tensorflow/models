# Copyright 2015 Paul Balanca. All Rights Reserved.
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
"""Converts Pascal VOC data to TFRecords file format with Example protos.

The raw Pascal VOC data set is expected to reside in JPEG files located in the
directory 'JPEGImages'. Similarly, bounding box annotations are supposed to be
stored in the 'Annotation directory'

This TensorFlow script converts the training and evaluation data into
a sharded data set consisting of 1024 and 128 TFRecord files, respectively.

Each validation TFRecord file contains ~500 records. Each training TFREcord
file contains ~1000 records. Each record within the TFRecord file is a
serialized Example proto. The Example proto contains the following fields:

    image/encoded: string containing JPEG encoded image in RGB colorspace
    image/height: integer, image height in pixels
    image/width: integer, image width in pixels
    image/channels: integer, specifying the number of channels, always 3
    image/format: string, specifying the format, always'JPEG'


    image/object/bbox/xmin: list of float specifying the 0+ human annotated
        bounding boxes
    image/object/bbox/xmax: list of float specifying the 0+ human annotated
        bounding boxes
    image/object/bbox/ymin: list of float specifying the 0+ human annotated
        bounding boxes
    image/object/bbox/ymax: list of float specifying the 0+ human annotated
        bounding boxes
    image/object/bbox/label: list of integer specifying the classification index.
    image/object/bbox/label_text: list of string descriptions.

Note that the length of xmin is identical to the length of xmax, ymin and ymax
for each example.
"""
import os
import sys
import random
import warnings

import cv2 as cv
import numpy as np
import tensorflow as tf

from gdxray_parser import get_images
from dataset_utils import int64_feature, float_feature, bytes_feature
from tensorflow import python_io# import TFRecordWriter

# TFRecords convertion parameters.
RANDOM_SEED = 4242

DEFECT_LABEL = 1
DEFECT_LABEL_TEXT = "defect"


def float_list_feature(value):
  return tf.train.Feature(float_list=tf.train.FloatList(value=value))

def _process_image(sess, image):
    """Process a image and annotation file.

    Args:
      filename: string, path to an image file e.g., '/path/to/example.JPG'.
      coder: instance of ImageCoder to provide TensorFlow image coding utils.
    Returns:
      image_buffer: string, JPEG encoding of RGB image.
      height: integer, image height in pixels.
      width: integer, image width in pixels.
    """
    # Read the image file.
    filename = image.filename
    image_data = _image_to_bytes(sess, image.pixels)

    # Image shape.
    shape = [int(image.height),
             int(image.width),
             int(image.depth)]
    # Find annotations.
    bboxes = []
    labels = []
    labels_text = []

    for row in range(image.boxes.shape[0]):
        labels.append(DEFECT_LABEL)
        labels_text.append(DEFECT_LABEL_TEXT.encode('ascii'))

        box = image.boxes[row,:]
        xmin = float(box[1]) / shape[1]
        xmax = float(box[2]) / shape[1]
        ymin = float(box[3]) / shape[0]
        ymax = float(box[4]) / shape[0]

        for test in [xmin,xmax,ymin,ymax]:
          if test<0 or test>1:
            warnings.warn("Box bounds must be between 0 and 1. Got %.4f"%test)

        xmin = min(xmin,1)
        xmax = min(xmax,1)
        ymin = min(ymin,1)
        ymax = min(ymax,1)

        bboxes.append((xmin, xmax, ymin, ymax))
    return image_data, shape, bboxes, labels, labels_text


def _convert_to_example(image_data, labels, labels_text, bboxes, shape):
    """Build an Example proto for an image example.

    Args:
      image_data: string, JPEG encoding of RGB image;
      labels: list of integers, identifier for the ground truth;
      labels_text: list of strings, human-readable labels;
      bboxes: list of bounding boxes; each box is a list of integers;
          specifying [xmin, ymin, xmax, ymax]. All boxes are assumed to belong
          to the same label as the image label.
      shape: 3 integers, image shapes in pixels.
    Returns:
      Example proto
    """
    xmin = []
    ymin = []
    xmax = []
    ymax = []
    for b in bboxes:
        assert len(b) == 4
        [l.append(point) for l, point in zip([xmin, xmax, ymin, ymax], b)]

    print('xmin:',xmin)
    print('xmax:',xmax)
    print('ymin:',ymin)
    print('ymax:',ymax)

    image_format = b'PNG'
    example = tf.train.Example(features=tf.train.Features(feature={
            'image/height': int64_feature(shape[0]),
            'image/width': int64_feature(shape[1]),
            'image/channels': int64_feature(shape[2]),
            'image/shape': int64_feature(shape),
            'image/object/bbox/xmin': float_list_feature(xmin),
            'image/object/bbox/xmax': float_list_feature(xmax),
            'image/object/bbox/ymin': float_list_feature(ymin),
            'image/object/bbox/ymax': float_list_feature(ymax),
            'image/object/class/label': int64_feature(labels),
            'image/object/class/text': bytes_feature(labels_text),
            'image/format': bytes_feature(image_format),
            'image/encoded': bytes_feature(image_data)}))
    return example


def _add_to_tfrecord(sess, image, tfrecord_writer):
    """Loads data from image and annotations files and add them to a TFRecord.

    Args:
      dataset_dir: Dataset directory;
      name: Image name to add to the TFRecord;
      tfrecord_writer: The TFRecord writer to use for writing.
    """
    image_data, shape, bboxes, labels, labels_text = _process_image(sess, image)
    example = _convert_to_example(image_data, labels, labels_text, bboxes, shape)
    tfrecord_writer.write(example.SerializeToString())
    _save_to_file(image, shape, bboxes)


def _save_to_file(image, shape, bboxes):
    """Save image to file with bounding boxes"""
    pixels = image.pixels
    filename = os.path.basename(image.filename)
    for box in bboxes:
        xmin = int(box[0] * shape[1])
        xmax = int(box[1] * shape[1])
        ymin = int(box[2] * shape[0])
        ymax = int(box[3] * shape[0])
        cv.rectangle(pixels, (xmin,ymin), (xmax,ymax), (0,0,255), 1)
    cv.imwrite(os.path.join('pictures','test',filename), pixels)


def _image_to_bytes(sess, image_data):
    """Return the image as png encoded bytes"""
    inputs = tf.placeholder(dtype=tf.uint8, shape=image_data.shape)
    print('shape:', image_data.shape)
    encoded_png = tf.image.encode_png(inputs)
    return sess.run(encoded_png, feed_dict={inputs: image_data})


def _get_output_filename(output_dir, train_or_eval, name):
    return '%s/%s_%s.tfrecord' % (output_dir, train_or_eval, name)


def _get_test_train_split():
    train_file = "data/gdxray_test_split.txt"
    test_file = "data/gdxray_train_split.txt"

    with open(train_file) as f1, open(test_file) as f2:
        train_split = [x.strip() for x in f1] 
        test_split = [x.strip() for x in f2] 
    return train_split, test_split


def run(output_dir, name='gdxray_train', shuffling=False):
    """Runs the conversion operation.

    Args:
      dataset_dir: The dataset directory where the dataset is stored.
      output_dir: Output directory.
    """

    # Process dataset files.
    tf_train_filename = _get_output_filename(output_dir, 'train', name)
    tf_eval_filename = _get_output_filename(output_dir, 'train', name)
    train_split, eval_split = ([],[])#_get_test_train_split()

    with tf.Session() as sess:
        
        i = 0
        print("Writing train file to", tf_train_filename)
        with python_io.TFRecordWriter(tf_train_filename) as train_writer:
            train_files = []
            test_files = []

            for image in get_images():
                print('\r>> Converting image [%i]: %s'%(i, image.filename))
                #if len(image.bboxes) and image.filename in train_split:
                #TEMP START
                if len(image.boxes):
                    if random.random()<0.8:
                        train_files.append(image.filename)
                        _add_to_tfrecord(sess, image, train_writer)
                        i += 1
                    else:
                        test_files.append(image.filename)


            with open("data/gdxray_train_split.txt", 'w') as f:
                print(train_files)
                f.write('\n'.join(train_files))

            with open("data/gdxray_eval_split.txt", 'w') as f:
                print(test_files)
                f.write('\n'.join(test_files))

        i = 0
        print("Writing eval file to", tf_eval_filename)
        with python_io.TFRecordWriter(tf_eval_filename) as eval_writer:
            for image in get_images():
                print('\r>> Converting image [%i]: %s'%(i, image.filename))
                if image.filename in eval_split:
                    _add_to_tfrecord(sess, image, eval_writer)
                    i += 1

    print('\nFinished converting the GDXray Dataset!')



if __name__=="__main__":
    output = os.path.expanduser("~/Data/GDXRay")
    run(output_dir=output)