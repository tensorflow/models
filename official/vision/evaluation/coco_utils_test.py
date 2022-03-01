# Copyright 2022 The TensorFlow Authors. All Rights Reserved.
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

"""Tests for coco_utils."""

import os

import tensorflow as tf

from official.vision.dataloaders import tfexample_utils
from official.vision.evaluation import coco_utils


class CocoUtilsTest(tf.test.TestCase):

  def test_scan_and_generator_annotation_file(self):
    num_samples = 10
    example = tfexample_utils.create_detection_test_example(
        image_height=512, image_width=512, image_channel=3, num_instances=10)
    tf_examples = [example] * num_samples
    data_file = os.path.join(self.create_tempdir(), 'test.tfrecord')
    tfexample_utils.dump_to_tfrecord(
        record_file=data_file, tf_examples=tf_examples)
    annotation_file = os.path.join(self.create_tempdir(), 'annotation.json')

    coco_utils.scan_and_generator_annotation_file(
        file_pattern=data_file,
        file_type='tfrecord',
        num_samples=num_samples,
        include_mask=True,
        annotation_file=annotation_file)
    self.assertTrue(
        tf.io.gfile.exists(annotation_file),
        msg='Annotation file {annotation_file} does not exists.')


if __name__ == '__main__':
  tf.test.main()
