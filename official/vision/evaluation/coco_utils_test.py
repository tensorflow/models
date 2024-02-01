# Copyright 2024 The TensorFlow Authors. All Rights Reserved.
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

import numpy as np
import tensorflow as tf, tf_keras

from official.vision.dataloaders import tfexample_utils
from official.vision.evaluation import coco_utils


class CocoUtilsTest(tf.test.TestCase):

  def test_scan_and_generator_annotation_file(self):
    num_samples = 10
    example = tfexample_utils.create_detection_test_example(
        image_height=512, image_width=512, image_channel=3, num_instances=10
    )
    tf_examples = [example] * num_samples
    data_file = os.path.join(self.create_tempdir(), 'test.tfrecord')
    tfexample_utils.dump_to_tfrecord(
        record_file=data_file, tf_examples=tf_examples
    )
    annotation_file = os.path.join(self.create_tempdir(), 'annotation.json')

    coco_utils.scan_and_generator_annotation_file(
        file_pattern=data_file,
        file_type='tfrecord',
        num_samples=num_samples,
        include_mask=True,
        annotation_file=annotation_file,
    )
    self.assertTrue(
        tf.io.gfile.exists(annotation_file),
        msg='Annotation file {annotation_file} does not exist.',
    )

  def test_convert_keypoint_predictions_to_coco_annotations(self):
    batch_size = 1
    max_num_detections = 3
    num_keypoints = 3
    image_size = 512

    source_id = [np.array([[1]], dtype=int)]
    detection_boxes = [
        np.random.random([batch_size, max_num_detections, 4]) * image_size
    ]
    detection_class = [
        np.random.randint(1, 5, [batch_size, max_num_detections])
    ]
    detection_scores = [np.random.random([batch_size, max_num_detections])]

    detection_keypoints = [
        np.random.random([batch_size, max_num_detections, num_keypoints, 2])
        * image_size
    ]

    predictions = {
        'source_id': source_id,
        'detection_boxes': detection_boxes,
        'detection_classes': detection_class,
        'detection_scores': detection_scores,
        'detection_keypoints': detection_keypoints,
    }
    anns = coco_utils.convert_predictions_to_coco_annotations(predictions)

    for i in range(max_num_detections):
      expected_keypoint_ann = np.concatenate(
          [
              np.expand_dims(detection_keypoints[0][0, i, :, 1], axis=-1),
              np.expand_dims(detection_keypoints[0][0, i, :, 0], axis=-1),
              np.expand_dims(np.ones(num_keypoints), axis=1),
          ],
          axis=1,
      ).astype(int)
      expected_keypoint_ann = expected_keypoint_ann.flatten().tolist()
      self.assertAllEqual(anns[i]['keypoints'], expected_keypoint_ann)


if __name__ == '__main__':
  tf.test.main()
