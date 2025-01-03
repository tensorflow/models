# Copyright 2020 The TensorFlow Authors. All Rights Reserved.
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
"""Tests for tensorflow_model.object_detection.metrics.lvis_tools."""
from lvis import results as lvis_results
import numpy as np
from pycocotools import mask
import tensorflow.compat.v1 as tf
from object_detection.metrics import lvis_tools


class LVISToolsTest(tf.test.TestCase):

  def setUp(self):
    super(LVISToolsTest, self).setUp()
    mask1 = np.pad(
        np.ones([100, 100], dtype=np.uint8),
        ((100, 56), (100, 56)), mode='constant')
    mask2 = np.pad(
        np.ones([50, 50], dtype=np.uint8),
        ((50, 156), (50, 156)), mode='constant')
    mask1_rle = lvis_tools.RleCompress(mask1)
    mask2_rle = lvis_tools.RleCompress(mask2)
    groundtruth_annotations_list = [
        {
            'id': 1,
            'image_id': 1,
            'category_id': 1,
            'bbox': [100., 100., 100., 100.],
            'area': 100.**2,
            'segmentation': mask1_rle
        },
        {
            'id': 2,
            'image_id': 2,
            'category_id': 1,
            'bbox': [50., 50., 50., 50.],
            'area': 50.**2,
            'segmentation': mask2_rle
        },
    ]
    image_list = [
        {
            'id': 1,
            'neg_category_ids': [],
            'not_exhaustive_category_ids': [],
            'height': 256,
            'width': 256
        },
        {
            'id': 2,
            'neg_category_ids': [],
            'not_exhaustive_category_ids': [],
            'height': 256,
            'width': 256
        }
    ]
    category_list = [{'id': 0, 'name': 'person', 'frequency': 'f'},
                     {'id': 1, 'name': 'cat', 'frequency': 'c'},
                     {'id': 2, 'name': 'dog', 'frequency': 'r'}]
    self._groundtruth_dict = {
        'annotations': groundtruth_annotations_list,
        'images': image_list,
        'categories': category_list
    }

    self._detections_list = [
        {
            'image_id': 1,
            'category_id': 1,
            'segmentation': mask1_rle,
            'score': .8
        },
        {
            'image_id': 2,
            'category_id': 1,
            'segmentation': mask2_rle,
            'score': .7
        },
    ]

  def testLVISWrappers(self):
    groundtruth = lvis_tools.LVISWrapper(self._groundtruth_dict)
    detections = lvis_results.LVISResults(groundtruth, self._detections_list)
    evaluator = lvis_tools.LVISEvalWrapper(groundtruth, detections,
                                           iou_type='segm')
    summary_metrics = evaluator.ComputeMetrics()
    self.assertAlmostEqual(1.0, summary_metrics['AP'])

  def testSingleImageDetectionMaskExport(self):
    masks = np.array(
        [[[1, 1,], [1, 1]],
         [[0, 0], [0, 1]],
         [[0, 0], [0, 0]]], dtype=np.uint8)
    classes = np.array([1, 2, 3], dtype=np.int32)
    scores = np.array([0.8, 0.2, 0.7], dtype=np.float32)
    lvis_annotations = lvis_tools.ExportSingleImageDetectionMasksToLVIS(
        image_id=1,
        category_id_set=set([1, 2, 3]),
        detection_classes=classes,
        detection_scores=scores,
        detection_masks=masks)
    expected_counts = ['04', '31', '4']
    for i, mask_annotation in enumerate(lvis_annotations):
      self.assertEqual(mask_annotation['segmentation']['counts'],
                       expected_counts[i])
      self.assertTrue(np.all(np.equal(mask.decode(
          mask_annotation['segmentation']), masks[i])))
      self.assertEqual(mask_annotation['image_id'], 1)
      self.assertEqual(mask_annotation['category_id'], classes[i])
      self.assertAlmostEqual(mask_annotation['score'], scores[i])

  def testSingleImageGroundtruthExport(self):
    masks = np.array(
        [[[1, 1,], [1, 1]],
         [[0, 0], [0, 1]],
         [[0, 0], [0, 0]]], dtype=np.uint8)
    boxes = np.array([[0, 0, 1, 1],
                      [0, 0, .5, .5],
                      [.5, .5, 1, 1]], dtype=np.float32)
    lvis_boxes = np.array([[0, 0, 1, 1],
                           [0, 0, .5, .5],
                           [.5, .5, .5, .5]], dtype=np.float32)
    classes = np.array([1, 2, 3], dtype=np.int32)
    next_annotation_id = 1
    expected_counts = ['04', '31', '4']

    lvis_annotations = lvis_tools.ExportSingleImageGroundtruthToLVIS(
        image_id=1,
        category_id_set=set([1, 2, 3]),
        next_annotation_id=next_annotation_id,
        groundtruth_boxes=boxes,
        groundtruth_classes=classes,
        groundtruth_masks=masks)
    for i, annotation in enumerate(lvis_annotations):
      self.assertEqual(annotation['segmentation']['counts'],
                       expected_counts[i])
      self.assertTrue(np.all(np.equal(mask.decode(
          annotation['segmentation']), masks[i])))
      self.assertTrue(np.all(np.isclose(annotation['bbox'], lvis_boxes[i])))
      self.assertEqual(annotation['image_id'], 1)
      self.assertEqual(annotation['category_id'], classes[i])
      self.assertEqual(annotation['id'], i + next_annotation_id)


if __name__ == '__main__':
  tf.test.main()
