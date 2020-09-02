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
"""Tests for coco_evaluator."""

import io
import os

# Import libraries

from absl import logging
from absl.testing import absltest
from absl.testing import parameterized
import numpy as np
from PIL import Image
from pycocotools.coco import COCO
import six
import tensorflow as tf

from official.vision.beta.evaluation import coco_evaluator
from official.vision.beta.evaluation import coco_utils

_COCO_JSON_FILE = '/placer/prod/home/snaggletooth/test/data/coco/instances_val2017.json'
_SAVED_COCO_JSON_FILE = 'tmp.json'


def get_groundtruth_annotations(image_id, coco, include_mask=False):
  anns = coco.loadAnns(coco.getAnnIds([image_id]))
  if not anns:
    return None

  image = coco.loadImgs([image_id])[0]

  groundtruths = {
      'boxes': [],
      'classes': [],
      'is_crowds': [],
      'areas': [],
  }
  if include_mask:
    groundtruths['masks'] = []
  for ann in anns:
    # Creates detections from groundtruths.
    # Converts [x, y, w, h] to [y1, x1, y2, x2] box format.
    box = [ann['bbox'][1],
           ann['bbox'][0],
           (ann['bbox'][1] + ann['bbox'][3]),
           (ann['bbox'][0] + ann['bbox'][2])]

    # Creates groundtruths.
    groundtruths['boxes'].append(box)
    groundtruths['classes'].append(ann['category_id'])
    groundtruths['is_crowds'].append(ann['iscrowd'])
    groundtruths['areas'].append(ann['area'])
    if include_mask:
      mask_img = Image.fromarray(coco.annToMask(ann).astype(np.uint8))
      with io.BytesIO() as stream:
        mask_img.save(stream, format='PNG')
        mask_bytes = stream.getvalue()
      groundtruths['masks'].append(mask_bytes)
  for key, val in groundtruths.items():
    groundtruths[key] = np.stack(val, axis=0)
  groundtruths['source_id'] = image['id']
  groundtruths['height'] = image['height']
  groundtruths['width'] = image['width']
  groundtruths['num_detections'] = len(anns)

  for k, v in six.iteritems(groundtruths):
    groundtruths[k] = np.expand_dims(v, axis=0)
  return groundtruths


def get_predictions(image_id, coco, include_mask=False):
  anns = coco.loadAnns(coco.getAnnIds([image_id]))
  if not anns:
    return None

  image = coco.loadImgs([image_id])[0]

  predictions = {
      'detection_boxes': [],
      'detection_classes': [],
      'detection_scores': [],
  }
  if include_mask:
    predictions['detection_masks'] = []
  for ann in anns:
    # Creates detections from groundtruths.
    # Converts [x, y, w, h] to [y1, x1, y2, x2] box format and
    # does the denormalization.
    box = [ann['bbox'][1],
           ann['bbox'][0],
           (ann['bbox'][1] + ann['bbox'][3]),
           (ann['bbox'][0] + ann['bbox'][2])]

    predictions['detection_boxes'].append(box)
    predictions['detection_classes'].append(ann['category_id'])
    predictions['detection_scores'].append(1)
    if include_mask:
      mask = coco.annToMask(ann)
      predictions['detection_masks'].append(mask)
  for key, val in predictions.items():
    predictions[key] = np.expand_dims(np.stack(val, axis=0), axis=0)

  predictions['source_id'] = np.array([image['id']])
  predictions['num_detections'] = np.array([len(anns)])
  predictions['image_info'] = np.array(
      [[[image['height'], image['width']],
        [image['height'], image['width']],
        [1, 1],
        [0, 0]]], dtype=np.float32)

  return predictions


def get_fake_predictions(image_id, coco, include_mask=False):
  anns = coco.loadAnns(coco.getAnnIds([image_id]))
  if not anns:
    return None

  label_id_max = max([ann['category_id'] for ann in anns])

  image = coco.loadImgs([image_id])[0]

  num_detections = 100
  xmin = np.random.randint(
      low=0, high=int(image['width'] / 2), size=(1, num_detections))
  xmax = np.random.randint(
      low=int(image['width'] / 2), high=image['width'],
      size=(1, num_detections))
  ymin = np.random.randint(
      low=0, high=int(image['height'] / 2), size=(1, num_detections))
  ymax = np.random.randint(
      low=int(image['height'] / 2), high=image['height'],
      size=(1, num_detections))
  predictions = {
      'detection_boxes': np.stack([ymin, xmin, ymax, xmax], axis=-1),
      'detection_classes': np.random.randint(
          low=0, high=(label_id_max + 1), size=(1, num_detections)),
      'detection_scores': np.random.random(size=(1, num_detections)),
  }
  if include_mask:
    predictions['detection_masks'] = np.random.randint(
        1, size=(1, num_detections, image['height'], image['width']))
  predictions['source_id'] = np.array([image['id']])
  predictions['num_detections'] = np.array([num_detections])
  predictions['image_info'] = np.array(
      [[[image['height'], image['width']],
        [image['height'], image['width']],
        [1, 1],
        [0, 0]]], dtype=np.float32)

  return predictions


class DummyGroundtruthGenerator(object):

  def __init__(self, include_mask, image_id, coco):
    self._include_mask = include_mask
    self._image_id = image_id
    self._coco = coco

  def __call__(self):
    yield get_groundtruth_annotations(
        self._image_id, self._coco, self._include_mask)


class COCOEvaluatorTest(parameterized.TestCase, absltest.TestCase):

  def setUp(self):
    super(COCOEvaluatorTest, self).setUp()
    temp = self.create_tempdir()
    self._saved_coco_json_file = os.path.join(temp.full_path,
                                              _SAVED_COCO_JSON_FILE)

  def tearDown(self):
    super(COCOEvaluatorTest, self).tearDown()

  @parameterized.parameters(
      (False, False), (False, True), (True, False), (True, True))
  def testEval(self, include_mask, use_fake_predictions):
    coco = COCO(annotation_file=_COCO_JSON_FILE)
    index = np.random.randint(len(coco.dataset['images']))
    image_id = coco.dataset['images'][index]['id']
    # image_id = 26564
    # image_id = 324158
    if use_fake_predictions:
      predictions = get_fake_predictions(
          image_id, coco, include_mask=include_mask)
    else:
      predictions = get_predictions(image_id, coco, include_mask=include_mask)

    if not predictions:
      logging.info('Empty predictions for index=%d', index)
      return

    predictions = tf.nest.map_structure(
        lambda x: tf.convert_to_tensor(x) if x is not None else None,
        predictions)

    evaluator_w_json = coco_evaluator.COCOEvaluator(
        annotation_file=_COCO_JSON_FILE, include_mask=include_mask)
    evaluator_w_json.update_state(groundtruths=None, predictions=predictions)
    results_w_json = evaluator_w_json.result()

    dummy_generator = DummyGroundtruthGenerator(
        include_mask=include_mask, image_id=image_id, coco=coco)
    coco_utils.generate_annotation_file(dummy_generator,
                                        self._saved_coco_json_file)
    evaluator_no_json = coco_evaluator.COCOEvaluator(
        annotation_file=self._saved_coco_json_file, include_mask=include_mask)
    evaluator_no_json.update_state(groundtruths=None, predictions=predictions)
    results_no_json = evaluator_no_json.result()

    for k, v in results_w_json.items():
      self.assertEqual(v, results_no_json[k])

  @parameterized.parameters(
      (False, False), (False, True), (True, False), (True, True))
  def testEvalOnTheFly(self, include_mask, use_fake_predictions):
    coco = COCO(annotation_file=_COCO_JSON_FILE)
    index = np.random.randint(len(coco.dataset['images']))
    image_id = coco.dataset['images'][index]['id']
    # image_id = 26564
    # image_id = 324158
    if use_fake_predictions:
      predictions = get_fake_predictions(
          image_id, coco, include_mask=include_mask)
    else:
      predictions = get_predictions(image_id, coco, include_mask=include_mask)

    if not predictions:
      logging.info('Empty predictions for index=%d', index)
      return

    predictions = tf.nest.map_structure(
        lambda x: tf.convert_to_tensor(x) if x is not None else None,
        predictions)
    evaluator_w_json = coco_evaluator.COCOEvaluator(
        annotation_file=_COCO_JSON_FILE, include_mask=include_mask)
    evaluator_w_json.update_state(groundtruths=None, predictions=predictions)
    results_w_json = evaluator_w_json.result()

    groundtruths = get_groundtruth_annotations(image_id, coco, include_mask)
    groundtruths = tf.nest.map_structure(
        lambda x: tf.convert_to_tensor(x) if x is not None else None,
        groundtruths)

    evaluator_no_json = coco_evaluator.COCOEvaluator(
        annotation_file=None, include_mask=include_mask)
    evaluator_no_json.update_state(groundtruths, predictions)
    results_no_json = evaluator_no_json.result()

    for k, v in results_w_json.items():
      self.assertEqual(v, results_no_json[k])


if __name__ == '__main__':
  absltest.main()
