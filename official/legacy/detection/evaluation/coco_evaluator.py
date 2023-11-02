# Copyright 2023 The TensorFlow Authors. All Rights Reserved.
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

"""The COCO-style evaluator.

The following snippet demonstrates the use of interfaces:

  evaluator = COCOEvaluator(...)
  for _ in range(num_evals):
    for _ in range(num_batches_per_eval):
      predictions, groundtruth = predictor.predict(...)  # pop a batch.
      evaluator.update(predictions, groundtruths)  # aggregate internal stats.
    evaluator.evaluate()  # finish one full eval.

See also: https://github.com/cocodataset/cocoapi/
"""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import atexit
import copy
import tempfile

from absl import logging
import numpy as np
from pycocotools import cocoeval
import six
import tensorflow as tf, tf_keras

from official.legacy.detection.evaluation import coco_utils
from official.legacy.detection.utils import class_utils


class OlnCOCOevalWrapper(cocoeval.COCOeval):
  """COCOeval wrapper class.

  Rewritten based on cocoapi: (pycocotools/cocoeval.py)

  This class wraps COCOEVAL API object, which provides the following additional
  functionalities:
    1. summarze 'all', 'seen', and 'novel' split output print-out, e.g., AR at
       different K proposals, AR and AP resutls for 'seen' and 'novel' class
       splits.
  """

  def __init__(self, coco_gt, coco_dt, iou_type='box'):
    super(OlnCOCOevalWrapper, self).__init__(
        cocoGt=coco_gt, cocoDt=coco_dt, iouType=iou_type)

  def summarize(self):
    """Compute and display summary metrics for evaluation results.

    Delta to the standard cocoapi function:
      More Averate Recall metrics are produced with different top-K proposals.
    Note this functin can *only* be applied on the default parameter
    setting.
    Raises:
      Exception: Please run accumulate() first.
    """

    def _summarize(ap=1, iou_thr=None, area_rng='all', max_dets=100):
      p = self.params
      i_str = (' {:<18} {} @[ IoU={:<9} | area={:>6s} | maxDets={:>3d} ] = '
               '{:0.3f}')
      title_str = 'Average Precision' if ap == 1 else 'Average Recall'
      type_str = '(AP)' if ap == 1 else '(AR)'
      iou_str = '{:0.2f}:{:0.2f}'.format(
          p.iouThrs[0],
          p.iouThrs[-1]) if iou_thr is None else '{:0.2f}'.format(iou_thr)

      aind = [i for i, a_rng in enumerate(p.areaRngLbl) if a_rng == area_rng]
      mind = [i for i, m_det in enumerate(p.maxDets) if m_det == max_dets]
      if ap == 1:
        # dimension of precision: [TxRxKxAxM]
        s = self.eval['precision']
        # IoU
        if iou_thr is not None:
          t = np.where(iou_thr == p.iouThrs)[0]
          s = s[t]
        s = s[:, :, :, aind, mind]
      else:
        # dimension of recall: [TxKxAxM]
        s = self.eval['recall']
        if iou_thr is not None:
          t = np.where(iou_thr == p.iouThrs)[0]
          s = s[t]
        s = s[:, :, aind, mind]

      if not (s[s > -1]).any():
        mean_s = -1
      else:
        mean_s = np.mean(s[s > -1])
        print(
            i_str.format(title_str, type_str, iou_str, area_rng, max_dets,
                         mean_s))
      return mean_s

    def _summarize_dets():
      stats = np.zeros((14,))
      stats[0] = _summarize(1)
      stats[1] = _summarize(
          1,
          iou_thr=.5,
      )
      stats[2] = _summarize(
          1,
          iou_thr=.75,
      )
      stats[3] = _summarize(
          1,
          area_rng='small',
      )
      stats[4] = _summarize(
          1,
          area_rng='medium',
      )
      stats[5] = _summarize(
          1,
          area_rng='large',
      )

      stats[6] = _summarize(0, max_dets=self.params.maxDets[0])  # 10
      stats[7] = _summarize(0, max_dets=self.params.maxDets[1])  # 20
      stats[8] = _summarize(0, max_dets=self.params.maxDets[2])  # 50
      stats[9] = _summarize(0, max_dets=self.params.maxDets[3])  # 100
      stats[10] = _summarize(0, max_dets=self.params.maxDets[4])  # 200

      stats[11] = _summarize(0, area_rng='small', max_dets=10)
      stats[12] = _summarize(0, area_rng='medium', max_dets=10)
      stats[13] = _summarize(0, area_rng='large', max_dets=10)
      return stats

    if not self.eval:
      raise Exception('Please run accumulate() first')
    summarize = _summarize_dets
    self.stats = summarize()


class OlnCOCOevalXclassWrapper(OlnCOCOevalWrapper):
  """COCOeval wrapper class.

  Rewritten based on cocoapi: (pycocotools/cocoeval.py)
  Delta to the standard cocoapi:
    Detections that hit the 'seen' class objects are ignored in top-K proposals.

  This class wraps COCOEVAL API object, which provides the following additional
  functionalities:
    1. Include ignore-class split (e.g., 'voc' or 'nonvoc').
    2. Do not count (or ignore) box proposals hitting ignore-class when
       evaluating Average Recall at top-K proposals.
  """

  def __init__(self, coco_gt, coco_dt, iou_type='box'):
    super(OlnCOCOevalXclassWrapper, self).__init__(
        coco_gt=coco_gt, coco_dt=coco_dt, iou_type=iou_type)

  def evaluateImg(self, img_id, cat_id, a_rng, max_det):
    p = self.params
    if p.useCats:
      gt = self._gts[img_id, cat_id]
      dt = self._dts[img_id, cat_id]
    else:
      gt, dt = [], []
      for c_id in p.catIds:
        gt.extend(self._gts[img_id, c_id])
        dt.extend(self._dts[img_id, c_id])

    if not gt and not dt:
      return None

    for g in gt:
      if g['ignore'] or (g['area'] < a_rng[0] or g['area'] > a_rng[1]):
        g['_ignore'] = 1
      else:
        g['_ignore'] = 0
      # Class manipulation: ignore the 'ignored_split'.
      if 'ignored_split' in g and g['ignored_split'] == 1:
        g['_ignore'] = 1

    # sort dt highest score first, sort gt ignore last
    gtind = np.argsort([g['_ignore'] for g in gt], kind='mergesort')
    gt = [gt[i] for i in gtind]
    dtind = np.argsort([-d['score'] for d in dt], kind='mergesort')
    dt = [dt[i] for i in dtind[0:max_det]]
    iscrowd = [int(o['iscrowd']) for o in gt]
    # load computed ious
    # ious = self.ious[img_id, cat_id][:, gtind] if len(
    #     self.ious[img_id, cat_id]) > 0 else self.ious[img_id, cat_id]
    if self.ious[img_id, cat_id].any():
      ious = self.ious[img_id, cat_id][:, gtind]
    else:
      ious = self.ious[img_id, cat_id]

    tt = len(p.iouThrs)
    gg = len(gt)
    dd = len(dt)
    gtm = np.zeros((tt, gg))
    dtm = np.zeros((tt, dd))
    gt_ig = np.array([g['_ignore'] for g in gt])
    dt_ig = np.zeros((tt, dd))
    # indicator of whether the gt object class is of ignored_split or not.
    gt_ig_split = np.array([g['ignored_split'] for g in gt])
    dt_ig_split = np.zeros((dd))

    if ious.any():
      for tind, t in enumerate(p.iouThrs):
        for dind, d in enumerate(dt):
          # information about best match so far (m=-1 -> unmatched)
          iou = min([t, 1 - 1e-10])
          m = -1
          for gind, g in enumerate(gt):
            # if this gt already matched, and not a crowd, continue
            if gtm[tind, gind] > 0 and not iscrowd[gind]:
              continue
            # if dt matched to reg gt, and on ignore gt, stop
            if m > -1 and gt_ig[m] == 0 and gt_ig[gind] == 1:
              break
            # continue to next gt unless better match made
            if ious[dind, gind] < iou:
              continue
            # if match successful and best so far, store appropriately
            iou = ious[dind, gind]
            m = gind
          # if match made store id of match for both dt and gt
          if m == -1:
            continue
          dt_ig[tind, dind] = gt_ig[m]
          dtm[tind, dind] = gt[m]['id']
          gtm[tind, m] = d['id']

          # Activate to ignore the seen-class detections.
          if tind == 0:  # Register just only once: tind > 0 is also fine.
            dt_ig_split[dind] = gt_ig_split[m]

    # set unmatched detections outside of area range to ignore
    a = np.array([d['area'] < a_rng[0] or d['area'] > a_rng[1] for d in dt
                 ]).reshape((1, len(dt)))
    dt_ig = np.logical_or(dt_ig, np.logical_and(dtm == 0, np.repeat(a, tt, 0)))

    # Activate to ignore the seen-class detections.
    # Take only eval_split (eg, nonvoc) and ignore seen_split (eg, voc).
    if dt_ig_split.sum() > 0:
      dtm = dtm[:, dt_ig_split == 0]
      dt_ig = dt_ig[:, dt_ig_split == 0]
      len_dt = min(max_det, len(dt))
      dt = [dt[i] for i in range(len_dt) if dt_ig_split[i] == 0]

    # store results for given image and category
    return {
        'image_id': img_id,
        'category_id': cat_id,
        'aRng': a_rng,
        'maxDet': max_det,
        'dtIds': [d['id'] for d in dt],
        'gtIds': [g['id'] for g in gt],
        'dtMatches': dtm,
        'gtMatches': gtm,
        'dtScores': [d['score'] for d in dt],
        'gtIgnore': gt_ig,
        'dtIgnore': dt_ig,
    }


class MetricWrapper(object):
  """Metric Wrapper of the COCO evaluator."""
  # This is only a wrapper for COCO metric and works on for numpy array. So it
  # doesn't inherit from tf_keras.layers.Layer or tf_keras.metrics.Metric.

  def __init__(self, evaluator):
    self._evaluator = evaluator

  def update_state(self, y_true, y_pred):
    """Update internal states."""
    labels = tf.nest.map_structure(lambda x: x.numpy(), y_true)
    outputs = tf.nest.map_structure(lambda x: x.numpy(), y_pred)
    groundtruths = {}
    predictions = {}
    for key, val in outputs.items():
      if isinstance(val, tuple):
        val = np.concatenate(val)
      predictions[key] = val
    for key, val in labels.items():
      if isinstance(val, tuple):
        val = np.concatenate(val)
      groundtruths[key] = val
    self._evaluator.update(predictions, groundtruths)

  def result(self):
    return self._evaluator.evaluate()

  def reset_states(self):
    return self._evaluator.reset()


class COCOEvaluator(object):
  """COCO evaluation metric class."""

  def __init__(self, annotation_file, include_mask, need_rescale_bboxes=True):
    """Constructs COCO evaluation class.

    The class provides the interface to metrics_fn in TPUEstimator. The
    _update_op() takes detections from each image and push them to
    self.detections. The _evaluate() loads a JSON file in COCO annotation format
    as the groundtruths and runs COCO evaluation.

    Args:
      annotation_file: a JSON file that stores annotations of the eval dataset.
        If `annotation_file` is None, groundtruth annotations will be loaded
        from the dataloader.
      include_mask: a boolean to indicate whether or not to include the mask
        eval.
      need_rescale_bboxes: If true bboxes in `predictions` will be rescaled back
        to absolute values (`image_info` is needed in this case).
    """
    if annotation_file:
      if annotation_file.startswith('gs://'):
        _, local_val_json = tempfile.mkstemp(suffix='.json')
        tf.io.gfile.remove(local_val_json)

        tf.io.gfile.copy(annotation_file, local_val_json)
        atexit.register(tf.io.gfile.remove, local_val_json)
      else:
        local_val_json = annotation_file
      self._coco_gt = coco_utils.COCOWrapper(
          eval_type=('mask' if include_mask else 'box'),
          annotation_file=local_val_json)
    self._annotation_file = annotation_file
    self._include_mask = include_mask
    self._metric_names = [
        'AP', 'AP50', 'AP75', 'APs', 'APm', 'APl', 'ARmax1', 'ARmax10',
        'ARmax100', 'ARs', 'ARm', 'ARl'
    ]
    self._required_prediction_fields = [
        'source_id', 'num_detections', 'detection_classes', 'detection_scores',
        'detection_boxes'
    ]
    self._need_rescale_bboxes = need_rescale_bboxes
    if self._need_rescale_bboxes:
      self._required_prediction_fields.append('image_info')
    self._required_groundtruth_fields = [
        'source_id', 'height', 'width', 'classes', 'boxes'
    ]
    if self._include_mask:
      mask_metric_names = ['mask_' + x for x in self._metric_names]
      self._metric_names.extend(mask_metric_names)
      self._required_prediction_fields.extend(['detection_masks'])
      self._required_groundtruth_fields.extend(['masks'])

    self.reset()

  def reset(self):
    """Resets internal states for a fresh run."""
    self._predictions = {}
    if not self._annotation_file:
      self._groundtruths = {}

  def evaluate(self):
    """Evaluates with detections from all images with COCO API.

    Returns:
      coco_metric: float numpy array with shape [24] representing the
        coco-style evaluation metrics (box and mask).
    """
    if not self._annotation_file:
      logging.info('Thre is no annotation_file in COCOEvaluator.')
      gt_dataset = coco_utils.convert_groundtruths_to_coco_dataset(
          self._groundtruths)
      coco_gt = coco_utils.COCOWrapper(
          eval_type=('mask' if self._include_mask else 'box'),
          gt_dataset=gt_dataset)
    else:
      logging.info('Using annotation file: %s', self._annotation_file)
      coco_gt = self._coco_gt
    coco_predictions = coco_utils.convert_predictions_to_coco_annotations(
        self._predictions)
    coco_dt = coco_gt.loadRes(predictions=coco_predictions)
    image_ids = [ann['image_id'] for ann in coco_predictions]

    coco_eval = cocoeval.COCOeval(coco_gt, coco_dt, iouType='bbox')
    coco_eval.params.imgIds = image_ids
    coco_eval.evaluate()
    coco_eval.accumulate()
    coco_eval.summarize()
    coco_metrics = coco_eval.stats

    if self._include_mask:
      mcoco_eval = cocoeval.COCOeval(coco_gt, coco_dt, iouType='segm')
      mcoco_eval.params.imgIds = image_ids
      mcoco_eval.evaluate()
      mcoco_eval.accumulate()
      mcoco_eval.summarize()
      mask_coco_metrics = mcoco_eval.stats

    if self._include_mask:
      metrics = np.hstack((coco_metrics, mask_coco_metrics))
    else:
      metrics = coco_metrics

    # Cleans up the internal variables in order for a fresh eval next time.
    self.reset()

    metrics_dict = {}
    for i, name in enumerate(self._metric_names):
      metrics_dict[name] = metrics[i].astype(np.float32)
    return metrics_dict

  def _process_predictions(self, predictions):
    image_scale = np.tile(predictions['image_info'][:, 2:3, :], (1, 1, 2))
    predictions['detection_boxes'] = (
        predictions['detection_boxes'].astype(np.float32))
    predictions['detection_boxes'] /= image_scale
    if 'detection_outer_boxes' in predictions:
      predictions['detection_outer_boxes'] = (
          predictions['detection_outer_boxes'].astype(np.float32))
      predictions['detection_outer_boxes'] /= image_scale

  def update(self, predictions, groundtruths=None):
    """Update and aggregate detection results and groundtruth data.

    Args:
      predictions: a dictionary of numpy arrays including the fields below. See
        different parsers under `../dataloader` for more details.
        Required fields:
          - source_id: a numpy array of int or string of shape [batch_size].
          - image_info [if `need_rescale_bboxes` is True]: a numpy array of
            float of shape [batch_size, 4, 2].
          - num_detections: a numpy array of int of shape [batch_size].
          - detection_boxes: a numpy array of float of shape [batch_size, K, 4].
          - detection_classes: a numpy array of int of shape [batch_size, K].
          - detection_scores: a numpy array of float of shape [batch_size, K].
        Optional fields:
          - detection_masks: a numpy array of float of shape [batch_size, K,
            mask_height, mask_width].
      groundtruths: a dictionary of numpy arrays including the fields below. See
        also different parsers under `../dataloader` for more details.
        Required fields:
          - source_id: a numpy array of int or string of shape [batch_size].
          - height: a numpy array of int of shape [batch_size].
          - width: a numpy array of int of shape [batch_size].
          - num_detections: a numpy array of int of shape [batch_size].
          - boxes: a numpy array of float of shape [batch_size, K, 4].
          - classes: a numpy array of int of shape [batch_size, K].
        Optional fields:
          - is_crowds: a numpy array of int of shape [batch_size, K]. If the
            field is absent, it is assumed that this instance is not crowd.
          - areas: a numy array of float of shape [batch_size, K]. If the field
            is absent, the area is calculated using either boxes or masks
            depending on which one is available.
          - masks: a numpy array of float of shape [batch_size, K, mask_height,
            mask_width],

    Raises:
      ValueError: if the required prediction or groundtruth fields are not
        present in the incoming `predictions` or `groundtruths`.
    """
    for k in self._required_prediction_fields:
      if k not in predictions:
        raise ValueError(
            'Missing the required key `{}` in predictions!'.format(k))
    if self._need_rescale_bboxes:
      self._process_predictions(predictions)
    for k, v in six.iteritems(predictions):
      if k not in self._predictions:
        self._predictions[k] = [v]
      else:
        self._predictions[k].append(v)

    if not self._annotation_file:
      assert groundtruths
      for k in self._required_groundtruth_fields:
        if k not in groundtruths:
          raise ValueError(
              'Missing the required key `{}` in groundtruths!'.format(k))
      for k, v in six.iteritems(groundtruths):
        if k not in self._groundtruths:
          self._groundtruths[k] = [v]
        else:
          self._groundtruths[k].append(v)


class OlnXclassEvaluator(COCOEvaluator):
  """COCO evaluation metric class."""

  def __init__(self, annotation_file, include_mask, need_rescale_bboxes=True,
               use_category=True, seen_class='all'):
    """Constructs COCO evaluation class.

    The class provides the interface to metrics_fn in TPUEstimator. The
    _update_op() takes detections from each image and push them to
    self.detections. The _evaluate() loads a JSON file in COCO annotation format
    as the groundtruths and runs COCO evaluation.

    Args:
      annotation_file: a JSON file that stores annotations of the eval dataset.
        If `annotation_file` is None, groundtruth annotations will be loaded
        from the dataloader.
      include_mask: a boolean to indicate whether or not to include the mask
        eval.
      need_rescale_bboxes: If true bboxes in `predictions` will be rescaled back
        to absolute values (`image_info` is needed in this case).
      use_category: if `False`, treat all object in all classes in one
        foreground category.
      seen_class: 'all' or 'voc' or 'nonvoc'
    """
    super(OlnXclassEvaluator, self).__init__(
        annotation_file=annotation_file,
        include_mask=include_mask,
        need_rescale_bboxes=need_rescale_bboxes)
    self._use_category = use_category
    self._seen_class = seen_class
    self._seen_class_ids = class_utils.coco_split_class_ids(seen_class)
    self._metric_names = [
        'AP', 'AP50', 'AP75',
        'APs', 'APm', 'APl',
        'ARmax10', 'ARmax20', 'ARmax50', 'ARmax100', 'ARmax200',
        'ARmax10s', 'ARmax10m', 'ARmax10l'
    ]
    if self._seen_class != 'all':
      self._metric_names.extend([
          'AP_seen', 'AP50_seen', 'AP75_seen',
          'APs_seen', 'APm_seen', 'APl_seen',
          'ARmax10_seen', 'ARmax20_seen', 'ARmax50_seen',
          'ARmax100_seen', 'ARmax200_seen',
          'ARmax10s_seen', 'ARmax10m_seen', 'ARmax10l_seen',

          'AP_novel', 'AP50_novel', 'AP75_novel',
          'APs_novel', 'APm_novel', 'APl_novel',
          'ARmax10_novel', 'ARmax20_novel', 'ARmax50_novel',
          'ARmax100_novel', 'ARmax200_novel',
          'ARmax10s_novel', 'ARmax10m_novel', 'ARmax10l_novel',
      ])
    if self._include_mask:
      mask_metric_names = ['mask_' + x for x in self._metric_names]
      self._metric_names.extend(mask_metric_names)
      self._required_prediction_fields.extend(['detection_masks'])
      self._required_groundtruth_fields.extend(['masks'])

    self.reset()

  def evaluate(self):
    """Evaluates with detections from all images with COCO API.

    Returns:
      coco_metric: float numpy array with shape [24] representing the
        coco-style evaluation metrics (box and mask).
    """
    if not self._annotation_file:
      logging.info('Thre is no annotation_file in COCOEvaluator.')
      gt_dataset = coco_utils.convert_groundtruths_to_coco_dataset(
          self._groundtruths)
      coco_gt = coco_utils.COCOWrapper(
          eval_type=('mask' if self._include_mask else 'box'),
          gt_dataset=gt_dataset)
    else:
      logging.info('Using annotation file: %s', self._annotation_file)
      coco_gt = self._coco_gt

    coco_predictions = coco_utils.convert_predictions_to_coco_annotations(
        self._predictions)
    coco_dt = coco_gt.loadRes(predictions=coco_predictions)
    image_ids = [ann['image_id'] for ann in coco_predictions]
    # Class manipulation: 'all' split samples -> ignored_split = 0.
    for idx, ann in enumerate(coco_gt.dataset['annotations']):
      coco_gt.dataset['annotations'][idx]['ignored_split'] = 0
    coco_eval = cocoeval.OlnCOCOevalXclassWrapper(
        coco_gt, coco_dt, iou_type='bbox')
    coco_eval.params.maxDets = [10, 20, 50, 100, 200]
    coco_eval.params.imgIds = image_ids
    coco_eval.params.useCats = 0 if not self._use_category else 1
    coco_eval.evaluate()
    coco_eval.accumulate()
    coco_eval.summarize()
    coco_metrics = coco_eval.stats

    if self._include_mask:
      mcoco_eval = cocoeval.OlnCOCOevalXclassWrapper(
          coco_gt, coco_dt, iou_type='segm')
      mcoco_eval.params.maxDets = [10, 20, 50, 100, 200]
      mcoco_eval.params.imgIds = image_ids
      mcoco_eval.params.useCats = 0 if not self._use_category else 1
      mcoco_eval.evaluate()
      mcoco_eval.accumulate()
      mcoco_eval.summarize()
      mask_coco_metrics = mcoco_eval.stats

    if self._include_mask:
      metrics = np.hstack((coco_metrics, mask_coco_metrics))
    else:
      metrics = coco_metrics

    if self._seen_class != 'all':
      # for seen class eval, samples of novel_class are ignored.
      coco_gt_seen = copy.deepcopy(coco_gt)
      for idx, ann in enumerate(coco_gt.dataset['annotations']):
        if ann['category_id'] in self._seen_class_ids:
          coco_gt_seen.dataset['annotations'][idx]['ignored_split'] = 0
        else:
          coco_gt_seen.dataset['annotations'][idx]['ignored_split'] = 1
      coco_eval_seen = cocoeval.OlnCOCOevalXclassWrapper(
          coco_gt_seen, coco_dt, iou_type='bbox')
      coco_eval_seen.params.maxDets = [10, 20, 50, 100, 200]
      coco_eval_seen.params.imgIds = image_ids
      coco_eval_seen.params.useCats = 0 if not self._use_category else 1
      coco_eval_seen.evaluate()
      coco_eval_seen.accumulate()
      coco_eval_seen.summarize()
      coco_metrics_seen = coco_eval_seen.stats
      if self._include_mask:
        mcoco_eval_seen = cocoeval.OlnCOCOevalXclassWrapper(
            coco_gt_seen, coco_dt, iou_type='segm')
        mcoco_eval_seen.params.maxDets = [10, 20, 50, 100, 200]
        mcoco_eval_seen.params.imgIds = image_ids
        mcoco_eval_seen.params.useCats = 0 if not self._use_category else 1
        mcoco_eval_seen.evaluate()
        mcoco_eval_seen.accumulate()
        mcoco_eval_seen.summarize()
        mask_coco_metrics_seen = mcoco_eval_seen.stats

      # for novel class eval, samples of seen_class are ignored.
      coco_gt_novel = copy.deepcopy(coco_gt)
      for idx, ann in enumerate(coco_gt.dataset['annotations']):
        if ann['category_id'] in self._seen_class_ids:
          coco_gt_novel.dataset['annotations'][idx]['ignored_split'] = 1
        else:
          coco_gt_novel.dataset['annotations'][idx]['ignored_split'] = 0
      coco_eval_novel = cocoeval.OlnCOCOevalXclassWrapper(
          coco_gt_novel, coco_dt, iou_type='bbox')
      coco_eval_novel.params.maxDets = [10, 20, 50, 100, 200]
      coco_eval_novel.params.imgIds = image_ids
      coco_eval_novel.params.useCats = 0 if not self._use_category else 1
      coco_eval_novel.evaluate()
      coco_eval_novel.accumulate()
      coco_eval_novel.summarize()
      coco_metrics_novel = coco_eval_novel.stats
      if self._include_mask:
        mcoco_eval_novel = cocoeval.OlnCOCOevalXclassWrapper(
            coco_gt_novel, coco_dt, iou_type='segm')
        mcoco_eval_novel.params.maxDets = [10, 20, 50, 100, 200]
        mcoco_eval_novel.params.imgIds = image_ids
        mcoco_eval_novel.params.useCats = 0 if not self._use_category else 1
        mcoco_eval_novel.evaluate()
        mcoco_eval_novel.accumulate()
        mcoco_eval_novel.summarize()
        mask_coco_metrics_novel = mcoco_eval_novel.stats

      # Combine all splits.
      if self._include_mask:
        metrics = np.hstack((
            coco_metrics, coco_metrics_seen, coco_metrics_novel,
            mask_coco_metrics, mask_coco_metrics_seen, mask_coco_metrics_novel))
      else:
        metrics = np.hstack((
            coco_metrics, coco_metrics_seen, coco_metrics_novel))

    # Cleans up the internal variables in order for a fresh eval next time.
    self.reset()

    metrics_dict = {}
    for i, name in enumerate(self._metric_names):
      metrics_dict[name] = metrics[i].astype(np.float32)
    return metrics_dict


class OlnXdataEvaluator(OlnXclassEvaluator):
  """COCO evaluation metric class."""

  def __init__(self, annotation_file, include_mask, need_rescale_bboxes=True,
               use_category=True, seen_class='all'):
    """Constructs COCO evaluation class.

    The class provides the interface to metrics_fn in TPUEstimator. The
    _update_op() takes detections from each image and push them to
    self.detections. The _evaluate() loads a JSON file in COCO annotation format
    as the groundtruths and runs COCO evaluation.

    Args:
      annotation_file: a JSON file that stores annotations of the eval dataset.
        If `annotation_file` is None, groundtruth annotations will be loaded
        from the dataloader.
      include_mask: a boolean to indicate whether or not to include the mask
        eval.
      need_rescale_bboxes: If true bboxes in `predictions` will be rescaled back
        to absolute values (`image_info` is needed in this case).
      use_category: if `False`, treat all object in all classes in one
        foreground category.
      seen_class: 'all' or 'voc' or 'nonvoc'
    """
    super(OlnXdataEvaluator, self).__init__(
        annotation_file=annotation_file,
        include_mask=include_mask,
        need_rescale_bboxes=need_rescale_bboxes,
        use_category=False,
        seen_class='all')

  def evaluate(self):
    """Evaluates with detections from all images with COCO API.

    Returns:
      coco_metric: float numpy array with shape [24] representing the
        coco-style evaluation metrics (box and mask).
    """
    if not self._annotation_file:
      logging.info('Thre is no annotation_file in COCOEvaluator.')
      gt_dataset = coco_utils.convert_groundtruths_to_coco_dataset(
          self._groundtruths)
      coco_gt = coco_utils.COCOWrapper(
          eval_type=('mask' if self._include_mask else 'box'),
          gt_dataset=gt_dataset)
    else:
      logging.info('Using annotation file: %s', self._annotation_file)
      coco_gt = self._coco_gt
    coco_predictions = coco_utils.convert_predictions_to_coco_annotations(
        self._predictions)
    coco_dt = coco_gt.loadRes(predictions=coco_predictions)
    image_ids = [ann['image_id'] for ann in coco_predictions]
    # Class manipulation: 'all' split samples -> ignored_split = 0.
    for idx, _ in enumerate(coco_gt.dataset['annotations']):
      coco_gt.dataset['annotations'][idx]['ignored_split'] = 0
    coco_eval = cocoeval.OlnCOCOevalWrapper(coco_gt, coco_dt, iou_type='bbox')
    coco_eval.params.maxDets = [10, 20, 50, 100, 200]
    coco_eval.params.imgIds = image_ids
    coco_eval.params.useCats = 0 if not self._use_category else 1
    coco_eval.evaluate()
    coco_eval.accumulate()
    coco_eval.summarize()
    coco_metrics = coco_eval.stats

    if self._include_mask:
      mcoco_eval = cocoeval.OlnCOCOevalWrapper(coco_gt, coco_dt,
                                               iou_type='segm')
      mcoco_eval.params.maxDets = [10, 20, 50, 100, 200]
      mcoco_eval.params.imgIds = image_ids
      mcoco_eval.params.useCats = 0 if not self._use_category else 1
      mcoco_eval.evaluate()
      mcoco_eval.accumulate()
      mcoco_eval.summarize()
      mask_coco_metrics = mcoco_eval.stats

    if self._include_mask:
      metrics = np.hstack((coco_metrics, mask_coco_metrics))
    else:
      metrics = coco_metrics

    # Cleans up the internal variables in order for a fresh eval next time.
    self.reset()

    metrics_dict = {}
    for i, name in enumerate(self._metric_names):
      metrics_dict[name] = metrics[i].astype(np.float32)
    return metrics_dict


class ShapeMaskCOCOEvaluator(COCOEvaluator):
  """COCO evaluation metric class for ShapeMask."""

  def __init__(self, mask_eval_class, **kwargs):
    """Constructs COCO evaluation class.

    The class provides the interface to metrics_fn in TPUEstimator. The
    _update_op() takes detections from each image and push them to
    self.detections. The _evaluate() loads a JSON file in COCO annotation format
    as the groundtruths and runs COCO evaluation.

    Args:
      mask_eval_class: the set of classes for mask evaluation.
      **kwargs: other keyword arguments passed to the parent class initializer.
    """
    super(ShapeMaskCOCOEvaluator, self).__init__(**kwargs)
    self._mask_eval_class = mask_eval_class
    self._eval_categories = class_utils.coco_split_class_ids(mask_eval_class)
    if mask_eval_class != 'all':
      self._metric_names = [
          x.replace('mask', 'novel_mask') for x in self._metric_names
      ]

  def evaluate(self):
    """Evaluates with detections from all images with COCO API.

    Returns:
      coco_metric: float numpy array with shape [24] representing the
        coco-style evaluation metrics (box and mask).
    """
    if not self._annotation_file:
      gt_dataset = coco_utils.convert_groundtruths_to_coco_dataset(
          self._groundtruths)
      coco_gt = coco_utils.COCOWrapper(
          eval_type=('mask' if self._include_mask else 'box'),
          gt_dataset=gt_dataset)
    else:
      coco_gt = self._coco_gt
    coco_predictions = coco_utils.convert_predictions_to_coco_annotations(
        self._predictions)
    coco_dt = coco_gt.loadRes(predictions=coco_predictions)
    image_ids = [ann['image_id'] for ann in coco_predictions]

    coco_eval = cocoeval.COCOeval(coco_gt, coco_dt, iouType='bbox')
    coco_eval.params.imgIds = image_ids
    coco_eval.evaluate()
    coco_eval.accumulate()
    coco_eval.summarize()
    coco_metrics = coco_eval.stats

    if self._include_mask:
      mcoco_eval = cocoeval.COCOeval(coco_gt, coco_dt, iouType='segm')
      mcoco_eval.params.imgIds = image_ids
      mcoco_eval.evaluate()
      mcoco_eval.accumulate()
      mcoco_eval.summarize()
      if self._mask_eval_class == 'all':
        metrics = np.hstack((coco_metrics, mcoco_eval.stats))
      else:
        mask_coco_metrics = mcoco_eval.category_stats
        val_catg_idx = np.isin(mcoco_eval.params.catIds, self._eval_categories)
        # Gather the valid evaluation of the eval categories.
        if np.any(val_catg_idx):
          mean_val_metrics = []
          for mid in range(len(self._metric_names) // 2):
            mean_val_metrics.append(
                np.nanmean(mask_coco_metrics[mid][val_catg_idx]))

          mean_val_metrics = np.array(mean_val_metrics)
        else:
          mean_val_metrics = np.zeros(len(self._metric_names) // 2)
        metrics = np.hstack((coco_metrics, mean_val_metrics))
    else:
      metrics = coco_metrics

    # Cleans up the internal variables in order for a fresh eval next time.
    self.reset()

    metrics_dict = {}
    for i, name in enumerate(self._metric_names):
      metrics_dict[name] = metrics[i].astype(np.float32)
    return metrics_dict
