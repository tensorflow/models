# Copyright 2019 The TensorFlow Authors All Rights Reserved.
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
"""Computes evaluation metrics on groundtruth and predictions in COCO format.

The Common Objects in Context (COCO) dataset defines a format for specifying
combined semantic and instance segmentations as "panoptic" segmentations. This
is done with the combination of JSON and image files as specified at:
http://cocodataset.org/#format-results
where the JSON file specifies the overall structure of the result,
including the categories for each annotation, and the images specify the image
region for each annotation in that image by its ID.

This script computes additional metrics such as Parsing Covering on datasets and
predictions in this format. An implementation of Panoptic Quality is also
provided for convenience.
"""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import collections
import json
import multiprocessing
import os

from absl import app
from absl import flags
from absl import logging
import numpy as np
from PIL import Image
import utils as panopticapi_utils
import six

from deeplab.evaluation import panoptic_quality
from deeplab.evaluation import parsing_covering

FLAGS = flags.FLAGS

flags.DEFINE_string(
    'gt_json_file', None,
    ' Path to a JSON file giving ground-truth annotations in COCO format.')
flags.DEFINE_string('pred_json_file', None,
                    'Path to a JSON file for the predictions to evaluate.')
flags.DEFINE_string(
    'gt_folder', None,
    'Folder containing panoptic-format ID images to match ground-truth '
    'annotations to image regions.')
flags.DEFINE_string('pred_folder', None,
                    'Folder containing ID images for predictions.')
flags.DEFINE_enum(
    'metric', 'pq', ['pq', 'pc'], 'Shorthand name of a metric to compute. '
    'Supported values are:\n'
    'Panoptic Quality (pq)\n'
    'Parsing Covering (pc)')
flags.DEFINE_integer(
    'num_categories', 201,
    'The number of segmentation categories (or "classes") in the dataset.')
flags.DEFINE_integer(
    'ignored_label', 0,
    'A category id that is ignored in evaluation, e.g. the void label as '
    'defined in COCO panoptic segmentation dataset.')
flags.DEFINE_integer(
    'max_instances_per_category', 256,
    'The maximum number of instances for each category. Used in ensuring '
    'unique instance labels.')
flags.DEFINE_integer('intersection_offset', None,
                     'The maximum number of unique labels.')
flags.DEFINE_bool(
    'normalize_by_image_size', True,
    'Whether to normalize groundtruth instance region areas by image size. If '
    'True, groundtruth instance areas and weighted IoUs will be divided by the '
    'size of the corresponding image before accumulated across the dataset. '
    'Only used for Parsing Covering (pc) evaluation.')
flags.DEFINE_integer(
    'num_workers', 0, 'If set to a positive number, will spawn child processes '
    'to compute parts of the metric in parallel by splitting '
    'the images between the workers. If set to -1, will use '
    'the value of multiprocessing.cpu_count().')
flags.DEFINE_integer('print_digits', 3,
                     'Number of significant digits to print in metrics.')


def _build_metric(metric,
                  num_categories,
                  ignored_label,
                  max_instances_per_category,
                  intersection_offset=None,
                  normalize_by_image_size=True):
  """Creates a metric aggregator objet of the given name."""
  if metric == 'pq':
    logging.warning('One should check Panoptic Quality results against the '
                    'official COCO API code. Small numerical differences '
                    '(< 0.1%) can be magnified by rounding.')
    return panoptic_quality.PanopticQuality(num_categories, ignored_label,
                                            max_instances_per_category,
                                            intersection_offset)
  elif metric == 'pc':
    return parsing_covering.ParsingCovering(
        num_categories, ignored_label, max_instances_per_category,
        intersection_offset, normalize_by_image_size)
  else:
    raise ValueError('No implementation for metric "%s"' % metric)


def _matched_annotations(gt_json, pred_json):
  """Yields a set of (groundtruth, prediction) image annotation pairs.."""
  image_id_to_pred_ann = {
      annotation['image_id']: annotation
      for annotation in pred_json['annotations']
  }
  for gt_ann in gt_json['annotations']:
    image_id = gt_ann['image_id']
    pred_ann = image_id_to_pred_ann[image_id]
    yield gt_ann, pred_ann


def _open_panoptic_id_image(image_path):
  """Loads a COCO-format panoptic ID image from file."""
  return panopticapi_utils.rgb2id(
      np.array(Image.open(image_path), dtype=np.uint32))


def _split_panoptic(ann_json, id_array, ignored_label, allow_crowds):
  """Given the COCO JSON and ID map, splits into categories and instances."""
  category = np.zeros(id_array.shape, np.uint16)
  instance = np.zeros(id_array.shape, np.uint16)
  next_instance_id = collections.defaultdict(int)
  # Skip instance label 0 for ignored label. That is reserved for void.
  next_instance_id[ignored_label] = 1
  for segment_info in ann_json['segments_info']:
    if allow_crowds and segment_info['iscrowd']:
      category_id = ignored_label
    else:
      category_id = segment_info['category_id']
    mask = np.equal(id_array, segment_info['id'])
    category[mask] = category_id
    instance[mask] = next_instance_id[category_id]
    next_instance_id[category_id] += 1
  return category, instance


def _category_and_instance_from_annotation(ann_json, folder, ignored_label,
                                           allow_crowds):
  """Given the COCO JSON annotations, finds maps of categories and instances."""
  panoptic_id_image = _open_panoptic_id_image(
      os.path.join(folder, ann_json['file_name']))
  return _split_panoptic(ann_json, panoptic_id_image, ignored_label,
                         allow_crowds)


def _compute_metric(metric_aggregator, gt_folder, pred_folder,
                    annotation_pairs):
  """Iterates over matched annotation pairs and computes a metric over them."""
  for gt_ann, pred_ann in annotation_pairs:
    # We only expect "iscrowd" to appear in the ground-truth, and not in model
    # output. In predicted JSON it is simply ignored, as done in official code.
    gt_category, gt_instance = _category_and_instance_from_annotation(
        gt_ann, gt_folder, metric_aggregator.ignored_label, True)
    pred_category, pred_instance = _category_and_instance_from_annotation(
        pred_ann, pred_folder, metric_aggregator.ignored_label, False)

    metric_aggregator.compare_and_accumulate(gt_category, gt_instance,
                                             pred_category, pred_instance)
  return metric_aggregator


def _iterate_work_queue(work_queue):
  """Creates an iterable that retrieves items from a queue until one is None."""
  task = work_queue.get(block=True)
  while task is not None:
    yield task
    task = work_queue.get(block=True)


def _run_metrics_worker(metric_aggregator, gt_folder, pred_folder, work_queue,
                        result_queue):
  result = _compute_metric(metric_aggregator, gt_folder, pred_folder,
                           _iterate_work_queue(work_queue))
  result_queue.put(result, block=True)


def _is_thing_array(categories_json, ignored_label):
  """is_thing[category_id] is a bool on if category is "thing" or "stuff"."""
  is_thing_dict = {}
  for category_json in categories_json:
    is_thing_dict[category_json['id']] = bool(category_json['isthing'])

  # Check our assumption that the category ids are consecutive.
  # Usually metrics should be able to handle this case, but adding a warning
  # here.
  max_category_id = max(six.iterkeys(is_thing_dict))
  if len(is_thing_dict) != max_category_id + 1:
    seen_ids = six.viewkeys(is_thing_dict)
    all_ids = set(six.moves.range(max_category_id + 1))
    unseen_ids = all_ids.difference(seen_ids)
    if unseen_ids != {ignored_label}:
      logging.warning(
          'Nonconsecutive category ids or no category JSON specified for ids: '
          '%s', unseen_ids)

  is_thing_array = np.zeros(max_category_id + 1)
  for category_id, is_thing in six.iteritems(is_thing_dict):
    is_thing_array[category_id] = is_thing

  return is_thing_array


def eval_coco_format(gt_json_file,
                     pred_json_file,
                     gt_folder=None,
                     pred_folder=None,
                     metric='pq',
                     num_categories=201,
                     ignored_label=0,
                     max_instances_per_category=256,
                     intersection_offset=None,
                     normalize_by_image_size=True,
                     num_workers=0,
                     print_digits=3):
  """Top-level code to compute metrics on a COCO-format result.

  Note that the default values are set for COCO panoptic segmentation dataset,
  and thus the users may want to change it for their own dataset evaluation.

  Args:
    gt_json_file: Path to a JSON file giving ground-truth annotations in COCO
      format.
    pred_json_file: Path to a JSON file for the predictions to evaluate.
    gt_folder: Folder containing panoptic-format ID images to match ground-truth
      annotations to image regions.
    pred_folder: Folder containing ID images for predictions.
    metric: Name of a metric to compute.
    num_categories: The number of segmentation categories (or "classes") in the
      dataset.
    ignored_label: A category id that is ignored in evaluation, e.g. the "void"
      label as defined in the COCO panoptic segmentation dataset.
    max_instances_per_category: The maximum number of instances for each
      category. Used in ensuring unique instance labels.
    intersection_offset: The maximum number of unique labels.
    normalize_by_image_size: Whether to normalize groundtruth instance region
      areas by image size. If True, groundtruth instance areas and weighted IoUs
      will be divided by the size of the corresponding image before accumulated
      across the dataset. Only used for Parsing Covering (pc) evaluation.
    num_workers: If set to a positive number, will spawn child processes to
      compute parts of the metric in parallel by splitting the images between
      the workers. If set to -1, will use the value of
      multiprocessing.cpu_count().
    print_digits: Number of significant digits to print in summary of computed
      metrics.

  Returns:
    The computed result of the metric as a float scalar.
  """
  with open(gt_json_file, 'r') as gt_json_fo:
    gt_json = json.load(gt_json_fo)
  with open(pred_json_file, 'r') as pred_json_fo:
    pred_json = json.load(pred_json_fo)
  if gt_folder is None:
    gt_folder = gt_json_file.replace('.json', '')
  if pred_folder is None:
    pred_folder = pred_json_file.replace('.json', '')
  if intersection_offset is None:
    intersection_offset = (num_categories + 1) * max_instances_per_category

  metric_aggregator = _build_metric(
      metric, num_categories, ignored_label, max_instances_per_category,
      intersection_offset, normalize_by_image_size)

  if num_workers == -1:
    logging.info('Attempting to get the CPU count to set # workers.')
    num_workers = multiprocessing.cpu_count()

  if num_workers > 0:
    logging.info('Computing metric in parallel with %d workers.', num_workers)
    work_queue = multiprocessing.Queue()
    result_queue = multiprocessing.Queue()
    workers = []
    worker_args = (metric_aggregator, gt_folder, pred_folder, work_queue,
                   result_queue)
    for _ in six.moves.range(num_workers):
      workers.append(
          multiprocessing.Process(target=_run_metrics_worker, args=worker_args))
    for worker in workers:
      worker.start()
    for ann_pair in _matched_annotations(gt_json, pred_json):
      work_queue.put(ann_pair, block=True)

    # Will cause each worker to return a result and terminate upon recieving a
    # None task.
    for _ in six.moves.range(num_workers):
      work_queue.put(None, block=True)

    # Retrieve results.
    for _ in six.moves.range(num_workers):
      metric_aggregator.merge(result_queue.get(block=True))

    for worker in workers:
      worker.join()
  else:
    logging.info('Computing metric in a single process.')
    annotation_pairs = _matched_annotations(gt_json, pred_json)
    _compute_metric(metric_aggregator, gt_folder, pred_folder, annotation_pairs)

  is_thing = _is_thing_array(gt_json['categories'], ignored_label)
  metric_aggregator.print_detailed_results(
      is_thing=is_thing, print_digits=print_digits)
  return metric_aggregator.detailed_results(is_thing=is_thing)


def main(argv):
  if len(argv) > 1:
    raise app.UsageError('Too many command-line arguments.')

  eval_coco_format(FLAGS.gt_json_file, FLAGS.pred_json_file, FLAGS.gt_folder,
                   FLAGS.pred_folder, FLAGS.metric, FLAGS.num_categories,
                   FLAGS.ignored_label, FLAGS.max_instances_per_category,
                   FLAGS.intersection_offset, FLAGS.normalize_by_image_size,
                   FLAGS.num_workers, FLAGS.print_digits)


if __name__ == '__main__':
  flags.mark_flags_as_required(
      ['gt_json_file', 'gt_folder', 'pred_json_file', 'pred_folder'])
  app.run(main)
