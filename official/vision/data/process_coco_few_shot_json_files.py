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

"""Processes the JSON files for COCO few-shot.

We assume that `workdir` mirrors the contents of
http://dl.yf.io/fs-det/datasets/cocosplit/, which contains the official JSON
files for the few-shot COCO evaluation procedure that Wang et al. (2020)'s
"Frustratingly Simple Few-Shot Object Detection" paper uses.
"""

import collections
import itertools
import json
import logging
import os

from absl import app
from absl import flags

import tensorflow as tf, tf_keras

logger = tf.get_logger()
logger.setLevel(logging.INFO)

flags.DEFINE_string('workdir', None, 'Working directory.')

FLAGS = flags.FLAGS
CATEGORIES = ['airplane', 'apple', 'backpack', 'banana', 'baseball bat',
              'baseball glove', 'bear', 'bed', 'bench', 'bicycle', 'bird',
              'boat', 'book', 'bottle', 'bowl', 'broccoli', 'bus', 'cake',
              'car', 'carrot', 'cat', 'cell phone', 'chair', 'clock', 'couch',
              'cow', 'cup', 'dining table', 'dog', 'donut', 'elephant',
              'fire hydrant', 'fork', 'frisbee', 'giraffe', 'hair drier',
              'handbag', 'horse', 'hot dog', 'keyboard', 'kite', 'knife',
              'laptop', 'microwave', 'motorcycle', 'mouse', 'orange', 'oven',
              'parking meter', 'person', 'pizza', 'potted plant',
              'refrigerator', 'remote', 'sandwich', 'scissors', 'sheep',
              'sink', 'skateboard', 'skis', 'snowboard', 'spoon', 'sports ball',
              'stop sign', 'suitcase', 'surfboard', 'teddy bear',
              'tennis racket', 'tie', 'toaster', 'toilet', 'toothbrush',
              'traffic light', 'train', 'truck', 'tv', 'umbrella', 'vase',
              'wine glass', 'zebra']
SEEDS = list(range(10))
SHOTS = [1, 3, 5, 10, 30]

FILE_SUFFIXES = collections.defaultdict(list)
for _seed, _shots in itertools.product(SEEDS, SHOTS):
  for _category in CATEGORIES:
    FILE_SUFFIXES[(_seed, _shots)].append(
        '{}full_box_{}shot_{}_trainval.json'.format(
            # http://dl.yf.io/fs-det/datasets/cocosplit/ is organized like so:
            #
            #   datasplit/
            #     trainvalno5k.json
            #     5k.json
            #   full_box_{1,2,3,5,10,30}shot_{category}_trainval.json
            #   seed{1-9}/
            #     full_box_{1,2,3,5,10,30}shot_{category}_trainval.json
            #
            # This means that the JSON files for seed0 are located in the root
            # directory rather than in a `seed?/` subdirectory, hence the
            # conditional expression below.
            '' if _seed == 0 else 'seed{}/'.format(_seed),
            _shots,
            _category))

# Base class IDs, as defined in
# https://github.com/ucbdrive/few-shot-object-detection/blob/master/fsdet/evaluation/coco_evaluation.py#L60-L65
BASE_CLASS_IDS = [8, 10, 11, 13, 14, 15, 22, 23, 24, 25, 27, 28, 31, 32, 33, 34,
                  35, 36, 37, 38, 39, 40, 41, 42, 43, 46, 47, 48, 49, 50, 51,
                  52, 53, 54, 55, 56, 57, 58, 59, 60, 61, 65, 70, 73, 74, 75,
                  76, 77, 78, 79, 80, 81, 82, 84, 85, 86, 87, 88, 89, 90]


def main(unused_argv):
  workdir = FLAGS.workdir

  # Filter novel class annotations from the training and validation sets.
  for name in ('trainvalno5k', '5k'):
    file_path = os.path.join(workdir, 'datasplit', '{}.json'.format(name))
    with tf.io.gfile.GFile(file_path, 'r') as f:
      json_dict = json.load(f)

    json_dict['annotations'] = [a for a in json_dict['annotations']
                                if a['category_id'] in BASE_CLASS_IDS]
    output_path = os.path.join(
        workdir, 'datasplit', '{}_base.json'.format(name))
    with tf.io.gfile.GFile(output_path, 'w') as f:
      json.dump(json_dict, f)

  for seed, shots in itertools.product(SEEDS, SHOTS):
    # Retrieve all examples for a given seed and shots setting.
    file_paths = [os.path.join(workdir, suffix)
                  for suffix in FILE_SUFFIXES[(seed, shots)]]
    json_dicts = []
    for file_path in file_paths:
      with tf.io.gfile.GFile(file_path, 'r') as f:
        json_dicts.append(json.load(f))

    # Make sure that all JSON files for a given seed and shots setting have the
    # same metadata. We count on this to fuse them later on.
    metadata_dicts = [{'info': d['info'], 'licenses': d['licenses'],
                       'categories': d['categories']} for d in json_dicts]
    if not all(d == metadata_dicts[0] for d in metadata_dicts[1:]):
      raise RuntimeError(
          'JSON files for {} shots (seed {}) '.format(shots, seed) +
          'have different info, licences, or categories fields')

    # Retrieve images across all JSON files.
    images = sum((d['images'] for d in json_dicts), [])
    # Remove duplicate image entries.
    images = list({image['id']: image for image in images}.values())

    output_dict = {
        'info': json_dicts[0]['info'],
        'licenses': json_dicts[0]['licenses'],
        'categories': json_dicts[0]['categories'],
        'images': images,
        'annotations': sum((d['annotations'] for d in json_dicts), [])
    }

    output_path = os.path.join(workdir,
                               '{}shot_seed{}.json'.format(shots, seed))
    with tf.io.gfile.GFile(output_path, 'w') as f:
      json.dump(output_dict, f)
    logger.info('Processed %d shots (seed %d) and saved to %s',
                shots, seed, output_path)


if __name__ == '__main__':
  flags.mark_flag_as_required('workdir')
  app.run(main)
