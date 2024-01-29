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

r"""Script to convert HierText to TFExamples.

This script is only intended to run locally.

python3 data_preprocess/convert.py \
--gt_file=/path/to/gt.jsonl \
--img_dir=/path/to/image \
--out_file=/path/to/tfrecords/file-prefix

"""

import json
import os
import random

from absl import app
from absl import flags
import tensorflow as tf
import tqdm
import utils


_GT_FILE = flags.DEFINE_string('gt_file', None, 'Path to the GT file')
_IMG_DIR = flags.DEFINE_string('img_dir', None, 'Path to the image folder.')
_OUT_FILE = flags.DEFINE_string('out_file', None, 'Path for the tfrecords.')
_NUM_SHARD = flags.DEFINE_integer(
    'num_shard', 100, 'The number of shards of tfrecords.')


def main(unused_argv) -> None:
  annotations = json.load(open(_GT_FILE.value))['annotations']
  random.shuffle(annotations)
  n_sample = len(annotations)
  n_shards = _NUM_SHARD.value
  n_sample_per_shard = (n_sample - 1) // n_shards + 1

  for shard in tqdm.tqdm(range(n_shards)):
    output_path = f'{_OUT_FILE.value}-{shard:05}-{n_shards:05}.tfrecords'
    annotation_subset = annotations[
        shard * n_sample_per_shard : (shard + 1) * n_sample_per_shard]

    with tf.io.TFRecordWriter(output_path) as file_writer:
      for annotation in annotation_subset:
        img_file_path = os.path.join(_IMG_DIR.value,
                                     f"{annotation['image_id']}.jpg")
        tfexample = utils.convert_to_tfe(img_file_path, annotation)
        file_writer.write(tfexample)


if __name__ == '__main__':
  flags.mark_flags_as_required(['gt_file', 'img_dir', 'out_file'])
  app.run(main)
