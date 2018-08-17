# Copyright 2017 The TensorFlow Authors. All Rights Reserved.
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

r"""Creates training and eval data from Quickdraw NDJSON files.

This tool reads the NDJSON files from https://quickdraw.withgoogle.com/data
and converts them into tensorflow.Example stored in TFRecord files.

The tensorflow example will contain 3 features:
 shape - contains the shape of the sequence [length, dim] where dim=3.
 class_index - the class index of the class for the example.
 ink - a length * dim vector of the ink.

It creates disjoint training and evaluation sets.

python create_dataset.py \
  --ndjson_path ${HOME}/ndjson \
  --output_path ${HOME}/tfrecord
"""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import argparse
import json
import os
import random
import sys
import numpy as np
import tensorflow as tf


def parse_line(ndjson_line):
  """Parse an ndjson line and return ink (as np array) and classname."""
  sample = json.loads(ndjson_line)
  class_name = sample["word"]
  if not class_name:
    print ("Empty classname")
    return None, None
  inkarray = sample["drawing"]
  stroke_lengths = [len(stroke[0]) for stroke in inkarray]
  total_points = sum(stroke_lengths)
  np_ink = np.zeros((total_points, 3), dtype=np.float32)
  current_t = 0
  if not inkarray:
    print("Empty inkarray")
    return None, None
  for stroke in inkarray:
    if len(stroke[0]) != len(stroke[1]):
      print("Inconsistent number of x and y coordinates.")
      return None, None
    for i in [0, 1]:
      np_ink[current_t:(current_t + len(stroke[0])), i] = stroke[i]
    current_t += len(stroke[0])
    np_ink[current_t - 1, 2] = 1  # stroke_end
  # Preprocessing.
  # 1. Size normalization.
  lower = np.min(np_ink[:, 0:2], axis=0)
  upper = np.max(np_ink[:, 0:2], axis=0)
  scale = upper - lower
  scale[scale == 0] = 1
  np_ink[:, 0:2] = (np_ink[:, 0:2] - lower) / scale
  # 2. Compute deltas.
  np_ink[1:, 0:2] -= np_ink[0:-1, 0:2]
  np_ink = np_ink[1:, :]
  return np_ink, class_name


def convert_data(trainingdata_dir,
                 observations_per_class,
                 output_file,
                 classnames,
                 output_shards=10,
                 offset=0):
  """Convert training data from ndjson files into tf.Example in tf.Record.

  Args:
   trainingdata_dir: path to the directory containin the training data.
     The training data is stored in that directory as ndjson files.
   observations_per_class: the number of items to load per class.
   output_file: path where to write the output.
   classnames: array with classnames - is auto created if not passed in.
   output_shards: the number of shards to write the output in.
   offset: the number of items to skip at the beginning of each file.

  Returns:
    classnames: the class names as strings. classnames[classes[i]] is the
      textual representation of the class of the i-th data point.
  """

  def _pick_output_shard():
    return random.randint(0, output_shards - 1)

  file_handles = []
  # Open all input files.
  for filename in sorted(tf.gfile.ListDirectory(trainingdata_dir)):
    if not filename.endswith(".ndjson"):
      print("Skipping", filename)
      continue
    file_handles.append(
        tf.gfile.GFile(os.path.join(trainingdata_dir, filename), "r"))
    if offset:  # Fast forward all files to skip the offset.
      count = 0
      for _ in file_handles[-1]:
        count += 1
        if count == offset:
          break

  writers = []
  for i in range(FLAGS.output_shards):
    writers.append(
        tf.python_io.TFRecordWriter("%s-%05i-of-%05i" % (output_file, i,
                                                         output_shards)))

  reading_order = list(range(len(file_handles))) * observations_per_class
  random.shuffle(reading_order)

  for c in reading_order:
    line = file_handles[c].readline()
    ink = None
    while ink is None:
      ink, class_name = parse_line(line)
      if ink is None:
        print ("Couldn't parse ink from '" + line + "'.")
    if class_name not in classnames:
      classnames.append(class_name)
    features = {}
    features["class_index"] = tf.train.Feature(int64_list=tf.train.Int64List(
        value=[classnames.index(class_name)]))
    features["ink"] = tf.train.Feature(float_list=tf.train.FloatList(
        value=ink.flatten()))
    features["shape"] = tf.train.Feature(int64_list=tf.train.Int64List(
        value=ink.shape))
    f = tf.train.Features(feature=features)
    example = tf.train.Example(features=f)
    writers[_pick_output_shard()].write(example.SerializeToString())

  # Close all files
  for w in writers:
    w.close()
  for f in file_handles:
    f.close()
  # Write the class list.
  with tf.gfile.GFile(output_file + ".classes", "w") as f:
    for class_name in classnames:
      f.write(class_name + "\n")
  return classnames


def main(argv):
  del argv
  classnames = convert_data(
      FLAGS.ndjson_path,
      FLAGS.train_observations_per_class,
      os.path.join(FLAGS.output_path, "training.tfrecord"),
      classnames=[],
      output_shards=FLAGS.output_shards,
      offset=0)
  convert_data(
      FLAGS.ndjson_path,
      FLAGS.eval_observations_per_class,
      os.path.join(FLAGS.output_path, "eval.tfrecord"),
      classnames=classnames,
      output_shards=FLAGS.output_shards,
      offset=FLAGS.train_observations_per_class)


if __name__ == "__main__":
  parser = argparse.ArgumentParser()
  parser.register("type", "bool", lambda v: v.lower() == "true")
  parser.add_argument(
      "--ndjson_path",
      type=str,
      default="",
      help="Directory where the ndjson files are stored.")
  parser.add_argument(
      "--output_path",
      type=str,
      default="",
      help="Directory where to store the output TFRecord files.")
  parser.add_argument(
      "--train_observations_per_class",
      type=int,
      default=10000,
      help="How many items per class to load for training.")
  parser.add_argument(
      "--eval_observations_per_class",
      type=int,
      default=1000,
      help="How many items per class to load for evaluation.")
  parser.add_argument(
      "--output_shards",
      type=int,
      default=10,
      help="Number of shards for the output.")

  FLAGS, unparsed = parser.parse_known_args()
  tf.app.run(main=main, argv=[sys.argv[0]] + unparsed)
