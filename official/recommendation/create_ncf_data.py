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

"""Binary to generate training/evaluation dataset for NCF model."""

import json

# pylint: disable=g-bad-import-order
# Import libraries
from absl import app
from absl import flags
import tensorflow as tf
# pylint: enable=g-bad-import-order

from official.recommendation import movielens
from official.recommendation import data_preprocessing

flags.DEFINE_string(
    "data_dir", None,
    "The input data dir at which training and evaluation tf record files "
    "will be saved.")
flags.DEFINE_string("meta_data_file_path", None,
                    "The path in which input meta data will be written.")
flags.DEFINE_enum("dataset", "ml-20m", ["ml-1m", "ml-20m"],
                  "Dataset to be trained/evaluated.")
flags.DEFINE_enum(
    "constructor_type", "bisection", ["bisection", "materialized"],
    "Strategy to use for generating false negatives. materialized has a "
    "precompute that scales badly, but a faster per-epoch construction "
    "time and can be faster on very large systems.")
flags.DEFINE_integer("num_train_epochs", 14,
                     "Total number of training epochs to generate.")
flags.DEFINE_integer(
    "num_negative_samples", 4,
    "Number of negative instances to pair with positive instance.")
flags.DEFINE_integer(
    "train_prebatch_size", 99000,
    "Batch size to be used for prebatching the dataset "
    "for training.")
flags.DEFINE_integer(
    "eval_prebatch_size", 99000,
    "Batch size to be used for prebatching the dataset "
    "for training.")

FLAGS = flags.FLAGS


def prepare_raw_data(flag_obj):
  """Downloads and prepares raw data for data generation."""
  movielens.download(flag_obj.dataset, flag_obj.data_dir)

  data_processing_params = {
      "train_epochs": flag_obj.num_train_epochs,
      "batch_size": flag_obj.train_prebatch_size,
      "eval_batch_size": flag_obj.eval_prebatch_size,
      "batches_per_step": 1,
      "stream_files": True,
      "num_neg": flag_obj.num_negative_samples,
  }

  num_users, num_items, producer = data_preprocessing.instantiate_pipeline(
      dataset=flag_obj.dataset,
      data_dir=flag_obj.data_dir,
      params=data_processing_params,
      constructor_type=flag_obj.constructor_type,
      epoch_dir=flag_obj.data_dir,
      generate_data_offline=True)

  # pylint: disable=protected-access
  input_metadata = {
      "num_users": num_users,
      "num_items": num_items,
      "constructor_type": flag_obj.constructor_type,
      "num_train_elements": producer._elements_in_epoch,
      "num_eval_elements": producer._eval_elements_in_epoch,
      "num_train_epochs": flag_obj.num_train_epochs,
      "train_prebatch_size": flag_obj.train_prebatch_size,
      "eval_prebatch_size": flag_obj.eval_prebatch_size,
      "num_train_steps": producer.train_batches_per_epoch,
      "num_eval_steps": producer.eval_batches_per_epoch,
  }
  # pylint: enable=protected-access

  return producer, input_metadata


def generate_data():
  """Creates NCF train/eval dataset and writes input metadata as a file."""
  producer, input_metadata = prepare_raw_data(FLAGS)
  producer.run()

  with tf.io.gfile.GFile(FLAGS.meta_data_file_path, "w") as writer:
    writer.write(json.dumps(input_metadata, indent=4) + "\n")


def main(_):
  generate_data()


if __name__ == "__main__":
  flags.mark_flag_as_required("data_dir")
  flags.mark_flag_as_required("meta_data_file_path")
  app.run(main)
