# Copyright 2021 The TensorFlow Authors. All Rights Reserved.
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

"""Rebalance a set of CSV/TFRecord shards to a target number of files.
"""

import argparse
import datetime
import os

import apache_beam as beam
import tensorflow as tf


parser = argparse.ArgumentParser()
parser.add_argument(
    "--input_path",
    default=None,
    required=True,
    help="Input path.")
parser.add_argument(
    "--output_path",
    default=None,
    required=True,
    help="Output path.")
parser.add_argument(
    "--num_output_files",
    type=int,
    default=256,
    help="Number of output file shards.")
parser.add_argument(
    "--filetype",
    default="tfrecord",
    help="File type, needs to be one of {tfrecord, csv}.")
parser.add_argument(
    "--project",
    default=None,
    help="ID (not name) of your project. Ignored by DirectRunner")
parser.add_argument(
    "--runner",
    help="Runner for Apache Beam, needs to be one of "
    "{DirectRunner, DataflowRunner}.",
    default="DirectRunner")
parser.add_argument(
    "--region",
    default=None,
    help="region")

args = parser.parse_args()


def rebalance_data_shards():
  """Rebalances data shards."""

  def csv_pipeline(pipeline: beam.Pipeline):
    """Rebalances CSV dataset.

    Args:
      pipeline: Beam pipeline object.
    """
    _ = (
        pipeline
        | beam.io.ReadFromText(args.input_path)
        | beam.io.WriteToText(args.output_path,
                              num_shards=args.num_output_files))

  def tfrecord_pipeline(pipeline: beam.Pipeline):
    """Rebalances TFRecords dataset.

    Args:
      pipeline: Beam pipeline object.
    """
    example_coder = beam.coders.ProtoCoder(tf.train.Example)
    _ = (
        pipeline
        | beam.io.ReadFromTFRecord(args.input_path, coder=example_coder)
        | beam.io.WriteToTFRecord(args.output_path, file_name_suffix="tfrecord",
                                  coder=example_coder,
                                  num_shards=args.num_output_files))

  job_name = (
      f"shard-rebalancer-{datetime.datetime.now().strftime('%y%m%d-%H%M%S')}")

  # set up Beam pipeline.
  options = {
      "staging_location": os.path.join(args.output_path, "tmp", "staging"),
      "temp_location": os.path.join(args.output_path, "tmp"),
      "job_name": job_name,
      "project": args.project,
      "save_main_session": True,
      "region": args.region,
  }

  opts = beam.pipeline.PipelineOptions(flags=[], **options)

  with beam.Pipeline(args.runner, options=opts) as pipeline:
    if args.filetype == "tfrecord":
      tfrecord_pipeline(pipeline)
    elif args.filetype == "csv":
      csv_pipeline(pipeline)


if __name__ == "__main__":
  rebalance_data_shards()
