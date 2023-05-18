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

"""A script to run waymo open dataset preprocessing."""

import os

from absl import app
from absl import flags
from absl import logging
import apache_beam as beam
from apache_beam.io import tfrecordio
import tensorflow as tf

from official.modeling import hyperparams
from official.projects.pointpillars.configs import pointpillars
from official.projects.pointpillars.utils.wod_processor import WodProcessor
from waymo_open_dataset import dataset_pb2


_SRC_DIR = flags.DEFINE_string(
    'src_dir', None,
    'The direcotry to read official wod tfrecords,')
_DST_DIR = flags.DEFINE_string(
    'dst_dir', None,
    'The direcotry to write processed tfrecords.')
_CONFIG_FILE = flags.DEFINE_string(
    'config_file', None,
    'YAML file to specify configurations.')
_PIPELINE_OPTIONS = flags.DEFINE_string(
    'pipeline_options', None,
    'Command line flags to use in constructing the Beam pipeline options. '
    'See https://beam.apache.org/documentation/#runners for available runners.')

# The --src_dir must contain these two sub-folders.
_SRC_FOLDERS = ['training', 'validation']


def read_dataset(pipeline: beam.Pipeline,
                 src_file_pattern: str) -> beam.PCollection:
  reader = tfrecordio.ReadFromTFRecord(
      src_file_pattern,
      coder=beam.coders.ProtoCoder(dataset_pb2.Frame))
  raw_frames = pipeline | f'Read frames: {src_file_pattern}' >> reader
  return raw_frames


def count_examples(examples: beam.PCollection, dst_path: str):
  writer = beam.io.WriteToText(
      dst_path,
      file_name_suffix='.stats.txt',
      num_shards=1)
  _ = (examples
       | 'Count examples' >> beam.combiners.Count.Globally()
       | 'Write statistics' >> writer)


def write_dataset(examples: beam.PCollection, dst_path: str):
  writer = tfrecordio.WriteToTFRecord(
      dst_path,
      coder=beam.coders.ProtoCoder(tf.train.Example),
      file_name_suffix='.tfrecord',
      compression_type='gzip')
  _ = examples | f'Write examples: {dst_path}' >> writer


def process_wod(pipeline: beam.Pipeline,
                src_file_pattern: str,
                dst_path: str,
                wod_processor: WodProcessor):
  """Creates the process WOD dataset pipeline."""
  raw_frames = read_dataset(pipeline, src_file_pattern)
  examples = (
      raw_frames
      | 'Reshuffle post read' >> beam.Reshuffle()
      | 'Process one frame' >> beam.Map(
          wod_processor.process_and_convert_to_tf_example)
      | 'Reshuffle post decode' >> beam.Reshuffle())
  count_examples(examples, dst_path)
  write_dataset(examples, dst_path)


def main(_):
  pipeline_options = beam.options.pipeline_options.PipelineOptions(
      _PIPELINE_OPTIONS.value.split(','))

  if _CONFIG_FILE.value:
    cfg = hyperparams.read_yaml_to_params_dict(_CONFIG_FILE.value)
    image_config = cfg.task.model.image
    pillars_config = cfg.task.model.pillars
  else:
    cfg = pointpillars
    image_config = cfg.ImageConfig()
    pillars_config = cfg.PillarsConfig()

  wod_processor = WodProcessor(image_config, pillars_config)
  for folder in _SRC_FOLDERS:
    src_file_pattern = os.path.join(_SRC_DIR.value, folder, '*.tfrecord')
    dst_path = os.path.join(_DST_DIR.value, folder)
    logging.info('Processing %s, writing to %s', src_file_pattern, dst_path)

    pipeline = beam.Pipeline(options=pipeline_options)
    process_wod(pipeline, src_file_pattern, dst_path, wod_processor)
    pipeline.run().wait_until_finish()


if __name__ == '__main__':
  app.run(main)
