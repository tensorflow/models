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

"""Downloads and prepares TriviaQA dataset."""
from unittest import mock

from absl import app
from absl import flags
from absl import logging
import apache_beam as beam
import tensorflow_datasets as tfds

from official.projects.triviaqa import dataset  # pylint: disable=unused-import

flags.DEFINE_integer('sequence_length', 4096, 'Max number of tokens.')

flags.DEFINE_integer(
    'global_sequence_length', None,
    'Max number of question tokens plus sentences. If not set, defaults to '
    'sequence_length // 16 + 64.')

flags.DEFINE_integer(
    'stride', 3072,
    'For documents longer than `sequence_length`, where to split them.')

flags.DEFINE_string(
    'sentencepiece_model_path', None,
    'SentencePiece model to use for tokenization.')

flags.DEFINE_string('data_dir', None, 'Data directory for TFDS.')

flags.DEFINE_string('runner', 'DirectRunner', 'Beam runner to use.')

FLAGS = flags.FLAGS


def main(argv):
  if len(argv) > 1:
    raise app.UsageError('Too many command-line arguments.')
  builder = tfds.builder(
      'bigbird_trivia_qa/rc_wiki.preprocessed',
      data_dir=FLAGS.data_dir,
      sentencepiece_model_path=FLAGS.sentencepiece_model_path,
      sequence_length=FLAGS.sequence_length,
      global_sequence_length=FLAGS.global_sequence_length,
      stride=FLAGS.stride)
  download_config = tfds.download.DownloadConfig(
      beam_options=beam.options.pipeline_options.PipelineOptions(flags=[
          f'--runner={FLAGS.runner}',
          '--direct_num_workers=8',
          '--direct_running_mode=multi_processing',
      ]))
  with mock.patch('tensorflow_datasets.core.download.extractor._normpath',
                  new=lambda x: x):
    builder.download_and_prepare(download_config=download_config)
  logging.info(builder.info.splits)


if __name__ == '__main__':
  flags.mark_flag_as_required('sentencepiece_model_path')
  app.run(main)
