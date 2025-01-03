#  Copyright 2018 The TensorFlow Authors. All Rights Reserved.
#
#  Licensed under the Apache License, Version 2.0 (the "License");
#  you may not use this file except in compliance with the License.
#  You may obtain a copy of the License at
#
#   http://www.apache.org/licenses/LICENSE-2.0
#
#  Unless required by applicable law or agreed to in writing, software
#  distributed under the License is distributed on an "AS IS" BASIS,
#  WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
#  See the License for the specific language governing permissions and
#  limitations under the License.
# ==============================================================================
"""Download and preprocess LibriSpeech dataset for DeepSpeech model."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import codecs
import fnmatch
import os
import sys
import tarfile
import tempfile
import unicodedata

from absl import app as absl_app
from absl import flags as absl_flags
import pandas
from six.moves import urllib
from sox import Transformer
import tensorflow as tf
from absl import logging

LIBRI_SPEECH_URLS = {
    "train-clean-100":
        "http://www.openslr.org/resources/12/train-clean-100.tar.gz",
    "train-clean-360":
        "http://www.openslr.org/resources/12/train-clean-360.tar.gz",
    "train-other-500":
        "http://www.openslr.org/resources/12/train-other-500.tar.gz",
    "dev-clean":
        "http://www.openslr.org/resources/12/dev-clean.tar.gz",
    "dev-other":
        "http://www.openslr.org/resources/12/dev-other.tar.gz",
    "test-clean":
        "http://www.openslr.org/resources/12/test-clean.tar.gz",
    "test-other":
        "http://www.openslr.org/resources/12/test-other.tar.gz"
}


def download_and_extract(directory, url):
  """Download and extract the given split of dataset.

  Args:
    directory: the directory where to extract the tarball.
    url: the url to download the data file.
  """

  if not tf.io.gfile.exists(directory):
    tf.io.gfile.makedirs(directory)

  _, tar_filepath = tempfile.mkstemp(suffix=".tar.gz")

  try:
    logging.info("Downloading %s to %s" % (url, tar_filepath))

    def _progress(count, block_size, total_size):
      sys.stdout.write("\r>> Downloading {} {:.1f}%".format(
          tar_filepath, 100.0 * count * block_size / total_size))
      sys.stdout.flush()

    urllib.request.urlretrieve(url, tar_filepath, _progress)
    print()
    statinfo = os.stat(tar_filepath)
    logging.info(
        "Successfully downloaded %s, size(bytes): %d" % (url, statinfo.st_size))
    with tarfile.open(tar_filepath, "r") as tar:
      tar.extractall(directory)
  finally:
    tf.io.gfile.remove(tar_filepath)


def convert_audio_and_split_transcript(input_dir, source_name, target_name,
                                       output_dir, output_file):
  """Convert FLAC to WAV and split the transcript.

  For audio file, convert the format from FLAC to WAV using the sox.Transformer
  library.
  For transcripts, each line contains the sequence id and the corresponding
  transcript (separated by space):
  Input data format: seq-id transcript_of_seq-id
  For example:
   1-2-0 transcript_of_1-2-0.flac
   1-2-1 transcript_of_1-2-1.flac
   ...

  Each sequence id has a corresponding .flac file.
  Parse the transcript file and generate a new csv file which has three columns:
  "wav_filename": the absolute path to a wav file.
  "wav_filesize": the size of the corresponding wav file.
  "transcript": the transcript for this audio segement.

  Args:
    input_dir: the directory which holds the input dataset.
    source_name: the name of the specified dataset. e.g. test-clean
    target_name: the directory name for the newly generated audio files.
                 e.g. test-clean-wav
    output_dir: the directory to place the newly generated csv files.
    output_file: the name of the newly generated csv file. e.g. test-clean.csv
  """

  logging.info("Preprocessing audio and transcript for %s" % source_name)
  source_dir = os.path.join(input_dir, source_name)
  target_dir = os.path.join(input_dir, target_name)

  if not tf.io.gfile.exists(target_dir):
    tf.io.gfile.makedirs(target_dir)

  files = []
  tfm = Transformer()
  # Convert all FLAC file into WAV format. At the same time, generate the csv
  # file.
  for root, _, filenames in tf.io.gfile.walk(source_dir):
    for filename in fnmatch.filter(filenames, "*.trans.txt"):
      trans_file = os.path.join(root, filename)
      with codecs.open(trans_file, "r", "utf-8") as fin:
        for line in fin:
          seqid, transcript = line.split(" ", 1)
          # We do a encode-decode transformation here because the output type
          # of encode is a bytes object, we need convert it to string.
          transcript = unicodedata.normalize("NFKD", transcript).encode(
              "ascii", "ignore").decode("ascii", "ignore").strip().lower()

          # Convert FLAC to WAV.
          flac_file = os.path.join(root, seqid + ".flac")
          wav_file = os.path.join(target_dir, seqid + ".wav")
          if not tf.io.gfile.exists(wav_file):
            tfm.build(flac_file, wav_file)
          wav_filesize = os.path.getsize(wav_file)

          files.append((os.path.abspath(wav_file), wav_filesize, transcript))

  # Write to CSV file which contains three columns:
  # "wav_filename", "wav_filesize", "transcript".
  csv_file_path = os.path.join(output_dir, output_file)
  df = pandas.DataFrame(
      data=files, columns=["wav_filename", "wav_filesize", "transcript"])
  df.to_csv(csv_file_path, index=False, sep="\t")
  logging.info("Successfully generated csv file {}".format(csv_file_path))


def download_and_process_datasets(directory, datasets):
  """Download and pre-process the specified list of LibriSpeech dataset.

  Args:
    directory: the directory to put all the downloaded and preprocessed data.
    datasets: list of dataset names that will be downloaded and processed.
  """

  logging.info("Preparing LibriSpeech dataset: {}".format(
      ",".join(datasets)))
  for dataset in datasets:
    logging.info("Preparing dataset %s", dataset)
    dataset_dir = os.path.join(directory, dataset)
    download_and_extract(dataset_dir, LIBRI_SPEECH_URLS[dataset])
    convert_audio_and_split_transcript(
        dataset_dir + "/LibriSpeech", dataset, dataset + "-wav",
        dataset_dir + "/LibriSpeech", dataset + ".csv")


def define_data_download_flags():
  """Define flags for data downloading."""
  absl_flags.DEFINE_string(
      "data_dir", "/tmp/librispeech_data",
      "Directory to download data and extract the tarball")
  absl_flags.DEFINE_bool("train_only", False,
                         "If true, only download the training set")
  absl_flags.DEFINE_bool("dev_only", False,
                         "If true, only download the dev set")
  absl_flags.DEFINE_bool("test_only", False,
                         "If true, only download the test set")


def main(_):
  if not tf.io.gfile.exists(FLAGS.data_dir):
    tf.io.gfile.makedirs(FLAGS.data_dir)

  if FLAGS.train_only:
    download_and_process_datasets(
        FLAGS.data_dir,
        ["train-clean-100", "train-clean-360", "train-other-500"])
  elif FLAGS.dev_only:
    download_and_process_datasets(FLAGS.data_dir, ["dev-clean", "dev-other"])
  elif FLAGS.test_only:
    download_and_process_datasets(FLAGS.data_dir, ["test-clean", "test-other"])
  else:
    # By default we download the entire dataset.
    download_and_process_datasets(FLAGS.data_dir, LIBRI_SPEECH_URLS.keys())


if __name__ == "__main__":
  logging.set_verbosity(logging.INFO)
  define_data_download_flags()
  FLAGS = absl_flags.FLAGS
  absl_app.run(main)
