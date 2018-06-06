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
"""tf.data.Dataset interface to the LibriSpeech dataset."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import codecs
import csv
import fnmatch
import os
import sys
import tarfile
import tempfile
import unicodedata

from absl import flags as absl_flags
from six.moves import urllib
from sox import Transformer
import tensorflow as tf

FLAGS = absl_flags.FLAGS

absl_flags.DEFINE_string(
    "data_dir", "/tmp/librispeech_data",
    "Directory to download data and extract the tarball")

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

  if not tf.gfile.Exists(directory):
    tf.gfile.MakeDirs(directory)
  _, tar_filepath = tempfile.mkstemp(suffix=".tar.gz")
  tf.logging.info("Downloading %s to %s" % (url, tar_filepath))
  urllib.request.urlretrieve(url, tar_filepath)
  with tarfile.open(tar_filepath, "r") as tar:
    try:
      tar.extractall(directory)
    finally:
      tf.gfile.Remove(tar_filepath)


def convert_audio_and_split_transcript(input_dir, source_name, target_name,
                                       output_dir, output_file):
  """Convert FLAC to WAV and split the transcript.

  For audio file, convert the format from FLAC to WAV using the sox.Transformer library.
  For transcripts, each line contains the sequence id and the corresponding transcript (separated by space):
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
    target_name: the directory name for the newly generated audio files. e.g. test-clean-wav
    output_dir: the directory to place the newly generated csv files.
    output_file: the name of the newly generated csv file. e.g. test-clean.csv
  """

  tf.logging.info("Preprocessing audio and transcript for %s" % source_name)
  source_dir = os.path.join(input_dir, source_name)
  target_dir = os.path.join(input_dir, target_name)

  if not tf.gfile.Exists(target_dir):
    tf.gfile.MakeDirs(target_dir)

  files = []
  tfm = Transformer()
  # Convert all FLAC file into WAV format. At the same time, generate the csv
  # file.
  for root, _, filenames in tf.gfile.Walk(source_dir):
    for filename in fnmatch.filter(filenames, "*.trans.txt"):
      trans_file = os.path.join(root, filename)
      with codecs.open(trans_file, "r", "utf-8") as fin:
        for line in fin:
          seqid, transcript = line.split(' ', 1)
          transcript = unicodedata.normalize("NFKD", transcript).encode(
              "ascii", "ignore").decode("ascii", "ignore").strip().lower()

          # Convert FLAC to WAV.
          flac_file = os.path.join(root, seqid + ".flac")
          wav_file = os.path.join(target_dir, seqid + ".wav")
          if not tf.gfile.Exists(wav_file):
            tfm.build(flac_file, wav_file)
          wav_filesize = os.path.getsize(wav_file)

          files.append((os.path.abspath(wav_file), wav_filesize, transcript))

  # Write to CSV file which contains three columns:
  # "wav_filename", "wav_filesize", "transcript".
  with open(os.path.join(output_dir, output_file), "w") as csvfile:
    fieldnames = ["wav_filename", "wav_filesize", "transcript"]
    writer = csv.DictWriter(csvfile, delimiter="\t", fieldnames=fieldnames)
    writer.writeheader()
    for item in files:
      writer.writerow({
          "wav_filename": item[0],
          "wav_filesize": item[1],
          "transcript": item[2]
      })


def download_and_process_entire_dataset(directory):
  """Download and pre-process all the LibriSpeech data."""
  for dataset in LIBRI_SPEECH_URLS:
    dataset_dir = os.path.join(directory, dataset)
    download_and_extract(dataset_dir, LIBRI_SPEECH_URLS[dataset])
    convert_audio_and_split_transcript(
        dataset_dir + "/LibriSpeech", dataset, dataset + "-wav",
        dataset_dir + "/LibriSpeech", dataset + ".csv")


def main(_):
  if not tf.gfile.Exists(FLAGS.data_dir):
    tf.gfile.MakeDirs(FLAGS.data_dir)
  download_and_process_entire_dataset(FLAGS.data_dir)


if __name__ == '__main__':
  tf.logging.set_verbosity(tf.logging.INFO)
  tf.app.run()
