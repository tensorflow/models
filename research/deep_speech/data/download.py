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
"""tf.data.Dataset interface to the LibriSpeech dataset."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import argparse
import codecs
import csv
import fnmatch
import os
import sys
import tarfile
import tempfile
import unicodedata

from six.moves import urllib
from sox import Transformer
import tensorflow as tf

parser = argparse.ArgumentParser()

parser.add_argument(
    '--data_dir', type=str, default='/tmp/librispeech_data',
    help='Directory to download data and extract the tarball')

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
  """Download (and unzip) the given split of the dataset."""
  if not tf.gfile.Exists(directory):
    tf.gfile.MakeDirs(directory)
  _, tar_filepath = tempfile.mkstemp(suffix=".tar.gz")
  print("Downloading %s to %s" % (url, tar_filepath))
  urllib.request.urlretrieve(url, tar_filepath)
  tar = tarfile.open(tar_filepath, "r")
  tar.extractall(directory)
  tar.close()
  os.remove(tar_filepath)


def convert_audio_and_split_transcript(input_dir, source_name, target_name,
                                       output_dir, output_file):
  """Convert FLAC to WAV and split the transcript."""
  print("Preprocessing audio and transcript for %s" % source_name)
  source_dir = os.path.join(input_dir, source_name)
  target_dir = os.path.join(input_dir, target_name)

  if not os.path.exists(target_dir):
    os.makedirs(target_dir)

  # Each transcript file has the following format:
  # seq-id transcript_of_seq-id
  # For example:
  # 1-2-0 transcript_of_1-2-0.flac
  # 1-2-1 transcript_of_1-2-1.flac
  # ....
  files = []
  for root, _, filenames in os.walk(source_dir):
    for filename in fnmatch.filter(filenames, "*.trans.txt"):
      trans_file = os.path.join(root, filename)
      with codecs.open(trans_file, "r", "utf-8") as fin:
        for line in fin:
          seqid, transcript = line.split()[0], " ".join(line.split()[1:])
          transcript = unicodedata.normalize("NFKD", transcript).encode(
              "ascii", "ignore").decode("ascii", "ignore").strip().lower()

          # Convert FLAC to WAV.
          flac_file = os.path.join(root, seqid + ".flac")
          wav_file = os.path.join(target_dir, seqid + ".wav")
          if not os.path.exists(wav_file):
            Transformer().build(flac_file, wav_file)
          wav_filesize = os.path.getsize(wav_file)

          files.append((os.path.abspath(wav_file), wav_filesize, transcript))

  # Write to CSV file which contains columns:
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
  if not os.path.exists(FLAGS.data_dir):
    os.makedirs(FLAGS.data_dir)
  download_and_process_entire_dataset(FLAGS.data_dir)


if __name__ == '__main__':
  FLAGS, unparsed = parser.parse_known_args()
  tf.app.run(argv=[sys.argv[0]] + unparsed)
