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

import os
import sys
import tempfile
import zipfile

from absl import app as absl_app
from absl import flags
from six.moves import urllib  # pylint: disable=redefined-builtin
import tensorflow as tf

from official.utils.flags import core as flags_core


# URL to download dataset
_DATA_URL = "http://files.grouplens.org/datasets/movielens/ml-latest.zip"
_FILES = ["movies.csv", "links.csv", "genome-scores.csv", "genome-tags.csv",
          "tags.csv", "ratings.csv", "README.txt"]


def download_and_extract(data_dir):
  if not tf.gfile.Exists(data_dir):
    tf.gfile.MakeDirs(data_dir)

  filename = _DATA_URL.split('/')[-1]
  filepath = os.path.join(data_dir, filename)

  skip = all([tf.gfile.Exists(os.path.join(data_dir, i)) for i in _FILES])

  if not skip:
    if tf.gfile.Exists(data_dir):
      tf.gfile.DeleteRecursively(data_dir)
    tf.gfile.MakeDirs(data_dir)

    def _progress(count, block_size, total_size):
      sys.stdout.write("\r>> Downloading {} {:.1f}%".format(
          filename, 100.0 * count * block_size / total_size))
      sys.stdout.flush()

    try:
      temp_filepath, _ = urllib.request.urlretrieve(
          url=_DATA_URL, reporthook=_progress)
      print()

      with tempfile.TemporaryDirectory() as temp_dir:
        zipfile.ZipFile(temp_filepath, "r").extractall(temp_dir)
        assert set(os.listdir(temp_dir)) == {"ml-latest"}
        files = os.listdir(os.path.join(temp_dir, "ml-latest"))
        assert set(files) == set(_FILES)
        for i in files:
          tf.gfile.Copy(os.path.join(temp_dir, "ml-latest", i),
                        os.path.join(data_dir, i))
          print(i.ljust(20), "copied")

    finally:
      tf.gfile.Remove(temp_filepath)


def define_data_download_flags():
  """Add flags specifying data download arguments."""
  flags.DEFINE_string(
      name="data_dir", default="/tmp/movielens-data-wide-deep/",
      help=flags_core.help_wrap(
          "Directory to download and extract data."))


def main(_):
  download_and_extract(flags.FLAGS.data_dir)


if __name__ == "__main__":
  tf.logging.set_verbosity(tf.logging.INFO)
  define_data_download_flags()
  absl_app.run(main)
