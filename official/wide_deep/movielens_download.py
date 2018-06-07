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
import tempfile

from absl import app as absl_app
from absl import flags
from six.moves import urllib  # pylint: disable=redefined-builtin
import tensorflow as tf

from official.utils.flags import core as flags_core


# URL to download dataset
_DATA_URL = "http://files.grouplens.org/datasets/movielens/ml-latest.zip"


def download_and_extract(data_dir):
  if not tf.gfile.Exists(data_dir):
    tf.gfile.MakeDirs(data_dir)

  filename = _DATA_URL.split('/')[-1]
  filepath = os.path.join(data_dir, filename)

  if not tf.gfile.Exists(filepath):
    def _progress(count, block_size, total_size):
      sys.stdout.write("\r>> Downloading {} {:.1f}%".format(
          file_path, 100.0 * count * block_size / total_size))
      sys.stdout.flush()

    temp_filename, _ = urllib.request.urlretrieve(
        url=_DATA_URL, reporthook=_progress)
    statinfo = os.stat(temp_filename)
    print(temp_filename)
    print(statinfo)



from urllib import request
request.urlretrieve()

def define_data_download_flags():
  """Add flags specifying data download arguments."""
  flags.DEFINE_string(
      name="data_dir", default="/tmp/movielens-data-wide-deep/",
      help=flags_core.help_wrap(
          "Directory to download and extract data."))


def main(flags_obj):
  download_and_extract(flags_obj.data_dir)


if __name__ == "__main__":
  tf.logging.set_verbosity(tf.logging.INFO)
  define_data_download_flags()
  absl_app.run(main)
