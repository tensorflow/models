# Copyright 2016 The TensorFlow Authors. All Rights Reserved.
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

"""Creates a text file with URLs to download FSNS dataset using aria2c.

The FSNS dataset has 640 files and takes 158Gb of the disk space. So it is
highly recommended to use some kind of a download manager to download it.

Aria2c is a powerful download manager which can download multiple files in
parallel, re-try if encounter an error and continue previously unfinished
downloads.
"""

import os

_FSNS_BASE_URL  = 'http://download.tensorflow.org/data/fsns-20160927/'
_SHARDS = {'test': 64, 'train': 512, 'validation':64}
_OUTPUT_FILE = "fsns_urls.txt"
_OUTPUT_DIR = "data/fsns"

def fsns_paths():
  paths = ['charset_size=134.txt']
  for name, shards in _SHARDS.items():
    for i in range(shards):
      paths.append('%s/%s-%05d-of-%05d' % (name, name, i, shards))
  return paths


if __name__ == "__main__":
  with open(_OUTPUT_FILE, "w") as f:
    for path in fsns_paths():
      url = _FSNS_BASE_URL + path
      dst_path = os.path.join(_OUTPUT_DIR, path)
      f.write("%s\n  out=%s\n" % (url, dst_path))
  print("To download FSNS dataset execute:")
  print("aria2c -c -j 20 -i %s" % _OUTPUT_FILE)
  print("The downloaded FSNS dataset will be stored under %s" % _OUTPUT_DIR)
