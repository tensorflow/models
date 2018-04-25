# Copyright 2018 The TensorFlow Authors. All Rights Reserved.
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
"""Helper functions to automate downloading of small datasets.

The download helper uses tf.gfile where appropriate, and as a result allows
downloading to distributed filesystems.
"""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import functools
import gzip
import os
import re
import shutil
import sys
import tarfile
import tempfile

from six.moves import urllib

import tensorflow as tf


def download_and_extract(dest_root, data_url):
  """(maybe) Download (and possibly uncompress) a file.

  If the eventual result already exists then this function will exit early. No
  correctness checking is performed; only existence checking.

  Example:
    download_and_extract("gs://my-bucket/foo", "http://bar.com/baz.txt.gz")
    produces "gs://my-bucket/foo/baz.txt"

  Args:
    dest_root:  The destination directory into which files should be placed.
    data_url: The remote url of the target file.

  Returns:
    The downloaded result. If the remote file is a tarball the result will be
    a directory; otherwise it will be a file.
  """
  target = os.path.split(data_url)[1]
  target = re.sub(r"(\.tar)?\.gz$", "", target)  # strip zip suffix
  if tf.gfile.Exists(os.path.join(dest_root, target)):
    tf.logging.info("{} already exists. {} will not be downloaded.".format(
        os.path.join(dest_root, target), data_url
    ))
    return os.path.join(dest_root, target)

  if data_url.endswith(".tar.gz"):
    _download_tarball(dest_root=dest_root, target=target, data_url=data_url)
  elif data_url.endswith(".gz"):
    _download_gzip(dest_root=dest_root, target=target, data_url=data_url)
  else:
    _download_raw(dest_root=dest_root, target=target, data_url=data_url)


def _progress(count, block_size, total_size, filename):
  sys.stdout.write("\r>> Downloading %s %.1f%%" % (
      filename, min([100.0 * count * block_size / total_size, 100.0])))
  sys.stdout.flush()


def _fetch(file_path, data_url):
  urllib.request.urlretrieve(
      data_url, file_path,
      functools.partial(_progress, filename=file_path)
  )
  print()  # There is no line break in progress to enable \r syntax


def _download_tarball(dest_root, target, data_url):
  download_dir = tempfile.mkdtemp()
  try:
    tarball_path = os.path.join(download_dir, os.path.split(data_url)[1])
    _fetch(file_path=tarball_path, data_url=data_url)

    tarfile.open(tarball_path, "r:gz").extractall(download_dir)
    os.remove(tarball_path)
    assert set(os.listdir(download_dir)) == {target}

    temp_root = os.path.join(download_dir, target)
    for root, _, fnames in tf.gfile.Walk(temp_root):
      destination_folder = os.path.join(dest_root, root[len(temp_root)+1:])
      tf.gfile.MakeDirs(destination_folder)
      for fname in fnames:
        source_path = os.path.join(root, fname)
        dest_path = os.path.join(destination_folder, fname)
        if dest_path.startswith("gs://"):
          print(">> Uploading {} to {}".format(source_path, dest_path))
        tf.gfile.Copy(source_path, dest_path)

  finally:
    shutil.rmtree(download_dir)


def _download_gzip(dest_root, target, data_url):
  _, zipped_path = tempfile.mkstemp(suffix=".gz")
  try:
    unzipped_path = os.path.join(dest_root, target)
    _fetch(file_path=zipped_path, data_url=data_url)

    tf.gfile.MakeDirs(dest_root)
    with gzip.open(zipped_path, "rb") as f_in, \
        tf.gfile.Open(unzipped_path, "wb") as f_out:
      if unzipped_path.startswith("gs://"):
        print(">> Uploading {} to {}".format(zipped_path, unzipped_path))
      shutil.copyfileobj(f_in, f_out)

  finally:
    os.remove(zipped_path)


def _download_raw(dest_root, target, data_url):
  _, temp_path = tempfile.mkstemp()
  try:
    _fetch(file_path=temp_path, data_url=data_url)
    dest_path = os.path.join(dest_root, target)
    tf.gfile.MakeDirs(dest_root)
    tf.gfile.Copy(temp_path, dest_path)

  finally:
    os.remove(temp_path)
