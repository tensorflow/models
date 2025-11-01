# Copyright 2021 The TensorFlow Authors All Rights Reserved.
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
"""Structure-from-Motion dataset (Sfm120k) download function."""

import os
import urllib3
import tarfile
import requests
from tqdm import tqdm

import tensorflow as tf

# Suppress SSL Warnings
urllib3.disable_warnings(urllib3.exceptions.InsecureRequestWarning)


def wget(src, dst, sha256checksum=False):
    """Download a file and calculate it's SHA256 chacksum"""

    # Writing in chunks helps in preventing running out of memory
    with requests.get(src, stream=True, verify=False) as r:
        r.raise_for_status()
        with open(dst, 'wb') as f, tqdm(desc="Downloading",
        unit='iB',
        unit_scale=True,
        unit_divisor=1024,
    ) as bar:
            for chunk in r.iter_content(chunk_size=8192):
                size = f.write(chunk)
                bar.update(size)

    if sha256checksum:
        from hashlib import sha256
        checksum = sha256(open(dst,"rb").read()).hexdigest()
        print("SHA256 checksum: ", checksum)
        f = open(f"{dst}.sha256","a")
        f.write(f"{checksum}  {dst}")
        f.close()


def download_train(data_dir):
  """Checks, and, if required, downloads the necessary files for the training.

  Checks if the data necessary for running the example training script exist.
  If not, it downloads it in the following folder structure:
    DATA_ROOT/train/retrieval-SfM-120k/ : folder with rsfm120k images and db
      files.
    DATA_ROOT/train/retrieval-SfM-30k/  : folder with rsfm30k images and db
      files.
  """

  # Create data folder if does not exist.
  if not tf.io.gfile.exists(data_dir):
    tf.io.gfile.mkdir(data_dir)

  # Create datasets folder if does not exist.
  datasets_dir = os.path.join(data_dir, 'train')
  if not tf.io.gfile.exists(datasets_dir):
    tf.io.gfile.mkdir(datasets_dir)

  # Download folder train/retrieval-SfM-120k/.
  src_dir = 'http://cmp.felk.cvut.cz/cnnimageretrieval/data/train/ims'
  dst_dir = os.path.join(datasets_dir, 'retrieval-SfM-120k', 'ims')
  download_file = 'ims.tar.gz'
  if not tf.io.gfile.exists(dst_dir):
    src_file = os.path.join(src_dir, download_file)
    dst_file = os.path.join(dst_dir, download_file)
    print('>> Image directory does not exist. Creating: {}'.format(dst_dir))
    tf.io.gfile.makedirs(dst_dir)
    print('>> Downloading ims.tar.gz...')
    wget(src_file, dst_file, sha256checksum=True)
    print('>> Extracting {}...'.format(dst_file))
    downloaded_tar = tarfile.open(dst_file)
    downloaded_tar.extractall(dst_dir)
    downloaded_tar.close()
    print('>> Extracted, deleting {}...'.format(dst_file))
    if os.path.exists(dst_file):
      os.remove(dst_file)


  # Create symlink for train/retrieval-SfM-30k/.
  dst_dir_old = os.path.join(datasets_dir, 'retrieval-SfM-120k', 'ims')
  dst_dir = os.path.join(datasets_dir, 'retrieval-SfM-30k', 'ims')
  if not (tf.io.gfile.exists(dst_dir) or os.path.islink(dst_dir)):
    tf.io.gfile.makedirs(os.path.join(datasets_dir, 'retrieval-SfM-30k'))
    os.symlink(dst_dir_old, dst_dir)
    print(
            '>> Created symbolic link from retrieval-SfM-120k/ims to '
            'retrieval-SfM-30k/ims')

  # Download db files.
  src_dir = 'http://cmp.felk.cvut.cz/cnnimageretrieval/data/train/dbs'
  datasets = ['retrieval-SfM-120k', 'retrieval-SfM-30k']
  for dataset in datasets:
    dst_dir = os.path.join(datasets_dir, dataset)
    if dataset == 'retrieval-SfM-120k':
      download_files = ['{}.pkl'.format(dataset),
                        '{}-whiten.pkl'.format(dataset)]
      download_eccv2020 = '{}-val-eccv2020.pkl'.format(dataset)
    elif dataset == 'retrieval-SfM-30k':
      download_files = ['{}-whiten.pkl'.format(dataset)]
      download_eccv2020 = None

    if not tf.io.gfile.exists(dst_dir):
      print('>> Dataset directory does not exist. Creating: {}'.format(
              dst_dir))
      tf.io.gfile.mkdir(dst_dir)

    for i in range(len(download_files)):
      src_file = os.path.join(src_dir, download_files[i])
      dst_file = os.path.join(dst_dir, download_files[i])
      if not os.path.isfile(dst_file):
        print('>> DB file {} does not exist. Downloading...'.format(
                download_files[i]))
        wget(src_file, dst_file)

      if download_eccv2020:
        eccv2020_dst_file = os.path.join(dst_dir, download_eccv2020)
        if not os.path.isfile(eccv2020_dst_file):
          eccv2020_src_dir = \
            "http://ptak.felk.cvut.cz/personal/toliageo/share/how/dataset/"
          eccv2020_dst_file = os.path.join(dst_dir, download_eccv2020)
          eccv2020_src_file = os.path.join(eccv2020_src_dir,
                                           download_eccv2020)
          wget(eccv2020_src_file, eccv2020_dst_file)
