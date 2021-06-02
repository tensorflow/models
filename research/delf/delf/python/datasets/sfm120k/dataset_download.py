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
"""Structure-from-Motion dataset (Sfm120k) download function.
"""

import os

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
  if not os.path.isdir(data_dir):
    os.mkdir(data_dir)

  # Create datasets folder if does not exist.
  datasets_dir = os.path.join(data_dir, 'train')
  if not os.path.isdir(datasets_dir):
    os.mkdir(datasets_dir)

  # Download folder train/retrieval-SfM-120k/.
  src_dir = os.path.join('http://cmp.felk.cvut.cz/cnnimageretrieval/data',
                         'train', 'ims')
  dst_dir = os.path.join(datasets_dir, 'retrieval-SfM-120k', 'ims')
  dl_file = 'ims.tar.gz'
  if not os.path.isdir(dst_dir):
    src_file = os.path.join(src_dir, dl_file)
    dst_file = os.path.join(dst_dir, dl_file)
    print('>> Image directory does not exist. Creating: {}'.format(dst_dir))
    os.makedirs(dst_dir)
    print('>> Downloading ims.tar.gz...')
    os.system('wget {} -O {}'.format(src_file, dst_file))
    print('>> Extracting {}...'.format(dst_file))
    os.system('tar -zxf {} -C {}'.format(dst_file, dst_dir))
    print('>> Extracted, deleting {}...'.format(dst_file))
    os.system('rm {}'.format(dst_file))

  # Create symlink for train/retrieval-SfM-30k/.
  dst_dir_old = os.path.join(datasets_dir, 'retrieval-SfM-120k', 'ims')
  dst_dir = os.path.join(datasets_dir, 'retrieval-SfM-30k', 'ims')
  if not (os.path.isdir(dst_dir) or os.path.islink(dst_dir)):
    os.makedirs(os.path.join(datasets_dir, 'retrieval-SfM-30k'))
    os.system('ln -s {} {}'.format(dst_dir_old, dst_dir))
    print(
            '>> Created symbolic link from retrieval-SfM-120k/ims to '
            'retrieval-SfM-30k/ims')

  # Download db files.
  src_dir = os.path.join('http://cmp.felk.cvut.cz/cnnimageretrieval/data',
                         'train', 'dbs')
  datasets = ['retrieval-SfM-120k', 'retrieval-SfM-30k']
  for dataset in datasets:
    dst_dir = os.path.join(datasets_dir, dataset)
    if dataset == 'retrieval-SfM-120k':
      dl_files = ['{}.pkl'.format(dataset),
                  '{}-whiten.pkl'.format(dataset)]
      dl_eccv2020 = '{}-val-eccv2020.pkl'.format(dataset)
    elif dataset == 'retrieval-SfM-30k':
      dl_files = ['{}-whiten.pkl'.format(dataset)]
      dl_eccv2020 = None

    if not os.path.isdir(dst_dir):
      print('>> Dataset directory does not exist. Creating: {}'.format(
              dst_dir))
      os.mkdir(dst_dir)

    for i in range(len(dl_files)):
      src_file = os.path.join(src_dir, dl_files[i])
      dst_file = os.path.join(dst_dir, dl_files[i])
      if not os.path.isfile(dst_file):
        print('>> DB file {} does not exist. Downloading...'.format(
                dl_files[i]))
        os.system('wget {} -O {}'.format(src_file, dst_file))

      if dl_eccv2020:
        eccv2020_dst_file = os.path.join(dst_dir, dl_eccv2020)
        if not os.path.isfile(eccv2020_dst_file):
          eccv2020_src_dir = \
            "http://ptak.felk.cvut.cz/personal/toliageo/share/how/dataset/"
          eccv2020_dst_file = os.path.join(dst_dir, dl_eccv2020)
          eccv2020_src_file = os.path.join(eccv2020_src_dir,
                                           dl_eccv2020)
          os.system('wget {} -O {}'.format(eccv2020_src_file,
                                           eccv2020_dst_file))
