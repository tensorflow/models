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
"""Structure-from-Motion dataset (Sfm120k)."""

import os
import pickle
import tensorflow as tf

from delf.python.datasets import tuples_dataset
from delf.python.datasets import utils


def id2filename(id, prefix):
  """Creates a training image path out of its CID name.

  Used for the image mapping in the Sfm120k datset.

  Args:
    id: String, image id.
    prefix: String, root directory where images are saved.

  Returns:
   filename: String, full image filename.
  """
  if prefix:
    return os.path.join(prefix, id[-2:], id[-4:-2], id[-6:-4], id)
  else:
    return os.path.join(id[-2:], id[-4:-2], id[-6:-4], id)


class _Sfm120k(tuples_dataset.TuplesDataset):
  """Structure-from-Motion (Sfm120k) dataset instance.

  The dataset contains the image names lists for training and validation,
  the cluster ID (3D model ID) for each image and indices forming
  query-positive pairs of images. The images are loaded per epoch and resized
  on the fly to the desired dimensionality. Extends
  tuples_dataset.TuplesDataset.
  """

  def __init__(self, mode, data_root, imsize=None, nnum=5, qsize=2000,
               poolsize=20000, loader=utils.default_loader, eccv2020=False):
    """Structure-from-Motion (Sfm120k) dataset initialization.

    Args:
      mode: Either 'train' or 'val'.
      data_root: Path to the root directory of the dataset.
      imsize: Integer, defines the maximum size of longer image side.
      nnum: Integer, number of negative images per one query.
      qsize: Integer, number of query images.
      poolsize: Integer, size of the negative image pool, from where the
        hard-negative images are chosen.
      loader: Callable, a function to load an image given its path.
      eccv2020: Bool, whether to use a new validation dataset used with ECCV
        2020 paper (https://arxiv.org/abs/2007.13172).

    Raises:
      ValueError: Raised if `mode` is not one of 'train' or 'val'.
    """
    if mode not in ['train', 'val']:
      raise ValueError(
        "`mode` argument should be either 'train' or 'val', passed as a "
        "String.")

    # Setting up the paths for the dataset.
    if eccv2020:
      name = "retrieval-SfM-120k-val-eccv2020"
    else:
      name = "retrieval-SfM-120k"
    db_root = os.path.join(data_root, 'train/retrieval-SfM-120k')
    ims_root = os.path.join(db_root, 'ims/')

    # Loading the dataset db file.
    db_fn = os.path.join(db_root, '{}.pkl'.format(name))

    with tf.io.gfile.GFile(db_fn, 'rb') as f:
      db = pickle.load(f)[mode]

    # Setting full paths for the dataset images.
    self.images = [id2filename(img_name, None) for
                   img_name in db['cids']]

    # Initializing tuples dataset.
    super().__init__(name, mode, db_root, imsize, nnum, qsize, poolsize,
                     loader, ims_root)

  def Sfm120kInfo(self):
    """Metadata for the Sfm120k dataset.

    The dataset contains the image names lists for training and
    validation, the cluster ID (3D model ID) for each image and indices
    forming query-positive pairs of images. The images are loaded per epoch
    and resized on the fly to the desired dimensionality.

    Returns:
      info: dictionary with the dataset parameters.
    """
    info = {'train': {'clusters': 91642, 'pidxs': 181697, 'qidxs': 181697},
            'val': {'clusters': 6403, 'pidxs': 1691, 'qidxs': 1691}}
    return info


def CreateDataset(mode, data_root, imsize=None, nnum=5, qsize=2000,
                  poolsize=20000, loader=utils.default_loader,
                  eccv2020=False):
  '''Creates Structure-from-Motion (Sfm120k) dataset.

  Args:
    mode: String, either 'train' or 'val'.
    data_root: Path to the root directory of the dataset.
    imsize: Integer, defines the maximum size of longer image side.
    nnum: Integer, number of negative images per one query.
    qsize: Integer, number of query images.
    poolsize: Integer, size of the negative image pool, from where the
      hard-negative images are chosen.
    loader: Callable, a function to load an image given its path.
    eccv2020: Bool, whether to use a new validation dataset used with ECCV
      2020 paper (https://arxiv.org/abs/2007.13172).

  Returns:
    sfm120k: Sfm120k dataset instance.
  '''
  return _Sfm120k(mode, data_root, imsize, nnum, qsize, poolsize, loader,
                  eccv2020)


def download_train(data_dir):
  """Checks, and, if required, downloads the necessary files for the training.

  download_train(DATA_ROOT) checks if the data necessary for running
  the example script exist.
  If not it downloads it in the following folder structure:
    DATA_ROOT/train/retrieval-SfM-120k/ : folder with rsfm120k images and db
      files.
    DATA_ROOT/train/retrieval-SfM-30k/  : folder with rsfm30k images and db
      files.
  """

  # Create data folder if it does not exist.
  if not os.path.isdir(data_dir):
    os.mkdir(data_dir)

  # Create datasets folder if it does not exist.
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
          