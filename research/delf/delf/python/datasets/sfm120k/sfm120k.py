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
"""Structure-from-Motion dataset (Sfm120k) module.

[1] From Single Image Query to Detailed 3D Reconstruction.
Johannes L. Schonberger, Filip Radenovic, Ondrej Chum, Jan-Michael Frahm.
The related paper can be found at: https://ieeexplore.ieee.org/document/7299148.
"""

import os
import pickle
import tensorflow as tf

from delf.python.datasets import tuples_dataset
from delf.python.datasets import utils


def id2filename(image_id, prefix):
  """Creates a training image path out of its id name.

  Used for the image mapping in the Sfm120k datset.

  Args:
    image_id: String, image id.
    prefix: String, root directory where images are saved.

  Returns:
   filename: String, full image filename.
  """
  if prefix:
    return os.path.join(prefix, image_id[-2:], image_id[-4:-2], image_id[-6:-4],
                        image_id)
  else:
    return os.path.join(image_id[-2:], image_id[-4:-2], image_id[-6:-4],
                        image_id)


class _Sfm120k(tuples_dataset.TuplesDataset):
  """Structure-from-Motion (Sfm120k) dataset instance.

  The dataset contains the image names lists for training and validation,
  the cluster ID (3D model ID) for each image and indices forming
  query-positive pairs of images. The images are loaded per epoch and resized
  on the fly to the desired dimensionality.
  """

  def __init__(self, mode, data_root, imsize=None, num_negatives=5,
               num_queries=2000, pool_size=20000, loader=utils.default_loader,
               eccv2020=False):
    """Structure-from-Motion (Sfm120k) dataset initialization.

    Args:
      mode: Either 'train' or 'val'.
      data_root: Path to the root directory of the dataset.
      imsize: Integer, defines the maximum size of longer image side.
      num_negatives: Integer, number of negative images per one query.
      num_queries: Integer, number of query images.
      pool_size: Integer, size of the negative image pool, from where the
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
    db_filename = os.path.join(db_root, '{}.pkl'.format(name))

    with tf.io.gfile.GFile(db_filename, 'rb') as f:
      db = pickle.load(f)[mode]

    # Setting full paths for the dataset images.
    self.images = [id2filename(img_name, None) for
                   img_name in db['cids']]

    # Initializing tuples dataset.
    super().__init__(name, mode, db_root, imsize, num_negatives, num_queries,
                     pool_size, loader, ims_root)

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


def CreateDataset(mode, data_root, imsize=None, num_negatives=5,
                  num_queries=2000, pool_size=20000,
                  loader=utils.default_loader, eccv2020=False):
  '''Creates Structure-from-Motion (Sfm120k) dataset.

  Args:
    mode: String, either 'train' or 'val'.
    data_root: Path to the root directory of the dataset.
    imsize: Integer, defines the maximum size of longer image side.
    num_negatives: Integer, number of negative images per one query.
    num_queries: Integer, number of query images.
    pool_size: Integer, size of the negative image pool, from where the
      hard-negative images are chosen.
    loader: Callable, a function to load an image given its path.
    eccv2020: Bool, whether to use a new validation dataset used with ECCV
      2020 paper (https://arxiv.org/abs/2007.13172).

  Returns:
    sfm120k: Sfm120k dataset instance.
  '''
  return _Sfm120k(mode, data_root, imsize, num_negatives, num_queries,
                  pool_size, loader, eccv2020)
