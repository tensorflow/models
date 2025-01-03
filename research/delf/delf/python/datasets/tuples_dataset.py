# Lint as: python3
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
"""Tuple dataset module.

Based on the Radenovic et al. ECCV16: CNN image retrieval learns from BoW.
For more information refer to https://arxiv.org/abs/1604.02426.
"""

import os
import pickle

import numpy as np
import tensorflow as tf

from delf.python.datasets import utils as image_loading_utils
from delf.python.training import global_features_utils
from delf.python.training.model import global_model


class TuplesDataset():
  """Data loader that loads training and validation tuples.

  After initialization, the function create_epoch_tuples() should be called to
  create the dataset tuples. After that, the dataset can be iterated through
  using next() function.
  Tuples are based on Radenovic et al. ECCV16 work: CNN image retrieval
  learns from BoW. For more information refer to
  https://arxiv.org/abs/1604.02426.
  """

  def __init__(self, name, mode, data_root, imsize=None, num_negatives=5,
               num_queries=2000, pool_size=20000,
               loader=image_loading_utils.default_loader, ims_root=None):
    """TuplesDataset object initialization.

    Args:
      name: String, dataset name. I.e. 'retrieval-sfm-120k'.
      mode: 'train' or 'val' for training and validation parts of dataset.
      data_root: Path to the root directory of the dataset.
      imsize: Integer, defines the maximum size of longer image side transform.
      num_negatives: Integer, number of negative images for a query image in a
        training tuple.
      num_queries: Integer, number of query images to be processed in one epoch.
      pool_size: Integer, size of the negative image pool, from where the
        hard-negative images are re-mined.
      loader: Callable, a function to load an image given its path.
      ims_root: String, image root directory.

    Raises:
      ValueError: If mode is not either 'train' or 'val'.
    """

    if mode not in ['train', 'val']:
      raise ValueError(
              "`mode` argument should be either 'train' or 'val', passed as a "
              "String.")

    # Loading db.
    db_filename = os.path.join(data_root, '{}.pkl'.format(name))
    with tf.io.gfile.GFile(db_filename, 'rb') as f:
      db = pickle.load(f)[mode]

    # Initializing tuples dataset.
    self._ims_root = data_root if ims_root is None else ims_root
    self._name = name
    self._mode = mode
    self._imsize = imsize
    self._clusters = db['cluster']
    self._query_pool = db['qidxs']
    self._positive_pool = db['pidxs']

    if not hasattr(self, 'images'):
      self.images = db['ids']

    # Size of training subset for an epoch.
    self._num_negatives = num_negatives
    self._num_queries = min(num_queries, len(self._query_pool))
    self._pool_size = min(pool_size, len(self.images))
    self._qidxs = None
    self._pidxs = None
    self._nidxs = None

    self._loader = loader
    self._print_freq = 10
    # Indexer for the iterator.
    self._n = 0

  def __iter__(self):
    """Function for making TupleDataset an iterator.

    Returns:
      iter: The iterator object itself (TupleDataset).
    """
    return self

  def __next__(self):
    """Function for making TupleDataset an iterator.

    Returns:
      next: The next item in the sequence (next dataset image tuple).
    """
    if self._n < len(self._qidxs):
      result = self.__getitem__(self._n)
      self._n += 1
      return result
    else:
      raise StopIteration

  def _img_names_to_full_path(self, image_list):
    """Converts list of image names to the list of full paths to the images.

    Args:
      image_list: Image names, either a list or a single image path.

    Returns:
      image_full_paths: List of full paths to the images.
    """
    if not isinstance(image_list, list):
      return os.path.join(self._ims_root, image_list)
    return [os.path.join(self._ims_root, img_name) for img_name in image_list]

  def __getitem__(self, index):
    """Called to load an image tuple at the given `index`.

    Args:
      index: Integer, index.

    Returns:
      output: Tuple [q,p,n1,...,nN, target], loaded 'train'/'val' tuple at
        index of qidxs. `q` is the query image tensor, `p` is the
        corresponding positive image tensor, `n1`,...,`nN` are the negatives
        associated with the query. `target` is a tensor (with the shape [2+N])
        of integer labels corresponding to the tuple list: query (-1),
        positive (1), negative (0).

    Raises:
      ValueError: Raised if the query indexes list `qidxs` is empty.
    """
    if self.__len__() == 0:
      raise ValueError(
              "List `qidxs` is empty. Run `dataset.create_epoch_tuples(net)` "
              "method to create subset for `train`/`val`.")

    output = []
    # Query image.
    output.append(self._loader(
            self._img_names_to_full_path(self.images[self._qidxs[index]]),
            self._imsize))
    # Positive image.
    output.append(self._loader(
            self._img_names_to_full_path(self.images[self._pidxs[index]]),
            self._imsize))
    # Negative images.
    for nidx in self._nidxs[index]:
      output.append(self._loader(
              self._img_names_to_full_path(self.images[nidx]),
              self._imsize))
    # Labels for the query (-1), positive (1), negative (0) images in the tuple.
    target = tf.convert_to_tensor([-1, 1] + [0] * self._num_negatives)
    output.append(target)

    return tuple(output)

  def __len__(self):
    """Called to implement the built-in function len().

    Returns:
      len: Integer, number of query images.
    """
    if self._qidxs is None:
      return 0
    return len(self._qidxs)

  def __repr__(self):
    """Metadata for the TupleDataset.

    Returns:
      meta: String, containing TupleDataset meta.
    """
    fmt_str = self.__class__.__name__ + '\n'
    fmt_str += '\tName and mode: {} {}\n'.format(self._name, self._mode)
    fmt_str += '\tNumber of images: {}\n'.format(len(self.images))
    fmt_str += '\tNumber of training tuples: {}\n'.format(len(self._query_pool))
    fmt_str += '\tNumber of negatives per tuple: {}\n'.format(
            self._num_negatives)
    fmt_str += '\tNumber of tuples processed in an epoch: {}\n'.format(
            self._num_queries)
    fmt_str += '\tPool size for negative remaining: {}\n'.format(self._pool_size)
    return fmt_str

  def create_epoch_tuples(self, net):
    """Creates epoch tuples with the hard-negative re-mining.

    Negative examples are selected from clusters different than the cluster
    of the query image, as the clusters are ideally non-overlapping. For
    every query image we choose  hard-negatives, that is, non-matching images
    with the most similar descriptor. Hard-negatives depend on the current
    CNN parameters. K-nearest neighbors from all non-matching images are
    selected. Query images are selected randomly. Positives examples are
    fixed for the related query image during the whole training process.

    Args:
      net: Model, network to be used for negative re-mining.

    Raises:
      ValueError: If the pool_size is smaller than the number of negative
        images per tuple.

    Returns:
      avg_l2: Float, average negative L2-distance.
    """
    self._n = 0

    if self._num_negatives < self._pool_size:
      raise ValueError("Unable to create epoch tuples. Negative pool_size "
                       "should be larger than the number of negative images "
                       "per tuple.")

    global_features_utils.debug_and_log(
            '>> Creating tuples for an epoch of {}-{}...'.format(self._name,
                                                                 self._mode),
            True)
    global_features_utils.debug_and_log(">> Used network: ", True)
    global_features_utils.debug_and_log(net.meta_repr(), True)

    ## Selecting queries.
    # Draw `num_queries` random queries for the tuples.
    idx_list = np.arange(len(self._query_pool))
    np.random.shuffle(idx_list)
    idxs2query_pool = idx_list[:self._num_queries]
    self._qidxs = [self._query_pool[i] for i in idxs2query_pool]

    ## Selecting positive pairs.
    # Positives examples are fixed for each query during the whole training
    # process.
    self._pidxs = [self._positive_pool[i] for i in idxs2query_pool]

    ## Selecting negative pairs.
    # If `num_negatives` = 0 create dummy nidxs.
    # Useful when only positives used for training.
    if self._num_negatives == 0:
      self._nidxs = [[] for _ in range(len(self._qidxs))]
      return 0

    # Draw pool_size random images for pool of negatives images.
    neg_idx_list = np.arange(len(self.images))
    np.random.shuffle(neg_idx_list)
    neg_images_idxs = neg_idx_list[:self._pool_size]

    global_features_utils.debug_and_log(
            '>> Extracting descriptors for query images...', debug=True)

    img_list = self._img_names_to_full_path([self.images[i] for i in
                                             self._qidxs])
    qvecs = global_model.extract_global_descriptors_from_list(
            net,
            images=img_list,
            image_size=self._imsize,
            print_freq=self._print_freq)

    global_features_utils.debug_and_log(
            '>> Extracting descriptors for negative pool...', debug=True)

    poolvecs = global_model.extract_global_descriptors_from_list(
            net,
            images=self._img_names_to_full_path([self.images[i] for i in
                                                 neg_images_idxs]),
            image_size=self._imsize,
            print_freq=self._print_freq)

    global_features_utils.debug_and_log('>> Searching for hard negatives...',
                                        debug=True)

    # Compute dot product scores and ranks.
    scores = tf.linalg.matmul(poolvecs, qvecs, transpose_a=True)
    ranks = tf.argsort(scores, axis=0, direction='DESCENDING')

    sum_ndist = 0.
    n_ndist = 0.

    # Selection of negative examples.
    self._nidxs = []

    for q, qidx in enumerate(self._qidxs):
      # We are not using the query cluster, those images are potentially
      # positive.
      qcluster = self._clusters[qidx]
      clusters = [qcluster]
      nidxs = []
      rank = 0

      while len(nidxs) < self._num_negatives:
        if rank >= tf.shape(ranks)[0]:
          raise ValueError("Unable to create epoch tuples. Number of required "
                           "negative images is larger than the number of "
                           "clusters in the dataset.")
        potential = neg_images_idxs[ranks[rank, q]]
        # Take at most one image from the same cluster.
        if not self._clusters[potential] in clusters:
          nidxs.append(potential)
          clusters.append(self._clusters[potential])
          dist = tf.norm(qvecs[:, q] - poolvecs[:, ranks[rank, q]],
                         axis=0).numpy()
          sum_ndist += dist
          n_ndist += 1
        rank += 1

      self._nidxs.append(nidxs)

    global_features_utils.debug_and_log(
            '>> Average negative l2-distance: {:.2f}'.format(
                    sum_ndist / n_ndist))

    # Return average negative L2-distance.
    return sum_ndist / n_ndist
