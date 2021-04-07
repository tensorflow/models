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
"""Tuple dataset based on the Radenovic et al. ECCV16: CNN image retrieval
learns from BoW.

For more information refer to https://arxiv.org/abs/1604.02426.
"""

import os
import pickle

from absl import flags
import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf

from delf.python.datasets import generic_dataset
from delf.python.datasets import utils as image_loading_utils
from delf.python.training import global_features_utilst
from delf.python.training import global_features_utils

FLAGS = flags.FLAGS


class TuplesDataset():
  """Data loader that loads training and validation tuples of Radenovic et
  al. ECCV16: CNN image retrieval learns from BoW.

  For more information refer to https://arxiv.org/abs/1604.02426.
  """

  def __init__(self, name, mode, data_root, imsize=None, nnum=5, qsize=2000,
               poolsize=20000, loader=image_loading_utils.default_loader,
               ims_root=None):
    """TuplesDataset object initialization.
    
    Args:
      name: String, dataset name. I.e. 'retrieval-sfm-120k'.
      mode: 'train' or 'val' for training and validation parts of dataset.
      data_root: Path to the root directory of the dataset.
      imsize: Integer, defines the maximum size of longer image side transform.
      nnum: Integer, number of negative images for a query image in a
        training tuple.
      qsize: Integer, number of query images to be processed in one epoch.
      poolsize: Integer, size of the negative image pool, from where the
        hard-negative images are re-mined.
      loader: Callable, a function to load an image given its path.

    Raises:
      ValueError: If mode is not either 'train' or 'val'.
    """

    if mode not in ['train', 'val']:
      raise ValueError(
        "`mode` argument should be either 'train' or 'val', passed as a "
        "String.")

    # Loading db.
    db_fn = os.path.join(data_root, '{}.pkl'.format(name))
    with tf.io.gfile.GFile(db_fn, 'rb') as f:
      db = pickle.load(f)[mode]

    # Initializing tuples dataset.
    self._ims_root = db_root if not ims_root else ims_root
    self._name = name
    self._mode = mode
    self._imsize = imsize
    self._clusters = db['cluster']
    self._qpool = db['qidxs']
    self._ppool = db['pidxs']

    if not hasattr(self, 'images'):
      self.images = db['ids']

    # Size of training subset for an epoch.
    self._nnum = nnum
    self._qsize = min(qsize, len(self._qpool))
    self._poolsize = min(poolsize, len(self.images))
    self._qidxs = None
    self._pidxs = None
    self._nidxs = None

    self._loader = loader
    self._print_freq = 10
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
    """Converrts list of image names to the list of full paths to the images.

    Args:
      image_list: List of image names.

    Returns:
      image_full_paths: List of full paths to the images.
    """
    if not isinstance(image_list, list):
      return os.path.join(self._ims_root, image_list)
    return [os.path.join(self._ims_root, img_name) for img_name in image_list]

  def __getitem__(self, index):
    """Called to load an image at the given `index`.

    Args:
      index: Integer, index.

    Returns:
      output: Tuple [q,p,n1,...,nN, target], loaded 'train'/'val' tuple at 
        index of qidxs. `q` is the query image tensor, `p` is the 
        corresponding positive image tensor, `n1`,...,`nN` are the negatives 
        associated with the query. `target` is the tensor of labels 
        corresponding to the tuple list: query (-1), positive (1), negative (0).

    Raises:
      ValueError: Raised if the query indexes list `qidxs` is empty.
    """
    if self.__len__() == 0:
      raise ValueError(
        "List `qidxs` is empty. Run `dataset.create_epoch_tuples(net)` "
        "method to create subset for `train`/`val`.")

    output = []
    # Query images.
    output.append(self._loader(
      self._img_names_to_full_path(self.images[self._qidxs[index]]),
      self._imsize))
    # Positive images.
    output.append(self._loader(
      self._img_names_to_full_path(self.images[self._pidxs[index]]),
      self._imsize))
    # Negative images.
    for nidx in self._nidxs[index]:
      output.append(self._loader(
        self._img_names_to_full_path(self.images[nidx]),
        self._imsize))
    # Labels for the query (-1), positive (1), negative (0) images in the tuple.
    target = tf.convert_to_tensor([-1, 1] + [0] * self._nnum)
    output.append(target)

    return tuple(output)

  def __len__(self):
    """Called to implement the built-in function len().

    Returns:
      len: Integer, number of query images.
    """
    if not self._qidxs:
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
    fmt_str += '\tNumber of training tuples: {}\n'.format(len(self._qpool))
    fmt_str += '\tNumber of negatives per tuple: {}\n'.format(self._nnum)
    fmt_str += '\tNumber of tuples processed in an epoch: {}\n'.format(
      self._qsize)
    fmt_str += '\tPool size for negative remining: {}\n'.format(self._poolsize)
    return fmt_str

  def create_epoch_tuples(self, net):
    """Creates epoch tuples with the hard-negative re-mining.
    
    Negative examples are selected from clusters different than the cluster 
    of the query image, as the clusters are ideally non-overlaping. For 
    every query image we choose  hard-negatives, that is, non-matching images 
    with the most similar descriptor. Hard-negatives depend on the current 
    CNN parameters. K-nearest neighbors from all non-matching images are 
    selected. Query images are selected randomly. Positives examples are
    fixed for the related query image during the whole training process.

    Args:
      net: Model, network to be used for negative re-mining.

    Returns:
      avg_l2: Float, average negative L2-distance.
    """
    self._n = 0
    global_features_utils.debug_and_log(
      '>> Creating tuples for an epoch of {}-{}...'.format(self._name,
                                                           self._mode), True)
    global_features_utils.debug_and_log(">> Used network: ", True)
    global_features_utils.debug_and_log(net.meta_repr(), True)

    ## Selecting queries.
    # Draw `qsize` random queries for the tuples.
    idx_list = np.arange(len(self._qpool))
    np.random.shuffle(idx_list)
    idxs2qpool = idx_list[:self._qsize]
    self._qidxs = [self._qpool[i] for i in idxs2qpool]

    ## Selecting positive pairs.
    # Positives examples are fixed for each query during the whole training
    # process.
    self._pidxs = [self._ppool[i] for i in idxs2qpool]

    ## Selecting negative pairs.
    # If nnum = 0 create dummy nidxs.
    # Useful when only positives used for training.
    if self._nnum == 0:
      self._nidxs = [[] for _ in range(len(self._qidxs))]
      return 0

    # Draw poolsize random images for pool of negatives images.
    idx_list = np.arange(len(self.images))
    np.random.shuffle(idx_list)
    idxs2images = idx_list[:self._poolsize]

    global_features_utils.debug_and_log(
      '>> Extracting descriptors for query images...', debug=True)

    img_list = self._img_names_to_full_path([self.images[i] for i in
                                             self._qidxs])
    qvecs = extract_descriptors_from_image_paths(
      net,
      image_paths=img_list,
      imsize=self._imsize,
      print_freq=self._print_freq)

    global_features_utils.debug_and_log(
      '>> Extracting descriptors for negative pool...', debug=True)

    poolvecs = extract_descriptors_from_image_paths(
      net,
      image_paths=self._img_names_to_full_path([self.images[i] for i in
                                                idxs2images]),
      imsize=self._imsize,
      print_freq=self._print_freq)

    global_features_utils.debug_and_log('>> Searching for hard negatives...',
                                        debug=True)

    # Compute dot product scores and ranks.
    scores = tf.linalg.matmul(poolvecs, qvecs, transpose_a=True)
    ranks = tf.argsort(scores, axis=0, direction='DESCENDING')

    avg_ndist = 0.
    n_ndist = 0.

    # Selection of negative examples.
    self._nidxs = []
    eps = 1e-6

    for q, qidx in enumerate(self._qidxs):
      # We are not using the query cluster, those images are potentially
      # positive.
      qcluster = self._clusters[qidx]
      clusters = [qcluster]
      nidxs = []
      r = 0

      while len(nidxs) < self._nnum:
        potential = idxs2images[ranks[r, q]]
        # Take at most one image from the same cluster.
        if not self._clusters[potential] in clusters:
          nidxs.append(potential)
          clusters.append(self._clusters[potential])
          dist = tf.norm(qvecs[:, q] - poolvecs[:, ranks[r, q]] + eps,
                      axis=0).numpy()
          avg_ndist += dist
          n_ndist += 1
        r += 1

      self._nidxs.append(nidxs)

    global_features_utils.debug_and_log(
      '>> Average negative l2-distance: {:.2f}'.format(avg_ndist / n_ndist))

    # Save the obtained descriptors to a file.
    filename_dataset_descriptors = os.path.join(FLAGS.directory,
                                                "data_descriptors.pkl")
    with tf.io.gfile.GFile(filename_dataset_descriptors, 'wb') as desc_file:
      pickle.dump({"qvecs": qvecs, "poolvecs": poolvecs}, desc_file)

    # Return average negative L2-distance.
    return (avg_ndist / n_ndist)


def extract_descriptors_from_image_paths(net, image_paths, imsize, print_freq):
  """Extracts descriptors of the images in the `image_paths`.

  Args:
    net: Model to be used for the descriptor extraction.
    image_paths: List of the paths to the images.

  Returns:
    vecs: List of the extracted descriptors.
  """
  # Prepare the loader.
  data = generic_dataset.ImagesFromList(
    root='',
    image_paths=image_paths,
    imsize=imsize)

  def images_gen():
    return (inst for inst in data)

  loader = tf.data.Dataset.from_generator(images_gen,
                                          output_types=(tf.float32))
  loader = loader.batch(batch_size=1)

  # Extract vectors.
  vecs = tf.zeros((net.meta['outputdim'], 0))

  for i, input in enumerate(loader):
    o = net(input, training=False)
    o = tf.transpose(o, perm=[1, 0])
    vecs = tf.concat([vecs, o], 1)
    if (i + 1) % print_freq == 0 or (i + 1) == len(image_paths):
      global_features_utils.debug_and_log('\r>>>> {}/{} done...'.format(
        i + 1, len(image_paths)), debug_on_the_same_line=True)

  global_features_utils.debug_and_log("", debug_on_the_same_line=True)
  return vecs
