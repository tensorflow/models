"""StarGAN Estimator data provider."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os
import numpy as np
import data_provider
from google3.pyglib import resources


provide_data = data_provider.provide_data


def provide_celeba_test_set():
  """Provide one example of every class, and labels.

  Returns:
    An `np.array` of shape (num_domains, H, W, C) representing the images.
      Values are in [-1, 1].
    An `np.array` of shape (num_domains, num_domains) representing the labels.

  Raises:
    ValueError: If test data is inconsistent or malformed.
  """
  base_dir = 'google3/third_party/tensorflow_models/gan/stargan_estimator/data'
  images_fn = os.path.join(base_dir, 'celeba_test_split_images.npy')
  with resources.GetResourceAsFile(images_fn) as f:
    images_np = np.load(f)
  labels_fn = os.path.join(base_dir, 'celeba_test_split_labels.npy')
  with resources.GetResourceAsFile(labels_fn) as f:
    labels_np = np.load(f)
  if images_np.shape[0] != labels_np.shape[0]:
    raise ValueError('Test data is malformed.')

  return images_np, labels_np
