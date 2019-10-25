# Copyright 2018 Google Inc. All Rights Reserved.
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

"""Library which creates datasets."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from datasets import imagenet_input
from datasets import tiny_imagenet_input


def get_dataset(dataset_name, split, batch_size, image_size, is_training):
  """Returns dataset.

  Args:
    dataset_name: name of the dataset, "imagenet" or "tiny_imagenet".
    split: name of the split, "train" or "validation".
    batch_size: size of the minibatch.
    image_size: size of the one side of the image. Output images will be
      resized to square shape image_size*image_size.
    is_training: if True then training preprocessing is done, otherwise eval
      preprocessing is done.

  Raises:
    ValueError: if dataset_name is invalid.

  Returns:
    dataset: instance of tf.data.Dataset with the dataset.
    num_examples: number of examples in given split of the dataset.
    num_classes: number of classes in the dataset.
    bounds: tuple with bounds of image values. All returned image pixels
      are between bounds[0] and bounds[1].
  """
  if dataset_name == 'tiny_imagenet':
    dataset = tiny_imagenet_input.tiny_imagenet_input(
        split, batch_size, image_size, is_training)
    num_examples = tiny_imagenet_input.num_examples_per_epoch(split)
    num_classes = 200
    bounds = (-1, 1)
  elif dataset_name == 'imagenet':
    dataset = imagenet_input.imagenet_input(
        split, batch_size, image_size, is_training)
    num_examples = imagenet_input.num_examples_per_epoch(split)
    num_classes = 1001
    bounds = (-1, 1)
  else:
    raise ValueError('Invalid dataset %s' % dataset_name)
  return dataset, num_examples, num_classes, bounds
