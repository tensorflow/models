# Copyright 2018 The TensorFlow Authors All Rights Reserved.
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

"""Utilities for working with Seq2Label datasets and models.

This library provides utilities for parsing and generating Seq2Label protos.
"""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import numpy as np

from protos import seq2label_pb2


def get_all_label_values(dataset_info):
  """Retrieves possible values for modeled labels from a `Seq2LabelDatasetInfo`.

  Args:
    dataset_info: a `Seq2LabelDatasetInfo` message.

  Returns:
    A dictionary mapping each label name to a tuple of its permissible values.
  """
  return {
      label_info.name: tuple(label_info.values)
      for label_info in dataset_info.labels
  }


def construct_seq2label_model_info(hparams, model_type, targets, metadata_path,
                                   batch_size, num_filters,
                                   training_noise_rate):
  """Constructs a Seq2LabelModelInfo proto with the given properties.

  Args:
    hparams: initialized tf.contrib.training.Hparams object.
    model_type: string; descriptive tag indicating type of model, ie. "conv".
    targets: list of names of the targets the model is trained to predict.
    metadata_path: string; full path to Seq2LabelDatasetInfo text proto used
      to initialize the model.
    batch_size: int; number of reads per mini-batch.
    num_filters: int; number of filters for convolutional model.
    training_noise_rate: float; rate [0.0, 1.0] of base-flipping noise injected
      into input read sequenced at training time.

  Returns:
    The Seq2LabelModelInfo proto with the hparams, model_type, targets,
    num_filters, batch_size, metadata_path, and training_noise_rate fields
    set to the given values.
  """
  return seq2label_pb2.Seq2LabelModelInfo(
      hparams_string=hparams.to_json(),
      model_type=model_type,
      targets=sorted(targets),
      num_filters=num_filters,
      batch_size=batch_size,
      metadata_path=metadata_path,
      training_noise_rate=training_noise_rate)


def add_read_noise(read, base_flip_probability=0.01):
  """Adds base-flipping noise to the given read sequence.

  Args:
    read: string; the read sequence to which to add noise.
    base_flip_probability: float; probability of a base flip at each position.

  Returns:
    The given read with base-flipping noise added at the provided
    base_flip_probability rate.
  """
  base_flips = np.random.binomial(1, base_flip_probability, len(read))
  if sum(base_flips) == 0:
    return read

  read = np.array(list(read))
  possible_mutations = np.char.replace(['ACTG'] * sum(base_flips),
                                       read[base_flips == 1], '')
  mutations = map(np.random.choice, map(list, possible_mutations))
  read[base_flips == 1] = mutations
  return ''.join(read)
