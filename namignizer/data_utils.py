# Copyright 2016 Google Inc. All Rights Reserved.
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
"""Utilities for parsing Kaggle baby names files."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import collections
import os

import numpy as np
import tensorflow as tf
import pandas as pd

# the default end of name rep will be zero
_EON = 0


def read_names(names_path):
    """read data from downloaded file. See SmallNames.txt for example format
    or go to https://www.kaggle.com/kaggle/us-baby-names for full lists

    Args:
        names_path: path to the csv file similar to the example type
    Returns:
        Dataset: a namedtuple of two elements: deduped names and their associated
            counts. The names contain only 26 chars and are all lower case
    """
    names_data = pd.read_csv(names_path)
    names_data.Name = names_data.Name.str.lower()

    name_data = names_data.groupby(by=["Name"])["Count"].sum()
    name_counts = np.array(name_data.tolist())
    names_deduped = np.array(name_data.index.tolist())

    Dataset = collections.namedtuple('Dataset', ['Name', 'Count'])
    return Dataset(names_deduped, name_counts)


def _letter_to_number(letter):
    """converts letters to numbers between 1 and 27"""
    # ord of lower case 'a' is 97
    return ord(letter) - 96


def namignizer_iterator(names, counts, batch_size, num_steps, epoch_size):
    """Takes a list of names and counts like those output from read_names, and
    makes an iterator yielding a batch_size by num_steps array of random names
    separated by an end of name token. The names are choosen randomly according
    to their counts. The batch may end mid-name

    Args:
        names: a set of lowercase names composed of 26 characters
        counts: a list of the frequency of those names
        batch_size: int
        num_steps: int
        epoch_size: number of batches to yield
    Yields:
        (x, y): a batch_size by num_steps array of ints representing letters, where
            x will be the input and y will be the target
    """
    name_distribution = counts / counts.sum()

    for i in range(epoch_size):
        data = np.zeros(batch_size * num_steps + 1)
        samples = np.random.choice(names, size=batch_size * num_steps // 2,
                                   replace=True, p=name_distribution)

        data_index = 0
        for sample in samples:
            if data_index >= batch_size * num_steps:
                break
            for letter in map(_letter_to_number, sample) + [_EON]:
                if data_index >= batch_size * num_steps:
                    break
                data[data_index] = letter
                data_index += 1

        x = data[:batch_size * num_steps].reshape((batch_size, num_steps))
        y = data[1:batch_size * num_steps + 1].reshape((batch_size, num_steps))

        yield (x, y)


def name_to_batch(name, batch_size, num_steps):
    """ Takes a single name and fills a batch with it

    Args:
        name: lowercase composed of 26 characters
        batch_size: int
        num_steps: int
    Returns:
        x, y: a batch_size by num_steps array of ints representing letters, where
            x will be the input and y will be the target. The array is filled up
            to the length of the string, the rest is filled with zeros
    """
    data = np.zeros(batch_size * num_steps + 1)

    data_index = 0
    for letter in map(_letter_to_number, name) + [_EON]:
        data[data_index] = letter
        data_index += 1

    x = data[:batch_size * num_steps].reshape((batch_size, num_steps))
    y = data[1:batch_size * num_steps + 1].reshape((batch_size, num_steps))

    return x, y
