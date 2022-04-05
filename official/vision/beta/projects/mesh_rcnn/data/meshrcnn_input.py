# Copyright 2022 The TensorFlow Authors. All Rights Reserved.
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

"""Data parser and processing for Mesh R-CNN."""

# Import libraries

import tensorflow as tf

from official.vision.beta.dataloaders import parser
from official.vision.beta.dataloaders import utils
from official.vision.beta.ops import anchor
from official.vision.beta.ops import box_ops
from official.vision.beta.ops import preprocess_ops

class Parser(parser.Parser):
  """Parser to parse an image and its annotations into a dictionary of tensors."""

  def __init__(self,
               aug_rand_hflip=False,
               dtype='float32'):

    """Initializes parameters for parsing annotations in the dataset."""

    # Data Augmentation: horizontal flip
    self.aug_rand_hflip = aug_rand_hflip

    # Image output dtype.
    self._dtype = dtype

  def _parse_train_data(self, data):
    """Parses data for training."""
    pass

  def _parse_eval_data(self, data):
    """Parses data for evaluation."""
    pass




