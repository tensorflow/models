# Copyright 2016 The TensorFlow Authors. All Rights Reserved.
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

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import numpy as np


def accuracy(logits, labels):
  """
  Return accuracy of the array of logits (or label predictions) wrt the labels
  :param logits: this can either be logits, probabilities, or a single label
  :param labels: the correct labels to match against
  :return: the accuracy as a float
  """
  assert len(logits) == len(labels)

  if len(np.shape(logits)) > 1:
    # Predicted labels are the argmax over axis 1
    predicted_labels = np.argmax(logits, axis=1)
  else:
    # Input was already labels
    assert len(np.shape(logits)) == 1
    predicted_labels = logits

  # Check against correct labels to compute correct guesses
  correct = np.sum(predicted_labels == labels.reshape(len(labels)))

  # Divide by number of labels to obtain accuracy
  accuracy = float(correct) / len(labels)

  # Return float value
  return accuracy


