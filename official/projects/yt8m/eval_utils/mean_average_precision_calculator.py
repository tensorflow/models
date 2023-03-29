# Copyright 2023 The TensorFlow Authors. All Rights Reserved.
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

"""Calculate the mean average precision.

It provides an interface for calculating mean average precision
for an entire list or the top-n ranked items.

Example usages:
We first call the function accumulate many times to process parts of the ranked
list. After processing all the parts, we call peek_map_at_n
to calculate the mean average precision.

```
import random

p = np.array([[random.random() for _ in xrange(50)] for _ in xrange(1000)])
a = np.array([[random.choice([0, 1]) for _ in xrange(50)]
     for _ in xrange(1000)])

# mean average precision for 50 classes.
calculator = mean_average_precision_calculator.MeanAveragePrecisionCalculator(
            num_class=50)
calculator.accumulate(p, a)
aps = calculator.peek_map_at_n()
```
"""

from official.projects.yt8m.eval_utils import average_precision_calculator


class MeanAveragePrecisionCalculator(object):
  """This class is to calculate mean average precision."""

  def __init__(self, num_class, filter_empty_classes=True, top_n=None):
    """Construct a calculator to calculate the (macro) average precision.

    Args:
      num_class: A positive Integer specifying the number of classes.
      filter_empty_classes: whether to filter classes without any positives.
      top_n: A positive Integer specifying the average precision at n, or None
        to use all provided data points.

    Raises:
      ValueError: An error occurred when num_class is not a positive integer;
      or the top_n_array is not a list of positive integers.
    """
    if not isinstance(num_class, int) or num_class <= 1:
      raise ValueError("num_class must be a positive integer.")

    self._ap_calculators = []  # member of AveragePrecisionCalculator
    self._num_class = num_class  # total number of classes
    self._filter_empty_classes = filter_empty_classes
    for _ in range(num_class):
      self._ap_calculators.append(
          average_precision_calculator.AveragePrecisionCalculator(top_n=top_n))

  def accumulate(self, predictions, actuals, num_positives=None):
    """Accumulate the predictions and their ground truth labels.

    Args:
      predictions: A list of lists storing the prediction scores. The outer
        dimension corresponds to classes.
      actuals: A list of lists storing the ground truth labels. The dimensions
        should correspond to the predictions input. Any value larger than 0 will
        be treated as positives, otherwise as negatives.
      num_positives: If provided, it is a list of numbers representing the
        number of true positives for each class. If not provided, the number of
        true positives will be inferred from the 'actuals' array.

    Raises:
      ValueError: An error occurred when the shape of predictions and actuals
      does not match.
    """
    if not num_positives:
      num_positives = [None for i in range(self._num_class)]

    calculators = self._ap_calculators
    for i in range(self._num_class):
      calculators[i].accumulate(predictions[i], actuals[i], num_positives[i])

  def clear(self):
    for calculator in self._ap_calculators:
      calculator.clear()

  def is_empty(self):
    return ([calculator.heap_size for calculator in self._ap_calculators
            ] == [0 for _ in range(self._num_class)])

  def peek_map_at_n(self):
    """Peek the non-interpolated mean average precision at n.

    Returns:
      An array of non-interpolated average precision at n (default 0) for each
      class.
    """
    aps = []
    for i in range(self._num_class):
      if (not self._filter_empty_classes or
          self._ap_calculators[i].num_accumulated_positives > 0):
        ap = self._ap_calculators[i].peek_ap_at_n()
        aps.append(ap)
    return aps
