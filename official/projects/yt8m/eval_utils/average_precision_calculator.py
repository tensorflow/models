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

"""Calculate or keep track of the interpolated average precision.

It provides an interface for calculating interpolated average precision for an
entire list or the top-n ranked items. For the definition of the
(non-)interpolated average precision:
http://trec.nist.gov/pubs/trec15/appendices/CE.MEASURES06.pdf

Example usages:
1) Use it as a static function call to directly calculate average precision for
a short ranked list in the memory.

```
import random

p = np.array([random.random() for _ in xrange(10)])
a = np.array([random.choice([0, 1]) for _ in xrange(10)])

ap = average_precision_calculator.AveragePrecisionCalculator.ap(p, a)
```

2) Use it as an object for long ranked list that cannot be stored in memory or
the case where partial predictions can be observed at a time (Tensorflow
predictions). In this case, we first call the function accumulate many times
to process parts of the ranked list. After processing all the parts, we call
peek_interpolated_ap_at_n.
```
p1 = np.array([random.random() for _ in xrange(5)])
a1 = np.array([random.choice([0, 1]) for _ in xrange(5)])
p2 = np.array([random.random() for _ in xrange(5)])
a2 = np.array([random.choice([0, 1]) for _ in xrange(5)])

# interpolated average precision at 10 using 1000 break points
calculator = average_precision_calculator.AveragePrecisionCalculator(10)
calculator.accumulate(p1, a1)
calculator.accumulate(p2, a2)
ap3 = calculator.peek_ap_at_n()
```
"""

import heapq
import numbers
import random

import numpy


class AveragePrecisionCalculator(object):
  """Calculate the average precision and average precision at n."""

  def __init__(self, top_n=None):
    """Construct an AveragePrecisionCalculator to calculate average precision.

    This class is used to calculate the average precision for a single label.

    Args:
      top_n: A positive Integer specifying the average precision at n, or None
        to use all provided data points.

    Raises:
      ValueError: An error occurred when the top_n is not a positive integer.
    """
    if not ((isinstance(top_n, int) and top_n >= 0) or top_n is None):
      raise ValueError("top_n must be a positive integer or None.")

    self._top_n = top_n  # average precision at n
    self._total_positives = 0  # total number of positives have seen
    self._heap = []  # max heap of (prediction, actual)

  @property
  def heap_size(self):
    """Gets the heap size maintained in the class."""
    return len(self._heap)

  @property
  def num_accumulated_positives(self):
    """Gets the number of positive samples that have been accumulated."""
    return self._total_positives

  def accumulate(self, predictions, actuals, num_positives=None):
    """Accumulate the predictions and their ground truth labels.

    After the function call, we may call peek_ap_at_n to actually calculate
    the average precision.
    Note predictions and actuals must have the same shape.

    Args:
      predictions: a list storing the prediction scores.
      actuals: a list storing the ground truth labels. Any value larger than 0
        will be treated as positives, otherwise as negatives. num_positives = If
        the 'predictions' and 'actuals' inputs aren't complete, then it's
        possible some true positives were missed in them. In that case, you can
        provide 'num_positives' in order to accurately track recall.
      num_positives: number of positive examples.

    Raises:
      ValueError: An error occurred when the format of the input is not the
      numpy 1-D array or the shape of predictions and actuals does not match.
    """
    if len(predictions) != len(actuals):
      raise ValueError("the shape of predictions and actuals does not match.")

    if num_positives is not None:
      if not isinstance(num_positives, numbers.Number) or num_positives < 0:
        raise ValueError(
            "'num_positives' was provided but it was a negative number.")

    if num_positives is not None:
      self._total_positives += num_positives
    else:
      self._total_positives += numpy.size(
          numpy.where(numpy.array(actuals) > 1e-5))
    topk = self._top_n
    heap = self._heap

    for i in range(numpy.size(predictions)):
      if topk is None or len(heap) < topk:
        heapq.heappush(heap, (predictions[i], actuals[i]))
      else:
        if predictions[i] > heap[0][0]:  # heap[0] is the smallest
          heapq.heappop(heap)
          heapq.heappush(heap, (predictions[i], actuals[i]))

  def clear(self):
    """Clear the accumulated predictions."""
    self._heap = []
    self._total_positives = 0

  def peek_ap_at_n(self):
    """Peek the non-interpolated average precision at n.

    Returns:
      The non-interpolated average precision at n (default 0).
      If n is larger than the length of the ranked list,
      the average precision will be returned.
    """
    if self.heap_size <= 0:
      return 0
    predlists = numpy.array(list(zip(*self._heap)))

    ap = self.ap_at_n(
        predlists[0],
        predlists[1],
        n=self._top_n,
        total_num_positives=self._total_positives)
    return ap

  @staticmethod
  def ap(predictions, actuals):
    """Calculate the non-interpolated average precision.

    Args:
      predictions: a numpy 1-D array storing the sparse prediction scores.
      actuals: a numpy 1-D array storing the ground truth labels. Any value
        larger than 0 will be treated as positives, otherwise as negatives.

    Returns:
      The non-interpolated average precision at n.
      If n is larger than the length of the ranked list,
      the average precision will be returned.

    Raises:
      ValueError: An error occurred when the format of the input is not the
      numpy 1-D array or the shape of predictions and actuals does not match.
    """
    return AveragePrecisionCalculator.ap_at_n(predictions, actuals, n=None)

  @staticmethod
  def ap_at_n(predictions, actuals, n=20, total_num_positives=None):
    """Calculate the non-interpolated average precision.

    Args:
      predictions: a numpy 1-D array storing the sparse prediction scores.
      actuals: a numpy 1-D array storing the ground truth labels. Any value
        larger than 0 will be treated as positives, otherwise as negatives.
      n: the top n items to be considered in ap@n.
      total_num_positives : (optionally) you can specify the number of total
        positive in the list. If specified, it will be used in calculation.

    Returns:
      The non-interpolated average precision at n.
      If n is larger than the length of the ranked list,
      the average precision will be returned.

    Raises:
      ValueError: An error occurred when
      1) the format of the input is not the numpy 1-D array;
      2) the shape of predictions and actuals does not match;
      3) the input n is not a positive integer.
    """
    if len(predictions) != len(actuals):
      raise ValueError("the shape of predictions and actuals does not match.")

    if n is not None:
      if not isinstance(n, int) or n <= 0:
        raise ValueError("n must be 'None' or a positive integer."
                         " It was '%s'." % n)

    ap = 0.0

    predictions = numpy.array(predictions)
    actuals = numpy.array(actuals)

    # add a shuffler to avoid overestimating the ap
    predictions, actuals = AveragePrecisionCalculator._shuffle(
        predictions, actuals)
    sortidx = sorted(
        range(len(predictions)), key=lambda k: predictions[k], reverse=True)

    if total_num_positives is None:
      numpos = numpy.size(numpy.where(actuals > 0))
    else:
      numpos = total_num_positives

    if numpos == 0:
      return 0

    if n is not None:
      numpos = min(numpos, n)
    delta_recall = 1.0 / numpos
    poscount = 0.0

    # calculate the ap
    r = len(sortidx)
    if n is not None:
      r = min(r, n)
    for i in range(r):
      if actuals[sortidx[i]] > 0:
        poscount += 1
        ap += poscount / (i + 1) * delta_recall
    return ap

  @staticmethod
  def _shuffle(predictions, actuals):
    random.seed(0)
    suffidx = random.sample(range(len(predictions)), len(predictions))
    predictions = predictions[suffidx]
    actuals = actuals[suffidx]
    return predictions, actuals

  @staticmethod
  def _zero_one_normalize(predictions, epsilon=1e-7):
    """Normalize the predictions to the range between 0.0 and 1.0.

    For some predictions like SVM predictions, we need to normalize them before
    calculate the interpolated average precision. The normalization will not
    change the rank in the original list and thus won't change the average
    precision.

    Args:
      predictions: a numpy 1-D array storing the sparse prediction scores.
      epsilon: a small constant to avoid denominator being zero.

    Returns:
      The normalized prediction.
    """
    denominator = numpy.max(predictions) - numpy.min(predictions)
    ret = (predictions - numpy.min(predictions)) / max(denominator, epsilon)
    return ret
