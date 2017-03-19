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

"""Some simple tools for error counting.

"""
import collections

# Named tuple Error counts describes the counts needed to accumulate errors
# over multiple trials:
#   false negatives (aka drops or deletions),
#   false positives: (aka adds or insertions),
#   truth_count: number of elements in ground truth = denominator for fn,
#   test_count:  number of elements in test string = denominator for fp,
# Note that recall = 1 - fn/truth_count, precision = 1 - fp/test_count,
# accuracy = 1 - (fn + fp) / (truth_count + test_count).
ErrorCounts = collections.namedtuple('ErrorCounts', ['fn', 'fp', 'truth_count',
                                                     'test_count'])

# Named tuple for error rates, as a percentage. Accuracies are just 100-error.
ErrorRates = collections.namedtuple('ErrorRates',
                                    ['label_error', 'word_recall_error',
                                     'word_precision_error', 'sequence_error'])


def CountWordErrors(ocr_text, truth_text):
  """Counts the word drop and add errors as a bag of words.

  Args:
    ocr_text:    OCR text string.
    truth_text:  Truth text string.

  Returns:
    ErrorCounts named tuple.
  """
  # Convert to lists of words.
  return CountErrors(ocr_text.split(), truth_text.split())


def CountErrors(ocr_text, truth_text):
  """Counts the drops and adds between 2 bags of iterables.

  Simple bag of objects count returns the number of dropped and added
  elements, regardless of order, from anything that is iterable, eg
  a pair of strings gives character errors, and a pair of word lists give
  word errors.
  Args:
    ocr_text:    OCR text iterable (eg string for chars, word list for words).
    truth_text:  Truth text iterable.

  Returns:
    ErrorCounts named tuple.
  """
  counts = collections.Counter(truth_text)
  counts.subtract(ocr_text)
  drops = sum(c for c in counts.values() if c > 0)
  adds = sum(-c for c in counts.values() if c < 0)
  return ErrorCounts(drops, adds, len(truth_text), len(ocr_text))


def AddErrors(counts1, counts2):
  """Adds the counts and returns a new sum tuple.

  Args:
    counts1: ErrorCounts named tuples to sum.
    counts2: ErrorCounts named tuples to sum.
  Returns:
    Sum of counts1, counts2.
  """
  return ErrorCounts(counts1.fn + counts2.fn, counts1.fp + counts2.fp,
                     counts1.truth_count + counts2.truth_count,
                     counts1.test_count + counts2.test_count)


def ComputeErrorRates(label_counts, word_counts, seq_errors, num_seqs):
  """Returns an ErrorRates corresponding to the given counts.

  Args:
    label_counts: ErrorCounts for the character labels
    word_counts:  ErrorCounts for the words
    seq_errors:   Number of sequence errors
    num_seqs:     Total sequences
  Returns:
    ErrorRates corresponding to the given counts.
  """
  label_errors = label_counts.fn + label_counts.fp
  num_labels = label_counts.truth_count + label_counts.test_count
  return ErrorRates(
      ComputeErrorRate(label_errors, num_labels),
      ComputeErrorRate(word_counts.fn, word_counts.truth_count),
      ComputeErrorRate(word_counts.fp, word_counts.test_count),
      ComputeErrorRate(seq_errors, num_seqs))


def ComputeErrorRate(error_count, truth_count):
  """Returns a sanitized percent error rate from the raw counts.

  Prevents div by 0 and clips return to 100%.
  Args:
    error_count: Number of errors.
    truth_count: Number to divide by.

  Returns:
    100.0 * error_count / truth_count clipped to 100.
  """
  if truth_count == 0:
    truth_count = 1
    error_count = 1
  elif error_count > truth_count:
    error_count = truth_count
  return error_count * 100.0 / truth_count
