# Copyright 2018 The TensorFlow Authors. All Rights Reserved.
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
"""Script to compute official BLEU score.

Source:
https://github.com/tensorflow/tensor2tensor/blob/master/tensor2tensor/utils/bleu_hook.py
"""

import collections
import math
import re
import sys
import unicodedata

import numpy as np
import tensorflow as tf


class UnicodeRegex(object):
  """Ad-hoc hack to recognize all punctuation and symbols."""

  def __init__(self):
    punctuation = self.property_chars("P")
    self.nondigit_punct_re = re.compile(r"([^\d])([" + punctuation + r"])")
    self.punct_nondigit_re = re.compile(r"([" + punctuation + r"])([^\d])")
    self.symbol_re = re.compile("([" + self.property_chars("S") + "])")

  def property_chars(self, prefix):
    return "".join(
        chr(x)
        for x in range(sys.maxunicode)
        if unicodedata.category(chr(x)).startswith(prefix))


uregex = UnicodeRegex()


def bleu_tokenize(string):
  r"""Tokenize a string following the official BLEU implementation.

  See https://github.com/moses-smt/mosesdecoder/'
           'blob/master/scripts/generic/mteval-v14.pl#L954-L983
  In our case, the input string is expected to be just one line
  and no HTML entities de-escaping is needed.
  So we just tokenize on punctuation and symbols,
  except when a punctuation is preceded and followed by a digit
  (e.g. a comma/dot as a thousand/decimal separator).

  Note that a numer (e.g. a year) followed by a dot at the end of sentence
  is NOT tokenized,
  i.e. the dot stays with the number because `s/(\p{P})(\P{N})/ $1 $2/g`
  does not match this case (unless we add a space after each sentence).
  However, this error is already in the original mteval-v14.pl
  and we want to be consistent with it.

  Args:
    string: the input string

  Returns:
    a list of tokens
  """
  string = uregex.nondigit_punct_re.sub(r"\1 \2 ", string)
  string = uregex.punct_nondigit_re.sub(r" \1 \2", string)
  string = uregex.symbol_re.sub(r" \1 ", string)
  return string.split()


def bleu_wrapper(ref_filename, hyp_filename, case_sensitive=False):
  """Compute BLEU for two files (reference and hypothesis translation)."""
  ref_lines = tf.io.gfile.GFile(ref_filename).read().strip().splitlines()
  hyp_lines = tf.io.gfile.GFile(hyp_filename).read().strip().splitlines()
  return bleu_on_list(ref_lines, hyp_lines, case_sensitive)


def _get_ngrams_with_counter(segment, max_order):
  """Extracts all n-grams up to a given maximum order from an input segment.

  Args:
    segment: text segment from which n-grams will be extracted.
    max_order: maximum length in tokens of the n-grams returned by this
        methods.

  Returns:
    The Counter containing all n-grams upto max_order in segment
    with a count of how many times each n-gram occurred.
  """
  ngram_counts = collections.Counter()
  for order in range(1, max_order + 1):
    for i in range(0, len(segment) - order + 1):
      ngram = tuple(segment[i:i + order])
      ngram_counts[ngram] += 1
  return ngram_counts


def compute_bleu(reference_corpus, translation_corpus, max_order=4,
                 use_bp=True):
  """Computes BLEU score of translated segments against one or more references.

  Args:
    reference_corpus: list of references for each translation. Each
        reference should be tokenized into a list of tokens.
    translation_corpus: list of translations to score. Each translation
        should be tokenized into a list of tokens.
    max_order: Maximum n-gram order to use when computing BLEU score.
    use_bp: boolean, whether to apply brevity penalty.

  Returns:
    BLEU score.
  """
  reference_length = 0
  translation_length = 0
  bp = 1.0
  geo_mean = 0

  matches_by_order = [0] * max_order
  possible_matches_by_order = [0] * max_order
  precisions = []

  for (references, translations) in zip(reference_corpus, translation_corpus):
    reference_length += len(references)
    translation_length += len(translations)
    ref_ngram_counts = _get_ngrams_with_counter(references, max_order)
    translation_ngram_counts = _get_ngrams_with_counter(translations, max_order)

    overlap = dict((ngram,
                    min(count, translation_ngram_counts[ngram]))
                   for ngram, count in ref_ngram_counts.items())

    for ngram in overlap:
      matches_by_order[len(ngram) - 1] += overlap[ngram]
    for ngram in translation_ngram_counts:
      possible_matches_by_order[len(ngram) - 1] += translation_ngram_counts[
          ngram]

  precisions = [0] * max_order
  smooth = 1.0

  for i in range(0, max_order):
    if possible_matches_by_order[i] > 0:
      precisions[i] = float(matches_by_order[i]) / possible_matches_by_order[i]
      if matches_by_order[i] > 0:
        precisions[i] = float(matches_by_order[i]) / possible_matches_by_order[
            i]
      else:
        smooth *= 2
        precisions[i] = 1.0 / (smooth * possible_matches_by_order[i])
    else:
      precisions[i] = 0.0

  if max(precisions) > 0:
    p_log_sum = sum(math.log(p) for p in precisions if p)
    geo_mean = math.exp(p_log_sum / max_order)

  if use_bp:
    ratio = translation_length / reference_length
    bp = math.exp(1 - 1. / ratio) if ratio < 1.0 else 1.0
  bleu = geo_mean * bp
  return np.float32(bleu)


def bleu_on_list(ref_lines, hyp_lines, case_sensitive=False):
  """Compute BLEU for two list of strings (reference and hypothesis)."""
  if len(ref_lines) != len(hyp_lines):
    raise ValueError(
        "Reference and translation files have different number of "
        "lines (%d VS %d). If training only a few steps (100-200), the "
        "translation may be empty." % (len(ref_lines), len(hyp_lines)))
  if not case_sensitive:
    ref_lines = [x.lower() for x in ref_lines]
    hyp_lines = [x.lower() for x in hyp_lines]
  ref_tokens = [bleu_tokenize(x) for x in ref_lines]
  hyp_tokens = [bleu_tokenize(x) for x in hyp_lines]
  return compute_bleu(ref_tokens, hyp_tokens) * 100
