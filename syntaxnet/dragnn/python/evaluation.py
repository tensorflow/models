# Copyright 2017 Google Inc. All Rights Reserved.
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
"""Parser evaluation utils."""

from __future__ import division

import tensorflow as tf

from syntaxnet import sentence_pb2
from syntaxnet.util import check


def calculate_parse_metrics(gold_corpus, annotated_corpus):
  """Calculate POS/UAS/LAS accuracy based on gold and annotated sentences."""
  check.Eq(len(gold_corpus), len(annotated_corpus), 'Corpora are not aligned')
  num_tokens = 0
  num_correct_pos = 0
  num_correct_uas = 0
  num_correct_las = 0
  for gold_str, annotated_str in zip(gold_corpus, annotated_corpus):
    gold = sentence_pb2.Sentence()
    annotated = sentence_pb2.Sentence()
    gold.ParseFromString(gold_str)
    annotated.ParseFromString(annotated_str)
    check.Eq(gold.text, annotated.text, 'Text is not aligned')
    check.Eq(len(gold.token), len(annotated.token), 'Tokens are not aligned')
    tokens = zip(gold.token, annotated.token)
    num_tokens += len(tokens)
    num_correct_pos += sum(1 for x, y in tokens if x.tag == y.tag)
    num_correct_uas += sum(1 for x, y in tokens if x.head == y.head)
    num_correct_las += sum(1 for x, y in tokens
                           if x.head == y.head and x.label == y.label)

  tf.logging.info('Total num documents: %d', len(annotated_corpus))
  tf.logging.info('Total num tokens: %d', num_tokens)
  pos = num_correct_pos * 100.0 / num_tokens
  uas = num_correct_uas * 100.0 / num_tokens
  las = num_correct_las * 100.0 / num_tokens
  tf.logging.info('POS: %.2f%%', pos)
  tf.logging.info('UAS: %.2f%%', uas)
  tf.logging.info('LAS: %.2f%%', las)
  return pos, uas, las


def parser_summaries(gold_corpus, annotated_corpus):
  """Computes parser evaluation summaries for gold and annotated sentences."""
  pos, uas, las = calculate_parse_metrics(gold_corpus, annotated_corpus)
  return {'POS': pos, 'LAS': las, 'UAS': uas, 'eval_metric': las}


def calculate_segmentation_metrics(gold_corpus, annotated_corpus):
  """Calculate precision/recall/f1 based on gold and annotated sentences."""
  check.Eq(len(gold_corpus), len(annotated_corpus), 'Corpora are not aligned')
  num_gold_tokens = 0
  num_test_tokens = 0
  num_correct_tokens = 0
  def token_span(token):
    check.Ge(token.end, token.start)
    return (token.start, token.end)

  def ratio(numerator, denominator):
    check.Ge(numerator, 0)
    check.Ge(denominator, 0)
    if denominator > 0:
      return numerator / denominator
    elif numerator == 0:
      return 0.0  # map 0/0 to 0
    else:
      return float('inf')  # map x/0 to inf

  for gold_str, annotated_str in zip(gold_corpus, annotated_corpus):
    gold = sentence_pb2.Sentence()
    annotated = sentence_pb2.Sentence()
    gold.ParseFromString(gold_str)
    annotated.ParseFromString(annotated_str)
    check.Eq(gold.text, annotated.text, 'Text is not aligned')
    gold_spans = set()
    test_spans = set()
    for token in gold.token:
      check.NotIn(token_span(token), gold_spans, 'Duplicate token')
      gold_spans.add(token_span(token))
    for token in annotated.token:
      check.NotIn(token_span(token), test_spans, 'Duplicate token')
      test_spans.add(token_span(token))
    num_gold_tokens += len(gold_spans)
    num_test_tokens += len(test_spans)
    num_correct_tokens += len(gold_spans.intersection(test_spans))

  tf.logging.info('Total num documents: %d', len(annotated_corpus))
  tf.logging.info('Total gold tokens: %d', num_gold_tokens)
  tf.logging.info('Total test tokens: %d', num_test_tokens)
  precision = 100 * ratio(num_correct_tokens, num_test_tokens)
  recall = 100 * ratio(num_correct_tokens, num_gold_tokens)
  f1 = ratio(2 * precision * recall, precision + recall)
  tf.logging.info('Precision: %.2f%%', precision)
  tf.logging.info('Recall: %.2f%%', recall)
  tf.logging.info('F1: %.2f%%', f1)

  return round(precision, 2), round(recall, 2), round(f1, 2)


def segmentation_summaries(gold_corpus, annotated_corpus):
  """Computes segmentation eval summaries for gold and annotated sentences."""
  prec, rec, f1 = calculate_segmentation_metrics(gold_corpus, annotated_corpus)
  return {'precision': prec, 'recall': rec, 'f1': f1, 'eval_metric': f1}
