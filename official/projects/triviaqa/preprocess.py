# Copyright 2025 The TensorFlow Authors. All Rights Reserved.
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

"""Utilities for preprocessing TriviaQA data."""
import bisect
import json
import operator
import os
import re
import string
from typing import Any, Dict, Generator, List, Optional, Set, Text, Tuple

from absl import logging
import apache_beam as beam
from apache_beam import metrics
import dataclasses
import nltk
import numpy as np
import tensorflow.io.gfile as gfile

import sentencepiece as spm
from official.projects.triviaqa import evaluation
from official.projects.triviaqa import sentencepiece_pb2


@dataclasses.dataclass
class Question(object):
  id: Text
  value: Text


@dataclasses.dataclass
class EvidenceInfo(object):
  id: Text
  source: Text
  title: Text


@dataclasses.dataclass
class Evidence(object):
  info: EvidenceInfo
  text: Text


@dataclasses.dataclass
class Answer(object):
  value: Text
  aliases: List[Text]
  normalized_aliases: List[Text]


@dataclasses.dataclass
class QuestionAnswer(object):
  question: Question
  evidence_info: List[EvidenceInfo]
  answer: Optional[Answer] = None


@dataclasses.dataclass
class QuestionAnswerEvidence(object):
  question: Question
  evidence: Evidence
  answer: Optional[Answer] = None


@dataclasses.dataclass
class Features(object):
  id: Text
  stride_index: int
  question_id: Text
  question: Text
  context: bytes
  token_ids: List[int]
  token_offsets: List[int]
  global_token_ids: List[int]
  segment_ids: List[int]


@dataclasses.dataclass
class Paragraph(object):
  sentences: List[sentencepiece_pb2.SentencePieceText]
  size: int


@dataclasses.dataclass
class AnswerSpan(object):
  begin: int  # inclusive
  end: int  # inclusive
  text: Text


def make_paragraph(
    sentence_tokenizer: nltk.tokenize.api.TokenizerI,
    processor: spm.SentencePieceProcessor,
    text: Text,
    paragraph_metric: Optional[metrics.Metrics.DelegatingDistribution] = None,
    sentence_metric: Optional[metrics.Metrics.DelegatingDistribution] = None
) -> Paragraph:
  """Tokenizes paragraphs."""
  paragraph_size = 0
  sentences = []
  for sentence in sentence_tokenizer.tokenize(text):
    sentencepiece_text = sentencepiece_pb2.SentencePieceText.FromString(
        processor.EncodeAsSerializedProto(sentence))
    paragraph_size += len(sentencepiece_text.pieces)
    sentences.append(sentencepiece_text)
    if sentence_metric:
      sentence_metric.update(len(sentencepiece_text.pieces))
  if paragraph_metric:
    paragraph_metric.update(paragraph_size)
  return Paragraph(sentences=sentences, size=paragraph_size)


def read_question_answers(json_path: Text) -> List[QuestionAnswer]:
  """Read question answers."""
  with gfile.GFile(json_path) as f:
    data = json.load(f)['Data']
  question_answers = []
  for datum in data:
    question = Question(id=datum['QuestionId'], value=datum['Question'])
    if 'Answer' in datum:
      answer = Answer(
          value=datum['Answer']['Value'],
          aliases=datum['Answer']['Aliases'],
          normalized_aliases=datum['Answer']['NormalizedAliases'])
    else:
      answer = None
    evidence_info = []
    for key in ['EntityPages', 'SearchResults']:
      for document in datum.get(key, []):
        evidence_info.append(
            EvidenceInfo(
                id=document['Filename'], title=document['Title'], source=key))
    question_answers.append(
        QuestionAnswer(
            question=question, evidence_info=evidence_info, answer=answer))
  return question_answers


def alias_answer(answer: Text, include=None):
  alias = answer.replace('_', ' ').lower()
  exclude = set(string.punctuation + ''.join(['‘', '’', '´', '`']))
  include = include or []
  alias = ''.join(c if c not in exclude or c in include else ' ' for c in alias)
  return ' '.join(alias.split()).strip()


def make_answer_set(answer: Answer) -> Set[Text]:
  """Apply less aggressive normalization to the answer aliases."""
  answers = []
  for alias in [answer.value] + answer.aliases:
    answers.append(alias_answer(alias))
    answers.append(alias_answer(alias, [',', '.']))
    answers.append(alias_answer(alias, ['-']))
    answers.append(alias_answer(alias, [',', '.', '-']))
    answers.append(alias_answer(alias, string.punctuation))
  return set(answers + answer.normalized_aliases)


def find_answer_spans(text: bytes, answer_set: Set[Text]) -> List[AnswerSpan]:
  """Find answer spans."""
  spans = []
  for answer in answer_set:
    answer_regex = re.compile(
        re.escape(answer).encode('utf-8').replace(b'\\ ', b'[ -]'),
        flags=re.IGNORECASE)
    for match in re.finditer(answer_regex, text):
      spans.append(
          AnswerSpan(
              begin=match.start(),
              end=match.end(),
              text=match.group(0).decode('utf-8')))
  return sorted(spans, key=operator.attrgetter('begin'))


def realign_answer_span(features: Features, answer_set: Optional[Set[Text]],
                        processor: spm.SentencePieceProcessor,
                        span: AnswerSpan) -> Optional[AnswerSpan]:
  """Align answer span to text with given tokens."""
  i = bisect.bisect_left(features.token_offsets, span.begin)
  if i == len(features.token_offsets) or span.begin < features.token_offsets[i]:
    i -= 1
  j = i + 1
  answer_end = span.begin + len(span.text.encode('utf-8'))
  while (j < len(features.token_offsets) and
         features.token_offsets[j] < answer_end):
    j += 1
  j -= 1
  sp_answer = (
      features.context[features.token_offsets[i]:features.token_offsets[j + 1]]
      if j + 1 < len(features.token_offsets) else
      features.context[features.token_offsets[i]:])
  if (processor.IdToPiece(features.token_ids[i]).startswith('▁') and
      features.token_offsets[i] > 0):
    sp_answer = sp_answer[1:]
  sp_answer = evaluation.normalize_answer(sp_answer.decode('utf-8'))
  if answer_set is not None and sp_answer not in answer_set:
    # No need to warn if the cause was breaking word boundaries.
    if len(sp_answer) and not len(sp_answer) > len(
        evaluation.normalize_answer(span.text)):
      logging.warning('%s: "%s" not in %s.', features.question_id, sp_answer,
                      answer_set)
    return None
  return AnswerSpan(begin=i, end=j, text=span.text)


def read_sentencepiece_model(path):
  with gfile.GFile(path, 'rb') as file:
    processor = spm.SentencePieceProcessor()
    processor.LoadFromSerializedProto(file.read())
  return processor


class ReadEvidence(beam.DoFn):
  """Function to read evidence."""

  def __init__(self, wikipedia_dir: Text, web_dir: Text):
    self._wikipedia_dir = wikipedia_dir
    self._web_dir = web_dir

  def process(
      self, question_answer: QuestionAnswer
  ) -> Generator[QuestionAnswerEvidence, None, None]:
    for info in question_answer.evidence_info:
      if info.source == 'EntityPages':
        evidence_path = os.path.join(self._wikipedia_dir, info.id)
      elif info.source == 'SearchResult':
        evidence_path = os.path.join(self._web_dir, info.id)
      else:
        raise ValueError(f'Unknown evidence source: {info.source}.')
      with gfile.GFile(evidence_path, 'rb') as f:
        text = f.read().decode('utf-8')
      metrics.Metrics.counter('_', 'documents').inc()
      yield QuestionAnswerEvidence(
          question=question_answer.question,
          evidence=Evidence(info=info, text=text),
          answer=question_answer.answer)


_CLS_PIECE = '<ans>'
_EOS_PIECE = '</s>'
_SEP_PIECE = '<sep_0>'
# _PARAGRAPH_SEP_PIECE = '<sep_1>'
_NULL_PIECE = '<empty>'
_QUESTION_PIECE = '<unused_34>'


class MakeFeatures(beam.DoFn):
  """Function to make features."""

  def __init__(self, sentencepiece_model_path: Text, max_num_tokens: int,
               max_num_global_tokens: int, stride: int):
    self._sentencepiece_model_path = sentencepiece_model_path
    self._max_num_tokens = max_num_tokens
    self._max_num_global_tokens = max_num_global_tokens
    self._stride = stride

  def setup(self):
    self._sentence_tokenizer = nltk.data.load('tokenizers/punkt/english.pickle')
    self._sentencepiece_processor = read_sentencepiece_model(
        self._sentencepiece_model_path)

  def _make_features(self, stride_index: int, paragraph_texts: List[Text],
                     paragraphs: List[Paragraph],
                     question_answer_evidence: QuestionAnswerEvidence,
                     ids: List[int],
                     paragraph_offset: int) -> Tuple[int, Features]:
    global_ids = (
        [self._sentencepiece_processor.PieceToId(_CLS_PIECE)] +
        [self._sentencepiece_processor.PieceToId(_QUESTION_PIECE)] * len(ids))
    segment_ids = [i + 1 for i in range(len(ids))]  # offset for CLS token
    token_ids, sentences = [], []
    offsets, offset, full_text = [-1] * len(ids), 0, True
    for i in range(paragraph_offset, len(paragraph_texts)):
      if i < len(paragraphs):
        paragraph = paragraphs[i]
      else:
        paragraphs.append(
            make_paragraph(
                self._sentence_tokenizer,
                self._sentencepiece_processor,
                paragraph_texts[i],
                paragraph_metric=metrics.Metrics.distribution(
                    '_', 'paragraphs'),
                sentence_metric=metrics.Metrics.distribution('_', 'sentences')))
        paragraph = paragraphs[-1]
      for sentence in paragraph.sentences:
        if (len(ids) + len(token_ids) + len(sentence.pieces) + 1 >=
            self._max_num_tokens or
            len(global_ids) >= self._max_num_global_tokens):
          full_text = False
          break
        for j, piece in enumerate(sentence.pieces):
          token_ids.append(piece.id)
          segment_ids.append(len(global_ids))
          offsets.append(offset + piece.begin)
          if j == 0 and sentences:
            offsets[-1] -= 1
        offset += len(sentence.text.encode('utf-8')) + 1
        global_ids.append(self._sentencepiece_processor.PieceToId(_EOS_PIECE))
        sentences.append(sentence.text)
      if not full_text:
        break
    context = ' '.join(sentences).encode('utf-8')
    token_ids.append(self._sentencepiece_processor.PieceToId(_NULL_PIECE))
    offsets.append(len(context))
    segment_ids.append(0)
    next_paragraph_index = len(paragraph_texts)
    if not full_text and self._stride > 0:
      shift = paragraphs[paragraph_offset].size
      next_paragraph_index = paragraph_offset + 1
      while (next_paragraph_index < len(paragraphs) and
             shift + paragraphs[next_paragraph_index].size <= self._stride):
        shift += paragraphs[next_paragraph_index].size
        next_paragraph_index += 1
    return next_paragraph_index, Features(
        id='{}--{}'.format(question_answer_evidence.question.id,
                           question_answer_evidence.evidence.info.id),
        stride_index=stride_index,
        question_id=question_answer_evidence.question.id,
        question=question_answer_evidence.question.value,
        context=context,
        token_ids=ids + token_ids,
        global_token_ids=global_ids,
        segment_ids=segment_ids,
        token_offsets=offsets)

  def process(
      self, question_answer_evidence: QuestionAnswerEvidence
  ) -> Generator[Features, None, None]:
    # Tokenize question which is shared among all examples.
    ids = (
        self._sentencepiece_processor.EncodeAsIds(
            question_answer_evidence.question.value) +
        [self._sentencepiece_processor.PieceToId(_SEP_PIECE)])
    paragraph_texts = list(
        filter(
            lambda p: p,
            map(lambda p: p.strip(),
                question_answer_evidence.evidence.text.split('\n'))))
    stride_index, paragraphs, paragraph_index = 0, [], 0
    while paragraph_index < len(paragraph_texts):
      paragraph_index, features = self._make_features(stride_index,
                                                      paragraph_texts,
                                                      paragraphs,
                                                      question_answer_evidence,
                                                      ids, paragraph_index)
      stride_index += 1
      yield features


def _handle_exceptional_examples(
    features: Features,
    processor: spm.SentencePieceProcessor) -> List[AnswerSpan]:
  """Special cases in data."""
  if features.id == 'qw_6687--Viola.txt':
    pattern = 'three strings in common—G, D, and A'.encode('utf-8')
    i = features.context.find(pattern)
    if i != -1:
      span = AnswerSpan(i + len(pattern) - 1, i + len(pattern), 'A')
      span = realign_answer_span(features, None, processor, span)
      assert span is not None, 'Span should exist.'
      return [span]
  if features.id == 'sfq_26183--Vitamin_A.txt':
    pattern = ('Vitamin A is a group of unsaturated nutritional organic '
               'compounds that includes retinol').encode('utf-8')
    i = features.context.find(pattern)
    if i != -1:
      span = AnswerSpan(i + pattern.find(b'A'), i + pattern.find(b'A') + 1, 'A')
      span = realign_answer_span(features, None, processor, span)
      assert span is not None, 'Span should exist.'
      spans = [span]
      span = AnswerSpan(i, i + pattern.find(b'A') + 1, 'Vitamin A')
      span = realign_answer_span(features, None, processor, span)
      return spans + [span]
  if features.id == 'odql_292--Colombia.txt':
    pattern = b'Colombia is the third-most populous country in Latin America'
    i = features.context.find(pattern)
    if i != -1:
      span = AnswerSpan(i, i + len(b'Colombia'), 'Colombia')
      span = realign_answer_span(features, None, processor, span)
      assert span is not None, 'Span should exist.'
      return [span]
  if features.id == 'tc_1648--Vietnam.txt':
    pattern = 'Bảo Đại'.encode('utf-8')
    i = features.context.find(pattern)
    if i != -1:
      span = AnswerSpan(i, i + len(pattern), 'Bảo Đại')
      span = realign_answer_span(features, None, processor, span)
      assert span is not None, 'Span should exist.'
      return [span]
  if features.id == 'sfq_22225--Irish_mythology.txt':
    pattern = 'Tír na nÓg'.encode('utf-8')
    spans = []
    i = 0
    while features.context.find(pattern, i) != -1:
      i = features.context.find(pattern)
      span = AnswerSpan(i, i + len(pattern), 'Tír na nÓg')
      span = realign_answer_span(features, None, processor, span)
      assert span is not None, 'Span should exist.'
      spans.append(span)
      i += len(pattern)
    return spans
  return []


class FindAnswerSpans(beam.DoFn):
  """Find answer spans in document."""

  def __init__(self, sentencepiece_model_path: Text):
    self._sentencepiece_model_path = sentencepiece_model_path

  def setup(self):
    self._sentencepiece_processor = read_sentencepiece_model(
        self._sentencepiece_model_path)

  def process(
      self,
      element: Tuple[Text, List[Features]],
      answer_sets: Dict[Text, Set[Text]],
  ) -> Generator[Tuple[Features, List[AnswerSpan]], None, None]:
    question_id, features = element
    answer_set = answer_sets[question_id]
    has_answer = False
    for feature in features:
      answer_spans = []
      for answer_span in find_answer_spans(feature.context, answer_set):
        realigned_answer_span = realign_answer_span(
            feature, answer_set, self._sentencepiece_processor, answer_span)
        if realigned_answer_span:
          answer_spans.append(realigned_answer_span)
      if not answer_spans:
        answer_spans = _handle_exceptional_examples(
            feature, self._sentencepiece_processor)
      if answer_spans:
        has_answer = True
      else:
        metrics.Metrics.counter('_', 'answerless_examples').inc()
      yield feature, answer_spans
    if not has_answer:
      metrics.Metrics.counter('_', 'answerless_questions').inc()
      logging.error('Question %s has no answer.', question_id)


def make_example(
    features: Features,
    labels: Optional[List[AnswerSpan]] = None) -> Tuple[Text, Dict[Text, Any]]:
  """Make an example."""
  feature = {
      'id': features.id,
      'qid': features.question_id,
      'question': features.question,
      'context': features.context,
      'token_ids': features.token_ids,
      'token_offsets': features.token_offsets,
      'segment_ids': features.segment_ids,
      'global_token_ids': features.global_token_ids,
  }
  if labels:
    answers = set((label.begin, label.end) for label in labels)
    feature['answers'] = np.array([list(answer) for answer in answers],
                                  np.int64)
  else:
    feature['answers'] = np.zeros([0, 2], np.int64)
  metrics.Metrics.counter('_', 'examples').inc()
  return f'{features.id}--{features.stride_index}', feature


def make_pipeline(root: beam.Pipeline, question_answers: List[QuestionAnswer],
                  answer: bool, max_num_tokens: int, max_num_global_tokens: int,
                  stride: int, sentencepiece_model_path: Text,
                  wikipedia_dir: Text, web_dir: Text):
  """Makes a Beam pipeline."""
  question_answers = (
      root | 'CreateQuestionAnswers' >> beam.Create(question_answers))
  features = (
      question_answers
      | 'ReadEvidence' >> beam.ParDo(
          ReadEvidence(wikipedia_dir=wikipedia_dir, web_dir=web_dir))
      | 'MakeFeatures' >> beam.ParDo(
          MakeFeatures(
              sentencepiece_model_path=sentencepiece_model_path,
              max_num_tokens=max_num_tokens,
              max_num_global_tokens=max_num_global_tokens,
              stride=stride)))
  if answer:
    features = features | 'KeyFeature' >> beam.Map(
        lambda feature: (feature.question_id, feature))
    # pylint: disable=g-long-lambda
    answer_sets = (
        question_answers
        | 'MakeAnswerSet' >>
        beam.Map(lambda qa: (qa.question.id, make_answer_set(qa.answer))))
    # pylint: enable=g-long-lambda
    examples = (
        features
        | beam.GroupByKey()
        | 'FindAnswerSpans' >> beam.ParDo(
            FindAnswerSpans(sentencepiece_model_path),
            answer_sets=beam.pvalue.AsDict(answer_sets))
        | 'MakeExamplesWithLabels' >> beam.MapTuple(make_example))
  else:
    examples = features | 'MakeExamples' >> beam.Map(make_example)
  return examples
