# Copyright 2021 The TensorFlow Authors. All Rights Reserved.
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

"""TriviaQA: A Reading Comprehension Dataset."""
import functools
import json
import os

from absl import logging
import apache_beam as beam
import six
import tensorflow as tf
import tensorflow_datasets.public_api as tfds

from official.projects.triviaqa import preprocess

_CITATION = """
@article{2017arXivtriviaqa,
       author = {{Joshi}, Mandar and {Choi}, Eunsol and {Weld},
                 Daniel and {Zettlemoyer}, Luke},
        title = "{triviaqa: A Large Scale Distantly Supervised Challenge Dataset for Reading Comprehension}",
      journal = {arXiv e-prints},
         year = 2017,
          eid = {arXiv:1705.03551},
        pages = {arXiv:1705.03551},
archivePrefix = {arXiv},
       eprint = {1705.03551},
}
"""
_DOWNLOAD_URL_TMPL = (
    "http://nlp.cs.washington.edu/triviaqa/data/triviaqa-{}.tar.gz")
_TRAIN_FILE_FORMAT = "*-train.json"
_VALIDATION_FILE_FORMAT = "*-dev.json"
_TEST_FILE_FORMAT = "*test-without-answers.json"
_WEB_EVIDENCE_DIR = "evidence/web"
_WIKI_EVIDENCE_DIR = "evidence/wikipedia"

_DESCRIPTION = """\
TriviaqQA is a reading comprehension dataset containing over 650K
question-answer-evidence triples. TriviaqQA includes 95K question-answer
pairs authored by trivia enthusiasts and independently gathered evidence
documents, six per question on average, that provide high quality distant
supervision for answering the questions.
"""

_RC_DESCRIPTION = """\
Question-answer pairs where all documents for a given question contain the
answer string(s).
"""

_UNFILTERED_DESCRIPTION = """\
110k question-answer pairs for open domain QA where not all documents for a
given question contain the answer string(s). This makes the unfiltered dataset
more appropriate for IR-style QA.
"""

_CONTEXT_ADDENDUM = "Includes context from Wikipedia and search results."


def _web_evidence_dir(tmp_dir):
  return tf.io.gfile.glob(os.path.join(tmp_dir, _WEB_EVIDENCE_DIR))


def _wiki_evidence_dir(tmp_dir):
  return tf.io.gfile.glob(os.path.join(tmp_dir, _WIKI_EVIDENCE_DIR))


class TriviaQAConfig(tfds.core.BuilderConfig):
  """BuilderConfig for TriviaQA."""

  def __init__(self, *, unfiltered=False, exclude_context=False, **kwargs):
    """BuilderConfig for TriviaQA.

    Args:
      unfiltered: bool, whether to use the unfiltered version of the dataset,
        intended for open-domain QA.
      exclude_context: bool, whether to exclude Wikipedia and search context for
        reduced size.
      **kwargs: keyword arguments forwarded to super.
    """
    name = "unfiltered" if unfiltered else "rc"
    if exclude_context:
      name += ".nocontext"
    description = _UNFILTERED_DESCRIPTION if unfiltered else _RC_DESCRIPTION
    if not exclude_context:
      description += _CONTEXT_ADDENDUM
    super(TriviaQAConfig, self).__init__(
        name=name,
        description=description,
        version=tfds.core.Version("1.1.1"),
        **kwargs)
    self.unfiltered = unfiltered
    self.exclude_context = exclude_context


class BigBirdTriviaQAConfig(tfds.core.BuilderConfig):
  """BuilderConfig for TriviaQA."""

  def __init__(self, **kwargs):
    """BuilderConfig for TriviaQA.

    Args:
      **kwargs: keyword arguments forwarded to super.
    """
    name = "rc_wiki.preprocessed"
    description = _RC_DESCRIPTION
    super(BigBirdTriviaQAConfig, self).__init__(
        name=name,
        description=description,
        version=tfds.core.Version("1.1.1"),
        **kwargs)
    self.unfiltered = False
    self.exclude_context = False

  def configure(self,
                sentencepiece_model_path,
                sequence_length,
                stride,
                global_sequence_length=None):
    """Configures additional user-specified arguments."""
    self.sentencepiece_model_path = sentencepiece_model_path
    self.sequence_length = sequence_length
    self.stride = stride
    if global_sequence_length is None and sequence_length is not None:
      self.global_sequence_length = sequence_length // 16 + 64
    else:
      self.global_sequence_length = global_sequence_length
    logging.info(
        """
        global_sequence_length: %s
        sequence_length: %s
        stride: %s
        sentencepiece_model_path: %s""",
        self.global_sequence_length, self.sequence_length,
        self.stride, self.sentencepiece_model_path)

  def validate(self):
    """Validates that user specifies valid arguments."""
    if self.sequence_length is None:
      raise ValueError("sequence_length must be specified for BigBird.")
    if self.stride is None:
      raise ValueError("stride must be specified for BigBird.")
    if self.sentencepiece_model_path is None:
      raise ValueError(
          "sentencepiece_model_path must be specified for BigBird.")


def filter_files_for_big_bird(files):
  filtered_files = [f for f in files if os.path.basename(f).startswith("wiki")]
  assert len(filtered_files) == 1, "There should only be one wikipedia file."
  return filtered_files


class TriviaQA(tfds.core.BeamBasedBuilder):
  """TriviaQA is a reading comprehension dataset.

  It containss over 650K question-answer-evidence triples.
  """
  name = "bigbird_trivia_qa"
  BUILDER_CONFIGS = [
      BigBirdTriviaQAConfig(),
      TriviaQAConfig(unfiltered=False, exclude_context=False),  # rc
      TriviaQAConfig(unfiltered=False, exclude_context=True),  # rc.nocontext
      TriviaQAConfig(unfiltered=True, exclude_context=False),  # unfiltered
      TriviaQAConfig(unfiltered=True, exclude_context=True),
      # unfilered.nocontext
  ]

  def __init__(self,
               *,
               sentencepiece_model_path=None,
               sequence_length=None,
               stride=None,
               global_sequence_length=None,
               **kwargs):
    super(TriviaQA, self).__init__(**kwargs)
    if isinstance(self.builder_config, BigBirdTriviaQAConfig):
      self.builder_config.configure(
          sentencepiece_model_path=sentencepiece_model_path,
          sequence_length=sequence_length,
          stride=stride,
          global_sequence_length=global_sequence_length)

  def _info(self):
    if isinstance(self.builder_config, BigBirdTriviaQAConfig):
      return tfds.core.DatasetInfo(
          builder=self,
          description=_DESCRIPTION,
          supervised_keys=None,
          homepage="http://nlp.cs.washington.edu/triviaqa/",
          citation=_CITATION,
          features=tfds.features.FeaturesDict({
              "id": tfds.features.Text(),
              "qid": tfds.features.Text(),
              "question": tfds.features.Text(),
              "context": tfds.features.Text(),
              # Sequence features.
              "token_ids": tfds.features.Tensor(shape=(None,), dtype=tf.int64),
              "token_offsets":
                  tfds.features.Tensor(shape=(None,), dtype=tf.int64),
              "segment_ids":
                  tfds.features.Tensor(shape=(None,), dtype=tf.int64),
              "global_token_ids":
                  tfds.features.Tensor(shape=(None,), dtype=tf.int64),
              # Start and end indices (inclusive).
              "answers":
                  tfds.features.Tensor(shape=(None, 2), dtype=tf.int64),
          }))

    return tfds.core.DatasetInfo(
        builder=self,
        description=_DESCRIPTION,
        features=tfds.features.FeaturesDict({
            "question":
                tfds.features.Text(),
            "question_id":
                tfds.features.Text(),
            "question_source":
                tfds.features.Text(),
            "entity_pages":
                tfds.features.Sequence({
                    "doc_source":
                        tfds.features.Text(),
                    "filename":
                        tfds.features.Text(),
                    "title":
                        tfds.features.Text(),
                    "wiki_context":
                        tfds.features.Text(),
                }),
            "search_results":
                tfds.features.Sequence({
                    "description":
                        tfds.features.Text(),
                    "filename":
                        tfds.features.Text(),
                    "rank":
                        tf.int32,
                    "title":
                        tfds.features.Text(),
                    "url":
                        tfds.features.Text(),
                    "search_context":
                        tfds.features.Text(),
                }),
            "answer":
                tfds.features.FeaturesDict({
                    "aliases":
                        tfds.features.Sequence(tfds.features.Text()),
                    "normalized_aliases":
                        tfds.features.Sequence(tfds.features.Text()),
                    "matched_wiki_entity_name":
                        tfds.features.Text(),
                    "normalized_matched_wiki_entity_name":
                        tfds.features.Text(),
                    "normalized_value":
                        tfds.features.Text(),
                    "type":
                        tfds.features.Text(),
                    "value":
                        tfds.features.Text(),
                }),
        }),

        supervised_keys=None,
        homepage="http://nlp.cs.washington.edu/triviaqa/",
        citation=_CITATION,
    )

  def _split_generators(self, dl_manager):
    """Returns SplitGenerators."""
    cfg = self.builder_config
    download_urls = dict()
    if not (cfg.unfiltered and cfg.exclude_context):
      download_urls["rc"] = _DOWNLOAD_URL_TMPL.format("rc")
    if cfg.unfiltered:
      download_urls["unfiltered"] = _DOWNLOAD_URL_TMPL.format("unfiltered")
    file_paths = dl_manager.download_and_extract(download_urls)

    qa_dir = (
        os.path.join(file_paths["unfiltered"], "triviaqa-unfiltered")
        if cfg.unfiltered else
        os.path.join(file_paths["rc"], "qa"))
    train_files = tf.io.gfile.glob(os.path.join(qa_dir, _TRAIN_FILE_FORMAT))
    valid_files = tf.io.gfile.glob(
        os.path.join(qa_dir, _VALIDATION_FILE_FORMAT))
    test_files = tf.io.gfile.glob(os.path.join(qa_dir, _TEST_FILE_FORMAT))

    if cfg.exclude_context:
      web_evidence_dir = None
      wiki_evidence_dir = None
    else:
      web_evidence_dir = os.path.join(file_paths["rc"], _WEB_EVIDENCE_DIR)
      wiki_evidence_dir = os.path.join(file_paths["rc"], _WIKI_EVIDENCE_DIR)

    if isinstance(cfg, BigBirdTriviaQAConfig):
      train_files = filter_files_for_big_bird(train_files)
      valid_files = filter_files_for_big_bird(valid_files)
      test_files = filter_files_for_big_bird(test_files)

    return [
        tfds.core.SplitGenerator(
            name=tfds.Split.TRAIN,
            gen_kwargs={"files": train_files,
                        "web_dir": web_evidence_dir,
                        "wiki_dir": wiki_evidence_dir,
                        "answer": True}),
        tfds.core.SplitGenerator(
            name=tfds.Split.VALIDATION,
            gen_kwargs={"files": valid_files,
                        "web_dir": web_evidence_dir,
                        "wiki_dir": wiki_evidence_dir,
                        "answer": True}),
        tfds.core.SplitGenerator(
            name=tfds.Split.TEST,
            gen_kwargs={"files": test_files,
                        "web_dir": web_evidence_dir,
                        "wiki_dir": wiki_evidence_dir,
                        "answer": False}),
    ]

  def _build_pcollection(self, pipeline, files, web_dir, wiki_dir, answer):
    if isinstance(self.builder_config, BigBirdTriviaQAConfig):
      self.builder_config.validate()
      question_answers = preprocess.read_question_answers(files[0])
      return preprocess.make_pipeline(
          pipeline,
          question_answers=question_answers,
          answer=answer,
          max_num_tokens=self.builder_config.sequence_length,
          max_num_global_tokens=self.builder_config.global_sequence_length,
          stride=self.builder_config.stride,
          sentencepiece_model_path=self.builder_config.sentencepiece_model_path,
          wikipedia_dir=wiki_dir,
          web_dir=web_dir)

    parse_example_fn = functools.partial(parse_example,
                                         self.builder_config.exclude_context,
                                         web_dir, wiki_dir)
    return (pipeline
            | beam.Create(files)
            | beam.ParDo(ReadQuestions())
            | beam.Reshuffle()
            | beam.Map(parse_example_fn))


class ReadQuestions(beam.DoFn):
  """Read questions from JSON."""

  def process(self, file):
    with tf.io.gfile.GFile(file) as f:
      data = json.load(f)
    for question in data["Data"]:
      example = {"SourceFile": os.path.basename(file)}
      example.update(question)
      yield example


def parse_example(exclude_context, web_dir, wiki_dir, article):
  """Return a single example from an article JSON record."""

  def _strip(collection):
    return [item.strip() for item in collection]

  if "Answer" in article:
    answer = article["Answer"]
    answer_dict = {
        "aliases":
            _strip(answer["Aliases"]),
        "normalized_aliases":
            _strip(answer["NormalizedAliases"]),
        "matched_wiki_entity_name":
            answer.get("MatchedWikiEntryName", "").strip(),
        "normalized_matched_wiki_entity_name":
            answer.get("NormalizedMatchedWikiEntryName", "").strip(),
        "normalized_value":
            answer["NormalizedValue"].strip(),
        "type":
            answer["Type"].strip(),
        "value":
            answer["Value"].strip(),
    }
  else:
    answer_dict = {
        "aliases": [],
        "normalized_aliases": [],
        "matched_wiki_entity_name": "<unk>",
        "normalized_matched_wiki_entity_name": "<unk>",
        "normalized_value": "<unk>",
        "type": "",
        "value": "<unk>",
    }

  if exclude_context:
    article["SearchResults"] = []
    article["EntityPages"] = []

  def _add_context(collection, context_field, file_dir):
    """Adds context from file, or skips if file does not exist."""
    new_items = []
    for item in collection:
      if "Filename" not in item:
        logging.info("Missing context 'Filename', skipping.")
        continue

      new_item = item.copy()
      fname = item["Filename"]
      try:
        with tf.io.gfile.GFile(os.path.join(file_dir, fname)) as f:
          new_item[context_field] = f.read()
      except (IOError, tf.errors.NotFoundError):
        logging.info("File does not exist, skipping: %s", fname)
        continue
      new_items.append(new_item)
    return new_items

  def _strip_if_str(v):
    return v.strip() if isinstance(v, six.string_types) else v

  def _transpose_and_strip_dicts(dicts, field_names):
    return {
        tfds.core.naming.camelcase_to_snakecase(k):
        [_strip_if_str(d[k]) for d in dicts] for k in field_names
    }

  search_results = _transpose_and_strip_dicts(
      _add_context(article.get("SearchResults", []), "SearchContext", web_dir),
      ["Description", "Filename", "Rank", "Title", "Url", "SearchContext"])

  entity_pages = _transpose_and_strip_dicts(
      _add_context(article.get("EntityPages", []), "WikiContext", wiki_dir),
      ["DocSource", "Filename", "Title", "WikiContext"])

  question = article["Question"].strip()
  question_id = article["QuestionId"]
  question_source = article["QuestionSource"].strip()

  return f"{article['SourceFile']}_{question_id}", {
      "entity_pages": entity_pages,
      "search_results": search_results,
      "question": question,
      "question_id": question_id,
      "question_source": question_source,
      "answer": answer_dict,
  }
