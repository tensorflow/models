# Copyright 2024 The TensorFlow Authors. All Rights Reserved.
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

"""Library for processing crawled content and generating tfrecords."""

import collections
import json
import multiprocessing.pool
import os
import urllib.parse

import tensorflow as tf, tf_keras

from official.nlp.data import classifier_data_lib
from official.nlp.tools import tokenization


class RawDataProcessor(object):
  """Data converter for story examples."""

  def __init__(self,
               vocab: str,
               do_lower_case: bool,
               len_title: int = 15,
               len_passage: int = 200,
               max_num_articles: int = 5,
               include_article_title_in_passage: bool = False,
               include_text_snippet_in_example: bool = False):
    """Constructs a RawDataProcessor.

    Args:
      vocab: Filepath of the BERT vocabulary.
      do_lower_case: Whether the vocabulary is uncased or not.
      len_title: Maximum number of tokens in story headline.
      len_passage: Maximum number of tokens in article passage.
      max_num_articles: Maximum number of articles in a story.
      include_article_title_in_passage: Whether to include article title in
        article passage.
      include_text_snippet_in_example: Whether to include text snippet (headline
        and article content) in generated tensorflow Examples, for debug usage.
        If include_article_title_in_passage=True, title and body will be
        separated by [SEP].
    """
    self.articles = dict()
    self.tokenizer = tokenization.FullTokenizer(
        vocab, do_lower_case=do_lower_case, split_on_punc=False)
    self.len_title = len_title
    self.len_passage = len_passage
    self.max_num_articles = max_num_articles
    self.include_article_title_in_passage = include_article_title_in_passage
    self.include_text_snippet_in_example = include_text_snippet_in_example
    # ex_index=5 deactivates printing inside convert_single_example.
    self.ex_index = 5
    # Parameters used in InputExample, not used in NHNet.
    self.label = 0
    self.guid = 0
    self.num_generated_examples = 0

  def read_crawled_articles(self, folder_path):
    """Reads crawled articles under folder_path."""
    for path, _, files in os.walk(folder_path):
      for name in files:
        if not name.endswith(".json"):
          continue
        url, article = self._get_article_content_from_json(
            os.path.join(path, name))
        if not article.text_a:
          continue
        self.articles[RawDataProcessor.normalize_url(url)] = article
        if len(self.articles) % 5000 == 0:
          print("Number of articles loaded: %d\r" % len(self.articles), end="")
    print()
    return len(self.articles)

  def generate_examples(self, input_file, output_files):
    """Loads story from input json file and exports examples in output_files."""
    writers = []
    story_partition = []
    for output_file in output_files:
      writers.append(tf.io.TFRecordWriter(output_file))
      story_partition.append(list())
    with tf.io.gfile.GFile(input_file, "r") as story_json_file:
      stories = json.load(story_json_file)
      writer_index = 0
      for story in stories:
        articles = []
        for url in story["urls"]:
          normalized_url = RawDataProcessor.normalize_url(url)
          if normalized_url in self.articles:
            articles.append(self.articles[normalized_url])
        if not articles:
          continue
        story_partition[writer_index].append((story["label"], articles))
        writer_index = (writer_index + 1) % len(writers)
    lock = multiprocessing.Lock()
    pool = multiprocessing.pool.ThreadPool(len(writers))
    data = [(story_partition[i], writers[i], lock) for i in range(len(writers))]
    pool.map(self._write_story_partition, data)
    return len(stories), self.num_generated_examples

  @classmethod
  def normalize_url(cls, url):
    """Normalize url for better matching."""
    url = urllib.parse.unquote(
        urllib.parse.urlsplit(url)._replace(query=None).geturl())
    output, part = [], None
    for part in url.split("//"):
      if part == "http:" or part == "https:":
        continue
      else:
        output.append(part)
    return "//".join(output)

  def _get_article_content_from_json(self, file_path):
    """Returns (url, InputExample) keeping content extracted from file_path."""
    with tf.io.gfile.GFile(file_path, "r") as article_json_file:
      article = json.load(article_json_file)
      if self.include_article_title_in_passage:
        return article["url"], classifier_data_lib.InputExample(
            guid=self.guid,
            text_a=article["title"],
            text_b=article["maintext"],
            label=self.label)
      else:
        return article["url"], classifier_data_lib.InputExample(
            guid=self.guid, text_a=article["maintext"], label=self.label)

  def _write_story_partition(self, data):
    """Writes stories in a partition into file."""
    for (story_headline, articles) in data[0]:
      story_example = tf.train.Example(
          features=tf.train.Features(
              feature=self._get_single_story_features(story_headline,
                                                      articles)))
      data[1].write(story_example.SerializeToString())
      data[2].acquire()
      try:
        self.num_generated_examples += 1
        if self.num_generated_examples % 1000 == 0:
          print(
              "Number of stories written: %d\r" % self.num_generated_examples,
              end="")
      finally:
        data[2].release()

  def _get_single_story_features(self, story_headline, articles):
    """Converts a list of articles to a tensorflow Example."""

    def get_text_snippet(article):
      if article.text_b:
        return " [SEP] ".join([article.text_a, article.text_b])
      else:
        return article.text_a

    story_features = collections.OrderedDict()
    story_headline_feature = classifier_data_lib.convert_single_example(
        ex_index=self.ex_index,
        example=classifier_data_lib.InputExample(
            guid=self.guid, text_a=story_headline, label=self.label),
        label_list=[self.label],
        max_seq_length=self.len_title,
        tokenizer=self.tokenizer)
    if self.include_text_snippet_in_example:
      story_headline_feature.label_id = story_headline
    self._add_feature_with_suffix(
        feature=story_headline_feature,
        suffix="a",
        story_features=story_features)
    for (article_index, article) in enumerate(articles):
      if article_index == self.max_num_articles:
        break
      article_feature = classifier_data_lib.convert_single_example(
          ex_index=self.ex_index,
          example=article,
          label_list=[self.label],
          max_seq_length=self.len_passage,
          tokenizer=self.tokenizer)
      if self.include_text_snippet_in_example:
        article_feature.label_id = get_text_snippet(article)
      suffix = chr(ord("b") + article_index)
      self._add_feature_with_suffix(
          feature=article_feature, suffix=suffix, story_features=story_features)

    # Adds empty features as placeholder.
    for article_index in range(len(articles), self.max_num_articles):
      suffix = chr(ord("b") + article_index)
      empty_article = classifier_data_lib.InputExample(
          guid=self.guid, text_a="", label=self.label)
      empty_feature = classifier_data_lib.convert_single_example(
          ex_index=self.ex_index,
          example=empty_article,
          label_list=[self.label],
          max_seq_length=self.len_passage,
          tokenizer=self.tokenizer)
      if self.include_text_snippet_in_example:
        empty_feature.label_id = ""
      self._add_feature_with_suffix(
          feature=empty_feature, suffix=suffix, story_features=story_features)
    return story_features

  def _add_feature_with_suffix(self, feature, suffix, story_features):
    """Appends suffix to feature names and fills in the corresponding values."""

    def _create_int_feature(values):
      return tf.train.Feature(int64_list=tf.train.Int64List(value=list(values)))

    def _create_string_feature(value):
      return tf.train.Feature(bytes_list=tf.train.BytesList(value=[value]))

    story_features["input_ids_%c" % suffix] = _create_int_feature(
        feature.input_ids)
    story_features["input_mask_%c" % suffix] = _create_int_feature(
        feature.input_mask)
    story_features["segment_ids_%c" % suffix] = _create_int_feature(
        feature.segment_ids)
    if self.include_text_snippet_in_example:
      story_features["text_snippet_%c" % suffix] = _create_string_feature(
          bytes(feature.label_id.encode()))
