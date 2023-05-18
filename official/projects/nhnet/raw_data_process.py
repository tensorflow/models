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

"""Processes crawled content from news URLs by generating tfrecords."""

import os

from absl import app
from absl import flags
from official.projects.nhnet import raw_data_processor

FLAGS = flags.FLAGS

flags.DEFINE_string("crawled_articles", "/tmp/nhnet/",
                    "Folder path to the crawled articles using news-please.")
flags.DEFINE_string("vocab", None, "Filepath of the BERT vocabulary.")
flags.DEFINE_bool("do_lower_case", True,
                  "Whether the vocabulary is uncased or not.")
flags.DEFINE_integer("len_title", 15,
                     "Maximum number of tokens in story headline.")
flags.DEFINE_integer("len_passage", 200,
                     "Maximum number of tokens in article passage.")
flags.DEFINE_integer("max_num_articles", 5,
                     "Maximum number of articles in a story.")
flags.DEFINE_bool("include_article_title_in_passage", False,
                  "Whether to include article title in article passage.")
flags.DEFINE_string("data_folder", None,
                    "Folder path to the downloaded data folder (output).")
flags.DEFINE_integer("num_tfrecords_shards", 20,
                     "Number of shards for train/valid/test.")


def transform_as_tfrecords(data_processor, filename):
  """Transforms story from json to tfrecord (sharded).

  Args:
    data_processor: Instance of RawDataProcessor.
    filename: 'train', 'valid', or 'test'.
  """
  print("Transforming json to tfrecord for %s..." % filename)
  story_filepath = os.path.join(FLAGS.data_folder, filename + ".json")
  output_folder = os.path.join(FLAGS.data_folder, "processed")
  os.makedirs(output_folder, exist_ok=True)
  output_filepaths = []
  for i in range(FLAGS.num_tfrecords_shards):
    output_filepaths.append(
        os.path.join(
            output_folder, "%s.tfrecord-%.5d-of-%.5d" %
            (filename, i, FLAGS.num_tfrecords_shards)))
  (total_num_examples,
   generated_num_examples) = data_processor.generate_examples(
       story_filepath, output_filepaths)
  print("For %s, %d examples have been generated from %d stories in json." %
        (filename, generated_num_examples, total_num_examples))


def main(_):
  if not FLAGS.data_folder:
    raise ValueError("data_folder must be set as the downloaded folder path.")
  if not FLAGS.vocab:
    raise ValueError("vocab must be set as the filepath of BERT vocabulary.")
  data_processor = raw_data_processor.RawDataProcessor(
      vocab=FLAGS.vocab,
      do_lower_case=FLAGS.do_lower_case,
      len_title=FLAGS.len_title,
      len_passage=FLAGS.len_passage,
      max_num_articles=FLAGS.max_num_articles,
      include_article_title_in_passage=FLAGS.include_article_title_in_passage,
      include_text_snippet_in_example=True)
  print("Loading crawled articles...")
  num_articles = data_processor.read_crawled_articles(FLAGS.crawled_articles)
  print("Total number of articles loaded: %d" % num_articles)
  print()
  transform_as_tfrecords(data_processor, "train")
  transform_as_tfrecords(data_processor, "valid")
  transform_as_tfrecords(data_processor, "test")


if __name__ == "__main__":
  app.run(main)
