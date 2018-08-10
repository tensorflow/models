#!/usr/bin/env python
# Copyright 2017, 2018 Google, Inc. All Rights Reserved.
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

"""Extracts paths that are indicative of each relation."""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os

import tensorflow as tf

from . import path_model
from . import lexnet_common

tf.flags.DEFINE_string(
    'dataset_dir', 'datasets',
    'Dataset base directory')

tf.flags.DEFINE_string(
    'dataset',
    'tratz/fine_grained',
    'Subdirectory containing the corpus directories: '
    'subdirectory of dataset_dir')

tf.flags.DEFINE_string(
    'corpus', 'random/wiki',
    'Subdirectory containing the corpus and split: '
    'subdirectory of dataset_dir/dataset')

tf.flags.DEFINE_string(
    'embeddings_base_path', 'embeddings',
    'Embeddings base directory')

tf.flags.DEFINE_string(
    'logdir', 'logdir',
    'Directory of model output files')

tf.flags.DEFINE_integer(
    'top_k', 20, 'Number of top paths to extract')

tf.flags.DEFINE_float(
    'threshold', 0.8, 'Threshold above which to consider paths as indicative')

FLAGS = tf.flags.FLAGS


def main(_):
  hparams = path_model.PathBasedModel.default_hparams()

  # First things first. Load the path data.
  path_embeddings_file = 'path_embeddings/{dataset}/{corpus}'.format(
      dataset=FLAGS.dataset,
      corpus=FLAGS.corpus)

  path_dim = (hparams.lemma_dim + hparams.pos_dim +
              hparams.dep_dim + hparams.dir_dim)

  path_embeddings, path_to_index = path_model.load_path_embeddings(
      os.path.join(FLAGS.embeddings_base_path, path_embeddings_file),
      path_dim)

  # Load and count the classes so we can correctly instantiate the model.
  classes_filename = os.path.join(
      FLAGS.dataset_dir, FLAGS.dataset, 'classes.txt')

  with open(classes_filename) as f_in:
    classes = f_in.read().splitlines()

  hparams.num_classes = len(classes)

  # We need the word embeddings to instantiate the model, too.
  print('Loading word embeddings...')
  lemma_embeddings = lexnet_common.load_word_embeddings(
      FLAGS.embeddings_base_path, hparams.lemma_embeddings_file)

  # Instantiate the model.
  with tf.Graph().as_default():
    with tf.variable_scope('lexnet'):
      instance = tf.placeholder(dtype=tf.string)
      model = path_model.PathBasedModel(
          hparams, lemma_embeddings, instance)

    with tf.Session() as session:
      model_dir = '{logdir}/results/{dataset}/path/{corpus}'.format(
          logdir=FLAGS.logdir,
          dataset=FLAGS.dataset,
          corpus=FLAGS.corpus)

      saver = tf.train.Saver()
      saver.restore(session, os.path.join(model_dir, 'best.ckpt'))

      path_model.get_indicative_paths(
          model, session, path_to_index, path_embeddings, classes,
          model_dir, FLAGS.top_k, FLAGS.threshold)

if __name__ == '__main__':
  tf.app.run()
