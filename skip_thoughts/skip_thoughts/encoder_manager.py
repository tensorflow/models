# Copyright 2017 The TensorFlow Authors. All Rights Reserved.
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
"""Manager class for loading and encoding with multiple skip-thoughts models.

If multiple models are loaded at once then the encode() function returns the
concatenation of the outputs of each model.

Example usage:
  manager = EncoderManager()
  manager.load_model(model_config_1, vocabulary_file_1, embedding_matrix_file_1,
                     checkpoint_path_1)
  manager.load_model(model_config_2, vocabulary_file_2, embedding_matrix_file_2,
                     checkpoint_path_2)
  encodings = manager.encode(data)
"""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import collections


import numpy as np
import tensorflow as tf

from skip_thoughts import skip_thoughts_encoder


class EncoderManager(object):
  """Manager class for loading and encoding with skip-thoughts models."""

  def __init__(self):
    self.encoders = []
    self.sessions = []

  def load_model(self, model_config, vocabulary_file, embedding_matrix_file,
                 checkpoint_path):
    """Loads a skip-thoughts model.

    Args:
      model_config: Object containing parameters for building the model.
      vocabulary_file: Path to vocabulary file containing a list of newline-
        separated words where the word id is the corresponding 0-based index in
        the file.
      embedding_matrix_file: Path to a serialized numpy array of shape
        [vocab_size, embedding_dim].
      checkpoint_path: SkipThoughtsModel checkpoint file or a directory
        containing a checkpoint file.
    """
    tf.logging.info("Reading vocabulary from %s", vocabulary_file)
    with tf.gfile.GFile(vocabulary_file, mode="r") as f:
      lines = list(f.readlines())
    reverse_vocab = [line.decode("utf-8").strip() for line in lines]
    tf.logging.info("Loaded vocabulary with %d words.", len(reverse_vocab))

    tf.logging.info("Loading embedding matrix from %s", embedding_matrix_file)
    # Note: tf.gfile.GFile doesn't work here because np.load() calls f.seek()
    # with 3 arguments.
    with open(embedding_matrix_file, "r") as f:
      embedding_matrix = np.load(f)
    tf.logging.info("Loaded embedding matrix with shape %s",
                    embedding_matrix.shape)

    word_embeddings = collections.OrderedDict(
        zip(reverse_vocab, embedding_matrix))

    g = tf.Graph()
    with g.as_default():
      encoder = skip_thoughts_encoder.SkipThoughtsEncoder(word_embeddings)
      restore_model = encoder.build_graph_from_config(model_config,
                                                      checkpoint_path)

    sess = tf.Session(graph=g)
    restore_model(sess)

    self.encoders.append(encoder)
    self.sessions.append(sess)

  def encode(self,
             data,
             use_norm=True,
             verbose=False,
             batch_size=128,
             use_eos=False):
    """Encodes a sequence of sentences as skip-thought vectors.

    Args:
      data: A list of input strings.
      use_norm: If True, normalize output skip-thought vectors to unit L2 norm.
      verbose: Whether to log every batch.
      batch_size: Batch size for the RNN encoders.
      use_eos: If True, append the end-of-sentence word to each input sentence.

    Returns:
      thought_vectors: A list of numpy arrays corresponding to 'data'.

    Raises:
      ValueError: If called before calling load_encoder.
    """
    if not self.encoders:
      raise ValueError(
          "Must call load_model at least once before calling encode.")

    encoded = []
    for encoder, sess in zip(self.encoders, self.sessions):
      encoded.append(
          np.array(
              encoder.encode(
                  sess,
                  data,
                  use_norm=use_norm,
                  verbose=verbose,
                  batch_size=batch_size,
                  use_eos=use_eos)))

    return np.concatenate(encoded, axis=1)

  def close(self):
    """Closes the active TensorFlow Sessions."""
    for sess in self.sessions:
      sess.close()
