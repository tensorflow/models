# Copyright 2018 The TensorFlow Authors All Rights Reserved.
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

"""Classes for storing hyperparameters, data locations, etc."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import json
from os.path import join
import tensorflow as tf


class Config(object):
  """Stores everything needed to train a model."""

  def __init__(self, **kwargs):
    # general
    self.data_dir = './data'  # top directory for data (corpora, models, etc.)
    self.model_name = 'default_model'  # name identifying the current model

    # mode
    self.mode = 'train'  # either "train" or "eval"
    self.task_names = ['chunk']  # list of tasks this model will learn
                                 # more than one trains a multi-task model
    self.is_semisup = True  # whether to use CVT or train purely supervised
    self.for_preprocessing = False  # is this for the preprocessing script

    # embeddings
    self.pretrained_embeddings = 'glove.6B.300d.txt'  # which pretrained
                                                      # embeddings to use
    self.word_embedding_size = 300  # size of each word embedding

    # encoder
    self.use_chars = True  # whether to include a character-level cnn
    self.char_embedding_size = 50  # size of character embeddings
    self.char_cnn_filter_widths = [2, 3, 4]  # filter widths for the char cnn
    self.char_cnn_n_filters = 100  # number of filters for each filter width
    self.unidirectional_sizes = [1024]  # size of first Bi-LSTM
    self.bidirectional_sizes = [512]  # size of second Bi-LSTM
    self.projection_size = 512  # projections size for LSTMs and hidden layers

    # dependency parsing
    self.depparse_projection_size = 128  # size of the representations used in
                                         # the bilinear classifier for parsing

    # tagging
    self.label_encoding = 'BIOES'  # label encoding scheme for entity-level
                                   # tagging tasks
    self.label_smoothing = 0.1  # label smoothing rate for tagging tasks

    # optimization
    self.lr = 0.5  # base learning rate
    self.momentum = 0.9  # momentum
    self.grad_clip = 1.0  # maximum gradient norm during optimization
    self.warm_up_steps = 5000.0  # linearly ramp up the lr for this many steps
    self.lr_decay = 0.005  # factor for gradually decaying the lr

    # EMA
    self.ema_decay = 0.998  # EMA coefficient for averaged model weights
    self.ema_test = True  # whether to use EMA weights at test time
    self.ema_teacher = False  # whether to use EMA weights for the teacher model

    # regularization
    self.labeled_keep_prob = 0.5  # 1 - dropout on labeled examples
    self.unlabeled_keep_prob = 0.8  # 1 - dropout on unlabeled examples

    # sizing
    self.max_sentence_length = 100  # maximum length of unlabeled sentences
    self.max_word_length = 20  # maximum length of words for char cnn
    self.train_batch_size = 64  # train batch size
    self.test_batch_size = 64  # test batch size
    self.buckets = [(0, 15), (15, 40), (40, 1000)]  # buckets for binning
                                                    # sentences by length

    # training
    self.print_every = 25  # how often to print out training progress
    self.eval_dev_every = 500  # how often to evaluate on the dev set
    self.eval_train_every = 2000  # how often to evaluate on the train set
    self.save_model_every = 1000  # how often to checkpoint the model

    # data set
    self.train_set_percent = 100  # how much of the train set to use

    for k, v in kwargs.iteritems():
      if k not in self.__dict__:
        raise ValueError("Unknown argument", k)
      self.__dict__[k] = v

    self.dev_set = self.mode == "train"  # whether to evaluate on the dev or
                                         # test set

    # locations of various data files
    self.raw_data_topdir = join(self.data_dir, 'raw_data')
    self.unsupervised_data = join(
        self.raw_data_topdir,
        'unlabeled_data',
        '1-billion-word-language-modeling-benchmark-r13output',
        'training-monolingual.tokenized.shuffled')
    self.pretrained_embeddings_file = join(
        self.raw_data_topdir, 'pretrained_embeddings',
        self.pretrained_embeddings)

    self.preprocessed_data_topdir = join(self.data_dir, 'preprocessed_data')
    self.embeddings_dir = join(self.preprocessed_data_topdir,
                               self.pretrained_embeddings.rsplit('.', 1)[0])
    self.word_vocabulary = join(self.embeddings_dir, 'word_vocabulary.pkl')
    self.word_embeddings = join(self.embeddings_dir, 'word_embeddings.pkl')

    self.model_dir = join(self.data_dir, "models", self.model_name)
    self.checkpoints_dir = join(self.model_dir, 'checkpoints')
    self.checkpoint = join(self.checkpoints_dir, 'checkpoint.ckpt')
    self.best_model_checkpoints_dir = join(
        self.model_dir, 'best_model_checkpoints')
    self.best_model_checkpoint = join(
        self.best_model_checkpoints_dir, 'checkpoint.ckpt')
    self.progress = join(self.checkpoints_dir, 'progress.pkl')
    self.summaries_dir = join(self.model_dir, 'summaries')
    self.history_file = join(self.model_dir, 'history.pkl')

  def write(self):
    tf.gfile.MakeDirs(self.model_dir)
    with open(join(self.model_dir, 'config.json'), 'w') as f:
      f.write(json.dumps(self.__dict__, sort_keys=True, indent=4,
                         separators=(',', ': ')))

