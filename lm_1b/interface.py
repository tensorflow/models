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
__email__ = "aphex@mit.edu"

"""
The defined class provides interface to the LM_1B model through it's methods.
"""

import os
import sys

import numpy as np
import tensorflow as tf

from google.protobuf import text_format
import data_utils

FLAGS = tf.flags.FLAGS

# General flags. Should work with the default installation.
tf.flags.DEFINE_string('pbtxt', '../data/graph-2016-09-10.pbtxt',
                       'GraphDef proto text file used to construct model '
                       'structure.')

tf.flags.DEFINE_string('ckpt', '../data/ckpt-*',
                       'Checkpoint directory used to fill model values.')

tf.flags.DEFINE_string('vocab_file', '../data/vocab-2016-09-10.txt',
                       'Vocabulary file.')

tf.flags.DEFINE_string('save_dir', '../output',
                       'Used for "dump_emb" mode to save word embeddings.')
                       
# sample mode flags.
tf.flags.DEFINE_string('prefix', 'I think that',
                       'Used for "sample" mode to predict next words.')

tf.flags.DEFINE_integer('max_sample_words', 140,
                        'Sampling stops either when </S> is met or this number '
                        'of steps has passed.')

tf.flags.DEFINE_integer('num_samples', 3,
                        'Number of samples to generate for the prefix.')

# dump_lstm_emb mode flags.
tf.flags.DEFINE_string('sentence', '',
                       'Used as input for "dump_lstm_emb" mode.')

# eval mode flags.
tf.flags.DEFINE_string('input_data', '../data/news.en.heldout-00000-of-00050',
                       'Input data files for eval model.')

tf.flags.DEFINE_integer('max_eval_steps', 3000,
                        'Maximum mumber of steps to run "eval" mode.')


# For saving demo resources, use batch size 1 and step 1.
BATCH_SIZE = 1
NUM_TIMESTEPS = 1
MAX_WORD_LEN = 50

class LM_1B(object):
    def __init__(self, model=FLAGS.pbtxt, checkpoint=FLAGS.ckpt, vocab=FLAGS.vocab_file):
        """Initialize the computational graph and the vocabulary."""
        
        self.sesh, self.t = self._load_model(model, checkpoint)
        self.vocab = data_utils.CharsVocabulary(vocab, MAX_WORD_LEN)

    def sample_text(self, prefix_words, num_samples=FLAGS.num_samples, verbose=True):
        """Predict next words using the given prefix words,
        until </S> is generated or length of string exceeds max_sample_words.

        Args:
          prefix_words: Prefix words.
          num_samples : How many sentences to predict.
          verbose     : Whether to log the sampling process to stderr (default=True).
          
        Returns:
          list of predicted sentences.
          
        """
        targets = np.zeros([BATCH_SIZE, NUM_TIMESTEPS], np.int32)
        weights = np.ones([BATCH_SIZE, NUM_TIMESTEPS], np.float32)

        if prefix_words.find('<S>') != 0:
            prefix_words = '<S> ' + prefix_words

        prefix = [self.vocab.word_to_id(w) for w in prefix_words.split()]
        prefix_char_ids = [self.vocab.word_to_char_ids(w) for w in prefix_words.split()]

        sents = []

        for _ in xrange(num_samples):

            inputs = np.zeros([BATCH_SIZE, NUM_TIMESTEPS], np.int32)
            char_ids_inputs = np.zeros([BATCH_SIZE, NUM_TIMESTEPS, self.vocab.max_word_length], np.int32)
            samples = prefix[:]
            char_ids_samples = prefix_char_ids[:]
            sent = ''

            while True:
                inputs[0, 0] = samples[0]
                char_ids_inputs[0, 0, :] = char_ids_samples[0]
                samples = samples[1:]
                char_ids_samples = char_ids_samples[1:]

                softmax = self.sesh.run(self.t['softmax_out'],
                                          feed_dict={self.t['char_inputs_in']: char_ids_inputs,
                                                self.t['inputs_in']: inputs,
                                                self.t['targets_in']: targets,
                                                self.t['target_weights_in']: weights})

                sample = _SampleSoftmax(softmax[0])
                sample_char_ids = self.vocab.word_to_char_ids(self.vocab.id_to_word(sample))

                if not samples:
                    samples = [sample]
                    char_ids_samples = [sample_char_ids]
                sent += self.vocab.id_to_word(samples[0]) + ' '
                if verbose: sys.stderr.write('%s\n' % sent)

                if (self.vocab.id_to_word(samples[0]) in ['</S>', '!']
                    or len(sent) > FLAGS.max_sample_words):
                    sents.append(sent)
                    break
                    
        return sents

    def encode_text(self, sentence):
        """Encode given string by returning the last LSTM state.
        
        Args:
          sentence: Sentence string.
        Returns:
          (1,1024)-shaped vector which is the last state of the last LSTM layer.
        """

        targets = np.zeros([BATCH_SIZE, NUM_TIMESTEPS], np.int32)
        weights = np.ones([BATCH_SIZE, NUM_TIMESTEPS], np.float32)

        if sentence.find('<S>') != 0:
            sentence = '<S> ' + sentence

        word_ids = [self.vocab.word_to_id(w) for w in sentence.split()]
        char_ids = [self.vocab.word_to_char_ids(w) for w in sentence.split()]

        inputs = np.zeros([BATCH_SIZE, NUM_TIMESTEPS], np.int32)
        char_ids_inputs = np.zeros(
              [BATCH_SIZE, NUM_TIMESTEPS, self.vocab.max_word_length], np.int32)
        for i in xrange(len(word_ids)):
            inputs[0, 0] = word_ids[i]
            char_ids_inputs[0, 0, :] = char_ids[i]

            # Add 'lstm/lstm_0/control_dependency' if you want to dump previous layer
            # LSTM.
            lstm_emb = self.sesh.run(self.t['lstm/lstm_1/control_dependency'],
                                     feed_dict={self.t['char_inputs_in']: char_ids_inputs,
                                           self.t['inputs_in']: inputs,
                                           self.t['targets_in']: targets,
                                           self.t['target_weights_in']: weights})
        self._reinitialize_states()
        
        return lstm_emb

    def dump_embeddings(self):
        """Dump the softmax weights and word embeddings to files."""
        
        assert FLAGS.save_dir, 'Must specify FLAGS.save_dir for dump_emb.'
        inputs = np.zeros([BATCH_SIZE, NUM_TIMESTEPS], np.int32)
        targets = np.zeros([BATCH_SIZE, NUM_TIMESTEPS], np.int32)
        weights = np.ones([BATCH_SIZE, NUM_TIMESTEPS], np.float32)

        softmax_weights = self.sesh.run(self.t['softmax_weights'])
        fname = FLAGS.save_dir + '/embeddings_softmax.npy'
        with tf.gfile.Open(fname, mode='w') as f:
            np.save(f, softmax_weights)
        sys.stderr.write('Finished softmax weights\n')

        all_embs = np.zeros([self.vocab.size, 1024])
        for i in range(self.vocab.size):
            input_dict = {self.t['inputs_in']: inputs,
                          self.t['targets_in']: targets,
                          self.t['target_weights_in']: weights}
            if 'char_inputs_in' in self.t:
                input_dict[self.t['char_inputs_in']] = (
                    self.vocab.word_char_ids[i].reshape([-1, 1, MAX_WORD_LEN]))
            embs = self.sesh.run(self.t['all_embs'], input_dict)
            all_embs[i, :] = embs
            sys.stderr.write('Finished word embedding %d/%d\n' % (i, self.vocab.size))

        fname = FLAGS.save_dir + '/embeddings_char_cnn.npy'
        with tf.gfile.Open(fname, mode='w') as f:
            np.save(f, all_embs)
        sys.stderr.write('Embedding file saved\n')

    def eval_model(self, data=FLAGS.input_data, steps=FLAGS.max_eval_steps, verbose=True):
        """Evaluate model perplexity using the provided dataset.

        Args:
          data: path to the text file to evaluate perplexity on.
          steps: how many steps to evaluate for.
        """
        current_step = self.t['global_step'].eval(session=self.sesh)
        if verbose: sys.stderr.write('Loaded step %d.\n' % current_step)

        dataset = data_utils.LM1BDataset(data, self.vocab)

        data_gen = dataset.get_batch(BATCH_SIZE, NUM_TIMESTEPS, forever=False)
        sum_num = 0.0
        sum_den = 0.0
        perplexity = 0.0
        ppx = []
        for i, (inputs, char_inputs, _, targets, weights) in enumerate(data_gen):
            input_dict = {self.t['inputs_in']: inputs,
                          self.t['targets_in']: targets,
                          self.t['target_weights_in']: weights}
            if 'char_inputs_in' in self.t:
                input_dict[self.t['char_inputs_in']] = char_inputs
            log_perp = self.sesh.run(self.t['log_perplexity_out'], feed_dict=input_dict)

            if np.isnan(log_perp):
                if verbose: sys.stderr.error('log_perplexity is Nan.\n')
            else:
                sum_num += log_perp * weights.mean()
                sum_den += weights.mean()
            if sum_den > 0:
                perplexity = np.exp(sum_num / sum_den)

            ppx.append(perplexity)
            if verbose: sys.stderr.write('Eval Step: %d, Average Perplexity: %f.\n' %
                             (i, perplexity))

            if i > steps:
                break
                
        return ppx

    def _reinitialize_states(self):
        self.sesh.run(self.t['states_init'])

    @staticmethod
    def _load_model(gd_file, ckpt_file):
        """Load the model from GraphDef and Checkpoint.

        Args:
          gd_file: GraphDef proto text file.
          ckpt_file: TensorFlow Checkpoint file.

        Returns:
          TensorFlow session and tensors dict.
        """
        with tf.Graph().as_default():
            sys.stderr.write('Recovering graph.\n')
            with tf.gfile.FastGFile(gd_file, 'r') as f:
                s = f.read()
                gd = tf.GraphDef()
                text_format.Merge(s, gd)

            tf.logging.info('Recovering Graph %s', gd_file)
            t = {}
            [t['states_init'], t['lstm/lstm_0/control_dependency'],
             t['lstm/lstm_1/control_dependency'], t['softmax_out'], t['class_ids_out'],
             t['class_weights_out'], t['log_perplexity_out'], t['inputs_in'],
             t['targets_in'], t['target_weights_in'], t['char_inputs_in'],
             t['all_embs'], t['softmax_weights'], t['global_step']
            ] = tf.import_graph_def(gd, {}, ['states_init',
                                             'lstm/lstm_0/control_dependency:0',
                                             'lstm/lstm_1/control_dependency:0',
                                             'softmax_out:0',
                                             'class_ids_out:0',
                                             'class_weights_out:0',
                                             'log_perplexity_out:0',
                                             'inputs_in:0',
                                             'targets_in:0',
                                             'target_weights_in:0',
                                             'char_inputs_in:0',
                                             'all_embs_out:0',
                                             'Reshape_3:0',
                                             'global_step:0'], name='')

            sys.stderr.write('Recovering checkpoint %s\n' % ckpt_file)
            sess = tf.Session(config=tf.ConfigProto(allow_soft_placement=True))
            sess.run('save/restore_all', {'save/Const:0': ckpt_file})
            sess.run(t['states_init'])

        return sess, t

def _SampleSoftmax(softmax):
    return min(np.sum(np.cumsum(softmax) < np.random.rand()), len(softmax) - 1)
