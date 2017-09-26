# Copyright 2016 The TensorFlow Authors All Rights Reserved.
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

import os
import sys

# Allow for base embeddings to be imported
sys.path.insert(0, '../')  # noqa

import tensorflow as tf
from base_embedding.word_embedding import WordEmbedding, flags, run_model, Options

glove = tf.load_op_library(
    os.path.join(os.path.dirname(os.path.realpath(__file__)), 'glove_ops.so'))

flags.DEFINE_float("alpha", 0.75, "Exponent term for weighting function")
flags.DEFINE_float("coocurrence_max", 100,
                   "Regularization term for weighting function.")

user_flags = flags.FLAGS


class GloVeOptions(Options):
    def __init__(self, user_flags):
        # Regularization term for weighting function
        self.coocurrence_max = user_flags.coocurrence_max

        # Exponent term for weighting function
        self.alpha = user_flags.alpha

        super(GloVeOptions, self).__init__(user_flags)


class GloVe(WordEmbedding):
    def forward(self, inputs, labels, **kwargs):
        opts = self._options
        init_width = 1.0

        input_embeddings = tf.Variable(tf.random_uniform(
                [opts.vocab_size, opts.emb_dim], -init_width, init_width),
            name="input_embeddings")

        # Transposed context embeddings
        context_embeddings = tf.Variable(
            tf.random_uniform(
                [opts.vocab_size, opts.emb_dim], -init_width, init_width),
            name="context_embeddings")

        input_biases = tf.Variable(
            tf.random_uniform([opts.vocab_size], -init_width, init_width),
            name="input_biases")

        context_biases = tf.Variable(
            tf.random_uniform([opts.vocab_size], -init_width, init_width),
            name="context_biases")

        self._embeddings = tf.add(input_embeddings,
                                  context_embeddings)

        # Embeddings for examples: [batch_size, emb_dim]
        inputs_embeddings = tf.nn.embedding_lookup(
            input_embeddings, inputs)

        # Embeddings for labels: [batch_size, vocab_size]
        labels_embeddings = tf.nn.embedding_lookup(
            context_embeddings, labels)

        # biases for examples: [batch_size]
        inputs_biases = tf.nn.embedding_lookup(
            input_biases, inputs)

        # biases for labels: [batch_size]
        labels_biases = tf.nn.embedding_lookup(
            context_biases, labels)

        self.global_step = tf.Variable(0, name="global_step")

        return (inputs_embeddings, inputs_biases,
                labels_embeddings, labels_biases)

    def loss(self, **kwargs):
        opts = self._options
        ccounts = kwargs['ccounts']
        inputs_embeddings = kwargs['inputs_embeddings']
        inputs_biases = kwargs['inputs_biases']
        labels_embeddings = kwargs['labels_embeddings']
        labels_biases = kwargs['labels_biases']

        alpha_value = tf.constant(opts.alpha, dtype=tf.float32)
        x_max = tf.constant(opts.coocurrence_max, dtype=tf.float32)

        # Co-ocurrences log
        log_coocurrences = tf.log(ccounts)

        embedding_product = tf.reduce_sum(
            tf.multiply(inputs_embeddings, labels_embeddings), 1)

        distance_score = tf.square(
                tf.add_n([embedding_product,
                          inputs_biases,
                          labels_biases,
                          tf.negative(log_coocurrences)]))

        weighting_factor = tf.minimum(
            1.0,
            tf.pow(tf.div(ccounts, x_max), alpha_value))

        loss = tf.reduce_sum(
            tf.multiply(weighting_factor, distance_score))

        return loss

    def optimize(self, loss):
        opts = self._options
        lr = opts.learning_rate

        optimizer = tf.train.AdagradOptimizer(lr)
        self._lr = tf.constant(lr)
        train = optimizer.minimize(loss,
                                   global_step=self.global_step,
                                   gate_gradients=optimizer.GATE_NONE)
        self._train = train

    def build_graph(self):
        opts = self._options

        print('Calculating Co-ocurrence matrix...')
        (words, _, _, words_per_epoch,
         self._epoch, self._words, examples, labels,
         ccounts) = glove.glove_model(filename=opts.train_data,
                                      batch_size=opts.batch_size,
                                      window_size=opts.window_size,
                                      min_count=opts.min_count)
        (opts.vocab_words, opts.words_per_epoch) = self._session.run(
                 [words, words_per_epoch])
        opts.vocab_size = len(opts.vocab_words)

        print("Data file: ", opts.train_data)
        print("Vocab size: ", opts.vocab_size - 1, " + UNK")
        print("Words per epoch: ", opts.words_per_epoch)

        self._examples = examples
        self._labels = labels
        self._ccounts = ccounts
        self._id2word = opts.vocab_words

        for i, w in enumerate(self._id2word):
            self._word2id[w] = i

        (inputs_embeddings, inputs_biases,
         labels_embeddings, labels_biases) = self.forward(examples, labels)

        loss_variables = {"ccounts": ccounts,
                          "inputs_embeddings": inputs_embeddings,
                          "inputs_biases": inputs_biases,
                          "labels_embeddings": labels_embeddings,
                          "labels_biases": labels_biases}
        loss = self.loss(**loss_variables)

        self._loss = loss
        tf.summary.scalar("GloVe loss", loss)
        self.optimize(loss)

        tf.global_variables_initializer().run()
        self.saver = tf.train.Saver()


def main(_):
    """Train a GloVe model."""
    if not user_flags.train_data or not user_flags.eval_data or not user_flags.save_path:
        print("--train_data --eval_data and --save_path must be specified.")
        sys.exit(1)

    run_model(GloVe, GloVeOptions, user_flags)


if __name__ == '__main__':
    tf.app.run()
