# Copyright 2016 Google Inc. All Rights Reserved.
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
"""A library showing off sequence recognition and generation with the simple
example of names.

We use recurrent neural nets to learn complex functions able to recogize and
generate sequences of a given form. This can be used for natural language
syntax recognition, dynamically generating maps or puzzles and of course
baby name generation.

Before using this module, it is recommended to read the Tensorflow tutorial on
recurrent neural nets, as it explains the basic concepts of this model, and
will show off another module, the PTB module on which this model bases itself.

Here is an overview of the functions available in this module:

* RNN Module for sequence functions based on PTB

* Name recognition specifically for recognizing names, but can be adapted to
    recognizing sequence patterns

* Name generations specifically for generating names, but can be adapted to
    generating arbitrary sequence patterns
"""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import time

import tensorflow as tf
import numpy as np

from model import NamignizerModel
import data_utils


class SmallConfig(object):
    """Small config."""
    init_scale = 0.1
    learning_rate = 1.0
    max_grad_norm = 5
    num_layers = 2
    num_steps = 20
    hidden_size = 200
    max_epoch = 4
    max_max_epoch = 13
    keep_prob = 1.0
    lr_decay = 0.5
    batch_size = 20
    vocab_size = 27
    epoch_size = 100


class LargeConfig(object):
    """Medium config."""
    init_scale = 0.05
    learning_rate = 1.0
    max_grad_norm = 5
    num_layers = 2
    num_steps = 35
    hidden_size = 650
    max_epoch = 6
    max_max_epoch = 39
    keep_prob = 0.5
    lr_decay = 0.8
    batch_size = 20
    vocab_size = 27
    epoch_size = 100


class TestConfig(object):
    """Tiny config, for testing."""
    init_scale = 0.1
    learning_rate = 1.0
    max_grad_norm = 1
    num_layers = 1
    num_steps = 2
    hidden_size = 2
    max_epoch = 1
    max_max_epoch = 1
    keep_prob = 1.0
    lr_decay = 0.5
    batch_size = 20
    vocab_size = 27
    epoch_size = 100


def run_epoch(session, m, names, counts, epoch_size, eval_op, verbose=False):
    """Runs the model on the given data for one epoch

    Args:
        session: the tf session holding the model graph
        m: an instance of the NamignizerModel
        names: a set of lowercase names of 26 characters
        counts: a list of the frequency of the above names
        epoch_size: the number of batches to run
        eval_op: whether to change the params or not, and how to do it
    Kwargs:
        verbose: whether to print out state of training during the epoch
    Returns:
        cost: the average cost during the last stage of the epoch
    """
    start_time = time.time()
    costs = 0.0
    iters = 0
    for step, (x, y) in enumerate(data_utils.namignizer_iterator(names, counts,
                                                                 m.batch_size, m.num_steps, epoch_size)):

        cost, _ = session.run([m.cost, eval_op],
                              {m.input_data: x,
                               m.targets: y,
                               m.initial_state: m.initial_state.eval(),
                               m.weights: np.ones(m.batch_size * m.num_steps)})
        costs += cost
        iters += m.num_steps

        if verbose and step % (epoch_size // 10) == 9:
            print("%.3f perplexity: %.3f speed: %.0f lps" %
                  (step * 1.0 / epoch_size, np.exp(costs / iters),
                   iters * m.batch_size / (time.time() - start_time)))

        if step >= epoch_size:
            break

    return np.exp(costs / iters)


def train(data_dir, checkpoint_path, config):
    """Trains the model with the given data

    Args:
        data_dir: path to the data for the model (see data_utils for data
            format)
        checkpoint_path: the path to save the trained model checkpoints
        config: one of the above configs that specify the model and how it
            should be run and trained
    Returns:
        None
    """
    # Prepare Name data.
    print("Reading Name data in %s" % data_dir)
    names, counts = data_utils.read_names(data_dir)

    with tf.Graph().as_default(), tf.Session() as session:
        initializer = tf.random_uniform_initializer(-config.init_scale,
                                                    config.init_scale)
        with tf.variable_scope("model", reuse=None, initializer=initializer):
            m = NamignizerModel(is_training=True, config=config)

        tf.initialize_all_variables().run()

        for i in range(config.max_max_epoch):
            lr_decay = config.lr_decay ** max(i - config.max_epoch, 0.0)
            m.assign_lr(session, config.learning_rate * lr_decay)

            print("Epoch: %d Learning rate: %.3f" % (i + 1, session.run(m.lr)))
            train_perplexity = run_epoch(session, m, names, counts, config.epoch_size, m.train_op,
                                         verbose=True)
            print("Epoch: %d Train Perplexity: %.3f" %
                  (i + 1, train_perplexity))

            m.saver.save(session, checkpoint_path, global_step=i)


def namignize(names, checkpoint_path, config):
    """Recognizes names and prints the Perplexity of the model for each names
    in the list

    Args:
        names: a list of names in the model format
        checkpoint_path: the path to restore the trained model from, should not
            include the model name, just the path to
        config: one of the above configs that specify the model and how it
            should be run and trained
    Returns:
        None
    """
    with tf.Graph().as_default(), tf.Session() as session:

        with tf.variable_scope("model"):
            m = NamignizerModel(is_training=False, config=config)

        m.saver.restore(session, checkpoint_path)

        for name in names:
            x, y = data_utils.name_to_batch(name, m.batch_size, m.num_steps)

            cost, loss, _ = session.run([m.cost, m.loss, tf.no_op()],
                                  {m.input_data: x,
                                   m.targets: y,
                                   m.initial_state: m.initial_state.eval(),
                                   m.weights: np.concatenate((
                                       np.ones(len(name)), np.zeros(m.batch_size * m.num_steps - len(name))))})

            print("Name {} gives us a perplexity of {}".format(
                name, np.exp(cost)))


def namignator(checkpoint_path, config):
    """Generates names randomly according to a given model

    Args:
        checkpoint_path: the path to restore the trained model from, should not
            include the model name, just the path to
        config: one of the above configs that specify the model and how it
            should be run and trained
    Returns:
        None
    """
    # mutate the config to become a name generator config
    config.num_steps = 1
    config.batch_size = 1

    with tf.Graph().as_default(), tf.Session() as session:

        with tf.variable_scope("model"):
            m = NamignizerModel(is_training=False, config=config)

        m.saver.restore(session, checkpoint_path)

        activations, final_state, _ = session.run([m.activations, m.final_state, tf.no_op()],
                                                  {m.input_data: np.zeros((1, 1)),
                                                   m.targets: np.zeros((1, 1)),
                                                   m.initial_state: m.initial_state.eval(),
                                                   m.weights: np.ones(1)})

        # sample from our softmax activations
        next_letter = np.random.choice(27, p=activations[0])
        name = [next_letter]
        while next_letter != 0:
            activations, final_state, _ = session.run([m.activations, m.final_state, tf.no_op()],
                                                      {m.input_data: [[next_letter]],
                                                       m.targets: np.zeros((1, 1)),
                                                       m.initial_state: final_state,
                                                       m.weights: np.ones(1)})

            next_letter = np.random.choice(27, p=activations[0])
            name += [next_letter]

        print(map(lambda x: chr(x + 96), name))


if __name__ == "__main__":
    # train("data/SmallNames.txt", "model/namignizer", SmallConfig)

    # namignize(["mary", "ida", "gazorbazorb", "mmmhmm", "bob"],
    #     tf.train.latest_checkpoint("model"), SmallConfig)

    # namignator(tf.train.latest_checkpoint("model"), SmallConfig)
