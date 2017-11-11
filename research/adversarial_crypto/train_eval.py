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

"""Adversarial training to learn trivial encryption functions,
from the paper "Learning to Protect Communications with
Adversarial Neural Cryptography", Abadi & Andersen, 2016.

https://arxiv.org/abs/1610.06918

This program creates and trains three neural networks,
termed Alice, Bob, and Eve.  Alice takes inputs
in_m (message), in_k (key) and outputs 'ciphertext'.

Bob takes inputs in_k, ciphertext and tries to reconstruct
the message.

Eve is an adversarial network that takes input ciphertext
and also tries to reconstruct the message.

The main function attempts to train these networks and then
evaluates them, all on random plaintext and key values.

"""

# TensorFlow Python 3 compatibility
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
import signal
import sys
from six.moves import xrange  # pylint: disable=redefined-builtin
import tensorflow as tf

flags = tf.app.flags

flags.DEFINE_float('learning_rate', 0.0008, 'Constant learning rate')
flags.DEFINE_integer('batch_size', 4096, 'Batch size')

FLAGS = flags.FLAGS

# Input and output configuration.
TEXT_SIZE = 16
KEY_SIZE = 16

# Training parameters.
ITERS_PER_ACTOR = 1
EVE_MULTIPLIER = 2  # Train Eve 2x for every step of Alice/Bob
# Train until either max loops or Alice/Bob "good enough":
MAX_TRAINING_LOOPS = 850000
BOB_LOSS_THRESH = 0.02  # Exit when Bob loss < 0.02 and Eve > 7.7 bits
EVE_LOSS_THRESH = 7.7

# Logging and evaluation.
PRINT_EVERY = 200  # In training, log every 200 steps.
EVE_EXTRA_ROUNDS = 2000  # At end, train eve a bit more.
RETRAIN_EVE_ITERS = 10000  # Retrain eve up to ITERS*LOOPS times.
RETRAIN_EVE_LOOPS = 25  # With an evaluation each loop
NUMBER_OF_EVE_RESETS = 5  # And do this up to 5 times with a fresh eve.
# Use EVAL_BATCHES samples each time we check accuracy.
EVAL_BATCHES = 1


def batch_of_random_bools(batch_size, n):
  """Return a batch of random "boolean" numbers.

  Args:
    batch_size:  Batch size dimension of returned tensor.
    n:  number of entries per batch.

  Returns:
    A [batch_size, n] tensor of "boolean" numbers, where each number is
    preresented as -1 or 1.
  """

  as_int = tf.random_uniform(
      [batch_size, n], minval=0, maxval=2, dtype=tf.int32)
  expanded_range = (as_int * 2) - 1
  return tf.cast(expanded_range, tf.float32)


class AdversarialCrypto(object):
  """Primary model implementation class for Adversarial Neural Crypto.

  This class contains the code for the model itself,
  and when created, plumbs the pathways from Alice to Bob and
  Eve, creates the optimizers and loss functions, etc.

  Attributes:
    eve_loss:  Eve's loss function.
    bob_loss:  Bob's loss function.  Different units from eve_loss.
    eve_optimizer:  A tf op that runs Eve's optimizer.
    bob_optimizer:  A tf op that runs Bob's optimizer.
    bob_reconstruction_loss:  Bob's message reconstruction loss,
      which is comparable to eve_loss.
    reset_eve_vars:  Execute this op to completely reset Eve.
  """

  def get_message_and_key(self):
    """Generate random pseudo-boolean key and message values."""

    batch_size = tf.placeholder_with_default(FLAGS.batch_size, shape=[])

    in_m = batch_of_random_bools(batch_size, TEXT_SIZE)
    in_k = batch_of_random_bools(batch_size, KEY_SIZE)
    return in_m, in_k

  def model(self, collection, message, key=None):
    """The model for Alice, Bob, and Eve.  If key=None, the first fully connected layer
    takes only the message as inputs.  Otherwise, it uses both the key
    and the message.

    Args:
      collection:  The graph keys collection to add new vars to.
      message:  The input message to process.
      key:  The input key (if any) to use.
    """

    if key is not None:
      combined_message = tf.concat(axis=1, values=[message, key])
    else:
      combined_message = message

    # Ensure that all variables created are in the specified collection.
    with tf.contrib.framework.arg_scope(
        [tf.contrib.layers.fully_connected, tf.contrib.layers.conv2d],
        variables_collections=[collection]):

      fc = tf.contrib.layers.fully_connected(
          combined_message,
          TEXT_SIZE + KEY_SIZE,
          biases_initializer=tf.constant_initializer(0.0),
          activation_fn=None)

      # Perform a sequence of 1D convolutions (by expanding the message out to 2D
      # and then squeezing it back down).
      fc = tf.expand_dims(fc, 2)
      # 2,1 -> 1,2
      conv = tf.contrib.layers.conv2d(
          fc, 2, 2, 2, 'SAME', activation_fn=tf.nn.sigmoid)
      # 1,2 -> 1, 2
      conv = tf.contrib.layers.conv2d(
          conv, 2, 1, 1, 'SAME', activation_fn=tf.nn.sigmoid)
      # 1,2 -> 1, 1
      conv = tf.contrib.layers.conv2d(
          conv, 1, 1, 1, 'SAME', activation_fn=tf.nn.tanh)
      conv = tf.squeeze(conv, 2)
      return conv

  def __init__(self):
    in_m, in_k = self.get_message_and_key()
    encrypted = self.model('alice', in_m, in_k)
    decrypted = self.model('bob', encrypted, in_k)
    eve_out = self.model('eve', encrypted, None)

    self.reset_eve_vars = tf.group(
        *[w.initializer for w in tf.get_collection('eve')])

    optimizer = tf.train.AdamOptimizer(learning_rate=FLAGS.learning_rate)

    # Eve's goal is to decrypt the entire message:
    eve_bits_wrong = tf.reduce_sum(
        tf.abs((eve_out + 1.0) / 2.0 - (in_m + 1.0) / 2.0), [1])
    self.eve_loss = tf.reduce_sum(eve_bits_wrong)
    self.eve_optimizer = optimizer.minimize(
        self.eve_loss, var_list=tf.get_collection('eve'))

    # Alice and Bob want to be accurate...
    self.bob_bits_wrong = tf.reduce_sum(
        tf.abs((decrypted + 1.0) / 2.0 - (in_m + 1.0) / 2.0), [1])
    # ... and to not let Eve do better than guessing.
    self.bob_reconstruction_loss = tf.reduce_sum(self.bob_bits_wrong)
    bob_eve_error_deviation = tf.abs(float(TEXT_SIZE) / 2.0 - eve_bits_wrong)
    # 7-9 bits wrong is OK too, so we squish the error function a bit.
    # Without doing this, we often tend to hang out at 0.25 / 7.5 error,
    # and it seems bad to have continued, high communication error.
    bob_eve_loss = tf.reduce_sum(
        tf.square(bob_eve_error_deviation) / (TEXT_SIZE / 2)**2)

    # Rescale the losses to [0, 1] per example and combine.
    self.bob_loss = (self.bob_reconstruction_loss / TEXT_SIZE + bob_eve_loss)

    self.bob_optimizer = optimizer.minimize(
        self.bob_loss,
        var_list=(tf.get_collection('alice') + tf.get_collection('bob')))


def doeval(s, ac, n, itercount):
  """Evaluate the current network on n batches of random examples.

  Args:
    s:  The current TensorFlow session
    ac: an instance of the AdversarialCrypto class
    n:  The number of iterations to run.
    itercount: Iteration count label for logging.

  Returns:
    Bob and Eve's loss, as a percent of bits incorrect.
  """

  bob_loss_accum = 0
  eve_loss_accum = 0
  for _ in xrange(n):
    bl, el = s.run([ac.bob_reconstruction_loss, ac.eve_loss])
    bob_loss_accum += bl
    eve_loss_accum += el
  bob_loss_percent = bob_loss_accum / (n * FLAGS.batch_size)
  eve_loss_percent = eve_loss_accum / (n * FLAGS.batch_size)
  print('%10d\t%20.2f\t%20.2f'%(itercount, bob_loss_percent, eve_loss_percent))
  sys.stdout.flush()
  return bob_loss_percent, eve_loss_percent


def train_until_thresh(s, ac):
  for j in xrange(MAX_TRAINING_LOOPS):
    for _ in xrange(ITERS_PER_ACTOR):
      s.run(ac.bob_optimizer)
    for _ in xrange(ITERS_PER_ACTOR * EVE_MULTIPLIER):
      s.run(ac.eve_optimizer)
    if j % PRINT_EVERY == 0:
      bob_avg_loss, eve_avg_loss = doeval(s, ac, EVAL_BATCHES, j)
      if (bob_avg_loss < BOB_LOSS_THRESH and eve_avg_loss > EVE_LOSS_THRESH):
        print('Target losses achieved.')
        return True
  return False


def train_and_evaluate():
  """Run the full training and evaluation loop."""

  ac = AdversarialCrypto()
  init = tf.global_variables_initializer()

  with tf.Session() as s:
    s.run(init)
    print('# Batch size: ', FLAGS.batch_size)
    print('# %10s\t%20s\t%20s'%("Iter","Bob_Recon_Error","Eve_Recon_Error"))

    if train_until_thresh(s, ac):
      for _ in xrange(EVE_EXTRA_ROUNDS):
        s.run(ac.eve_optimizer)
      print('Loss after eve extra training:')
      doeval(s, ac, EVAL_BATCHES * 2, 0)
      for _ in xrange(NUMBER_OF_EVE_RESETS):
        print('Resetting Eve')
        s.run(ac.reset_eve_vars)
        eve_counter = 0
        for _ in xrange(RETRAIN_EVE_LOOPS):
          for _ in xrange(RETRAIN_EVE_ITERS):
            eve_counter += 1
            s.run(ac.eve_optimizer)
          doeval(s, ac, EVAL_BATCHES, eve_counter)
        doeval(s, ac, EVAL_BATCHES, eve_counter)


def main(unused_argv):
  # Exit more quietly with Ctrl-C.
  signal.signal(signal.SIGINT, signal.SIG_DFL)
  train_and_evaluate()


if __name__ == '__main__':
  tf.app.run()
