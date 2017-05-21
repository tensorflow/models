r"""A basic end-to-end LSTM language model for melody next-note prediction.

This model provides baselines for the application of language modeling to melody
generation. This code also serves as a working example for implementing a
language model in TensorFlow.

This code can be run out of the box with default flags on your local machine:
$ python minimal_lstm.py

There are approaches to building a recurrent model in TensorFlow:
1) Dynamic looping - implemented in tf.nn.dynamic_rnn:
   Use a dynamic loop to unroll the recurrent model at graph eval time.
   A while_loop control flow op is used under the hood, which has a circular
   connection to itself in the graph. Only one instance of the graph inside the
   loop is constructed.
2) Chunking - implemented in tf.nn.state_saving_rnn:
   Unroll a chunk of the recurrent model at graph construction time using a
   Python loop. Training examples are trained in chunks on the unrolled portion
   of the loop.

With each method there is a set of built in ops for constructing the pipeline
that reads training samples from disk, and for constructing the recurrent
pipeline.

================================================================================
The data
================================================================================

The training data is stored as tf.SequenceExample protos. Each training sample
is a sequence of notes given as one-hot vectors and a sequence of labels. In
language modeling the label is just the next correct item in the sequence, so
the labels are the input sequence shifted backward by one step. The last label
is a rest.

0 = no-event
1 = note-off event
2 = note-on event for pitch 48
3 = note-on event for pitch 49
...
37 = note-on event for pitch 83

A no-event continues the previous state, whether thats continuing to hold a note
on or continuing a rest.

The 'inputs' represent the current note, which is stored as a one-hot vector,
and 'labels' represent the next note, which is stored as an int. For example,
the first 9 16th notes of Twinkle Twinkle Little Star would be encoded:

14, 0, 14, 0, 21, 0, 21, 0, 23

So if batch_size = 1, and num_unroll = 8, batch.sequences['inputs'] would be
the tensor:

[[[0.0, 0.0, ... 1.0 (14th index), ... 0.0, 0.0],
  [1.0, 0.0, ... 0.0, 0.0],
  [0.0, 0.0, ... 1.0 (14th index), ... 0.0, 0.0],
  [1.0, 0.0, ... 0.0, 0.0],
  [0.0, 0.0, ... 1.0 (21st index), ... 0.0, 0.0],
  [1.0, 0.0, ... 0.0, 0.0],
  [0.0, 0.0, ... 1.0 (21st index), ... 0.0, 0.0],
  [1.0, 0.0, ... 0.0, 0.0]]]

And batch.sequences['labels'] would be the tensor:

[[0, 14, 0, 21, 0, 21, 0, 23]]

The first dimension of the tensors is the batch_size, and since
batch_size = 1 in this example, the batch only contains one sequence.

Heres a brief description of each method:

================================================================================
1) Dynamic looping method
================================================================================

The data reading pipeline is implemented with a tf.PaddingFIFOQueue.
Most of the complicated data reading code has been collected into a single
function, minimal_lstm_ops.dynamic_rnn_batch(), which returns a batch queue:

  (inputs, labels, lengths) = minimal_lstm_ops.dynamic_rnn_batch(*args)

The recurrent model is constructed with tf.nn.dynamic_rnn(). This code is inside
minimal_lstm_ops.dynamic_rnn():

  hidden, final_state = tf.nn.dynamic_rnn(
      cell,
      inputs,
      sequence_length=lengths,
      initial_state=initial_state)


================================================================================
2) Chunking method
================================================================================

The data reading pipeline is implemented with a tf.SequenceQueueingStateSaver.
The data reading code is collected into a single function,
minimal_lstm_ops.state_saving_rnn_batch(), which returns a
NextQueuedSequenceBatch object. This object provides access to batched data:

  batch = state_saving_rnn_batch(*args)
  inputs = batch.sequences['inputs']
  labels = batch.sequences['labels']

The recurrent model is constructed with tf.nn.state_saving_rnn(). This code is
inside minimal_lstm_ops.state_saving_rnn_inference().
The NextQueuedSequenceBatch object is also used by tf.nn.state_saving_rnn to
automatically save the LSTM outputs and hidden states between batches:

  outputs_by_time, _ = tf.nn.state_saving_rnn(cell, inputs_by_time, batch,
                                              LSTM_STATE_NAME)
================================================================================
"""

import collections
import logging
import minimal_lstm_ops
import os
import os.path
import shutil
import tensorflow as tf


FLAGS = tf.app.flags.FLAGS
tf.app.flags.DEFINE_string('master', 'local',
                           'BNS name of the TensorFlow master to use.')
tf.app.flags.DEFINE_integer('task', 0,
                            'Task id of the replica running the training.')
tf.app.flags.DEFINE_integer('ps_tasks', 0,
                            'Number of tasks in the ps job. If 0 no ps job is '
                            'used.')
tf.app.flags.DEFINE_string('sequence_examples_file', '/cns/iy-d/home/elliotwaite'
                           '/midiworld_melodies_pitch_48_to_83.recordio',
                           'The path of the RecordIO file containing the '
                           'tf.SequenceExample records for training.')
tf.app.flags.DEFINE_string('output_dir', '/tmp/minimal_lstm/',
                           'Optional path to a directory where checkpoint files and '
                           'event files for TensorBoard will be saved. The directory '
                           'will be created if it does not exist.')
tf.app.flags.DEFINE_bool('start_fresh', False,
                         'If true, the contents of output_dir will be deleted and a '
                         'new model will be trained with reinitialized variables.'
                         'If false, and output_dir contains a checkpoint file, that '
                         'checkpoint file will be used to restore variables.')
tf.app.flags.DEFINE_string('hparams', '',
                           'Comma separated list of name=value pairs. For '
                           'example, "batch_size=64,rnn_layer_sizes=[100,100],'
                           'use_dynamic_rnn=". To set something False, just '
                           'set it to the empty string: "use_dynamic_rnn=".')
tf.app.flags.DEFINE_integer('num_training_steps', 10000,
                            'The the number of training steps to take in this '
                            'training session.')
tf.app.flags.DEFINE_integer('summary_frequency', 10,
                            'A summary statement will be logged every '
                            '`summary_frequency` steps of training.')
tf.app.flags.DEFINE_integer('steps_to_average', 20,
                            'Accuracy averaged over the last `steps_to_average` steps '
                            'is reported.')


# TODO(danabo): Determine `NUM_CLASSES` from dataset at runtime.
# Number of classification classes. This is a property of the dataset.
NUM_CLASSES = 38


def default_hparams():
  return tf.HParams(use_dynamic_rnn=True,
                    batch_size=128,
                    lr=0.0002,
                    l2_reg=2.5e-5,
                    clip_norm=5,
                    initial_learning_rate=0.5,
                    decay_steps=1000,
                    decay_rate=0.85,
                    rnn_layer_sizes=[100],
                    skip_first_n_losses=32,  # dynamic_rnn param
                    num_unroll=16,  # state_saving_rnn param
                    one_hot_length=NUM_CLASSES,
                    exponentially_decay_learning_rate=True)


def make_graph_from_flags():
  """Construct the model from flags and return the graph.

  Constructs the graph using the FLAGS settings as parameters. Hyperparameters
  are given in the hparams flag as a string of comma seperated key value pairs.
  For example: "batch_size=64,rnn_layer_sizes=[100,100],use_dynamic_rnn="

  Returns:
    tf.Graph instance which contains the TF ops.
  """
  with tf.Graph().as_default() as graph:
    with tf.device(tf.ReplicaDeviceSetter(FLAGS.ps_tasks)):
      hparams = default_hparams()
      hparams = hparams.parse(FLAGS.hparams)
      logging.info('hparams = %s', hparams.values())

      with tf.variable_scope('rnn_model'):
        # Define the type of RNN cell to use.
        cell = minimal_lstm_ops.make_cell(hparams)

        # There are two ways to construct a variable length RNN in TensorFlow:
        # dynamic_rnn, and state_saving_rnn. The code below demonstrates how to
        # construct an end-to-end samples on disk to labels and logits pipeline
        # for both types of RNNs.
        if hparams.use_dynamic_rnn:
          # Construct dynamic_rnn reader and inference.

          # Get a batch queue.
          (melody_sequence,
           melody_labels,
           lengths) = minimal_lstm_ops.dynamic_rnn_batch(
               [FLAGS.sequence_examples_file], hparams)

          # Make inference graph. That is, inputs to logits.
          # Note: long sequences need a lot of memory on GPU because all forward
          # pass activations are needed to compute backprop. Additionally
          # multiple steps are computed simultaneously (the parts of each step
          # which don't depend on other steps). The `parallel_iterations`
          # and `swap_memory` arguments given here trade lower GPU memory
          # footprint for speed decrease.
          logits, _, _ = minimal_lstm_ops.dynamic_rnn_inference(
              melody_sequence, lengths, cell, hparams, zero_initial_state=True,
              parallel_iterations=1, swap_memory=True)

          # The first hparams.skip_first_n_losses steps of the logits tensor is
          # removed. Those first steps are given to the model as a primer during
          # generation. The model does not get penalized for incorrect
          # predictions in those first steps so the loss does not include those
          # logits.
          truncated_logits = logits[:, hparams.skip_first_n_losses:, :]

          # Reshape logits from [batch_size, sequence_length, one_hot_length] to
          # [batch_size * sequence_length, one_hot_length].
          flat_logits = tf.reshape(truncated_logits,
                                   [-1, hparams.one_hot_length])

          # Reshape labels from [batch_size, num_unroll] to
          # [batch_size * sequence_length]. Also truncate first steps to match
          # truncated_logits.
          flat_labels = tf.reshape(
              melody_labels[:, hparams.skip_first_n_losses:], [-1])
        else:
          # Construct state_saving_rnn reader and inference.

          # Get a NextQueuedSequenceBatch object. Note: If queue_capacity is too
          # small, the tf.Supervisor will sometimes hang when trying to close
          # the managed_session. This may be due to a bug in the way the
          # Supervisor closes threads.
          batch = minimal_lstm_ops.state_saving_rnn_batch(
              [FLAGS.sequence_examples_file], cell.state_size, hparams)

          # Make inference graph. That is, inputs to logits.
          logging.info('making state_saving_rnn')
          inputs = batch.sequences['inputs']
          logits = minimal_lstm_ops.state_saving_rnn_inference(
              inputs, cell, batch, hparams)

          # Reshape logits from [batch_size, num_unroll, one_hot_length] to
          # [batch_size * num_unroll, one_hot_length].
          flat_logits = tf.reshape(logits, [-1, hparams.one_hot_length])

          # Reshape labels from [batch_size, num_unroll] to
          # [batch_size * num_unroll].
          labels = batch.sequences['labels']
          flat_labels = tf.reshape(labels, [-1])

        # Compute loss and gradients for training, and accuracy for evaluation.
        loss = minimal_lstm_ops.cross_entropy_loss(flat_logits, flat_labels)
        training_op, learning_rate, global_step = minimal_lstm_ops.train_op(
            loss, hparams)
        accuracy = minimal_lstm_ops.eval_accuracy(flat_logits, flat_labels)

      tf.scalar_summary('loss', loss)
      tf.scalar_summary('learning_rate', learning_rate)
      tf.scalar_summary('accuracy', accuracy)
      tf.scalar_summary('global_step', global_step)

      tf.add_to_collection('hparams', hparams)
      tf.add_to_collection('logits', logits)
      tf.add_to_collection('loss', loss)
      tf.add_to_collection('learning_rate', learning_rate)
      tf.add_to_collection('accuracy', accuracy)
      tf.add_to_collection('training_op', training_op)
      tf.add_to_collection('global_step', global_step)

  return graph


def training_loop(graph):
  """A generator which runs training steps at each output.

  Args:
    graph: A tf.Graph object containing the model.

  Yields:
    A dict of training metrics, and runs FLAGS.summary_frequency training steps
      between each yield.
  """
  loss = graph.get_collection('loss')[0]
  accuracy = graph.get_collection('accuracy')[0]
  global_step = graph.get_collection('global_step')[0]
  learning_rate = graph.get_collection('learning_rate')[0]
  training_op = graph.get_collection('training_op')[0]

  # Prepare output_dir.
  if FLAGS.start_fresh:
    if os.path.exists(FLAGS.output_dir):
      shutil.rmtree(FLAGS.output_dir)
  if not os.path.exists(FLAGS.output_dir):
    os.makedirs(FLAGS.output_dir)

  # Run training loop.
  sv = tf.Supervisor(graph=graph,
                     logdir=FLAGS.output_dir,
                     is_chief=(FLAGS.task == 0),
                     save_model_secs=30,
                     global_step=global_step)
  session = sv.PrepareSession(FLAGS.master)
  sv.StartQueueRunners(session)
  step = 0

  logging.info('Starting training loop')
  try:
    accuracies = collections.deque(maxlen=FLAGS.steps_to_average)
    while not sv.ShouldStop() and step < FLAGS.num_training_steps:
      step += 1
      l, a, gs, lr, _ = session.run([loss, accuracy, global_step, learning_rate,
                                     training_op])

      accuracies.append(a)
      if step % FLAGS.summary_frequency == 0:
        avg_accuracy = sum(accuracies) / len(accuracies)
        logging.info('Session Step: %s - Global Step: %s - Loss: %.3f - Step '
                     'Accuracy: %.2f - Avg Accuracy (last %d summaries): '
                     '%.2f - Learning Rate: %f', '{:,}'.format(step),
                     '{:,}'.format(gs), l, a, FLAGS.steps_to_average,
                     avg_accuracy, lr)
        yield {'step': step, 'global_step': gs, 'loss': l, 'accuracy': a,
               'average_accuracy': avg_accuracy, 'learning_rate': lr}
    sv.saver.save(session, sv.save_path, global_step=sv.global_step)
  except tf.errors.OutOfRangeError as e:
    logging.warn('Got error reported to coordinator: %s', e)
  finally:
    try:
      sv.Stop()
    except RuntimeError as e:
      logging.warn('Got runtime error: %s', e)


def run():
  graph = make_graph_from_flags()
  for _ in training_loop(graph):
    pass


if __name__ == '__main__':
  run()

