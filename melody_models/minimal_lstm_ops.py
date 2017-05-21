"""Graph building functions for state_saving_rnn and dynamic_rnn LSTMs.

Each graph component is produced by a seperate function.
"""

import tensorflow as tf


LSTM_STATE_NAME = 'lstm'


def input_sequence_example(file_list, hparams):
  """Deserializes SequenceExamples from RecordIO.

  Args:
    file_list: List of recordIO files of SequenceExamples.
    hparams: HParams instance containing model hyperparameters.

  Returns:
    seq_key: Key of SequenceExample as a string.
    context: Context of SequenceExample as dictionary key -> Tensor.
    sequence: Sequence of SequenceExample as dictionary key -> Tensor.
  """
  file_queue = tf.train.string_input_producer(file_list)
  reader = tf.RecordIOReader()
  seq_key, serialized_example = reader.read(file_queue)

  sequence_features = {
      'inputs': tf.FixedLenSequenceFeature(shape=[hparams.one_hot_length],
                                           dtype=tf.float32),
      'labels': tf.FixedLenSequenceFeature(shape=[],
                                           dtype=tf.int64)
  }

  context, sequence = tf.parse_single_sequence_example(
      serialized_example,
      sequence_features=sequence_features)
  return seq_key, context, sequence


def state_saving_rnn_batch(file_list, cell_state_size, hparams):
  """Initializes the SequenceQueueingStateSaver (SQSS).

  Args:
    file_list: A list of strings. Each string should be a path to a
        RecordIO file containing tf.SequenceExample records.
    cell_state_size: An int. The size of state used by the rnn cell. Easiest
        way to obtain this number is to instantiate an RNNCell first, and then
        call cell.state_size.
    hparams: HParams instance containing model hyperparameters.

  Returns:
    The `NextQueuedSequenceBatch` providing access to batched output data.
    Also provides access to the `state` and `save_state` methods used by the
    state_saving_rnn.
  """
  seq_key, context, sequences = input_sequence_example(file_list, hparams)
  initial_state_values = tf.zeros([cell_state_size], dtype=tf.float32)
  initial_states = {LSTM_STATE_NAME: initial_state_values}
  stateful_reader = tf.SequenceQueueingStateSaver(
      hparams.batch_size,
      hparams.num_unroll,
      input_length=tf.shape(sequences['inputs'])[0],
      input_key=seq_key,
      input_sequences=sequences,
      input_context=context,
      initial_states=initial_states,
      capacity=hparams.batch_size * 100,
      allow_small_batch=True)
  queue_runner = tf.queue_runner.QueueRunner(stateful_reader.barrier,
                                             [stateful_reader.prefetch_op])
  tf.train.add_queue_runner(queue_runner)
  batch = stateful_reader.next_batch
  return batch


def state_saving_rnn_inference(inputs, cell, batch, hparams):
  """Creates possibly layered LSTM cells with a linear projection layer.

  Uses state_saving_rnn which unrolls for a fixed number of steps. Samples
  are chunked into sequences of that fixed length and fed into the model at
  seperate training steps.

  Let num_unroll = chunk size.

  Args:
    inputs: A tensor of shape [batch_size, num_unroll, input_size].
    cell: An RNNCell instance.
    batch: A NextQueuedSequenceBatch instance.
    hparams: HParams instance containing model hyperparameters.

  Returns:
    Logits of shape [batch_size, num_unroll, one_hot_length]
  """
  # Transpose inputs from [batch_size, num_unroll, input_size] to
  # [num_unroll, batch_size, input_size], then unpack it into a
  # num_unroll length array of [batch_size, input_size] tensors.
  inputs_by_time = tf.unpack(tf.transpose(inputs, [1, 0, 2]))

  # Run inputs_by_time through the rnn. Any unfinished sequences will have
  # their outputs and hidden states automatically saved for the next batch.
  # The returned outputs_by_time is a num_unroll length array of
  # [batch_size, lstm_num_units] tensors.
  outputs_by_time, _ = tf.nn.state_saving_rnn(cell, inputs_by_time, batch,
                                              LSTM_STATE_NAME)

  # Concat outputs_by_time along the second dimension to get a tensor of shape:
  # [batch_size, num_unroll, one_hot_length]
  outputs = tf.concat(1, outputs_by_time)

  # create projection layer to logits.
  outputs_flat = tf.reshape(outputs, [-1, hparams.rnn_layer_sizes[-1]])
  logits_flat = tf.contrib.layers.linear(outputs_flat, hparams.one_hot_length)
  logits = tf.reshape(logits_flat,
                      [hparams.batch_size, -1, hparams.one_hot_length])

  return logits


def dynamic_rnn_batch(file_list, hparams):
  """Reads batches of SequenceExamples from recordIO and pads them.

  Can deal with variable length SequenceExamples by padding each batch to the
  length of the longest sequence with zeros.

  Args:
    file_list: List of recordIO files of SequenceExamples.
    hparams: HParams instance containing model hyperparameters.

  Returns:
    inputs: Tensor of shape [batch_size, examples_per_sequence, one_hot_length]
        with floats indicating the next note event.
    labels: Tensor of shape [batch_size, examples_per_sequence] with int64s
        indicating the prediction for next note event given the notes up to this
        point in the inputs sequence.
    lengths: Tensor vector of shape [batch_size] with the length of the
        SequenceExamples before padding.
  """
  _, _, sequences = input_sequence_example(file_list, hparams)

  length = tf.shape(sequences['inputs'])[0]

  queue = tf.PaddingFIFOQueue(
      capacity=1000,
      dtypes=[tf.float32, tf.int64, tf.int32],
      shapes=[(None, hparams.one_hot_length), (None,), ()])

  # The number of threads for enqueuing.
  num_threads = 4
  enqueue_ops = [queue.enqueue([sequences['inputs'],
                                sequences['labels'],
                                length])] * num_threads
  tf.train.add_queue_runner(tf.queue_runner.QueueRunner(queue, enqueue_ops))
  return queue.dequeue_many(hparams.batch_size)


def dynamic_rnn_inference(inputs, lengths, cell, hparams,
                          zero_initial_state=True, parallel_iterations=1,
                          swap_memory=True):
  """Creates possibly layered LSTM cells with a linear projection layer.

  Uses dynamic_rnn which dynamically unrolls for each minibatch allowing truely
  variable length minibatches.

  Args:
    inputs: Tensor of shape [batch_size, batch_sequence_length, one_hot_length).
    lengths: Tensor of shape [batch_size] with the length of the
        SequenceExample before padding.
    cell: An RNNCell instance.
    hparams: HParams instance containing model hyperparameters.
    zero_initial_state: If true, a constant tensor of 0s is used as the initial
        RNN state. If false, a placeholder is created to hold the initial state.
    parallel_iterations: The number of iterations to run in parallel. Those
        operations which do not have any temporal dependency
        and can be run in parallel, will be. This parameter trades off
        time for space. Values >> 1 use more memory but take less time,
        while smaller values use less memory but computations take longer.
    swap_memory: Transparently swap the tensors produced in forward inference
        but needed for back prop from GPU to CPU. This allows training RNNs
        which would typically not fit on a single GPU, with very minimal (or no)
        performance penalty.

  Returns:
    logits: Output logits. A tensor of shape
        [batch_size, batch_sequence_length, one_hot_length].
    initial_state: The tensor fed into dynamic_rnn as the initial state. When
        zero_initial_state is true, this will be the placeholder.
    final_state: The final internal state after computing the number of steps
        given in lengths for each sample. Same shape as cell.state_size.
  """
  if zero_initial_state:
    initial_state = cell.zero_state(batch_size=hparams.batch_size,
                                    dtype=tf.float32)
  else:
    initial_state = tf.placeholder(tf.float32,
                                   [hparams.batch_size, cell.state_size])

  outputs, final_state = tf.nn.dynamic_rnn(
      cell,
      inputs,
      sequence_length=lengths,
      initial_state=initial_state,
      swap_memory=swap_memory,
      parallel_iterations=parallel_iterations)

  # create projection layer to logits.
  outputs_flat = tf.reshape(outputs, [-1, hparams.rnn_layer_sizes[-1]])
  logits_flat = tf.contrib.layers.linear(outputs_flat, hparams.one_hot_length)
  logits = tf.reshape(logits_flat,
                      [hparams.batch_size, -1, hparams.one_hot_length])

  return logits, initial_state, final_state


def make_cell(hparams):
  """Instantiates an RNNCell object.

  Will construct an appropriate RNN cell given hyperparameters. This will
  specifically be a stack of LSTM cells. The height of the stack is specified in
  hparams.

  Args:
    hparams: HParams instance containing model hyperparameters.

  Returns:
    RNNCell instance.
  """
  lstm_layers = [
      tf.nn.rnn_cell.LSTMCell(num_units=layer_size)
      for layer_size in hparams.rnn_layer_sizes
  ]
  multi_cell = tf.nn.rnn_cell.MultiRNNCell(lstm_layers)
  return multi_cell


def cross_entropy_loss(predictions, labels):
  """Computes the cross entropy between the normalized logits and the labels.

  Args:
    predictions: Output tensor of a linear layer of shape
      [batch * batch_sequence_length, one_hot_length].
    labels: tensor of ints between 0 and one_hot_length-1 of shape
      [batch * batch_sequence_length].

  Returns:
    The cross entropy between the labels and the predictions p_0, p_1, ...
    normalized as p_normalized_i = ln(e^{p_i} / sum_j e^{p_j}).
  """

  losses = tf.nn.sparse_softmax_cross_entropy_with_logits(predictions, labels)
  return tf.reduce_sum(losses)


def train_op(loss, hparams):
  """Uses a gradient descent optimizer to minimize loss.

  Gradient descent is applied to the loss function with an exponentially
  decreasing learning rate.

  Args:
    loss: loss tensor to minimize.
    hparams: HParams instance containing model hyperparameters.

  Returns:
    training_op: An op that performs weight updates on the model.
    learning_rate: An op that decays learning rate, if that option is set in
        `hparams`.
    global_step: An op that increments the global step counter.
  """
  global_step = tf.Variable(0, trainable=False)
  if hparams.exponentially_decay_learning_rate:
    learning_rate = tf.train.exponential_decay(hparams.initial_learning_rate,
                                               global_step,
                                               hparams.decay_steps,
                                               hparams.decay_rate,
                                               staircase=True,
                                               name='learning_rate')
  else:
    learning_rate = tf.Variable(hparams.initial_learning_rate, trainable=False)
  opt = tf.train.AdagradOptimizer(learning_rate)
  params = tf.trainable_variables()
  gradients = tf.gradients(loss, params)
  clipped_gradients, _ = tf.clip_by_global_norm(gradients, hparams.clip_norm)
  training_op = opt.apply_gradients(zip(clipped_gradients, params),
                                    global_step=global_step)

  return training_op, learning_rate, global_step


def eval_accuracy(predictions, labels):
  """Evaluates the accuracy of the predictions.

  Checks how often the prediciton with the highest weight is correct on average.

  Args:
    predictions: Output tensor of a linear layer of shape
      [batch * batch_sequence_length, one_hot_length].
    labels: tensor of ints between 0 and one_hot_length-1 of shape
      [batch * batch_sequence_length].

  Returns:
    The precision of the highest weighted predicted class.
  """
  correct_predictions = tf.nn.in_top_k(predictions, labels, 1)
  return tf.reduce_mean(tf.to_float(correct_predictions))

