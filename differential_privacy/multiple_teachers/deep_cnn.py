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

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from datetime import datetime
import math
import numpy as np
import tensorflow as tf
import time

from differential_privacy.multiple_teachers import utils

FLAGS = tf.app.flags.FLAGS

# Basic model parameters.
tf.app.flags.DEFINE_integer('dropout_seed', 123, """seed for dropout.""")
tf.app.flags.DEFINE_integer('batch_size', 128, """Nb of images in a batch.""")
tf.app.flags.DEFINE_integer('epochs_per_decay', 350, """Nb epochs per decay""")
tf.app.flags.DEFINE_integer('learning_rate', 5, """100 * learning rate""")
tf.app.flags.DEFINE_boolean('log_device_placement', False, """see TF doc""")


# Constants describing the training process.
MOVING_AVERAGE_DECAY = 0.9999     # The decay to use for the moving average.
LEARNING_RATE_DECAY_FACTOR = 0.1  # Learning rate decay factor.


def _variable_on_cpu(name, shape, initializer):
  """Helper to create a Variable stored on CPU memory.

  Args:
    name: name of the variable
    shape: list of ints
    initializer: initializer for Variable

  Returns:
    Variable Tensor
  """
  with tf.device('/cpu:0'):
    var = tf.get_variable(name, shape, initializer=initializer)
  return var


def _variable_with_weight_decay(name, shape, stddev, wd):
  """Helper to create an initialized Variable with weight decay.

  Note that the Variable is initialized with a truncated normal distribution.
  A weight decay is added only if one is specified.

  Args:
    name: name of the variable
    shape: list of ints
    stddev: standard deviation of a truncated Gaussian
    wd: add L2Loss weight decay multiplied by this float. If None, weight
        decay is not added for this Variable.

  Returns:
    Variable Tensor
  """
  var = _variable_on_cpu(name, shape,
                         tf.truncated_normal_initializer(stddev=stddev))
  if wd is not None:
    weight_decay = tf.mul(tf.nn.l2_loss(var), wd, name='weight_loss')
    tf.add_to_collection('losses', weight_decay)
  return var


def inference(images, dropout=False):
  """Build the CNN model.
  Args:
    images: Images returned from distorted_inputs() or inputs().
    dropout: Boolean controling whether to use dropout or not
  Returns:
    Logits
  """
  if FLAGS.dataset == 'mnist':
    first_conv_shape = [5, 5, 1, 64]
  else:
    first_conv_shape = [5, 5, 3, 64]

  # conv1
  with tf.variable_scope('conv1') as scope:
    kernel = _variable_with_weight_decay('weights', 
                                         shape=first_conv_shape,
                                         stddev=1e-4, 
                                         wd=0.0)
    conv = tf.nn.conv2d(images, kernel, [1, 1, 1, 1], padding='SAME')
    biases = _variable_on_cpu('biases', [64], tf.constant_initializer(0.0))
    bias = tf.nn.bias_add(conv, biases)
    conv1 = tf.nn.relu(bias, name=scope.name)
    if dropout:
      conv1 = tf.nn.dropout(conv1, 0.3, seed=FLAGS.dropout_seed)


  # pool1
  pool1 = tf.nn.max_pool(conv1, 
                         ksize=[1, 3, 3, 1], 
                         strides=[1, 2, 2, 1],
                         padding='SAME', 
                         name='pool1')
  
  # norm1
  norm1 = tf.nn.lrn(pool1, 
                    4, 
                    bias=1.0, 
                    alpha=0.001 / 9.0, 
                    beta=0.75,
                    name='norm1')

  # conv2
  with tf.variable_scope('conv2') as scope:
    kernel = _variable_with_weight_decay('weights', 
                                         shape=[5, 5, 64, 128],
                                         stddev=1e-4, 
                                         wd=0.0)
    conv = tf.nn.conv2d(norm1, kernel, [1, 1, 1, 1], padding='SAME')
    biases = _variable_on_cpu('biases', [128], tf.constant_initializer(0.1))
    bias = tf.nn.bias_add(conv, biases)
    conv2 = tf.nn.relu(bias, name=scope.name)
    if dropout:
      conv2 = tf.nn.dropout(conv2, 0.3, seed=FLAGS.dropout_seed)


  # norm2
  norm2 = tf.nn.lrn(conv2, 
                    4, 
                    bias=1.0, 
                    alpha=0.001 / 9.0, 
                    beta=0.75,
                    name='norm2')
  
  # pool2
  pool2 = tf.nn.max_pool(norm2, 
                         ksize=[1, 3, 3, 1],
                         strides=[1, 2, 2, 1], 
                         padding='SAME', 
                         name='pool2')

  # local3
  with tf.variable_scope('local3') as scope:
    # Move everything into depth so we can perform a single matrix multiply.
    reshape = tf.reshape(pool2, [FLAGS.batch_size, -1])
    dim = reshape.get_shape()[1].value
    weights = _variable_with_weight_decay('weights', 
                                          shape=[dim, 384],
                                          stddev=0.04, 
                                          wd=0.004)
    biases = _variable_on_cpu('biases', [384], tf.constant_initializer(0.1))
    local3 = tf.nn.relu(tf.matmul(reshape, weights) + biases, name=scope.name)
    if dropout:
      local3 = tf.nn.dropout(local3, 0.5, seed=FLAGS.dropout_seed)

  # local4
  with tf.variable_scope('local4') as scope:
    weights = _variable_with_weight_decay('weights', 
                                          shape=[384, 192],
                                          stddev=0.04, 
                                          wd=0.004)
    biases = _variable_on_cpu('biases', [192], tf.constant_initializer(0.1))
    local4 = tf.nn.relu(tf.matmul(local3, weights) + biases, name=scope.name)
    if dropout:
      local4 = tf.nn.dropout(local4, 0.5, seed=FLAGS.dropout_seed)

  # compute logits
  with tf.variable_scope('softmax_linear') as scope:
    weights = _variable_with_weight_decay('weights', 
                                          [192, FLAGS.nb_labels],
                                          stddev=1/192.0, 
                                          wd=0.0)
    biases = _variable_on_cpu('biases', 
                              [FLAGS.nb_labels],
                              tf.constant_initializer(0.0))
    logits = tf.add(tf.matmul(local4, weights), biases, name=scope.name)

  return logits


def inference_deeper(images, dropout=False):
  """Build a deeper CNN model.
  Args:
    images: Images returned from distorted_inputs() or inputs().
    dropout: Boolean controling whether to use dropout or not
  Returns:
    Logits
  """
  if FLAGS.dataset == 'mnist':
    first_conv_shape = [3, 3, 1, 96]
  else:
    first_conv_shape = [3, 3, 3, 96]

  # conv1
  with tf.variable_scope('conv1') as scope:
    kernel = _variable_with_weight_decay('weights',
                                         shape=first_conv_shape,
                                         stddev=0.05,
                                         wd=0.0)
    conv = tf.nn.conv2d(images, kernel, [1, 1, 1, 1], padding='SAME')
    biases = _variable_on_cpu('biases', [96], tf.constant_initializer(0.0))
    bias = tf.nn.bias_add(conv, biases)
    conv1 = tf.nn.relu(bias, name=scope.name)

  # conv2
  with tf.variable_scope('conv2') as scope:
    kernel = _variable_with_weight_decay('weights',
                                         shape=[3, 3, 96, 96],
                                         stddev=0.05,
                                         wd=0.0)
    conv = tf.nn.conv2d(conv1, kernel, [1, 1, 1, 1], padding='SAME')
    biases = _variable_on_cpu('biases', [96], tf.constant_initializer(0.0))
    bias = tf.nn.bias_add(conv, biases)
    conv2 = tf.nn.relu(bias, name=scope.name)

  # conv3
  with tf.variable_scope('conv3') as scope:
    kernel = _variable_with_weight_decay('weights',
                                         shape=[3, 3, 96, 96],
                                         stddev=0.05,
                                         wd=0.0)
    conv = tf.nn.conv2d(conv2, kernel, [1, 2, 2, 1], padding='SAME')
    biases = _variable_on_cpu('biases', [96], tf.constant_initializer(0.0))
    bias = tf.nn.bias_add(conv, biases)
    conv3 = tf.nn.relu(bias, name=scope.name)
    if dropout:
      conv3 = tf.nn.dropout(conv3, 0.5, seed=FLAGS.dropout_seed)

  # conv4
  with tf.variable_scope('conv4') as scope:
    kernel = _variable_with_weight_decay('weights',
                                         shape=[3, 3, 96, 192],
                                         stddev=0.05,
                                         wd=0.0)
    conv = tf.nn.conv2d(conv3, kernel, [1, 1, 1, 1], padding='SAME')
    biases = _variable_on_cpu('biases', [192], tf.constant_initializer(0.0))
    bias = tf.nn.bias_add(conv, biases)
    conv4 = tf.nn.relu(bias, name=scope.name)

  # conv5
  with tf.variable_scope('conv5') as scope:
    kernel = _variable_with_weight_decay('weights',
                                         shape=[3, 3, 192, 192],
                                         stddev=0.05,
                                         wd=0.0)
    conv = tf.nn.conv2d(conv4, kernel, [1, 1, 1, 1], padding='SAME')
    biases = _variable_on_cpu('biases', [192], tf.constant_initializer(0.0))
    bias = tf.nn.bias_add(conv, biases)
    conv5 = tf.nn.relu(bias, name=scope.name)

  # conv6
  with tf.variable_scope('conv6') as scope:
    kernel = _variable_with_weight_decay('weights',
                                         shape=[3, 3, 192, 192],
                                         stddev=0.05,
                                         wd=0.0)
    conv = tf.nn.conv2d(conv5, kernel, [1, 2, 2, 1], padding='SAME')
    biases = _variable_on_cpu('biases', [192], tf.constant_initializer(0.0))
    bias = tf.nn.bias_add(conv, biases)
    conv6 = tf.nn.relu(bias, name=scope.name)
    if dropout:
      conv6 = tf.nn.dropout(conv6, 0.5, seed=FLAGS.dropout_seed)


  # conv7
  with tf.variable_scope('conv7') as scope:
    kernel = _variable_with_weight_decay('weights',
                                         shape=[5, 5, 192, 192],
                                         stddev=1e-4,
                                         wd=0.0)
    conv = tf.nn.conv2d(conv6, kernel, [1, 1, 1, 1], padding='SAME')
    biases = _variable_on_cpu('biases', [192], tf.constant_initializer(0.1))
    bias = tf.nn.bias_add(conv, biases)
    conv7 = tf.nn.relu(bias, name=scope.name)


  # local1
  with tf.variable_scope('local1') as scope:
    # Move everything into depth so we can perform a single matrix multiply.
    reshape = tf.reshape(conv7, [FLAGS.batch_size, -1])
    dim = reshape.get_shape()[1].value
    weights = _variable_with_weight_decay('weights',
                                          shape=[dim, 192],
                                          stddev=0.05,
                                          wd=0)
    biases = _variable_on_cpu('biases', [192], tf.constant_initializer(0.1))
    local1 = tf.nn.relu(tf.matmul(reshape, weights) + biases, name=scope.name)

  # local2
  with tf.variable_scope('local2') as scope:
    weights = _variable_with_weight_decay('weights',
                                          shape=[192, 192],
                                          stddev=0.05,
                                          wd=0)
    biases = _variable_on_cpu('biases', [192], tf.constant_initializer(0.1))
    local2 = tf.nn.relu(tf.matmul(local1, weights) + biases, name=scope.name)
    if dropout:
      local2 = tf.nn.dropout(local2, 0.5, seed=FLAGS.dropout_seed)

  # compute logits
  with tf.variable_scope('softmax_linear') as scope:
    weights = _variable_with_weight_decay('weights',
                                          [192, FLAGS.nb_labels],
                                          stddev=0.05,
                                          wd=0.0)
    biases = _variable_on_cpu('biases',
                              [FLAGS.nb_labels],
                              tf.constant_initializer(0.0))
    logits = tf.add(tf.matmul(local2, weights), biases, name=scope.name)

  return logits


def loss_fun(logits, labels):
  """Add L2Loss to all the trainable variables.

  Add summary for "Loss" and "Loss/avg".
  Args:
    logits: Logits from inference().
    labels: Labels from distorted_inputs or inputs(). 1-D tensor
            of shape [batch_size]
    distillation: if set to True, use probabilities and not class labels to
                  compute softmax loss

  Returns:
    Loss tensor of type float.
  """

  # Calculate the cross entropy between labels and predictions
  labels = tf.cast(labels, tf.int64)
  cross_entropy = tf.nn.sparse_softmax_cross_entropy_with_logits(
      logits, labels, name='cross_entropy_per_example')

  # Calculate the average cross entropy loss across the batch.
  cross_entropy_mean = tf.reduce_mean(cross_entropy, name='cross_entropy')

  # Add to TF collection for losses
  tf.add_to_collection('losses', cross_entropy_mean)

  # The total loss is defined as the cross entropy loss plus all of the weight
  # decay terms (L2 loss).
  return tf.add_n(tf.get_collection('losses'), name='total_loss')


def moving_av(total_loss):
  """
  Generates moving average for all losses

  Args:
    total_loss: Total loss from loss().
  Returns:
    loss_averages_op: op for generating moving averages of losses.
  """
  # Compute the moving average of all individual losses and the total loss.
  loss_averages = tf.train.ExponentialMovingAverage(0.9, name='avg')
  losses = tf.get_collection('losses')
  loss_averages_op = loss_averages.apply(losses + [total_loss])

  return loss_averages_op


def train_op_fun(total_loss, global_step):
  """Train model.

  Create an optimizer and apply to all trainable variables. Add moving
  average for all trainable variables.

  Args:
    total_loss: Total loss from loss().
    global_step: Integer Variable counting the number of training steps
      processed.
  Returns:
    train_op: op for training.
  """
  # Variables that affect learning rate.
  nb_ex_per_train_epoch = int(60000 / FLAGS.nb_teachers)
  
  num_batches_per_epoch = nb_ex_per_train_epoch / FLAGS.batch_size
  decay_steps = int(num_batches_per_epoch * FLAGS.epochs_per_decay)

  initial_learning_rate = float(FLAGS.learning_rate) / 100.0

  # Decay the learning rate exponentially based on the number of steps.
  lr = tf.train.exponential_decay(initial_learning_rate,
                                  global_step,
                                  decay_steps,
                                  LEARNING_RATE_DECAY_FACTOR,
                                  staircase=True)
  tf.scalar_summary('learning_rate', lr)

  # Generate moving averages of all losses and associated summaries.
  loss_averages_op = moving_av(total_loss)

  # Compute gradients.
  with tf.control_dependencies([loss_averages_op]):
    opt = tf.train.GradientDescentOptimizer(lr)
    grads = opt.compute_gradients(total_loss)

  # Apply gradients.
  apply_gradient_op = opt.apply_gradients(grads, global_step=global_step)

  # Add histograms for trainable variables.
  for var in tf.trainable_variables():
    tf.histogram_summary(var.op.name, var)

  # Track the moving averages of all trainable variables.
  variable_averages = tf.train.ExponentialMovingAverage(
      MOVING_AVERAGE_DECAY, global_step)
  variables_averages_op = variable_averages.apply(tf.trainable_variables())

  with tf.control_dependencies([apply_gradient_op, variables_averages_op]):
    train_op = tf.no_op(name='train')

  return train_op


def _input_placeholder():
  """
  This helper function declares a TF placeholder for the graph input data
  :return: TF placeholder for the graph input data
  """
  if FLAGS.dataset == 'mnist':
    image_size = 28
    num_channels = 1
  else:
    image_size = 32
    num_channels = 3

  # Declare data placeholder
  train_node_shape = (FLAGS.batch_size, image_size, image_size, num_channels)
  return tf.placeholder(tf.float32, shape=train_node_shape)


def train(images, labels, ckpt_path, dropout=False):
  """
  This function contains the loop that actually trains the model.
  :param images: a numpy array with the input data
  :param labels: a numpy array with the output labels
  :param ckpt_path: a path (including name) where model checkpoints are saved
  :param dropout: Boolean, whether to use dropout or not
  :return: True if everything went well
  """

  # Check training data
  assert len(images) == len(labels)
  assert images.dtype == np.float32
  assert labels.dtype == np.int32

  # Set default TF graph
  with tf.Graph().as_default():
    global_step = tf.Variable(0, trainable=False)

    # Declare data placeholder
    train_data_node = _input_placeholder()

    # Create a placeholder to hold labels
    train_labels_shape = (FLAGS.batch_size,)
    train_labels_node = tf.placeholder(tf.int32, shape=train_labels_shape)

    print("Done Initializing Training Placeholders")

    # Build a Graph that computes the logits predictions from the placeholder
    if FLAGS.deeper:
      logits = inference_deeper(train_data_node, dropout=dropout)
    else:
      logits = inference(train_data_node, dropout=dropout)

    # Calculate loss
    loss = loss_fun(logits, train_labels_node)

    # Build a Graph that trains the model with one batch of examples and
    # updates the model parameters.
    train_op = train_op_fun(loss, global_step)

    # Create a saver.
    saver = tf.train.Saver(tf.all_variables())

    print("Graph constructed and saver created")

    # Build an initialization operation to run below.
    init = tf.initialize_all_variables()

    # Create and init sessions
    sess = tf.Session(config=tf.ConfigProto(log_device_placement=FLAGS.log_device_placement)) #NOLINT(long-line)
    sess.run(init)

    print("Session ready, beginning training loop")

    # Initialize the number of batches
    data_length = len(images)
    nb_batches = math.ceil(data_length / FLAGS.batch_size)

    for step in xrange(FLAGS.max_steps):
      # for debug, save start time
      start_time = time.time()

      # Current batch number
      batch_nb = step % nb_batches

      # Current batch start and end indices
      start, end = utils.batch_indices(batch_nb, data_length, FLAGS.batch_size)

      # Prepare dictionnary to feed the session with
      feed_dict = {train_data_node: images[start:end],
                   train_labels_node: labels[start:end]}

      # Run training step
      _, loss_value = sess.run([train_op, loss], feed_dict=feed_dict)

      # Compute duration of training step
      duration = time.time() - start_time

      # Sanity check
      assert not np.isnan(loss_value), 'Model diverged with loss = NaN'

      # Echo loss once in a while
      if step % 100 == 0:
        num_examples_per_step = FLAGS.batch_size
        examples_per_sec = num_examples_per_step / duration
        sec_per_batch = float(duration)

        format_str = ('%s: step %d, loss = %.2f (%.1f examples/sec; %.3f '
                      'sec/batch)')
        print (format_str % (datetime.now(), step, loss_value,
                             examples_per_sec, sec_per_batch))

      # Save the model checkpoint periodically.
      if step % 1000 == 0 or (step + 1) == FLAGS.max_steps:
        saver.save(sess, ckpt_path, global_step=step)

  return True


def softmax_preds(images, ckpt_path, return_logits=False):
  """
  Compute softmax activations (probabilities) with the model saved in the path
  specified as an argument
  :param images: a np array of images
  :param ckpt_path: a TF model checkpoint
  :param logits: if set to True, return logits instead of probabilities
  :return: probabilities (or logits if logits is set to True)
  """
  # Compute nb samples and deduce nb of batches
  data_length = len(images)
  nb_batches = math.ceil(len(images) / FLAGS.batch_size)

  # Declare data placeholder
  train_data_node = _input_placeholder()

  # Build a Graph that computes the logits predictions from the placeholder
  if FLAGS.deeper:
    logits = inference_deeper(train_data_node)
  else:
    logits = inference(train_data_node)

  if return_logits:
    # We are returning the logits directly (no need to apply softmax)
    output = logits
  else:
    # Add softmax predictions to graph: will return probabilities
    output = tf.nn.softmax(logits)

  # Restore the moving average version of the learned variables for eval.
  variable_averages = tf.train.ExponentialMovingAverage(MOVING_AVERAGE_DECAY)
  variables_to_restore = variable_averages.variables_to_restore()
  saver = tf.train.Saver(variables_to_restore)

  # Will hold the result
  preds = np.zeros((data_length, FLAGS.nb_labels), dtype=np.float32)

  # Create TF session
  with tf.Session() as sess:
    # Restore TF session from checkpoint file
    saver.restore(sess, ckpt_path)

    # Parse data by batch
    for batch_nb in xrange(0, int(nb_batches+1)):
      # Compute batch start and end indices
      start, end = utils.batch_indices(batch_nb, data_length, FLAGS.batch_size)

      # Prepare feed dictionary
      feed_dict = {train_data_node: images[start:end]}

      # Run session ([0] because run returns a batch with len 1st dim == 1)
      preds[start:end, :] = sess.run([output], feed_dict=feed_dict)[0]

  # Reset graph to allow multiple calls
  tf.reset_default_graph()

  return preds


