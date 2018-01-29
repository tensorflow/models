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
"""Example differentially private trainer and evaluator for MNIST.
"""
from __future__ import division

import json
import os
import sys
import time

import numpy as np
from six.moves import xrange
import tensorflow as tf

from differential_privacy.dp_sgd.dp_optimizer import dp_optimizer
from differential_privacy.dp_sgd.dp_optimizer import dp_pca
from differential_privacy.dp_sgd.dp_optimizer import sanitizer
from differential_privacy.dp_sgd.dp_optimizer import utils
from differential_privacy.privacy_accountant.tf import accountant

# parameters for the training
tf.flags.DEFINE_integer("batch_size", 600,
                        "The training batch size.")
tf.flags.DEFINE_integer("batches_per_lot", 1,
                        "Number of batches per lot.")
# Together, batch_size and batches_per_lot determine lot_size.
tf.flags.DEFINE_integer("num_training_steps", 50000,
                        "The number of training steps."
                        "This counts number of lots.")

tf.flags.DEFINE_bool("randomize", True,
                     "If true, randomize the input data; otherwise use a fixed "
                     "seed and non-randomized input.")
tf.flags.DEFINE_bool("freeze_bottom_layers", False,
                     "If true, only train on the logit layer.")
tf.flags.DEFINE_bool("save_mistakes", False,
                     "If true, save the mistakes made during testing.")
tf.flags.DEFINE_float("lr", 0.05, "start learning rate")
tf.flags.DEFINE_float("end_lr", 0.05, "end learning rate")
tf.flags.DEFINE_float("lr_saturate_epochs", 0,
                      "learning rate saturate epochs; set to 0 for a constant "
                      "learning rate of --lr.")

# For searching parameters
tf.flags.DEFINE_integer("projection_dimensions", 60,
                        "PCA projection dimensions, or 0 for no projection.")
tf.flags.DEFINE_integer("num_hidden_layers", 1,
                        "Number of hidden layers in the network")
tf.flags.DEFINE_integer("hidden_layer_num_units", 1000,
                        "Number of units per hidden layer")
tf.flags.DEFINE_float("default_gradient_l2norm_bound", 4.0, "norm clipping")
tf.flags.DEFINE_integer("num_conv_layers", 0,
                        "Number of convolutional layers to use.")

tf.flags.DEFINE_string("training_data_path",
                       "/tmp/mnist/mnist_train.tfrecord",
                       "Location of the training data.")
tf.flags.DEFINE_string("eval_data_path",
                       "/tmp/mnist/mnist_test.tfrecord",
                       "Location of the eval data.")
tf.flags.DEFINE_integer("eval_steps", 10,
                        "Evaluate the model every eval_steps")

# Parameters for privacy spending. We allow linearly varying eps during
# training.
tf.flags.DEFINE_string("accountant_type", "Moments", "Moments, Amortized.")

# Flags that control privacy spending during training.
tf.flags.DEFINE_float("eps", 1.0,
                      "Start privacy spending for one epoch of training, "
                      "used if accountant_type is Amortized.")
tf.flags.DEFINE_float("end_eps", 1.0,
                      "End privacy spending for one epoch of training, "
                      "used if accountant_type is Amortized.")
tf.flags.DEFINE_float("eps_saturate_epochs", 0,
                      "Stop varying epsilon after eps_saturate_epochs. Set to "
                      "0 for constant eps of --eps. "
                      "Used if accountant_type is Amortized.")
tf.flags.DEFINE_float("delta", 1e-5,
                      "Privacy spending for training. Constant through "
                      "training, used if accountant_type is Amortized.")
tf.flags.DEFINE_float("sigma", 4.0,
                      "Noise sigma, used only if accountant_type is Moments")


# Flags that control privacy spending for the pca projection
# (only used if --projection_dimensions > 0).
tf.flags.DEFINE_float("pca_eps", 0.5,
                      "Privacy spending for PCA, used if accountant_type is "
                      "Amortized.")
tf.flags.DEFINE_float("pca_delta", 0.005,
                      "Privacy spending for PCA, used if accountant_type is "
                      "Amortized.")

tf.flags.DEFINE_float("pca_sigma", 7.0,
                      "Noise sigma for PCA, used if accountant_type is Moments")

tf.flags.DEFINE_string("target_eps", "0.125,0.25,0.5,1,2,4,8",
                       "Log the privacy loss for the target epsilon's. Only "
                       "used when accountant_type is Moments.")
tf.flags.DEFINE_float("target_delta", 1e-5,
                      "Maximum delta for --terminate_based_on_privacy.")
tf.flags.DEFINE_bool("terminate_based_on_privacy", False,
                     "Stop training if privacy spent exceeds "
                     "(max(--target_eps), --target_delta), even "
                     "if --num_training_steps have not yet been completed.")

tf.flags.DEFINE_string("save_path", "/tmp/mnist_dir",
                       "Directory for saving model outputs.")

FLAGS = tf.flags.FLAGS
NUM_TRAINING_IMAGES = 60000
NUM_TESTING_IMAGES = 10000
IMAGE_SIZE = 28


def MnistInput(mnist_data_file, batch_size, randomize):
  """Create operations to read the MNIST input file.

  Args:
    mnist_data_file: Path of a file containing the MNIST images to process.
    batch_size: size of the mini batches to generate.
    randomize: If true, randomize the dataset.

  Returns:
    images: A tensor with the formatted image data. shape [batch_size, 28*28]
    labels: A tensor with the labels for each image.  shape [batch_size]
  """
  file_queue = tf.train.string_input_producer([mnist_data_file])
  reader = tf.TFRecordReader()
  _, value = reader.read(file_queue)
  example = tf.parse_single_example(
      value,
      features={"image/encoded": tf.FixedLenFeature(shape=(), dtype=tf.string),
                "image/class/label": tf.FixedLenFeature([1], tf.int64)})

  image = tf.cast(tf.image.decode_png(example["image/encoded"], channels=1),
                  tf.float32)
  image = tf.reshape(image, [IMAGE_SIZE * IMAGE_SIZE])
  image /= 255
  label = tf.cast(example["image/class/label"], dtype=tf.int32)
  label = tf.reshape(label, [])

  if randomize:
    images, labels = tf.train.shuffle_batch(
        [image, label], batch_size=batch_size,
        capacity=(batch_size * 100),
        min_after_dequeue=(batch_size * 10))
  else:
    images, labels = tf.train.batch([image, label], batch_size=batch_size)

  return images, labels


def Eval(mnist_data_file, network_parameters, num_testing_images,
         randomize, load_path, save_mistakes=False):
  """Evaluate MNIST for a number of steps.

  Args:
    mnist_data_file: Path of a file containing the MNIST images to process.
    network_parameters: parameters for defining and training the network.
    num_testing_images: the number of images we will evaluate on.
    randomize: if false, randomize; otherwise, read the testing images
      sequentially.
    load_path: path where to load trained parameters from.
    save_mistakes: save the mistakes if True.

  Returns:
    The evaluation accuracy as a float.
  """
  batch_size = 100
  # Like for training, we need a session for executing the TensorFlow graph.
  with tf.Graph().as_default(), tf.Session() as sess:
    # Create the basic Mnist model.
    images, labels = MnistInput(mnist_data_file, batch_size, randomize)
    logits, _, _ = utils.BuildNetwork(images, network_parameters)
    softmax = tf.nn.softmax(logits)

    # Load the variables.
    ckpt_state = tf.train.get_checkpoint_state(load_path)
    if not (ckpt_state and ckpt_state.model_checkpoint_path):
      raise ValueError("No model checkpoint to eval at %s\n" % load_path)

    saver = tf.train.Saver()
    saver.restore(sess, ckpt_state.model_checkpoint_path)
    coord = tf.train.Coordinator()
    _ = tf.train.start_queue_runners(sess=sess, coord=coord)

    total_examples = 0
    correct_predictions = 0
    image_index = 0
    mistakes = []
    for _ in xrange((num_testing_images + batch_size - 1) // batch_size):
      predictions, label_values = sess.run([softmax, labels])

      # Count how many were predicted correctly.
      for prediction, label_value in zip(predictions, label_values):
        total_examples += 1
        if np.argmax(prediction) == label_value:
          correct_predictions += 1
        elif save_mistakes:
          mistakes.append({"index": image_index,
                           "label": label_value,
                           "pred": np.argmax(prediction)})
        image_index += 1

  return (correct_predictions / total_examples,
          mistakes if save_mistakes else None)


def Train(mnist_train_file, mnist_test_file, network_parameters, num_steps,
          save_path, eval_steps=0):
  """Train MNIST for a number of steps.

  Args:
    mnist_train_file: path of MNIST train data file.
    mnist_test_file: path of MNIST test data file.
    network_parameters: parameters for defining and training the network.
    num_steps: number of steps to run. Here steps = lots
    save_path: path where to save trained parameters.
    eval_steps: evaluate the model every eval_steps.

  Returns:
    the result after the final training step.

  Raises:
    ValueError: if the accountant_type is not supported.
  """
  batch_size = FLAGS.batch_size

  params = {"accountant_type": FLAGS.accountant_type,
            "task_id": 0,
            "batch_size": FLAGS.batch_size,
            "projection_dimensions": FLAGS.projection_dimensions,
            "default_gradient_l2norm_bound":
            network_parameters.default_gradient_l2norm_bound,
            "num_hidden_layers": FLAGS.num_hidden_layers,
            "hidden_layer_num_units": FLAGS.hidden_layer_num_units,
            "num_examples": NUM_TRAINING_IMAGES,
            "learning_rate": FLAGS.lr,
            "end_learning_rate": FLAGS.end_lr,
            "learning_rate_saturate_epochs": FLAGS.lr_saturate_epochs
           }
  # Log different privacy parameters dependent on the accountant type.
  if FLAGS.accountant_type == "Amortized":
    params.update({"flag_eps": FLAGS.eps,
                   "flag_delta": FLAGS.delta,
                   "flag_pca_eps": FLAGS.pca_eps,
                   "flag_pca_delta": FLAGS.pca_delta,
                  })
  elif FLAGS.accountant_type == "Moments":
    params.update({"sigma": FLAGS.sigma,
                   "pca_sigma": FLAGS.pca_sigma,
                  })

  with tf.Graph().as_default(), tf.Session() as sess, tf.device('/cpu:0'):
    # Create the basic Mnist model.
    images, labels = MnistInput(mnist_train_file, batch_size, FLAGS.randomize)

    logits, projection, training_params = utils.BuildNetwork(
        images, network_parameters)

    cost = tf.nn.softmax_cross_entropy_with_logits(
        logits=logits, labels=tf.one_hot(labels, 10))

    # The actual cost is the average across the examples.
    cost = tf.reduce_sum(cost, [0]) / batch_size

    if FLAGS.accountant_type == "Amortized":
      priv_accountant = accountant.AmortizedAccountant(NUM_TRAINING_IMAGES)
      sigma = None
      pca_sigma = None
      with_privacy = FLAGS.eps > 0
    elif FLAGS.accountant_type == "Moments":
      priv_accountant = accountant.GaussianMomentsAccountant(
          NUM_TRAINING_IMAGES)
      sigma = FLAGS.sigma
      pca_sigma = FLAGS.pca_sigma
      with_privacy = FLAGS.sigma > 0
    else:
      raise ValueError("Undefined accountant type, needs to be "
                       "Amortized or Moments, but got %s" % FLAGS.accountant)
    # Note: Here and below, we scale down the l2norm_bound by
    # batch_size. This is because per_example_gradients computes the
    # gradient of the minibatch loss with respect to each individual
    # example, and the minibatch loss (for our model) is the *average*
    # loss over examples in the minibatch. Hence, the scale of the
    # per-example gradients goes like 1 / batch_size.
    gaussian_sanitizer = sanitizer.AmortizedGaussianSanitizer(
        priv_accountant,
        [network_parameters.default_gradient_l2norm_bound / batch_size, True])

    for var in training_params:
      if "gradient_l2norm_bound" in training_params[var]:
        l2bound = training_params[var]["gradient_l2norm_bound"] / batch_size
        gaussian_sanitizer.set_option(var,
                                      sanitizer.ClipOption(l2bound, True))
    lr = tf.placeholder(tf.float32)
    eps = tf.placeholder(tf.float32)
    delta = tf.placeholder(tf.float32)

    init_ops = []
    if network_parameters.projection_type == "PCA":
      with tf.variable_scope("pca"):
        # Compute differentially private PCA.
        all_data, _ = MnistInput(mnist_train_file, NUM_TRAINING_IMAGES, False)
        pca_projection = dp_pca.ComputeDPPrincipalProjection(
            all_data, network_parameters.projection_dimensions,
            gaussian_sanitizer, [FLAGS.pca_eps, FLAGS.pca_delta], pca_sigma)
        assign_pca_proj = tf.assign(projection, pca_projection)
        init_ops.append(assign_pca_proj)

    # Add global_step
    global_step = tf.Variable(0, dtype=tf.int32, trainable=False,
                              name="global_step")

    if with_privacy:
      gd_op = dp_optimizer.DPGradientDescentOptimizer(
          lr,
          [eps, delta],
          gaussian_sanitizer,
          sigma=sigma,
          batches_per_lot=FLAGS.batches_per_lot).minimize(
              cost, global_step=global_step)
    else:
      gd_op = tf.train.GradientDescentOptimizer(lr).minimize(cost)

    saver = tf.train.Saver()
    coord = tf.train.Coordinator()
    _ = tf.train.start_queue_runners(sess=sess, coord=coord)

    # We need to maintain the intialization sequence.
    for v in tf.trainable_variables():
      sess.run(tf.variables_initializer([v]))
    sess.run(tf.global_variables_initializer())
    sess.run(init_ops)

    results = []
    start_time = time.time()
    prev_time = start_time
    filename = "results-0.json"
    log_path = os.path.join(save_path, filename)

    target_eps = [float(s) for s in FLAGS.target_eps.split(",")]
    if FLAGS.accountant_type == "Amortized":
      # Only matters if --terminate_based_on_privacy is true.
      target_eps = [max(target_eps)]
    max_target_eps = max(target_eps)

    lot_size = FLAGS.batches_per_lot * FLAGS.batch_size
    lots_per_epoch = NUM_TRAINING_IMAGES / lot_size
    for step in xrange(num_steps):
      epoch = step / lots_per_epoch
      curr_lr = utils.VaryRate(FLAGS.lr, FLAGS.end_lr,
                               FLAGS.lr_saturate_epochs, epoch)
      curr_eps = utils.VaryRate(FLAGS.eps, FLAGS.end_eps,
                                FLAGS.eps_saturate_epochs, epoch)
      for _ in xrange(FLAGS.batches_per_lot):
        _ = sess.run(
            [gd_op], feed_dict={lr: curr_lr, eps: curr_eps, delta: FLAGS.delta})
      sys.stderr.write("step: %d\n" % step)

      # See if we should stop training due to exceeded privacy budget:
      should_terminate = False
      terminate_spent_eps_delta = None
      if with_privacy and FLAGS.terminate_based_on_privacy:
        terminate_spent_eps_delta = priv_accountant.get_privacy_spent(
            sess, target_eps=[max_target_eps])[0]
        # For the Moments accountant, we should always have
        # spent_eps == max_target_eps.
        if (terminate_spent_eps_delta.spent_delta > FLAGS.target_delta or
            terminate_spent_eps_delta.spent_eps > max_target_eps):
          should_terminate = True

      if (eval_steps > 0 and (step + 1) % eval_steps == 0) or should_terminate:
        if with_privacy:
          spent_eps_deltas = priv_accountant.get_privacy_spent(
              sess, target_eps=target_eps)
        else:
          spent_eps_deltas = [accountant.EpsDelta(0, 0)]
        for spent_eps, spent_delta in spent_eps_deltas:
          sys.stderr.write("spent privacy: eps %.4f delta %.5g\n" % (
              spent_eps, spent_delta))

        saver.save(sess, save_path=save_path + "/ckpt")
        train_accuracy, _ = Eval(mnist_train_file, network_parameters,
                                 num_testing_images=NUM_TESTING_IMAGES,
                                 randomize=True, load_path=save_path)
        sys.stderr.write("train_accuracy: %.2f\n" % train_accuracy)
        test_accuracy, mistakes = Eval(mnist_test_file, network_parameters,
                                       num_testing_images=NUM_TESTING_IMAGES,
                                       randomize=False, load_path=save_path,
                                       save_mistakes=FLAGS.save_mistakes)
        sys.stderr.write("eval_accuracy: %.2f\n" % test_accuracy)

        curr_time = time.time()
        elapsed_time = curr_time - prev_time
        prev_time = curr_time

        results.append({"step": step+1,  # Number of lots trained so far.
                        "elapsed_secs": elapsed_time,
                        "spent_eps_deltas": spent_eps_deltas,
                        "train_accuracy": train_accuracy,
                        "test_accuracy": test_accuracy,
                        "mistakes": mistakes})
        loginfo = {"elapsed_secs": curr_time-start_time,
                   "spent_eps_deltas": spent_eps_deltas,
                   "train_accuracy": train_accuracy,
                   "test_accuracy": test_accuracy,
                   "num_training_steps": step+1,  # Steps so far.
                   "mistakes": mistakes,
                   "result_series": results}
        loginfo.update(params)
        if log_path:
          with tf.gfile.Open(log_path, "w") as f:
            json.dump(loginfo, f, indent=2)
            f.write("\n")
            f.close()

      if should_terminate:
        break


def main(_):
  network_parameters = utils.NetworkParameters()

  # If the ASCII proto isn't specified, then construct a config protobuf based
  # on 3 flags.
  network_parameters.input_size = IMAGE_SIZE ** 2
  network_parameters.default_gradient_l2norm_bound = (
      FLAGS.default_gradient_l2norm_bound)
  if FLAGS.projection_dimensions > 0 and FLAGS.num_conv_layers > 0:
    raise ValueError("Currently you can't do PCA and have convolutions"
                     "at the same time. Pick one")

    # could add support for PCA after convolutions.
    # Currently BuildNetwork can build the network with conv followed by
    # projection, but the PCA training works on data, rather than data run
    # through a few layers. Will need to init the convs before running the
    # PCA, and need to change the PCA subroutine to take a network and perhaps
    # allow for batched inputs, to handle larger datasets.
  if FLAGS.num_conv_layers > 0:
    conv = utils.ConvParameters()
    conv.name = "conv1"
    conv.in_channels = 1
    conv.out_channels = 128
    conv.num_outputs = 128 * 14 * 14
    network_parameters.conv_parameters.append(conv)
    # defaults for the rest: 5x5,stride 1, relu, maxpool 2x2,stride 2.
    # insize 28x28, bias, stddev 0.1, non-trainable.
  if FLAGS.num_conv_layers > 1:
    conv = network_parameters.ConvParameters()
    conv.name = "conv2"
    conv.in_channels = 128
    conv.out_channels = 128
    conv.num_outputs = 128 * 7 * 7
    conv.in_size = 14
    # defaults for the rest: 5x5,stride 1, relu, maxpool 2x2,stride 2.
    # bias, stddev 0.1, non-trainable.
    network_parameters.conv_parameters.append(conv)

  if FLAGS.num_conv_layers > 2:
    raise ValueError("Currently --num_conv_layers must be 0,1 or 2."
                     "Manually create a network_parameters proto for more.")

  if FLAGS.projection_dimensions > 0:
    network_parameters.projection_type = "PCA"
    network_parameters.projection_dimensions = FLAGS.projection_dimensions
  for i in xrange(FLAGS.num_hidden_layers):
    hidden = utils.LayerParameters()
    hidden.name = "hidden%d" % i
    hidden.num_units = FLAGS.hidden_layer_num_units
    hidden.relu = True
    hidden.with_bias = False
    hidden.trainable = not FLAGS.freeze_bottom_layers
    network_parameters.layer_parameters.append(hidden)

  logits = utils.LayerParameters()
  logits.name = "logits"
  logits.num_units = 10
  logits.relu = False
  logits.with_bias = False
  network_parameters.layer_parameters.append(logits)

  Train(FLAGS.training_data_path,
        FLAGS.eval_data_path,
        network_parameters,
        FLAGS.num_training_steps,
        FLAGS.save_path,
        eval_steps=FLAGS.eval_steps)


if __name__ == "__main__":
  tf.app.run()
