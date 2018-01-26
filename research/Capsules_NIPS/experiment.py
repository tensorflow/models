# Copyright 2017 The TensorFlow Authors All Rights Reserved.
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

"""Framework for training and evaluating models."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os
import sys
import time

import numpy as np
import tensorflow as tf


from input_data.cifar10 import cifar10_input
from input_data.mnist import mnist_input_record
from input_data.norb import norb_input_record
from models import capsule_model
from models import conv_model

FLAGS = tf.flags.FLAGS

tf.flags.DEFINE_string('data_dir', None, 'The data directory.')
tf.flags.DEFINE_integer('eval_size', 10000, 'Size of the test dataset.')
tf.flags.DEFINE_string('hparams_override', None,
                       'A string of form key=value,key=value to override the'
                       'hparams of this experiment.')
tf.flags.DEFINE_integer('max_steps', 1000, 'Number of steps to train.')
tf.flags.DEFINE_string('model', 'capsule',
                       'The model to use for the experiment.'
                       'capsule or baseline')
tf.flags.DEFINE_string('dataset', 'mnist',
                       'The dataset to use for the experiment.'
                       'mnist, norb, cifar10.')
tf.flags.DEFINE_integer('num_gpus', 1, 'Number of gpus to use.')
tf.flags.DEFINE_integer('num_targets', 1,
                        'Number of targets to detect (1 or 2).')
tf.flags.DEFINE_integer('num_trials', 1,
                        'Number of trials for ensemble evaluation.')
tf.flags.DEFINE_integer('save_step', 1500, 'How often to save checkpoints.')
tf.flags.DEFINE_string('summary_dir', None,
                       'Main directory for the experiments.')
tf.flags.DEFINE_string('checkpoint', None,
                       'The model checkpoint for evaluation.')
tf.flags.DEFINE_bool('train', True, 'Either train the model or test the model.')
tf.flags.DEFINE_bool('validate', False, 'Run trianing/eval in validation mode.')

models = {
    'capsule': capsule_model.CapsuleModel,
    'baseline': conv_model.ConvModel,
}


def get_features(split, total_batch_size, num_gpus, data_dir, num_targets,
                 dataset, validate=False):
  """Reads the input data and distributes it over num_gpus GPUs.

  Each tower of data has 1/FLAGS.num_gpus of the total_batch_size.

  Args:
    split: 'train' or 'test', split of the data to read.
    total_batch_size: total number of data entries over all towers.
    num_gpus: Number of GPUs to distribute the data on.
    data_dir: Directory containing the input data.
    num_targets: Number of objects present in the image.
    dataset: The name of the dataset, either norb or mnist.
    validate: If set, subset training data into training and test.

  Returns:
    A list of batched feature dictionaries.

  Raises:
    ValueError: If dataset is not mnist or norb.
  """

  batch_size = total_batch_size // max(1, num_gpus)
  features = []
  for i in range(num_gpus):
    with tf.device('/gpu:%d' % i):
      if dataset == 'mnist':
        features.append(
            mnist_input_record.inputs(
                data_dir=data_dir,
                batch_size=batch_size,
                split=split,
                num_targets=num_targets,
                validate=validate,
            ))
      elif dataset == 'norb':
        features.append(
            norb_input_record.inputs(
                data_dir=data_dir, batch_size=batch_size, split=split,
            ))
      elif dataset == 'cifar10':
        data_dir = os.path.join(data_dir, 'cifar-10-batches-bin')
        features.append(
            cifar10_input.inputs(
                split=split, data_dir=data_dir, batch_size=batch_size))
      else:
        raise ValueError(
            'Unexpected dataset {!r}, must be mnist, norb, or cifar10.'.format(
                dataset))
  return features


def extract_step(path):
  """Returns the step from the file format name of Tensorflow checkpoints.

  Args:
    path: The checkpoint path returned by tf.train.get_checkpoint_state.
      The format is: {ckpnt_name}-{step}

  Returns:
    The last training step number of the checkpoint.
  """
  file_name = os.path.basename(path)
  return int(file_name.split('-')[-1])


def load_training(saver, session, load_dir):
  """Loads a saved model into current session or initializes the directory.

  If there is no functioning saved model or FLAGS.restart is set, cleans the
  load_dir directory. Otherwise, loads the latest saved checkpoint in load_dir
  to session.

  Args:
    saver: An instance of tf.train.saver to load the model in to the session.
    session: An instance of tf.Session with the built-in model graph.
    load_dir: The directory which is used to load the latest checkpoint.

  Returns:
    The latest saved step.
  """
  if tf.gfile.Exists(load_dir):
    ckpt = tf.train.get_checkpoint_state(load_dir)
    if ckpt and ckpt.model_checkpoint_path:
      saver.restore(session, ckpt.model_checkpoint_path)
      prev_step = extract_step(ckpt.model_checkpoint_path)
    else:
      tf.gfile.DeleteRecursively(load_dir)
      tf.gfile.MakeDirs(load_dir)
      prev_step = 0
  else:
    tf.gfile.MakeDirs(load_dir)
    prev_step = 0
  return prev_step


def train_experiment(session, result, writer, last_step, max_steps, saver,
                     summary_dir, save_step):
  """Runs training for up to max_steps and saves the model and summaries.

  Args:
    session: The loaded tf.session with the initialized model.
    result: The resultant operations of the model including train_op.
    writer: The summary writer file.
    last_step: The last trained step.
    max_steps: Maximum number of training iterations.
    saver: An instance of tf.train.saver to save the current model.
    summary_dir: The directory to save the model in it.
    save_step: How often to save the model ckpt.
  """
  step = 0
  for i in range(last_step, max_steps):
    step += 1
    summary, _ = session.run([result.summary, result.train_op])
    writer.add_summary(summary, i)
    if (i + 1) % save_step == 0:
      saver.save(
          session, os.path.join(summary_dir, 'model.ckpt'), global_step=i + 1)


def load_eval(saver, session, load_dir):
  """Loads the latest saved model to the given session.

  Args:
    saver: An instance of tf.train.saver to load the model in to the session.
    session: An instance of tf.Session with the built-in model graph.
    load_dir: The path to the latest checkpoint.

  Returns:
    The latest saved step.
  """
  saver.restore(session, load_dir)
  print('model loaded successfully')
  return extract_step(load_dir)


def eval_experiment(session, result, writer, last_step, max_steps, **kwargs):
  """Evaluates the current model on the test dataset once.

  Evaluates the loaded model on the test data set with batch sizes of 100.
  Aggregates the results and writes one summary point to the summary file.

  Args:
    session: The loaded tf.session with the trained model.
    result: The resultant operations of the model including evaluation metrics.
    writer: The summary writer file.
    last_step: The last trained step.
    max_steps: Maximum number of evaluation iterations.
    **kwargs: Arguments passed by run_experiment but not used in this function.
  """
  del kwargs

  total_correct = 0
  total_almost = 0
  for _ in range(max_steps):
    summary_i, correct, almost = session.run(
        [result.summary, result.correct, result.almost])
    total_correct += correct
    total_almost += almost

  total_false = max_steps * 100 - total_correct
  total_almost_false = max_steps * 100 - total_almost
  summary = tf.Summary.FromString(summary_i)
  summary.value.add(tag='correct_prediction', simple_value=total_correct)
  summary.value.add(tag='wrong_prediction', simple_value=total_false)
  summary.value.add(
      tag='almost_wrong_prediction', simple_value=total_almost_false)
  print('Total wrong predictions: {}, wrong percent: {}%'.format(
      total_false, total_false / max_steps))
  tf.logging.info('Total wrong predictions: {}, wrong percent: {}%'.format(
      total_false, total_false / max_steps))
  writer.add_summary(summary, last_step)


def run_experiment(loader,
                   load_dir,
                   writer,
                   experiment,
                   result,
                   max_steps,
                   save_step=0):
  """Starts a session, loads the model and runs the given experiment on it.

  This is a general wrapper to load a saved model and run an experiment on it.
  An experiment can be a training experiment or an evaluation experiment.
  It starts session, threads and queues and closes them before returning.

  Args:
    loader: A function of prototype (saver, session, load_dir) to load a saved
      checkpoint in load_dir given a session and saver.
    load_dir: The directory to load the previously saved model from it and to
      save the current model in it.
    writer: A tf.summary.FileWriter to add summaries.
    experiment: The function of prototype (session, result, writer, last_step,
      max_steps, saver, load_dir, save_step) which will execute the experiment
      steps from result on the given session.
    result: The resultant final operations of the built model.
    max_steps: Maximum number of experiment iterations.
    save_step: How often the training model should be saved.
  """
  session = tf.Session(config=tf.ConfigProto(allow_soft_placement=True))
  init_op = tf.group(tf.global_variables_initializer(),
                     tf.local_variables_initializer())
  session.run(init_op)
  saver = tf.train.Saver(max_to_keep=1000)
  last_step = loader(saver, session, load_dir)
  coord = tf.train.Coordinator()
  threads = tf.train.start_queue_runners(sess=session, coord=coord)
  try:
    experiment(
        session=session,
        result=result,
        writer=writer,
        last_step=last_step,
        max_steps=max_steps,
        saver=saver,
        summary_dir=load_dir,
        save_step=save_step)
  except tf.errors.OutOfRangeError:
    tf.logging.info('Finished experiment.')
  finally:
    coord.request_stop()
  coord.join(threads)
  session.close()


def train(hparams, summary_dir, num_gpus, model_type, max_steps, save_step,
          data_dir, num_targets, dataset, validate):
  """Trains a model with batch sizes of 128 to FLAGS.max_steps steps.

  It will initialize the model with either previously saved model in the
  summary directory or start from scratch if FLAGS.restart is set or the
  directory is empty.
  The training is distributed on num_gpus GPUs. It writes a summary at every
  step and saves the model every 1500 iterations.

  Args:
    hparams: The hyper parameters to build the model graph.
    summary_dir: The directory to save model and write training summaries.
    num_gpus: Number of GPUs to use for reading data and computation.
    model_type: The model architecture category.
    max_steps: Maximum number of training iterations.
    save_step: How often the training model should be saved.
    data_dir: Directory containing the input data.
    num_targets: Number of objects present in the image.
    dataset: Name of the dataset for the experiments.
    validate: If set, use training-validation set for training.
  """
  summary_dir += '/train/'
  with tf.Graph().as_default():
    # Build model
    features = get_features('train', 128, num_gpus, data_dir, num_targets,
                            dataset, validate)
    model = models[model_type](hparams)
    result, _ = model.multi_gpu(features, num_gpus)
    # Print stats
    param_stats = tf.contrib.tfprof.model_analyzer.print_model_analysis(
        tf.get_default_graph(),
        tfprof_options=tf.contrib.tfprof.model_analyzer.
        TRAINABLE_VARS_PARAMS_STAT_OPTIONS)
    sys.stdout.write('total_params: %d\n' % param_stats.total_parameters)
    writer = tf.summary.FileWriter(summary_dir)
    run_experiment(load_training, summary_dir, writer, train_experiment, result,
                   max_steps, save_step)
    writer.close()


def find_checkpoint(load_dir, seen_step):
  """Finds the global step for the latest written checkpoint to the load_dir.

  Args:
    load_dir: The directory address to look for the training checkpoints.
    seen_step: Latest step which evaluation has been done on it.
  Returns:
    The latest new step in the load_dir and the file path of the latest model
    in load_dir. If no new file is found returns -1 and None.

  """
  ckpt = tf.train.get_checkpoint_state(load_dir)
  if ckpt and ckpt.model_checkpoint_path:
    global_step = extract_step(ckpt.model_checkpoint_path)
    if int(global_step) != seen_step:
      return int(global_step), ckpt.model_checkpoint_path
  return -1, None


def evaluate(hparams, summary_dir, num_gpus, model_type, eval_size, data_dir,
             num_targets, dataset, validate, checkpoint=None):
  """Continuously evaluates the latest trained model or a specific checkpoint.

  Regularly (every 2 min, maximum 6 hours) checks the training directory for
  the latest model. If it finds any new model, it outputs the total number of
  correct and wrong predictions for the test data set to the summary file.
  If a checkpoint is provided performs the evaluation only on the specific
  checkpoint.

  Args:
    hparams: The hyperparameters for building the model graph.
    summary_dir: The directory to load training model and write test summaries.
    num_gpus: Number of GPUs to use for reading data and computation.
    model_type: The model architecture category.
    eval_size: Total number of examples in the test dataset.
    data_dir: Directory containing the input data.
    num_targets: Number of objects present in the image.
    dataset: The name of the dataset for the experiment.
    validate: If set, use validation set for continuous evaluation.
    checkpoint: (optional) The checkpoint file name.
  """
  load_dir = summary_dir + '/train/'
  summary_dir += '/test/'
  with tf.Graph().as_default():
    features = get_features('test', 100, num_gpus, data_dir, num_targets,
                            dataset, validate)
    model = models[model_type](hparams)
    result, _ = model.multi_gpu(features, num_gpus)
    test_writer = tf.summary.FileWriter(summary_dir)
    seen_step = -1
    paused = 0
    while paused < 360:
      print('start evaluation, model defined')
      if checkpoint:
        step = extract_step(checkpoint)
        last_checkpoint = checkpoint
      else:
        step, last_checkpoint = find_checkpoint(load_dir, seen_step)
      if step == -1:
        time.sleep(60)
        paused += 1
      else:
        paused = 0
        seen_step = step
        run_experiment(load_eval, last_checkpoint, test_writer, eval_experiment,
                       result, eval_size // 100)
        if checkpoint:
          break

    test_writer.close()


def get_placeholder_data(num_steps, batch_size, features, session):
  """Reads the features into a numpy array and replaces them with placeholders.

  Loads all the images and labels of the features queue in memory. Replaces
  the feature queue reader handle with placeholders to switch input method from
  queue to placeholders. Using placeholders gaurantees the order of datapoints
  to stay exactly the same during each epoch.

  Args:
    num_steps: The number of times to read from the features queue.
    batch_size: The number of datapoints at each step.
    features: The dictionary containing the data queues such as images.
    session: The session handle to use for running tensors.

  Returns:
    data: List of numpy arrays containing all the queued data in features.
    targets: List of all the labels in range [0...num_classes].
  """
  image_size = features['height']
  depth = features['depth']
  num_classes = features['num_classes']
  data = []
  targets = []
  for i in range(num_steps):
    data.append(
        session.run({
            'recons_label': features['recons_label'],
            'labels': features['labels'],
            'images': features['images'],
            'recons_image': features['recons_image']
        }))
    targets.append(data[i]['recons_label'])
  image_shape = (batch_size, depth, image_size, image_size)
  features['images'] = tf.placeholder(tf.float32, shape=image_shape)
  features['labels'] = tf.placeholder(
      tf.float32, shape=(batch_size, num_classes))
  features['recons_image'] = tf.placeholder(tf.float32, shape=image_shape)
  features['recons_label'] = tf.placeholder(tf.int32, shape=(batch_size))
  return data, targets


def infer_ensemble_logits(features, model, checkpoints, session, num_steps,
                          data):
  """Extracts the logits for the whole dataset and all the trained models.

  Loads all the checkpoints. For each checkpoint stores the logits for the whole
  dataset.

  Args:
    features: The dictionary of the input handles.
    model: The model operation graph.
    checkpoints: The list of all checkpoint paths.
    session: The session handle to use for running tensors.
    num_steps: The number of steps to run the experiment.
    data: The num_steps list of loaded data to be fed to placeholders.

  Returns:
    logits: List of all the final layer logits for different checkpoints.
  """
  _, inferred = model.multi_gpu([features], 1)
  logits = []
  saver = tf.train.Saver()
  for checkpoint in checkpoints:
    saver.restore(session, checkpoint)
    for i in range(num_steps):
      logits.append(
          session.run(
              inferred[0].logits,
              feed_dict={
                  features['recons_label']: data[i]['recons_label'],
                  features['labels']: data[i]['labels'],
                  features['images']: data[i]['images'],
                  features['recons_image']: data[i]['recons_image']
              }))
  return logits


def evaluate_ensemble(hparams, model_type, eval_size, data_dir, num_targets,
                      dataset, checkpoint, num_trials):
  """Evaluates an ensemble of trained models.

  Loads a series of checkpoints and aggregates the output logit of them on the
  test data. Selects the class with maximum aggregated logit as the prediction.
  Prints the total number of wrong predictions.

  Args:
    hparams: The hyperparameters for building the model graph.
    model_type: The model architecture category.
    eval_size: Total number of examples in the test dataset.
    data_dir: Directory containing the input data.
    num_targets: Number of objects present in the image.
    dataset: The name of the dataset for the experiment.
    checkpoint: The file format of the checkpoints to be loaded.
    num_trials: Number of trained models to ensemble.
  """
  checkpoints = []
  for i in range(num_trials):
    file_name = checkpoint.format(i)
    if tf.train.checkpoint_exists(file_name):
      checkpoints.append(file_name)

  with tf.Graph().as_default():
    batch_size = 100
    features = get_features('test', batch_size, 1, data_dir, num_targets,
                            dataset)[0]
    model = models[model_type](hparams)

    session = tf.Session(config=tf.ConfigProto(allow_soft_placement=True))
    coord = tf.train.Coordinator()
    threads = tf.train.start_queue_runners(sess=session, coord=coord)
    num_steps = eval_size // batch_size
    data, targets = get_placeholder_data(num_steps, batch_size, features,
                                         session)
    logits = infer_ensemble_logits(features, model, checkpoints, session,
                                   num_steps, data)
    coord.request_stop()
    coord.join(threads)
    session.close()

    logits = np.reshape(logits, (num_trials, num_steps, batch_size, -1))
    logits = np.sum(logits, axis=0)
    predictions = np.argmax(logits, axis=2)
    total_wrong = np.sum(np.not_equal(predictions, targets))
    print('Total wrong predictions: {}, wrong percent: {}%'.format(
        total_wrong, total_wrong / eval_size * 100))


def default_hparams():
  """Builds an HParam object with default hyperparameters."""
  return tf.contrib.training.HParams(
      decay_rate=0.96,
      decay_steps=2000,
      leaky=False,
      learning_rate=0.001,
      loss_type='margin',
      num_prime_capsules=32,
      padding='VALID',
      remake=True,
      routing=3,
      verbose=False,
  )


def main(_):
  hparams = default_hparams()
  if FLAGS.hparams_override:
    hparams.parse(FLAGS.hparams_override)

  if FLAGS.train:
    train(hparams, FLAGS.summary_dir, FLAGS.num_gpus, FLAGS.model,
          FLAGS.max_steps, FLAGS.save_step, FLAGS.data_dir, FLAGS.num_targets,
          FLAGS.dataset, FLAGS.validate)
  else:
    if FLAGS.num_trials == 1:
      evaluate(hparams, FLAGS.summary_dir, FLAGS.num_gpus, FLAGS.model,
               FLAGS.eval_size, FLAGS.data_dir, FLAGS.num_targets,
               FLAGS.dataset, FLAGS.validate, FLAGS.checkpoint)
    else:
      evaluate_ensemble(hparams, FLAGS.model, FLAGS.eval_size, FLAGS.data_dir,
                        FLAGS.num_targets, FLAGS.dataset, FLAGS.checkpoint,
                        FLAGS.num_trials)


if __name__ == '__main__':
  tf.app.run()
