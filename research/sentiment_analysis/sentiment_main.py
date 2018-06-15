"""The model makes use of concatenation of two CNN layers with different kernel sizes.
`sentiment_model.py` for more details about the models.
"""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

# pylint: disable=g-bad-import-order
from absl import app as absl_app
from absl import flags
import tensorflow as tf
# pylint: enable=g-bad-import-order

from data import dataset
from data.dataset import imdb
import sentiment_model
from official.utils.flags import core as flags_core
from official.utils.logs import hooks_helper
from official.utils.logs import logger

def evaluate_model(estimator, num_gpus):
  # Define prediction input function
  def pred_input_fn():
    return dataset.input_fn(
        FLAGS.dataset, True, per_device_batch_size(FLAGS.batch_size, num_gpus),
        FLAGS.vocabulary_size, FLAGS.sentence_length)

  return estimator.evaluate(input_fn=pred_input_fn)

def convert_keras_to_estimator(keras_model, num_gpus, model_dir=None):
  keras_model.compile(optimizer='rmsprop', loss="categorical_crossentropy", metrics=['accuracy'])

  if num_gpus == 0:
    distribution = tf.contrib.distribute.OneDeviceStrategy("device:CPU:0")
  elif num_gpus == 1:
    distribution = tf.contrib.distribute.OneDeviceStrategy("device:GPU:0")
  else:
    distribution = tf.contrib.distribute.MirroredStrategy(num_gpus=num_gpus)

  run_config = tf.estimator.RunConfig(train_distribute=distribution)

  estimator = tf.keras.estimator.model_to_estimator(
      keras_model=keras_model, model_dir=model_dir, config=run_config)

  return estimator


def per_device_batch_size(batch_size, num_gpus):
  """For multi-gpu, batch-size must be a multiple of the number of GPUs.
  Note that this should eventually be handled by DistributionStrategies
  directly. Multi-GPU support is currently experimental, however,
  so doing the work here until that feature is in place.
  Args:
    batch_size: Global batch size to be divided among devices. This should be
      equal to num_gpus times the single-GPU batch_size for multi-gpu training.
    num_gpus: How many GPUs are used with DistributionStrategies.
  Returns:
    Batch size per device.
  Raises:
    ValueError: if batch_size is not divisible by number of devices
  """
  if num_gpus <= 1:
    return batch_size

  remainder = batch_size % num_gpus
  if remainder:
    err = ("When running with multiple GPUs, batch size "
           "must be a multiple of the number of available GPUs. Found {} "
           "GPUs with a batch size of {}; try --batch_size={} instead."
          ).format(num_gpus, batch_size, batch_size - remainder)
    raise ValueError(err)
  return int(batch_size / num_gpus)

def run_model():
  """Run training and eval loop."""

  tf.logging.info("Creating Estimator from Keras model...")
  num_class = imdb.NUM_CLASS

  keras_model = sentiment_model.CNN(
      FLAGS.embedding_dim, FLAGS.vocabulary_size, FLAGS.sentence_length,
      FLAGS.cnn_filters, num_class, FLAGS.dropout_rate)
  num_gpus = flags_core.get_num_gpus(FLAGS)
  estimator = convert_keras_to_estimator(keras_model, num_gpus, FLAGS.model_dir)

  # Create hooks that log information about the training and metric values
  train_hooks = hooks_helper.get_train_hooks(
      FLAGS.hooks,
      batch_size=FLAGS.batch_size  # for ExamplesPerSecondHook
  )
  run_params = {
      "batch_size": FLAGS.batch_size,
      "train_epochs": FLAGS.train_epochs,
  }
  benchmark_logger = logger.get_benchmark_logger()
  benchmark_logger.log_run_info(
      model_name="sentiment_analysis",
      dataset_name=FLAGS.dataset,
      run_params=run_params,
      test_id=FLAGS.benchmark_test_id)

  # Training and evaluation cycle
  def train_input_fn():
    return dataset.input_fn(
        FLAGS.dataset, True, per_device_batch_size(FLAGS.batch_size, num_gpus),
        FLAGS.vocabulary_size, FLAGS.sentence_length, repeat=FLAGS.epochs_between_evals)

  total_training_cycle = FLAGS.train_epochs // FLAGS.epochs_between_evals

  for cycle_index in range(total_training_cycle):
    tf.logging.info("Starting a training cycle: {}/{}".format(
        cycle_index + 1, total_training_cycle))

    # Train the model
    estimator.train(input_fn=train_input_fn, hooks=train_hooks)

    # Evaluate the model
    eval_results = evaluate_model(estimator, num_gpus)

    # Benchmark the evaluation results
    benchmark_logger.log_evaluation_result(eval_results)

    tf.logging.info("Iteration {}".format(eval_results))

  # Clear the session explicitly to avoid session delete error
  tf.keras.backend.clear_session()

def main(_):
  with logger.benchmark_context(FLAGS):
    run_model()

def define_flags():
  """Add flags to run sentiment_main.py"""
  # Add common flags
  flags_core.define_base(export_dir=False)
  flags_core.define_performance(
      num_parallel_calls=False,
      inter_op=False,
      intra_op=False,
      synthetic_data=False,
      max_train_steps=False,
      dtype=False
  )
  flags_core.define_benchmark()

  flags.adopt_module_key_flags(flags_core)

  flags_core.set_defaults(
      model_dir=None,
      train_epochs=30,
      batch_size=30,
      hooks="ProfilerHook")

  # Add domain-specific flags
  flags.DEFINE_enum(
      name="dataset", default="imdb",
      enum_values=["imdb"], case_sensitive=False,
      help=flags_core.help_wrap(
          "Dataset to be trained and evaluated."))

  flags.DEFINE_integer(
      name="vocabulary_size", default=6000,
      help=flags_core.help_wrap(
          "The number of the most frequent tokens to be used from the corpus."))

  flags.DEFINE_integer(
      name="sentence_length", default=200,
      help=flags_core.help_wrap(
          "The number of words in each sentence. Longer sentences get cut, shorter ones padded."))

  flags.DEFINE_integer(
      name="embedding_dim", default=256,
      help=flags_core.help_wrap("The dimension of the Embedding layer."))

  flags.DEFINE_integer(
      name="cnn_filters", default=512,
      help=flags_core.help_wrap("The number of the CNN layer filters."))

  flags.DEFINE_float(
      name="dropout_rate", default=0.7,
      help=flags_core.help_wrap("The rate for the Dropout layer."))

if __name__ == "__main__":
  tf.logging.set_verbosity(tf.logging.INFO)
  define_flags()
  FLAGS = flags.FLAGS
  absl_app.run(main)
