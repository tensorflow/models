"""The main module for sentiment analysis.

The model makes use of concatenation of two CNN layers
with different kernel sizes.
See `sentiment_model.py` for more details about the models.
"""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from absl import app as absl_app
from absl import flags
from data import dataset
from official.utils.flags import core as flags_core
from official.utils.logs import hooks_helper
from official.utils.logs import logger
from official.utils.misc import distribution_utils
import sentiment_model
import tensorflow as tf


def convert_keras_to_estimator(keras_model, num_gpus, model_dir=None):
  """Convert keras model into tensorflow estimator."""

  keras_model.compile(optimizer="rmsprop",
                      loss="categorical_crossentropy", metrics=["accuracy"])

  distribution = distribution_utils.get_distribution_strategy(
      num_gpus, all_reduce_alg=None)
  run_config = tf.estimator.RunConfig(train_distribute=distribution)

  estimator = tf.keras.estimator.model_to_estimator(
      keras_model=keras_model, model_dir=model_dir, config=run_config)

  return estimator


def run_model(flags_obj):
  """Run training and eval loop."""

  num_class = dataset.get_num_class(flags_obj.dataset)

  tf.logging.info("Loading the dataset...")

  train_input_fn, eval_input_fn = dataset.construct_input_fns(
      flags_obj.dataset, flags_obj.batch_size, flags_obj.vocabulary_size,
      flags_obj.sentence_length, repeat=flags_obj.epochs_between_evals)

  keras_model = sentiment_model.CNN(
      flags_obj.embedding_dim, flags_obj.vocabulary_size,
      flags_obj.sentence_length,
      flags_obj.cnn_filters, num_class, flags_obj.dropout_rate)
  num_gpus = flags_core.get_num_gpus(FLAGS)
  tf.logging.info("Creating Estimator from Keras model...")
  estimator = convert_keras_to_estimator(
      keras_model, num_gpus, flags_obj.model_dir)

  # Create hooks that log information about the training and metric values
  train_hooks = hooks_helper.get_train_hooks(
      flags_obj.hooks,
      batch_size=flags_obj.batch_size  # for ExamplesPerSecondHook
  )
  run_params = {
      "batch_size": flags_obj.batch_size,
      "train_epochs": flags_obj.train_epochs,
  }
  benchmark_logger = logger.get_benchmark_logger()
  benchmark_logger.log_run_info(
      model_name="sentiment_analysis",
      dataset_name=flags_obj.dataset,
      run_params=run_params,
      test_id=flags_obj.benchmark_test_id)

  # Training and evaluation cycle
  total_training_cycle = flags_obj.train_epochs\
    // flags_obj.epochs_between_evals

  for cycle_index in range(total_training_cycle):
    tf.logging.info("Starting a training cycle: {}/{}".format(
        cycle_index + 1, total_training_cycle))

    # Train the model
    estimator.train(input_fn=train_input_fn, hooks=train_hooks)

    # Evaluate the model
    eval_results = estimator.evaluate(input_fn=eval_input_fn)

    # Benchmark the evaluation results
    benchmark_logger.log_evaluation_result(eval_results)

    tf.logging.info("Iteration {}".format(eval_results))

  # Clear the session explicitly to avoid session delete error
  tf.keras.backend.clear_session()


def main(_):
  with logger.benchmark_context(FLAGS):
    run_model(FLAGS)


def define_flags():
  """Add flags to run the main function."""

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
      hooks="")

  # Add domain-specific flags
  flags.DEFINE_enum(
      name="dataset", default=dataset.DATASET_IMDB,
      enum_values=[dataset.DATASET_IMDB], case_sensitive=False,
      help=flags_core.help_wrap(
          "Dataset to be trained and evaluated."))

  flags.DEFINE_integer(
      name="vocabulary_size", default=6000,
      help=flags_core.help_wrap(
          "The number of the most frequent tokens"
          "to be used from the corpus."))

  flags.DEFINE_integer(
      name="sentence_length", default=200,
      help=flags_core.help_wrap(
          "The number of words in each sentence. Longer sentences get cut,"
          "shorter ones padded."))

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
