# Copyright 2018 The TensorFlow Authors. All Rights Reserved.
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
"""Main entry to train and evaluate DeepSpeech model."""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os
# pylint: disable=g-bad-import-order
from absl import app as absl_app
from absl import flags
import tensorflow as tf
# pylint: enable=g-bad-import-order

import data.dataset as dataset
import decoder
import deep_speech_model
from official.utils.flags import core as flags_core
from official.utils.logs import hooks_helper
from official.utils.logs import logger
from official.utils.misc import distribution_utils
from official.utils.misc import model_helpers

# Default vocabulary file
_VOCABULARY_FILE = os.path.join(
    os.path.dirname(__file__), "data/vocabulary.txt")
# Evaluation metrics
_WER_KEY = "WER"
_CER_KEY = "CER"


def evaluate_model(
    estimator, batch_size, speech_labels, targets, input_fn_eval):
  """Evaluate the model performance using WER anc CER as metrics.

  WER: Word Error Rate
  CER: Character Error Rate

  Args:
    estimator: estimator to evaluate.
    batch_size: size of a mini-batch.
    speech_labels: a string specifying all the character in the vocabulary.
    targets: a list of list of integers for the featurized transcript.
    input_fn_eval: data input function for evaluation.

  Returns:
    Evaluation result containing 'wer' and 'cer' as two metrics.
  """
  # Get predictions
  predictions = estimator.predict(
      input_fn=input_fn_eval, yield_single_examples=False)

  y_preds = []
  input_lengths = []
  for p in predictions:
    y_preds.append(p["y_pred"])
    input_lengths.append(p["ctc_input_length"])

  num_of_examples = len(targets)
  total_wer, total_cer = 0, 0
  greedy_decoder = decoder.GreedyDecoder(speech_labels)
  for i in range(len(y_preds)):
    # Compute the CER and WER for the current batch,
    # and aggregate to total_cer, total_wer.
    y_pred_tensor = tf.convert_to_tensor(y_preds[i])
    batch_targets = targets[i * batch_size : (i + 1) * batch_size]
    seq_len = tf.squeeze(input_lengths[i], axis=1)

    # Perform decoding
    _, decoded_output = greedy_decoder.decode(
        y_pred_tensor, seq_len)

    # Compute CER.
    batch_cer = greedy_decoder.batch_cer(decoded_output, batch_targets)
    total_cer += batch_cer
    # Compute WER.
    batch_wer = greedy_decoder.batch_wer(decoded_output, batch_targets)
    total_wer += batch_wer

  # Get mean value
  total_cer /= num_of_examples
  total_wer /= num_of_examples

  global_step = estimator.get_variable_value(tf.GraphKeys.GLOBAL_STEP)
  eval_results = {
      _WER_KEY: total_wer,
      _CER_KEY: total_cer,
      tf.GraphKeys.GLOBAL_STEP: global_step,
  }

  return eval_results


def convert_keras_to_estimator(keras_model, num_gpus):
  """Configure and convert keras model to Estimator.

  Args:
    keras_model: A Keras model object.
    num_gpus: An integer, the number of GPUs.

  Returns:
    estimator: The converted Estimator.
  """
  # keras optimizer is not compatible with distribution strategy.
  # Use tf optimizer instead
  optimizer = tf.train.MomentumOptimizer(
      learning_rate=flags_obj.learning_rate, momentum=flags_obj.momentum,
      use_nesterov=True)

  # ctc_loss is wrapped as a Lambda layer in the model.
  keras_model.compile(
      optimizer=optimizer, loss={"ctc_loss": lambda y_true, y_pred: y_pred})

  distribution_strategy = distribution_utils.get_distribution_strategy(
      num_gpus)
  run_config = tf.estimator.RunConfig(
      train_distribute=distribution_strategy)

  estimator = tf.keras.estimator.model_to_estimator(
      keras_model=keras_model, model_dir=flags_obj.model_dir, config=run_config)

  return estimator


def generate_dataset(data_dir):
  """Generate a speech dataset."""
  audio_conf = dataset.AudioConfig(
      flags_obj.sample_rate, flags_obj.frame_length, flags_obj.frame_step)
  train_data_conf = dataset.DatasetConfig(
      audio_conf,
      data_dir,
      flags_obj.vocabulary_file,
  )
  speech_dataset = dataset.DeepSpeechDataset(train_data_conf)
  return speech_dataset


def run_deep_speech(_):
  """Run deep speech training and eval loop."""
  # Data preprocessing
  # The file name of training and test dataset
  tf.logging.info("Data preprocessing...")

  train_speech_dataset = generate_dataset(flags_obj.train_data_dir)
  eval_speech_dataset = generate_dataset(flags_obj.eval_data_dir)

  # Number of label classes. Label string is "[a-z]' -"
  num_classes = len(train_speech_dataset.speech_labels)

  # Input shape of each data example:
  # [time_steps (T), feature_bins(F), channel(C)]
  # Channel is set as 1 by default.
  input_shape = (None, train_speech_dataset.num_feature_bins, 1)

  # Create deep speech model and convert it to Estimator
  tf.logging.info("Creating Estimator from Keras model...")
  keras_model = deep_speech_model.DeepSpeech(
      input_shape, flags_obj.rnn_hidden_layers, flags_obj.rnn_type,
      flags_obj.is_bidirectional, flags_obj.rnn_hidden_size,
      flags_obj.rnn_activation, num_classes, flags_obj.use_bias)

  # Convert to estimator
  num_gpus = flags_core.get_num_gpus(flags_obj)
  estimator = convert_keras_to_estimator(keras_model, num_gpus)

  # Benchmark logging
  run_params = {
      "batch_size": flags_obj.batch_size,
      "train_epochs": flags_obj.train_epochs,
      "rnn_hidden_size": flags_obj.rnn_hidden_size,
      "rnn_hidden_layers": flags_obj.rnn_hidden_layers,
      "rnn_activation": flags_obj.rnn_activation,
      "rnn_type": flags_obj.rnn_type,
      "is_bidirectional": flags_obj.is_bidirectional,
      "use_bias": flags_obj.use_bias
  }

  dataset_name = "LibriSpeech"
  benchmark_logger = logger.get_benchmark_logger()
  benchmark_logger.log_run_info("deep_speech", dataset_name, run_params,
                                test_id=flags_obj.benchmark_test_id)

  train_hooks = hooks_helper.get_train_hooks(
      flags_obj.hooks,
      batch_size=flags_obj.batch_size)

  per_device_batch_size = distribution_utils.per_device_batch_size(
      flags_obj.batch_size, num_gpus)

  def input_fn_train():
    return dataset.input_fn(
        per_device_batch_size, train_speech_dataset)

  def input_fn_eval():
    return dataset.input_fn(
        per_device_batch_size, eval_speech_dataset)

  total_training_cycle = (flags_obj.train_epochs //
                          flags_obj.epochs_between_evals)
  for cycle_index in range(total_training_cycle):
    tf.logging.info("Starting a training cycle: %d/%d",
                    cycle_index + 1, total_training_cycle)

    estimator.train(input_fn=input_fn_train, hooks=train_hooks)

    # Evaluation
    tf.logging.info("Starting to evaluate...")

    eval_results = evaluate_model(
        estimator, flags_obj.batch_size, eval_speech_dataset.speech_labels,
        eval_speech_dataset.labels, input_fn_eval)

    # Log the WER and CER results.
    benchmark_logger.log_evaluation_result(eval_results)
    tf.logging.info(
        "Iteration {}: WER = {:.2f}, CER = {:.2f}".format(
            cycle_index + 1, eval_results[_WER_KEY], eval_results[_CER_KEY]))

    # If some evaluation threshold is met
    if model_helpers.past_stop_threshold(
        flags_obj.wer_threshold, eval_results[_WER_KEY]):
      break

  # Clear the session explicitly to avoid session delete error
  tf.keras.backend.clear_session()


def define_deep_speech_flags():
  """Add flags for run_deep_speech."""
  # Add common flags
  flags_core.define_base(
      data_dir=False  # we use train_data_dir and eval_data_dir instead
  )
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
      model_dir="/tmp/deep_speech_model/",
      export_dir="/tmp/deep_speech_saved_model/",
      train_epochs=2,
      batch_size=4,
      hooks="")

  # Deep speech flags
  flags.DEFINE_string(
      name="train_data_dir",
      default="/tmp/librispeech_data/test-clean/LibriSpeech/test-clean-20.csv",
      help=flags_core.help_wrap("The csv file path of train dataset."))

  flags.DEFINE_string(
      name="eval_data_dir",
      default="/tmp/librispeech_data/test-clean/LibriSpeech/test-clean-20.csv",
      help=flags_core.help_wrap("The csv file path of evaluation dataset."))

  flags.DEFINE_integer(
      name="sample_rate", default=16000,
      help=flags_core.help_wrap("The sample rate for audio."))

  flags.DEFINE_integer(
      name="frame_length", default=25,
      help=flags_core.help_wrap("The frame length for spectrogram."))

  flags.DEFINE_integer(
      name="frame_step", default=10,
      help=flags_core.help_wrap("The frame step."))

  flags.DEFINE_string(
      name="vocabulary_file", default=_VOCABULARY_FILE,
      help=flags_core.help_wrap("The file path of vocabulary file."))

  # RNN related flags
  flags.DEFINE_integer(
      name="rnn_hidden_size", default=256,
      help=flags_core.help_wrap("The hidden size of RNNs."))

  flags.DEFINE_integer(
      name="rnn_hidden_layers", default=3,
      help=flags_core.help_wrap("The number of RNN layers."))

  flags.DEFINE_bool(
      name="use_bias", default=True,
      help=flags_core.help_wrap("Use bias in the last fully-connected layer"))

  flags.DEFINE_bool(
      name="is_bidirectional", default=True,
      help=flags_core.help_wrap("If rnn unit is bidirectional"))

  flags.DEFINE_enum(
      name="rnn_type", default="gru",
      enum_values=deep_speech_model.SUPPORTED_RNNS.keys(),
      case_sensitive=False,
      help=flags_core.help_wrap("Type of RNN cell."))

  flags.DEFINE_enum(
      name="rnn_activation", default="tanh",
      enum_values=["tanh", "relu"], case_sensitive=False,
      help=flags_core.help_wrap("Type of the activation within RNN."))

  # Training related flags
  flags.DEFINE_float(
      name="learning_rate", default=0.0003,
      help=flags_core.help_wrap("The initial learning rate."))

  flags.DEFINE_float(
      name="momentum", default=0.9,
      help=flags_core.help_wrap("Momentum to accelerate SGD optimizer."))

  # Evaluation metrics threshold
  flags.DEFINE_float(
      name="wer_threshold", default=None,
      help=flags_core.help_wrap(
          "If passed, training will stop when the evaluation metric WER is "
          "greater than or equal to wer_threshold. For libri speech dataset "
          "the desired wer_threshold is 0.23 which is the result achieved by "
          "MLPerf implementation."))


def main(_):
  with logger.benchmark_context(flags_obj):
    run_deep_speech(flags_obj)


if __name__ == "__main__":
  tf.logging.set_verbosity(tf.logging.INFO)
  define_deep_speech_flags()
  flags_obj = flags.FLAGS
  absl_app.run(main)
