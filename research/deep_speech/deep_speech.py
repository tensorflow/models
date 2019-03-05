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


def compute_length_after_conv(max_time_steps, ctc_time_steps, input_length):
  """Computes the time_steps/ctc_input_length after convolution.

  Suppose that the original feature contains two parts:
  1) Real spectrogram signals, spanning input_length steps.
  2) Padded part with all 0s.
  The total length of those two parts is denoted as max_time_steps, which is
  the padded length of the current batch. After convolution layers, the time
  steps of a spectrogram feature will be decreased. As we know the percentage
  of its original length within the entire length, we can compute the time steps
  for the signal after conv as follows (using ctc_input_length to denote):
  ctc_input_length = (input_length / max_time_steps) * output_length_of_conv.
  This length is then fed into ctc loss function to compute loss.

  Args:
    max_time_steps: max_time_steps for the batch, after padding.
    ctc_time_steps: number of timesteps after convolution.
    input_length: actual length of the original spectrogram, without padding.

  Returns:
    the ctc_input_length after convolution layer.
  """
  ctc_input_length = tf.to_float(tf.multiply(
      input_length, ctc_time_steps))
  return tf.to_int32(tf.floordiv(
      ctc_input_length, tf.to_float(max_time_steps)))


def ctc_loss(label_length, ctc_input_length, labels, logits):
  """Computes the ctc loss for the current batch of predictions."""
  label_length = tf.to_int32(tf.squeeze(label_length))
  ctc_input_length = tf.to_int32(tf.squeeze(ctc_input_length))
  sparse_labels = tf.to_int32(
      tf.keras.backend.ctc_label_dense_to_sparse(labels, label_length))
  y_pred = tf.log(tf.transpose(
      logits, perm=[1, 0, 2]) + tf.keras.backend.epsilon())

  return tf.expand_dims(
      tf.nn.ctc_loss(labels=sparse_labels, inputs=y_pred,
                     sequence_length=ctc_input_length),
      axis=1)


def evaluate_model(estimator, speech_labels, entries, input_fn_eval):
  """Evaluate the model performance using WER anc CER as metrics.

  WER: Word Error Rate
  CER: Character Error Rate

  Args:
    estimator: estimator to evaluate.
    speech_labels: a string specifying all the character in the vocabulary.
    entries: a list of data entries (audio_file, file_size, transcript) for the
      given dataset.
    input_fn_eval: data input function for evaluation.

  Returns:
    Evaluation result containing 'wer' and 'cer' as two metrics.
  """
  # Get predictions
  predictions = estimator.predict(input_fn=input_fn_eval)

  # Get probabilities of each predicted class
  probs = [pred["probabilities"] for pred in predictions]

  num_of_examples = len(probs)
  targets = [entry[2] for entry in entries]  # The ground truth transcript

  total_wer, total_cer = 0, 0
  greedy_decoder = decoder.DeepSpeechDecoder(speech_labels)
  for i in range(num_of_examples):
    # Decode string.
    decoded_str = greedy_decoder.decode(probs[i])
    # Compute CER.
    total_cer += greedy_decoder.cer(decoded_str, targets[i]) / float(
        len(targets[i]))
    # Compute WER.
    total_wer += greedy_decoder.wer(decoded_str, targets[i]) / float(
        len(targets[i].split()))

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


def model_fn(features, labels, mode, params):
  """Define model function for deep speech model.

  Args:
    features: a dictionary of input_data features. It includes the data
      input_length, label_length and the spectrogram features.
    labels: a list of labels for the input data.
    mode: current estimator mode; should be one of
      `tf.estimator.ModeKeys.TRAIN`, `EVALUATE`, `PREDICT`.
    params: a dict of hyper parameters to be passed to model_fn.

  Returns:
    EstimatorSpec parameterized according to the input params and the
    current mode.
  """
  num_classes = params["num_classes"]
  input_length = features["input_length"]
  label_length = features["label_length"]
  features = features["features"]

  # Create DeepSpeech2 model.
  model = deep_speech_model.DeepSpeech2(
      flags_obj.rnn_hidden_layers, flags_obj.rnn_type,
      flags_obj.is_bidirectional, flags_obj.rnn_hidden_size,
      num_classes, flags_obj.use_bias)

  if mode == tf.estimator.ModeKeys.PREDICT:
    logits = model(features, training=False)
    predictions = {
        "classes": tf.argmax(logits, axis=2),
        "probabilities": tf.nn.softmax(logits),
        "logits": logits
    }
    return tf.estimator.EstimatorSpec(
        mode=mode,
        predictions=predictions)

  # In training mode.
  logits = model(features, training=True)
  probs = tf.nn.softmax(logits)
  ctc_input_length = compute_length_after_conv(
      tf.shape(features)[1], tf.shape(probs)[1], input_length)
  # Compute CTC loss
  loss = tf.reduce_mean(ctc_loss(
      label_length, ctc_input_length, labels, probs))

  optimizer = tf.train.AdamOptimizer(learning_rate=flags_obj.learning_rate)
  global_step = tf.train.get_or_create_global_step()
  minimize_op = optimizer.minimize(loss, global_step=global_step)
  update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS)
  # Create the train_op that groups both minimize_ops and update_ops
  train_op = tf.group(minimize_op, update_ops)

  return tf.estimator.EstimatorSpec(
      mode=mode,
      loss=loss,
      train_op=train_op)


def generate_dataset(data_dir):
  """Generate a speech dataset."""
  audio_conf = dataset.AudioConfig(sample_rate=flags_obj.sample_rate,
                                   window_ms=flags_obj.window_ms,
                                   stride_ms=flags_obj.stride_ms,
                                   normalize=True)
  train_data_conf = dataset.DatasetConfig(
      audio_conf,
      data_dir,
      flags_obj.vocabulary_file,
      flags_obj.sortagrad
  )
  speech_dataset = dataset.DeepSpeechDataset(train_data_conf)
  return speech_dataset


def run_deep_speech(_):
  """Run deep speech training and eval loop."""
  tf.set_random_seed(flags_obj.seed)
  # Data preprocessing
  tf.logging.info("Data preprocessing...")
  train_speech_dataset = generate_dataset(flags_obj.train_data_dir)
  eval_speech_dataset = generate_dataset(flags_obj.eval_data_dir)

  # Number of label classes. Label string is "[a-z]' -"
  num_classes = len(train_speech_dataset.speech_labels)

  # Use distribution strategy for multi-gpu training
  num_gpus = flags_core.get_num_gpus(flags_obj)
  distribution_strategy = distribution_utils.get_distribution_strategy(num_gpus)
  run_config = tf.estimator.RunConfig(
      train_distribute=distribution_strategy)

  estimator = tf.estimator.Estimator(
      model_fn=model_fn,
      model_dir=flags_obj.model_dir,
      config=run_config,
      params={
          "num_classes": num_classes,
      }
  )

  # Benchmark logging
  run_params = {
      "batch_size": flags_obj.batch_size,
      "train_epochs": flags_obj.train_epochs,
      "rnn_hidden_size": flags_obj.rnn_hidden_size,
      "rnn_hidden_layers": flags_obj.rnn_hidden_layers,
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
      model_dir=flags_obj.model_dir,
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

    # Perform batch_wise dataset shuffling
    train_speech_dataset.entries = dataset.batch_wise_dataset_shuffle(
        train_speech_dataset.entries, cycle_index, flags_obj.sortagrad,
        flags_obj.batch_size)

    estimator.train(input_fn=input_fn_train, hooks=train_hooks)

    # Evaluation
    tf.logging.info("Starting to evaluate...")

    eval_results = evaluate_model(
        estimator, eval_speech_dataset.speech_labels,
        eval_speech_dataset.entries, input_fn_eval)

    # Log the WER and CER results.
    benchmark_logger.log_evaluation_result(eval_results)
    tf.logging.info(
        "Iteration {}: WER = {:.2f}, CER = {:.2f}".format(
            cycle_index + 1, eval_results[_WER_KEY], eval_results[_CER_KEY]))

    # If some evaluation threshold is met
    if model_helpers.past_stop_threshold(
        flags_obj.wer_threshold, eval_results[_WER_KEY]):
      break


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
      train_epochs=10,
      batch_size=128,
      hooks="")

  # Deep speech flags
  flags.DEFINE_integer(
      name="seed", default=1,
      help=flags_core.help_wrap("The random seed."))

  flags.DEFINE_string(
      name="train_data_dir",
      default="/tmp/librispeech_data/test-clean/LibriSpeech/test-clean.csv",
      help=flags_core.help_wrap("The csv file path of train dataset."))

  flags.DEFINE_string(
      name="eval_data_dir",
      default="/tmp/librispeech_data/test-clean/LibriSpeech/test-clean.csv",
      help=flags_core.help_wrap("The csv file path of evaluation dataset."))

  flags.DEFINE_bool(
      name="sortagrad", default=True,
      help=flags_core.help_wrap(
          "If true, sort examples by audio length and perform no "
          "batch_wise shuffling for the first epoch."))

  flags.DEFINE_integer(
      name="sample_rate", default=16000,
      help=flags_core.help_wrap("The sample rate for audio."))

  flags.DEFINE_integer(
      name="window_ms", default=20,
      help=flags_core.help_wrap("The frame length for spectrogram."))

  flags.DEFINE_integer(
      name="stride_ms", default=10,
      help=flags_core.help_wrap("The frame step."))

  flags.DEFINE_string(
      name="vocabulary_file", default=_VOCABULARY_FILE,
      help=flags_core.help_wrap("The file path of vocabulary file."))

  # RNN related flags
  flags.DEFINE_integer(
      name="rnn_hidden_size", default=800,
      help=flags_core.help_wrap("The hidden size of RNNs."))

  flags.DEFINE_integer(
      name="rnn_hidden_layers", default=5,
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

  # Training related flags
  flags.DEFINE_float(
      name="learning_rate", default=5e-4,
      help=flags_core.help_wrap("The initial learning rate."))

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

