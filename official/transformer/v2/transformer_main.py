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
"""Train and evaluate the Transformer model.

See README for description of setting the training schedule and evaluating the
BLEU score.
"""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os
import tempfile

from absl import app as absl_app
from absl import flags
import tensorflow as tf

# pylint: disable=g-bad-import-order
from official.transformer import compute_bleu
from official.transformer.utils import tokenizer
from official.transformer.v2 import data_pipeline
from official.transformer.v2 import misc
from official.transformer.v2 import optimizer
from official.transformer.v2 import transformer
from official.transformer.v2 import translate
from official.utils.flags import core as flags_core
from official.utils.logs import logger


INF = int(1e9)
BLEU_DIR = "bleu"
_SINGLE_SAMPLE = 1


def translate_and_compute_bleu(model, subtokenizer, bleu_source, bleu_ref):
  """Translate file and report the cased and uncased bleu scores."""
  # Create temporary file to store translation.
  tmp = tempfile.NamedTemporaryFile(delete=False)
  tmp_filename = tmp.name

  translate.translate_file(
      model,
      subtokenizer,
      bleu_source,
      output_file=tmp_filename,
      print_all_translations=False)

  # Compute uncased and cased bleu scores.
  uncased_score = compute_bleu.bleu_wrapper(bleu_ref, tmp_filename, False)
  cased_score = compute_bleu.bleu_wrapper(bleu_ref, tmp_filename, True)
  os.remove(tmp_filename)
  return uncased_score, cased_score


def evaluate_and_log_bleu(model, bleu_source, bleu_ref, vocab_file):
  """Calculate and record the BLEU score."""
  subtokenizer = tokenizer.Subtokenizer(vocab_file)

  uncased_score, cased_score = translate_and_compute_bleu(
      model, subtokenizer, bleu_source, bleu_ref)

  tf.compat.v1.logging.info("Bleu score (uncased): %s", uncased_score)
  tf.compat.v1.logging.info("Bleu score (cased): %s", cased_score)
  return uncased_score, cased_score


class TransformerTask(object):
  """Main entry of Transformer model."""

  def __init__(self, flags_obj):
    """Init function of TransformerMain.

    Args:
      flags_obj: Object containing parsed flag values, i.e., FLAGS.
    """
    self.flags_obj = flags_obj
    self.predict_model = None

    # Add flag-defined parameters to params object
    num_gpus = flags_core.get_num_gpus(flags_obj)
    self.params = params = misc.get_model_params(flags_obj.param_set, num_gpus)

    params["data_dir"] = flags_obj.data_dir
    params["model_dir"] = flags_obj.model_dir
    params["num_parallel_calls"] = (
        flags_obj.num_parallel_calls or tf.data.experimental.AUTOTUNE)

    params["use_synthetic_data"] = flags_obj.use_synthetic_data
    params["batch_size"] = flags_obj.batch_size or params["default_batch_size"]
    params["repeat_dataset"] = None

  def train(self):
    """Trains the model."""
    params, flags_obj, is_train = self.params, self.flags_obj, True
    _ensure_dir(flags_obj.model_dir)
    model = transformer.create_model(params, is_train)
    opt = self._create_optimizer()

    model.compile(opt, target_tensors=[])
    model.summary()

    map_data_fn = data_pipeline.map_data_for_transformer_fn
    train_ds = data_pipeline.train_input_fn(params)
    train_ds = train_ds.map(
        map_data_fn, num_parallel_calls=params["num_parallel_calls"])

    callbacks = self._create_callbacks(flags_obj.model_dir, 0, params)

    if flags_obj.train_steps < flags_obj.steps_between_evals:
      flags_obj.steps_between_evals = flags_obj.train_steps
    iterations = flags_obj.train_steps // flags_obj.steps_between_evals

    cased_score, uncased_score = None, None
    for i in range(1, iterations + 1):
      print("Start train iteration:{}/{}".format(i, iterations))
      history = model.fit(
          train_ds,
          initial_epoch=i-1,
          epochs=i,
          steps_per_epoch=flags_obj.steps_between_evals,
          callbacks=callbacks,
          verbose=2)
      print("End train iteration:{}/{} global step:{}".format(
          i,
          iterations,
          i*flags_obj.steps_between_evals))
      tf.compat.v1.logging.info("Train history: {}".format(history.history))
      stats = misc.build_stats(history, callbacks)

      if (flags_obj.bleu_source and flags_obj.bleu_ref):
        uncased_score, cased_score = self.eval()

    stats = misc.build_stats(history, callbacks)
    if uncased_score and cased_score:
      stats["bleu_uncased"] = uncased_score
      stats["bleu_cased"] = cased_score
    return stats

  def eval(self):
    """Evaluates the model."""
    if not self.predict_model:
      self.predict_model = transformer.create_model(self.params, False)
    self._load_weights_if_possible(
        self.predict_model,
        tf.train.latest_checkpoint(self.flags_obj.model_dir))
    self.predict_model.summary()
    return evaluate_and_log_bleu(self.predict_model,
                                 self.flags_obj.bleu_source,
                                 self.flags_obj.bleu_ref,
                                 self.flags_obj.vocab_file)

  def predict(self):
    """Predicts result from the model."""
    params, flags_obj, is_train = self.params, self.flags_obj, False

    with tf.name_scope("model"):
      model = transformer.create_model(params, is_train)
      self._load_weights_if_possible(model, flags_obj.init_weight_path)
      model.summary()
    subtokenizer = tokenizer.Subtokenizer(flags_obj.vocab_file)

    ds = data_pipeline.eval_input_fn(params)
    ds = ds.map(lambda x, y: x).take(_SINGLE_SAMPLE)
    ret = model.predict(ds)
    val_outputs, _ = ret
    length = len(val_outputs)
    for i in range(length):
      translate.translate_from_input(val_outputs[i], subtokenizer)

  def _create_callbacks(self, cur_log_dir, init_steps, params):
    """Creates a list of callbacks."""
    sfunc = optimizer.LearningRateFn(params["learning_rate"],
                                     params["hidden_size"],
                                     params["learning_rate_warmup_steps"])
    scheduler_callback = optimizer.LearningRateScheduler(sfunc, init_steps)
    callbacks = misc.get_callbacks()
    callbacks.append(scheduler_callback)
    ckpt_full_path = os.path.join(cur_log_dir, "cp-{epoch:04d}.ckpt")
    callbacks.append(tf.keras.callbacks.ModelCheckpoint(ckpt_full_path,
                                                        save_weights_only=True))
    return callbacks

  def _load_weights_if_possible(self, model, init_weight_path=None):
    """Loads model weights when it is provided."""
    if init_weight_path:
      tf.compat.v1.logging.info("Load weights: {}".format(init_weight_path))
      model.load_weights(init_weight_path)
    else:
      print("Weights not loaded from path:{}".format(init_weight_path))

  def _create_optimizer(self):
    """Creates optimizer."""
    params = self.params
    opt = optimizer.LazyAdam(
        params["learning_rate"],
        params["optimizer_adam_beta1"],
        params["optimizer_adam_beta2"],
        epsilon=params["optimizer_adam_epsilon"])
    return opt


def _ensure_dir(log_dir):
  """Makes log dir if not existed."""
  if not os.path.exists(log_dir):
    os.makedirs(log_dir)


def main(_):
  flags_obj = flags.FLAGS
  with logger.benchmark_context(flags_obj):
    task = TransformerTask(flags_obj)
    if flags_obj.mode == "train":
      task.train()
    elif flags_obj.mode == "predict":
      task.predict()
    elif flags_obj.mode == "eval":
      task.eval()
    else:
      raise ValueError("Invalid mode {}".format(flags_obj.mode))


if __name__ == "__main__":
  tf.compat.v1.logging.set_verbosity(tf.compat.v1.logging.INFO)
  misc.define_transformer_flags()
  absl_app.run(main)
