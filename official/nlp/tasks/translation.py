# Lint as: python3
# Copyright 2020 The TensorFlow Authors. All Rights Reserved.
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
"""Defines the translation task."""
import os
from typing import Optional

from absl import logging
import dataclasses
import sacrebleu
import tensorflow as tf
import tensorflow_text as tftxt

from official.core import base_task
from official.core import config_definitions as cfg
from official.core import task_factory
from official.modeling.hyperparams import base_config
from official.nlp.data import data_loader_factory
from official.nlp.metrics import bleu
from official.nlp.modeling import models


def _pad_tensors_to_same_length(x, y):
  """Pad x and y so that the results have the same length (second dimension)."""
  x_length = tf.shape(x)[1]
  y_length = tf.shape(y)[1]

  max_length = tf.maximum(x_length, y_length)

  x = tf.pad(x, [[0, 0], [0, max_length - x_length], [0, 0]])
  y = tf.pad(y, [[0, 0], [0, max_length - y_length]])
  return x, y


def _padded_cross_entropy_loss(logits, labels, smoothing, vocab_size):
  """Calculate cross entropy loss while ignoring padding.

  Args:
    logits: Tensor of size [batch_size, length_logits, vocab_size]
    labels: Tensor of size [batch_size, length_labels]
    smoothing: Label smoothing constant, used to determine the on and off values
    vocab_size: int size of the vocabulary

  Returns:
    Returns the cross entropy loss and weight tensors: float32 tensors with
      shape [batch_size, max(length_logits, length_labels)]
  """
  logits, labels = _pad_tensors_to_same_length(logits, labels)

  # Calculate smoothing cross entropy
  confidence = 1.0 - smoothing
  low_confidence = (1.0 - confidence) / tf.cast(vocab_size - 1, tf.float32)
  soft_targets = tf.one_hot(
      tf.cast(labels, tf.int32),
      depth=vocab_size,
      on_value=confidence,
      off_value=low_confidence)
  xentropy = tf.nn.softmax_cross_entropy_with_logits(
      logits=logits, labels=soft_targets)

  # Calculate the best (lowest) possible value of cross entropy, and
  # subtract from the cross entropy loss.
  normalizing_constant = -(
      confidence * tf.math.log(confidence) + tf.cast(vocab_size - 1, tf.float32)
      * low_confidence * tf.math.log(low_confidence + 1e-20))
  xentropy -= normalizing_constant

  weights = tf.cast(tf.not_equal(labels, 0), tf.float32)
  return xentropy * weights, weights


@dataclasses.dataclass
class EncDecoder(base_config.Config):
  """Configurations for Encoder/Decoder."""
  num_layers: int = 6
  num_attention_heads: int = 8
  intermediate_size: int = 2048
  activation: str = "relu"
  dropout_rate: float = 0.1
  attention_dropout_rate: float = 0.1
  intermediate_dropout: float = 0.1
  use_bias: bool = False
  norm_first: bool = True
  norm_epsilon: float = 1e-6


@dataclasses.dataclass
class ModelConfig(base_config.Config):
  """A base Seq2Seq model configuration."""
  encoder: EncDecoder = EncDecoder()
  decoder: EncDecoder = EncDecoder()

  embedding_width: int = 512
  dropout_rate: float = 0.1

  # Decoding.
  padded_decode: bool = False
  decode_max_length: Optional[int] = None
  beam_size: int = 4
  alpha: float = 0.6

  # Training.
  label_smoothing: float = 0.1


@dataclasses.dataclass
class TranslationConfig(cfg.TaskConfig):
  """The translation task config."""
  model: ModelConfig = ModelConfig()
  train_data: cfg.DataConfig = cfg.DataConfig()
  validation_data: cfg.DataConfig = cfg.DataConfig()
  # Tokenization
  sentencepiece_model_path: str = ""
  # Evaluation.
  print_translations: Optional[bool] = None


def write_test_record(params, model_dir):
  """Writes the test input to a tfrecord."""
  # Get raw data from tfds.
  params = params.replace(transform_and_batch=False)
  dataset = data_loader_factory.get_data_loader(params).load()
  references = []
  total_samples = 0
  output_file = os.path.join(model_dir, "eval.tf_record")
  writer = tf.io.TFRecordWriter(output_file)
  for d in dataset:
    references.append(d[params.tgt_lang].numpy().decode())
    example = tf.train.Example(
        features=tf.train.Features(
            feature={
                "unique_id": tf.train.Feature(
                    int64_list=tf.train.Int64List(value=[total_samples])),
                params.src_lang: tf.train.Feature(
                    bytes_list=tf.train.BytesList(
                        value=[d[params.src_lang].numpy()])),
                params.tgt_lang: tf.train.Feature(
                    bytes_list=tf.train.BytesList(
                        value=[d[params.tgt_lang].numpy()])),
            }))
    writer.write(example.SerializeToString())
    total_samples += 1
  batch_size = params.global_batch_size
  num_dummy_example = batch_size - total_samples % batch_size
  for i in range(num_dummy_example):
    example = tf.train.Example(
        features=tf.train.Features(
            feature={
                "unique_id": tf.train.Feature(
                    int64_list=tf.train.Int64List(value=[total_samples + i])),
                params.src_lang: tf.train.Feature(
                    bytes_list=tf.train.BytesList(value=[b""])),
                params.tgt_lang: tf.train.Feature(
                    bytes_list=tf.train.BytesList(value=[b""])),
            }))
    writer.write(example.SerializeToString())
  writer.close()
  return references, output_file


@task_factory.register_task_cls(TranslationConfig)
class TranslationTask(base_task.Task):
  """A single-replica view of training procedure.

  Tasks provide artifacts for training/evalution procedures, including
  loading/iterating over Datasets, initializing the model, calculating the loss
  and customized metrics with reduction.
  """

  def __init__(self, params: cfg.TaskConfig, logging_dir=None, name=None):
    super().__init__(params, logging_dir, name=name)
    self._sentencepiece_model_path = params.sentencepiece_model_path
    if params.sentencepiece_model_path:
      self._sp_tokenizer = tftxt.SentencepieceTokenizer(
          model=tf.io.gfile.GFile(params.sentencepiece_model_path, "rb").read(),
          add_eos=True)
      try:
        empty_str_tokenized = self._sp_tokenizer.tokenize("").numpy()
      except tf.errors.InternalError:
        raise ValueError(
            "EOS token not in tokenizer vocab."
            "Please make sure the tokenizer generates a single token for an "
            "empty string.")
      self._eos_id = empty_str_tokenized.item()
      self._vocab_size = self._sp_tokenizer.vocab_size().numpy()
    else:
      raise ValueError("Setencepiece model path not provided.")
    if (params.validation_data.input_path or
        params.validation_data.tfds_name) and self._logging_dir:
      self._references, self._tf_record_input_path = write_test_record(
          params.validation_data, self.logging_dir)

  def build_model(self) -> tf.keras.Model:
    """Creates model architecture.

    Returns:
      A model instance.
    """
    model_cfg = self.task_config.model
    encoder_kwargs = model_cfg.encoder.as_dict()
    encoder_layer = models.TransformerEncoder(**encoder_kwargs)
    decoder_kwargs = model_cfg.decoder.as_dict()
    decoder_layer = models.TransformerDecoder(**decoder_kwargs)

    return models.Seq2SeqTransformer(
        vocab_size=self._vocab_size,
        embedding_width=model_cfg.embedding_width,
        dropout_rate=model_cfg.dropout_rate,
        padded_decode=model_cfg.padded_decode,
        decode_max_length=model_cfg.decode_max_length,
        beam_size=model_cfg.beam_size,
        alpha=model_cfg.alpha,
        encoder_layer=encoder_layer,
        decoder_layer=decoder_layer,
        eos_id=self._eos_id)

  def build_inputs(self,
                   params: cfg.DataConfig,
                   input_context: Optional[tf.distribute.InputContext] = None):
    """Returns a dataset."""
    if params.is_training:
      dataloader_params = params
    else:
      input_path = self._tf_record_input_path
      # Read from padded tf records instead.
      dataloader_params = params.replace(
          input_path=input_path,
          tfds_name="",
          tfds_split="",
          has_unique_id=True)
    dataloader_params = dataloader_params.replace(
        sentencepiece_model_path=self._sentencepiece_model_path)
    return data_loader_factory.get_data_loader(dataloader_params).load(
        input_context)

  def build_losses(self, labels, model_outputs, aux_losses=None) -> tf.Tensor:
    """Standard interface to compute losses.

    Args:
      labels: optional label tensors.
      model_outputs: a nested structure of output tensors.
      aux_losses: auxiliary loss tensors, i.e. `losses` in keras.Model.

    Returns:
      The total loss tensor.
    """
    del aux_losses

    smoothing = self.task_config.model.label_smoothing
    xentropy, weights = _padded_cross_entropy_loss(model_outputs, labels,
                                                   smoothing, self._vocab_size)
    return tf.reduce_sum(xentropy) / tf.reduce_sum(weights)

  def train_step(self,
                 inputs,
                 model: tf.keras.Model,
                 optimizer: tf.keras.optimizers.Optimizer,
                 metrics=None):
    """Does forward and backward.

    With distribution strategies, this method runs on devices.

    Args:
      inputs: a dictionary of input tensors.
      model: the model, forward pass definition.
      optimizer: the optimizer for this training step.
      metrics: a nested structure of metrics objects.

    Returns:
      A dictionary of logs.
    """
    with tf.GradientTape() as tape:
      outputs = model(inputs, training=True)
      # Computes per-replica loss.
      loss = self.build_losses(labels=inputs["targets"], model_outputs=outputs)
      # Scales loss as the default gradients allreduce performs sum inside the
      # optimizer.
      scaled_loss = loss / tf.distribute.get_strategy().num_replicas_in_sync

      # For mixed precision, when a LossScaleOptimizer is used, the loss is
      # scaled to avoid numeric underflow.
      if isinstance(optimizer, tf.keras.mixed_precision.LossScaleOptimizer):
        scaled_loss = optimizer.get_scaled_loss(scaled_loss)

    tvars = model.trainable_variables
    grads = tape.gradient(scaled_loss, tvars)

    if isinstance(optimizer, tf.keras.mixed_precision.LossScaleOptimizer):
      grads = optimizer.get_unscaled_gradients(grads)
    optimizer.apply_gradients(list(zip(grads, tvars)))
    logs = {self.loss: loss}
    if metrics:
      self.process_metrics(metrics, inputs["targets"], outputs)
      logs.update({m.name: m.result() for m in metrics})
    return logs

  def validation_step(self, inputs, model: tf.keras.Model, metrics=None):
    unique_ids = inputs.pop("unique_id")
    # Validation loss
    outputs = model(inputs, training=False)
    # Computes per-replica loss to help understand if we are overfitting.
    loss = self.build_losses(labels=inputs["targets"], model_outputs=outputs)
    inputs.pop("targets")
    # Beam search to calculate metrics.
    model_outputs = model(inputs, training=False)
    outputs = model_outputs
    logs = {
        self.loss: loss,
        "inputs": inputs["inputs"],
        "unique_ids": unique_ids,
    }
    logs.update(outputs)
    return logs

  def aggregate_logs(self, state=None, step_outputs=None):
    """Aggregates over logs returned from a validation step."""
    if state is None:
      state = {}

    for in_token_ids, out_token_ids, unique_ids in zip(
        step_outputs["inputs"],
        step_outputs["outputs"],
        step_outputs["unique_ids"]):
      for in_ids, out_ids, u_id in zip(
          in_token_ids.numpy(), out_token_ids.numpy(), unique_ids.numpy()):
        state[u_id] = (in_ids, out_ids)
    return state

  def reduce_aggregated_logs(self, aggregated_logs):

    def _decode(ids):
      return self._sp_tokenizer.detokenize(ids).numpy().decode()

    def _trim_and_decode(ids):
      """Trim EOS and PAD tokens from ids, and decode to return a string."""
      try:
        index = list(ids).index(self._eos_id)
        return _decode(ids[:index])
      except ValueError:  # No EOS found in sequence
        return _decode(ids)

    translations = []
    for u_id in sorted(aggregated_logs):
      if u_id >= len(self._references):
        continue
      src = _trim_and_decode(aggregated_logs[u_id][0])
      translation = _trim_and_decode(aggregated_logs[u_id][1])
      translations.append(translation)
      if self.task_config.print_translations:
        # Deccoding the in_ids to reflect what the model sees.
        logging.info("Translating:\n\tInput: %s\n\tOutput: %s\n\tReference: %s",
                     src, translation, self._references[u_id])
    sacrebleu_score = sacrebleu.corpus_bleu(
        translations, [self._references]).score
    bleu_score = bleu.bleu_on_list(self._references, translations)
    return {"sacrebleu_score": sacrebleu_score,
            "bleu_score": bleu_score}
