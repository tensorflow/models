# Copyright 2021 The TensorFlow Authors. All Rights Reserved.
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

"""Library of components of export_tfhub.py. See docstring there for more."""

import contextlib
import hashlib
import os
import tempfile

from typing import Optional, Text, Tuple

# Import libraries
from absl import logging
import tensorflow as tf
# pylint: disable=g-direct-tensorflow-import  TODO(b/175369555): Remove these.
from tensorflow.core.protobuf import saved_model_pb2
from tensorflow.python.ops import control_flow_ops
# pylint: enable=g-direct-tensorflow-import
from official.modeling import tf_utils
from official.nlp.bert import configs
from official.nlp.configs import encoders
from official.nlp.modeling import layers
from official.nlp.modeling import models
from official.nlp.modeling import networks


def get_bert_encoder(bert_config):
  """Returns a BertEncoder with dict outputs."""
  bert_encoder = networks.BertEncoder(
      vocab_size=bert_config.vocab_size,
      hidden_size=bert_config.hidden_size,
      num_layers=bert_config.num_hidden_layers,
      num_attention_heads=bert_config.num_attention_heads,
      intermediate_size=bert_config.intermediate_size,
      activation=tf_utils.get_activation(bert_config.hidden_act),
      dropout_rate=bert_config.hidden_dropout_prob,
      attention_dropout_rate=bert_config.attention_probs_dropout_prob,
      max_sequence_length=bert_config.max_position_embeddings,
      type_vocab_size=bert_config.type_vocab_size,
      initializer=tf.keras.initializers.TruncatedNormal(
          stddev=bert_config.initializer_range),
      embedding_width=bert_config.embedding_size,
      dict_outputs=True)

  return bert_encoder


def get_do_lower_case(do_lower_case, vocab_file=None, sp_model_file=None):
  """Returns do_lower_case, replacing None by a guess from vocab file name."""
  if do_lower_case is not None:
    return do_lower_case
  elif vocab_file:
    do_lower_case = "uncased" in vocab_file
    logging.info("Using do_lower_case=%s based on name of vocab_file=%s",
                 do_lower_case, vocab_file)
    return do_lower_case
  elif sp_model_file:
    do_lower_case = True  # All public ALBERTs (as of Oct 2020) do it.
    logging.info("Defaulting to do_lower_case=%s for Sentencepiece tokenizer",
                 do_lower_case)
    return do_lower_case
  else:
    raise ValueError("Must set vocab_file or sp_model_file.")


def _create_model(
    *,
    bert_config: Optional[configs.BertConfig] = None,
    encoder_config: Optional[encoders.EncoderConfig] = None,
    with_mlm: bool,
) -> Tuple[tf.keras.Model, tf.keras.Model]:
  """Creates the model to export and the model to restore the checkpoint.

  Args:
    bert_config: A legacy `BertConfig` to create a `BertEncoder` object.
      Exactly one of encoder_config and bert_config must be set.
    encoder_config: An `EncoderConfig` to create an encoder of the configured
      type (`BertEncoder` or other).
    with_mlm: A bool to control the second component of the result.
      If True, will create a `BertPretrainerV2` object; otherwise, will
      create a `BertEncoder` object.

  Returns:
    A Tuple of (1) a Keras model that will be exported, (2) a `BertPretrainerV2`
    object or `BertEncoder` object depending on the value of `with_mlm`
    argument, which contains the first model and will be used for restoring
    weights from the checkpoint.
  """
  if (bert_config is not None) == (encoder_config is not None):
    raise ValueError("Exactly one of `bert_config` and `encoder_config` "
                     "can be specified, but got %s and %s" %
                     (bert_config, encoder_config))

  if bert_config is not None:
    encoder = get_bert_encoder(bert_config)
  else:
    encoder = encoders.build_encoder(encoder_config)

  # Convert from list of named inputs to dict of inputs keyed by name.
  # Only the latter accepts a dict of inputs after restoring from SavedModel.
  encoder_inputs_dict = {x.name: x for x in encoder.inputs}
  encoder_output_dict = encoder(encoder_inputs_dict)
  # For interchangeability with other text representations,
  # add "default" as an alias for BERT's whole-input reptesentations.
  encoder_output_dict["default"] = encoder_output_dict["pooled_output"]
  core_model = tf.keras.Model(
      inputs=encoder_inputs_dict, outputs=encoder_output_dict)

  if with_mlm:
    if bert_config is not None:
      hidden_act = bert_config.hidden_act
    else:
      assert encoder_config is not None
      hidden_act = encoder_config.get().hidden_activation

    pretrainer = models.BertPretrainerV2(
        encoder_network=encoder,
        mlm_activation=tf_utils.get_activation(hidden_act))

    pretrainer_inputs_dict = {x.name: x for x in pretrainer.inputs}
    pretrainer_output_dict = pretrainer(pretrainer_inputs_dict)
    mlm_model = tf.keras.Model(
        inputs=pretrainer_inputs_dict, outputs=pretrainer_output_dict)
    # Set `_auto_track_sub_layers` to False, so that the additional weights
    # from `mlm` sub-object will not be included in the core model.
    # TODO(b/169210253): Use a public API when available.
    core_model._auto_track_sub_layers = False  # pylint: disable=protected-access
    core_model.mlm = mlm_model
    return core_model, pretrainer
  else:
    return core_model, encoder


def export_model(export_path: Text,
                 *,
                 bert_config: Optional[configs.BertConfig] = None,
                 encoder_config: Optional[encoders.EncoderConfig] = None,
                 model_checkpoint_path: Text,
                 with_mlm: bool,
                 copy_pooler_dense_to_encoder: bool = False,
                 vocab_file: Optional[Text] = None,
                 sp_model_file: Optional[Text] = None,
                 do_lower_case: Optional[bool] = None) -> None:
  """Exports an Encoder as SavedModel after restoring pre-trained weights.

  The exported SavedModel implements a superset of the Encoder API for
  Text embeddings with Transformer Encoders described at
  https://www.tensorflow.org/hub/common_saved_model_apis/text.

  In particular, the exported SavedModel can be used in the following way:

  ```
  # Calls default interface (encoder only).

  encoder = hub.load(...)
  encoder_inputs = dict(
      input_word_ids=...,  # Shape [batch, seq_length], dtype=int32
      input_mask=...,      # Shape [batch, seq_length], dtype=int32
      input_type_ids=...,  # Shape [batch, seq_length], dtype=int32
  )
  encoder_outputs = encoder(encoder_inputs)
  assert encoder_outputs.keys() == {
    "pooled_output",   # Shape [batch_size, width], dtype=float32
    "default",         # Alias for "pooled_output" (aligns with other models).
    "sequence_output"  # Shape [batch_size, seq_length, width], dtype=float32
    "encoder_outputs", # List of Tensors with outputs of all transformer layers.
  }
  ```

  If `with_mlm` is True, the exported SavedModel can also be called in the
  following way:

  ```
  # Calls expanded interface that includes logits of the Masked Language Model.
  mlm_inputs = dict(
      input_word_ids=...,       # Shape [batch, seq_length], dtype=int32
      input_mask=...,           # Shape [batch, seq_length], dtype=int32
      input_type_ids=...,       # Shape [batch, seq_length], dtype=int32
      masked_lm_positions=...,  # Shape [batch, num_predictions], dtype=int32
  )
  mlm_outputs = encoder.mlm(mlm_inputs)
  assert mlm_outputs.keys() == {
    "pooled_output",   # Shape [batch, width], dtype=float32
    "sequence_output", # Shape [batch, seq_length, width], dtype=float32
    "encoder_outputs", # List of Tensors with outputs of all transformer layers.
    "mlm_logits"    # Shape [batch, num_predictions, vocab_size], dtype=float32
  }
  ```

  Args:
    export_path: The SavedModel output directory.
    bert_config: An optional `configs.BertConfig` object. Note: exactly one of
      `bert_config` and following `encoder_config` must be specified.
    encoder_config: An optional `encoders.EncoderConfig` object.
    model_checkpoint_path: The path to the checkpoint.
    with_mlm: Whether to export the additional mlm sub-object.
    copy_pooler_dense_to_encoder: Whether to copy the pooler's dense layer
      used in the next sentence prediction task to the encoder.
    vocab_file: The path to the wordpiece vocab file, or None.
    sp_model_file: The path to the sentencepiece model file, or None.
      Exactly one of vocab_file and sp_model_file must be set.
    do_lower_case: Whether to lower-case text before tokenization.
  """
  if with_mlm:
    core_model, pretrainer = _create_model(bert_config=bert_config,
                                           encoder_config=encoder_config,
                                           with_mlm=with_mlm)
    encoder = pretrainer.encoder_network
    # It supports both the new pretrainer checkpoint produced by TF-NLP and
    # the checkpoint converted from TF1 (original BERT, SmallBERTs).
    checkpoint_items = pretrainer.checkpoint_items
    checkpoint = tf.train.Checkpoint(**checkpoint_items)
  else:
    core_model, encoder = _create_model(bert_config=bert_config,
                                        encoder_config=encoder_config,
                                        with_mlm=with_mlm)
    checkpoint = tf.train.Checkpoint(
        model=encoder,  # Legacy checkpoints.
        encoder=encoder)
  checkpoint.restore(model_checkpoint_path).assert_existing_objects_matched()

  if copy_pooler_dense_to_encoder:
    logging.info("Copy pooler's dense layer to the encoder.")
    pooler_checkpoint = tf.train.Checkpoint(
        **{"next_sentence.pooler_dense": encoder.pooler_layer})
    pooler_checkpoint.restore(
        model_checkpoint_path).assert_existing_objects_matched()

  # Before SavedModels for preprocessing appeared in Oct 2020, the encoders
  # provided this information to let users do preprocessing themselves.
  # We keep doing that for now. It helps users to upgrade incrementally.
  # Moreover, it offers an escape hatch for advanced users who want the
  # full vocab, not the high-level operations from the preprocessing model.
  if vocab_file:
    core_model.vocab_file = tf.saved_model.Asset(vocab_file)
    if do_lower_case is None:
      raise ValueError("Must pass do_lower_case if passing vocab_file.")
    core_model.do_lower_case = tf.Variable(do_lower_case, trainable=False)
  elif sp_model_file:
    # This was used by ALBERT, with implied values of do_lower_case=True
    # and strip_diacritics=True.
    core_model.sp_model_file = tf.saved_model.Asset(sp_model_file)
  else:
    raise ValueError("Must set vocab_file or sp_model_file")
  core_model.save(export_path, include_optimizer=False, save_format="tf")


class BertPackInputsSavedModelWrapper(tf.train.Checkpoint):
  """Wraps a BertPackInputs layer for export to SavedModel.

  The wrapper object is suitable for use with `tf.saved_model.save()` and
  `.load()`. The wrapper object is callable with inputs and outputs like the
  BertPackInputs layer, but differs from saving an unwrapped Keras object:

    - The inputs can be a list of 1 or 2 RaggedTensors of dtype int32 and
      ragged rank 1 or 2. (In Keras, saving to a tf.function in a SavedModel
      would fix the number of RaggedTensors and their ragged rank.)
    - The call accepts an optional keyword argument `seq_length=` to override
      the layer's .seq_length hyperparameter. (In Keras, a hyperparameter
      could not be changed after saving to a tf.function in a SavedModel.)
  """

  def __init__(self, bert_pack_inputs: layers.BertPackInputs):
    super().__init__()

    # Preserve the layer's configured seq_length as a default but make it
    # overridable. Having this dynamically determined default argument
    # requires self.__call__ to be defined in this indirect way.
    default_seq_length = bert_pack_inputs.seq_length
    @tf.function(autograph=False)
    def call(inputs, seq_length=default_seq_length):
      return layers.BertPackInputs.bert_pack_inputs(
          inputs, seq_length=seq_length,
          start_of_sequence_id=bert_pack_inputs.start_of_sequence_id,
          end_of_segment_id=bert_pack_inputs.end_of_segment_id,
          padding_id=bert_pack_inputs.padding_id)
    self.__call__ = call

    for ragged_rank in range(1, 3):
      for num_segments in range(1, 3):
        _ = self.__call__.get_concrete_function(
            [tf.RaggedTensorSpec([None] * (ragged_rank + 1), dtype=tf.int32)
             for _ in range(num_segments)],
            seq_length=tf.TensorSpec([], tf.int32))


def create_preprocessing(*,
                         vocab_file: Optional[str] = None,
                         sp_model_file: Optional[str] = None,
                         do_lower_case: bool,
                         tokenize_with_offsets: bool,
                         default_seq_length: int) -> tf.keras.Model:
  """Returns a preprocessing Model for given tokenization parameters.

  This function builds a Keras Model with attached subobjects suitable for
  saving to a SavedModel. The resulting SavedModel implements the Preprocessor
  API for Text embeddings with Transformer Encoders described at
  https://www.tensorflow.org/hub/common_saved_model_apis/text.

  Args:
    vocab_file: The path to the wordpiece vocab file, or None.
    sp_model_file: The path to the sentencepiece model file, or None.
      Exactly one of vocab_file and sp_model_file must be set.
      This determines the type of tokenzer that is used.
    do_lower_case: Whether to do lower case.
    tokenize_with_offsets: Whether to include the .tokenize_with_offsets
      subobject.
    default_seq_length: The sequence length of preprocessing results from
      root callable. This is also the default sequence length for the
      bert_pack_inputs subobject.

  Returns:
    A tf.keras.Model object with several attached subobjects, suitable for
    saving as a preprocessing SavedModel.
  """
  # Select tokenizer.
  if bool(vocab_file) == bool(sp_model_file):
    raise ValueError("Must set exactly one of vocab_file, sp_model_file")
  if vocab_file:
    tokenize = layers.BertTokenizer(
        vocab_file=vocab_file,
        lower_case=do_lower_case,
        tokenize_with_offsets=tokenize_with_offsets)
  else:
    tokenize = layers.SentencepieceTokenizer(
        model_file_path=sp_model_file,
        lower_case=do_lower_case,
        strip_diacritics=True,  #  Strip diacritics to follow ALBERT model.
        tokenize_with_offsets=tokenize_with_offsets)

  # The root object of the preprocessing model can be called to do
  # one-shot preprocessing for users with single-sentence inputs.
  sentences = tf.keras.layers.Input(shape=(), dtype=tf.string, name="sentences")
  if tokenize_with_offsets:
    tokens, start_offsets, limit_offsets = tokenize(sentences)
  else:
    tokens = tokenize(sentences)
  pack = layers.BertPackInputs(
      seq_length=default_seq_length,
      special_tokens_dict=tokenize.get_special_tokens_dict())
  model_inputs = pack(tokens)
  preprocessing = tf.keras.Model(sentences, model_inputs)

  # Individual steps of preprocessing are made available as named subobjects
  # to enable more general preprocessing. For saving, they need to be Models
  # in their own right.
  preprocessing.tokenize = tf.keras.Model(sentences, tokens)
  # Provide an equivalent to tokenize.get_special_tokens_dict().
  preprocessing.tokenize.get_special_tokens_dict = tf.train.Checkpoint()
  preprocessing.tokenize.get_special_tokens_dict.__call__ = tf.function(
      lambda: tokenize.get_special_tokens_dict(),  # pylint: disable=[unnecessary-lambda]
      input_signature=[])
  if tokenize_with_offsets:
    preprocessing.tokenize_with_offsets = tf.keras.Model(
        sentences, [tokens, start_offsets, limit_offsets])
    preprocessing.tokenize_with_offsets.get_special_tokens_dict = (
        preprocessing.tokenize.get_special_tokens_dict)
  # Conceptually, this should be
  # preprocessing.bert_pack_inputs = tf.keras.Model(tokens, model_inputs)
  # but technicalities require us to use a wrapper (see comments there).
  # In particular, seq_length can be overridden when calling this.
  preprocessing.bert_pack_inputs = BertPackInputsSavedModelWrapper(pack)

  return preprocessing


def _move_to_tmpdir(file_path: Optional[Text], tmpdir: Text) -> Optional[Text]:
  """Returns new path with same basename and hash of original path."""
  if file_path is None: return None
  olddir, filename = os.path.split(file_path)
  hasher = hashlib.sha1()
  hasher.update(olddir.encode("utf-8"))
  target_dir = os.path.join(tmpdir, hasher.hexdigest())
  target_file = os.path.join(target_dir, filename)
  tf.io.gfile.mkdir(target_dir)
  tf.io.gfile.copy(file_path, target_file)
  return target_file


def export_preprocessing(export_path: Text,
                         *,
                         vocab_file: Optional[Text] = None,
                         sp_model_file: Optional[Text] = None,
                         do_lower_case: bool,
                         tokenize_with_offsets: bool,
                         default_seq_length: int,
                         experimental_disable_assert: bool = False) -> None:
  """Exports preprocessing to a SavedModel for TF Hub."""
  with tempfile.TemporaryDirectory() as tmpdir:
    # TODO(b/175369555): Remove experimental_disable_assert and its use.
    with _maybe_disable_assert(experimental_disable_assert):
      preprocessing = create_preprocessing(
          vocab_file=_move_to_tmpdir(vocab_file, tmpdir),
          sp_model_file=_move_to_tmpdir(sp_model_file, tmpdir),
          do_lower_case=do_lower_case,
          tokenize_with_offsets=tokenize_with_offsets,
          default_seq_length=default_seq_length)
      preprocessing.save(export_path, include_optimizer=False, save_format="tf")
    if experimental_disable_assert:
      _check_no_assert(export_path)
  # It helps the unit test to prevent stray copies of the vocab file.
  if tf.io.gfile.exists(tmpdir):
    raise IOError("Failed to clean up TemporaryDirectory")


# TODO(b/175369555): Remove all workarounds for this bug of TensorFlow 2.4
# when this bug is no longer a concern for publishing new models.
# TensorFlow 2.4 has a placement issue with Assert ops in tf.functions called
# from Dataset.map() on a TPU worker. They end up on the TPU coordinator,
# and invoking them from the TPU worker is either inefficient (when possible)
# or impossible (notably when using "headless" TPU workers on Cloud that do not
# have a channel to the coordinator). The bug has been fixed in time for TF 2.5.
# To work around this, the following code avoids Assert ops in the exported
# SavedModels. It monkey-patches calls to tf.Assert from inside TensorFlow and
# replaces them by a no-op while building the exported model. This is fragile,
# so _check_no_assert() validates the result. The resulting model should be fine
# to read on future versions of TF, even if this workaround at export time
# may break eventually. (Failing unit tests will tell.)


def _dont_assert(condition, data, summarize=None, name="Assert"):
  """The no-op version of tf.Assert installed by _maybe_disable_assert."""
  del condition, data, summarize  # Unused.
  if tf.executing_eagerly():
    return
  with tf.name_scope(name):
    return tf.no_op(name="dont_assert")


@contextlib.contextmanager
def _maybe_disable_assert(disable_assert):
  """Scoped monkey patch of control_flow_ops.Assert to a no-op."""
  if not disable_assert:
    yield
    return

  original_assert = control_flow_ops.Assert
  control_flow_ops.Assert = _dont_assert
  yield
  control_flow_ops.Assert = original_assert


def _check_no_assert(saved_model_path):
  """Raises AssertionError if SavedModel contains Assert ops."""
  saved_model_filename = os.path.join(saved_model_path, "saved_model.pb")
  with tf.io.gfile.GFile(saved_model_filename, "rb") as f:
    saved_model = saved_model_pb2.SavedModel.FromString(f.read())

  assert_nodes = []
  graph_def = saved_model.meta_graphs[0].graph_def
  assert_nodes += ["node '{}' in global graph".format(n.name)
                   for n in graph_def.node if n.op == "Assert"]
  for fdef in graph_def.library.function:
    assert_nodes += [
        "node '{}' in function '{}'".format(n.name, fdef.signature.name)
        for n in fdef.node_def if n.op == "Assert"]
  if assert_nodes:
    raise AssertionError(
        "Internal tool error: "
        "failed to suppress {} Assert ops in SavedModel:\n{}".format(
            len(assert_nodes), "\n".join(assert_nodes[:10])))
