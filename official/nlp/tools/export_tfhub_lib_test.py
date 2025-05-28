# Copyright 2025 The TensorFlow Authors. All Rights Reserved.
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

"""Tests export_tfhub_lib."""

import os
import tempfile

from absl.testing import parameterized
import numpy as np
import tensorflow as tf, tf_keras
from tensorflow import estimator as tf_estimator
import tensorflow_hub as hub
import tensorflow_text as text

from sentencepiece import SentencePieceTrainer
from official.legacy.bert import configs
from official.modeling import tf_utils
from official.nlp.configs import encoders
from official.nlp.modeling import layers
from official.nlp.modeling import models
from official.nlp.tools import export_tfhub_lib


def _get_bert_config_or_encoder_config(use_bert_config,
                                       hidden_size,
                                       num_hidden_layers,
                                       encoder_type="albert",
                                       vocab_size=100):
  """Generates config args for export_tfhub_lib._create_model().

  Args:
    use_bert_config: bool. If True, returns legacy BertConfig.
    hidden_size: int.
    num_hidden_layers: int.
    encoder_type: str. Can be ['albert', 'bert', 'bert_v2']. If use_bert_config
      == True, then model_type is not used.
    vocab_size: int.

  Returns:
    bert_config, encoder_config. Only one is not None. If
      `use_bert_config` == True, the first config is valid. Otherwise
      `bert_config` == None.
  """
  if use_bert_config:
    bert_config = configs.BertConfig(
        vocab_size=vocab_size,
        hidden_size=hidden_size,
        intermediate_size=32,
        max_position_embeddings=128,
        num_attention_heads=2,
        num_hidden_layers=num_hidden_layers)
    encoder_config = None
  else:
    bert_config = None
    if encoder_type == "albert":
      encoder_config = encoders.EncoderConfig(
          type="albert",
          albert=encoders.AlbertEncoderConfig(
              vocab_size=vocab_size,
              embedding_width=16,
              hidden_size=hidden_size,
              intermediate_size=32,
              max_position_embeddings=128,
              num_attention_heads=2,
              num_layers=num_hidden_layers,
              dropout_rate=0.1))
    else:
      # encoder_type can be 'bert' or 'bert_v2'.
      model_config = encoders.BertEncoderConfig(
          vocab_size=vocab_size,
          embedding_size=16,
          hidden_size=hidden_size,
          intermediate_size=32,
          max_position_embeddings=128,
          num_attention_heads=2,
          num_layers=num_hidden_layers,
          dropout_rate=0.1)
      kwargs = {"type": encoder_type, encoder_type: model_config}
      encoder_config = encoders.EncoderConfig(**kwargs)

  return bert_config, encoder_config


def _get_vocab_or_sp_model_dummy(temp_dir, use_sp_model):
  """Returns tokenizer asset args for export_tfhub_lib.export_model()."""
  dummy_file = os.path.join(temp_dir, "dummy_file.txt")
  with tf.io.gfile.GFile(dummy_file, "w") as f:
    f.write("dummy content")
  if use_sp_model:
    vocab_file, sp_model_file = None, dummy_file
  else:
    vocab_file, sp_model_file = dummy_file, None
  return vocab_file, sp_model_file


def _read_asset(asset: tf.saved_model.Asset):
  return tf.io.gfile.GFile(asset.asset_path.numpy()).read()


def _find_lambda_layers(layer):
  """Returns list of all Lambda layers in a Keras model."""
  if isinstance(layer, tf_keras.layers.Lambda):
    return [layer]
  elif hasattr(layer, "layers"):  # It's nested, like a Model.
    result = []
    for l in layer.layers:
      result += _find_lambda_layers(l)
    return result
  else:
    return []


class ExportModelTest(tf.test.TestCase, parameterized.TestCase):
  """Tests exporting a Transformer Encoder model as a SavedModel.

  This covers export from an Encoder checkpoint to a SavedModel without
  the .mlm subobject. This is no longer preferred, but still useful
    for models like Electra that are trained without the MLM task.

  The export code is generic. This test focuses on two main cases
  (the most important ones in practice when this was written in 2020):
    - BERT built from a legacy BertConfig, for use with BertTokenizer.
    - ALBERT built from an EncoderConfig (as a representative of all other
      choices beyond BERT, for use with SentencepieceTokenizer (the one
      alternative to BertTokenizer).
  """

  @parameterized.named_parameters(
      ("Bert_Legacy", True, None), ("Albert", False, "albert"),
      ("BertEncoder", False, "bert"), ("BertEncoderV2", False, "bert_v2"))
  def test_export_model(self, use_bert, encoder_type):
    # Create the encoder and export it.
    hidden_size = 16
    num_hidden_layers = 1
    bert_config, encoder_config = _get_bert_config_or_encoder_config(
        use_bert,
        hidden_size=hidden_size,
        num_hidden_layers=num_hidden_layers,
        encoder_type=encoder_type)
    bert_model, encoder = export_tfhub_lib._create_model(
        bert_config=bert_config, encoder_config=encoder_config, with_mlm=False)
    self.assertEmpty(
        _find_lambda_layers(bert_model),
        "Lambda layers are non-portable since they serialize Python bytecode.")
    model_checkpoint_dir = os.path.join(self.get_temp_dir(), "checkpoint")
    checkpoint = tf.train.Checkpoint(encoder=encoder)
    checkpoint.save(os.path.join(model_checkpoint_dir, "test"))
    model_checkpoint_path = tf.train.latest_checkpoint(model_checkpoint_dir)

    vocab_file, sp_model_file = _get_vocab_or_sp_model_dummy(
        self.get_temp_dir(), use_sp_model=not use_bert)
    export_path = os.path.join(self.get_temp_dir(), "hub")
    export_tfhub_lib.export_model(
        export_path=export_path,
        bert_config=bert_config,
        encoder_config=encoder_config,
        model_checkpoint_path=model_checkpoint_path,
        with_mlm=False,
        vocab_file=vocab_file,
        sp_model_file=sp_model_file,
        do_lower_case=True)

    # Restore the exported model.
    hub_layer = hub.KerasLayer(export_path, trainable=True)

    # Check legacy tokenization data.
    if use_bert:
      self.assertTrue(hub_layer.resolved_object.do_lower_case.numpy())
      self.assertEqual("dummy content",
                       _read_asset(hub_layer.resolved_object.vocab_file))
      self.assertFalse(hasattr(hub_layer.resolved_object, "sp_model_file"))
    else:
      self.assertFalse(hasattr(hub_layer.resolved_object, "do_lower_case"))
      self.assertFalse(hasattr(hub_layer.resolved_object, "vocab_file"))
      self.assertEqual("dummy content",
                       _read_asset(hub_layer.resolved_object.sp_model_file))

    # Check restored weights.
    self.assertEqual(
        len(bert_model.trainable_weights), len(hub_layer.trainable_weights))
    for source_weight, hub_weight in zip(bert_model.trainable_weights,
                                         hub_layer.trainable_weights):
      self.assertAllClose(source_weight.numpy(), hub_weight.numpy())

    # Check computation.
    seq_length = 10
    dummy_ids = np.zeros((2, seq_length), dtype=np.int32)
    input_dict = dict(
        input_word_ids=dummy_ids,
        input_mask=dummy_ids,
        input_type_ids=dummy_ids)
    hub_output = hub_layer(input_dict)
    source_output = bert_model(input_dict)
    encoder_output = encoder(input_dict)
    self.assertEqual(hub_output["pooled_output"].shape, (2, hidden_size))
    self.assertEqual(hub_output["sequence_output"].shape,
                     (2, seq_length, hidden_size))
    self.assertLen(hub_output["encoder_outputs"], num_hidden_layers)

    for key in ("pooled_output", "sequence_output", "encoder_outputs"):
      self.assertAllClose(source_output[key], hub_output[key])
      self.assertAllClose(source_output[key], encoder_output[key])

    # The "default" output of BERT as a text representation is pooled_output.
    self.assertAllClose(hub_output["pooled_output"], hub_output["default"])

    # Test that training=True makes a difference (activates dropout).
    def _dropout_mean_stddev(training, num_runs=20):
      input_ids = np.array([[14, 12, 42, 95, 99]], np.int32)
      input_dict = dict(
          input_word_ids=input_ids,
          input_mask=np.ones_like(input_ids),
          input_type_ids=np.zeros_like(input_ids))
      outputs = np.concatenate([
          hub_layer(input_dict, training=training)["pooled_output"]
          for _ in range(num_runs)
      ])
      return np.mean(np.std(outputs, axis=0))

    self.assertLess(_dropout_mean_stddev(training=False), 1e-6)
    self.assertGreater(_dropout_mean_stddev(training=True), 1e-3)

    # Test propagation of seq_length in shape inference.
    input_word_ids = tf_keras.layers.Input(shape=(seq_length,), dtype=tf.int32)
    input_mask = tf_keras.layers.Input(shape=(seq_length,), dtype=tf.int32)
    input_type_ids = tf_keras.layers.Input(shape=(seq_length,), dtype=tf.int32)
    input_dict = dict(
        input_word_ids=input_word_ids,
        input_mask=input_mask,
        input_type_ids=input_type_ids)
    output_dict = hub_layer(input_dict)
    pooled_output = output_dict["pooled_output"]
    sequence_output = output_dict["sequence_output"]
    encoder_outputs = output_dict["encoder_outputs"]

    self.assertEqual(pooled_output.shape.as_list(), [None, hidden_size])
    self.assertEqual(sequence_output.shape.as_list(),
                     [None, seq_length, hidden_size])
    self.assertLen(encoder_outputs, num_hidden_layers)


class ExportModelWithMLMTest(tf.test.TestCase, parameterized.TestCase):
  """Tests exporting a Transformer Encoder model as a SavedModel.

  This covers export from a Pretrainer checkpoint to a SavedModel including
  the .mlm subobject, which is the preferred way since 2020.

  The export code is generic. This test focuses on two main cases
  (the most important ones in practice when this was written in 2020):
    - BERT built from a legacy BertConfig, for use with BertTokenizer.
    - ALBERT built from an EncoderConfig (as a representative of all other
      choices beyond BERT, for use with SentencepieceTokenizer (the one
      alternative to BertTokenizer).
  """

  def test_copy_pooler_dense_to_encoder(self):
    encoder_config = encoders.EncoderConfig(
        type="bert",
        bert=encoders.BertEncoderConfig(
            hidden_size=24, intermediate_size=48, num_layers=2))
    cls_heads = [
        layers.ClassificationHead(
            inner_dim=24, num_classes=2, name="next_sentence")
    ]
    encoder = encoders.build_encoder(encoder_config)
    pretrainer = models.BertPretrainerV2(
        encoder_network=encoder,
        classification_heads=cls_heads,
        mlm_activation=tf_utils.get_activation(
            encoder_config.get().hidden_activation))
    # Makes sure the pretrainer variables are created.
    _ = pretrainer(pretrainer.inputs)
    checkpoint = tf.train.Checkpoint(**pretrainer.checkpoint_items)
    model_checkpoint_dir = os.path.join(self.get_temp_dir(), "checkpoint")
    checkpoint.save(os.path.join(model_checkpoint_dir, "test"))

    vocab_file, sp_model_file = _get_vocab_or_sp_model_dummy(
        self.get_temp_dir(), use_sp_model=True)
    export_path = os.path.join(self.get_temp_dir(), "hub")
    export_tfhub_lib.export_model(
        export_path=export_path,
        encoder_config=encoder_config,
        model_checkpoint_path=tf.train.latest_checkpoint(model_checkpoint_dir),
        with_mlm=True,
        copy_pooler_dense_to_encoder=True,
        vocab_file=vocab_file,
        sp_model_file=sp_model_file,
        do_lower_case=True)
    # Restores a hub KerasLayer.
    hub_layer = hub.KerasLayer(export_path, trainable=True)
    dummy_ids = np.zeros((2, 10), dtype=np.int32)
    input_dict = dict(
        input_word_ids=dummy_ids,
        input_mask=dummy_ids,
        input_type_ids=dummy_ids)
    hub_pooled_output = hub_layer(input_dict)["pooled_output"]
    encoder_outputs = encoder(input_dict)
    # Verify that hub_layer's pooled_output is the same as the output of next
    # sentence prediction's dense layer.
    pretrained_pooled_output = cls_heads[0].dense(
        (encoder_outputs["sequence_output"][:, 0, :]))
    self.assertAllClose(hub_pooled_output, pretrained_pooled_output)
    # But the pooled_output between encoder and hub_layer are not the same.
    encoder_pooled_output = encoder_outputs["pooled_output"]
    self.assertNotAllClose(hub_pooled_output, encoder_pooled_output)

  @parameterized.named_parameters(
      ("Bert", True),
      ("Albert", False),
  )
  def test_export_model_with_mlm(self, use_bert):
    # Create the encoder and export it.
    hidden_size = 16
    num_hidden_layers = 2
    bert_config, encoder_config = _get_bert_config_or_encoder_config(
        use_bert, hidden_size, num_hidden_layers)
    bert_model, pretrainer = export_tfhub_lib._create_model(
        bert_config=bert_config, encoder_config=encoder_config, with_mlm=True)
    self.assertEmpty(
        _find_lambda_layers(bert_model),
        "Lambda layers are non-portable since they serialize Python bytecode.")
    bert_model_with_mlm = bert_model.mlm
    model_checkpoint_dir = os.path.join(self.get_temp_dir(), "checkpoint")

    checkpoint = tf.train.Checkpoint(**pretrainer.checkpoint_items)

    checkpoint.save(os.path.join(model_checkpoint_dir, "test"))
    model_checkpoint_path = tf.train.latest_checkpoint(model_checkpoint_dir)

    vocab_file, sp_model_file = _get_vocab_or_sp_model_dummy(
        self.get_temp_dir(), use_sp_model=not use_bert)
    export_path = os.path.join(self.get_temp_dir(), "hub")
    export_tfhub_lib.export_model(
        export_path=export_path,
        bert_config=bert_config,
        encoder_config=encoder_config,
        model_checkpoint_path=model_checkpoint_path,
        with_mlm=True,
        vocab_file=vocab_file,
        sp_model_file=sp_model_file,
        do_lower_case=True)

    # Restore the exported model.
    hub_layer = hub.KerasLayer(export_path, trainable=True)

    # Check legacy tokenization data.
    if use_bert:
      self.assertTrue(hub_layer.resolved_object.do_lower_case.numpy())
      self.assertEqual("dummy content",
                       _read_asset(hub_layer.resolved_object.vocab_file))
      self.assertFalse(hasattr(hub_layer.resolved_object, "sp_model_file"))
    else:
      self.assertFalse(hasattr(hub_layer.resolved_object, "do_lower_case"))
      self.assertFalse(hasattr(hub_layer.resolved_object, "vocab_file"))
      self.assertEqual("dummy content",
                       _read_asset(hub_layer.resolved_object.sp_model_file))

    # Check restored weights.
    # Note that we set `_auto_track_sub_layers` to False when exporting the
    # SavedModel, so hub_layer has the same number of weights as bert_model;
    # otherwise, hub_layer will have extra weights from its `mlm` subobject.
    self.assertEqual(
        len(bert_model.trainable_weights), len(hub_layer.trainable_weights))
    for source_weight, hub_weight in zip(bert_model.trainable_weights,
                                         hub_layer.trainable_weights):
      self.assertAllClose(source_weight, hub_weight)

    # Check computation.
    seq_length = 10
    dummy_ids = np.zeros((2, seq_length), dtype=np.int32)
    input_dict = dict(
        input_word_ids=dummy_ids,
        input_mask=dummy_ids,
        input_type_ids=dummy_ids)
    hub_outputs_dict = hub_layer(input_dict)
    source_outputs_dict = bert_model(input_dict)
    encoder_outputs_dict = pretrainer.encoder_network(
        [dummy_ids, dummy_ids, dummy_ids])
    self.assertEqual(hub_outputs_dict["pooled_output"].shape, (2, hidden_size))
    self.assertEqual(hub_outputs_dict["sequence_output"].shape,
                     (2, seq_length, hidden_size))
    for output_key in ("pooled_output", "sequence_output", "encoder_outputs"):
      self.assertAllClose(source_outputs_dict[output_key],
                          hub_outputs_dict[output_key])
      self.assertAllClose(source_outputs_dict[output_key],
                          encoder_outputs_dict[output_key])

    # The "default" output of BERT as a text representation is pooled_output.
    self.assertAllClose(hub_outputs_dict["pooled_output"],
                        hub_outputs_dict["default"])

    # Test that training=True makes a difference (activates dropout).
    def _dropout_mean_stddev(training, num_runs=20):
      input_ids = np.array([[14, 12, 42, 95, 99]], np.int32)
      input_dict = dict(
          input_word_ids=input_ids,
          input_mask=np.ones_like(input_ids),
          input_type_ids=np.zeros_like(input_ids))
      outputs = np.concatenate([
          hub_layer(input_dict, training=training)["pooled_output"]
          for _ in range(num_runs)
      ])
      return np.mean(np.std(outputs, axis=0))

    self.assertLess(_dropout_mean_stddev(training=False), 1e-6)
    self.assertGreater(_dropout_mean_stddev(training=True), 1e-3)

    # Checks sub-object `mlm`.
    self.assertTrue(hasattr(hub_layer.resolved_object, "mlm"))

    self.assertLen(hub_layer.resolved_object.mlm.trainable_variables,
                   len(bert_model_with_mlm.trainable_weights))
    self.assertLen(hub_layer.resolved_object.mlm.trainable_variables,
                   len(pretrainer.trainable_weights))
    for source_weight, hub_weight, pretrainer_weight in zip(
        bert_model_with_mlm.trainable_weights,
        hub_layer.resolved_object.mlm.trainable_variables,
        pretrainer.trainable_weights):
      self.assertAllClose(source_weight, hub_weight)
      self.assertAllClose(source_weight, pretrainer_weight)

    max_predictions_per_seq = 4
    mlm_positions = np.zeros((2, max_predictions_per_seq), dtype=np.int32)
    input_dict = dict(
        input_word_ids=dummy_ids,
        input_mask=dummy_ids,
        input_type_ids=dummy_ids,
        masked_lm_positions=mlm_positions)
    hub_mlm_outputs_dict = hub_layer.resolved_object.mlm(input_dict)
    source_mlm_outputs_dict = bert_model_with_mlm(input_dict)
    for output_key in ("pooled_output", "sequence_output", "mlm_logits",
                       "encoder_outputs"):
      self.assertAllClose(hub_mlm_outputs_dict[output_key],
                          source_mlm_outputs_dict[output_key])

    pretrainer_mlm_logits_output = pretrainer(input_dict)["mlm_logits"]
    self.assertAllClose(hub_mlm_outputs_dict["mlm_logits"],
                        pretrainer_mlm_logits_output)

    # Test that training=True makes a difference (activates dropout).
    def _dropout_mean_stddev_mlm(training, num_runs=20):
      input_ids = np.array([[14, 12, 42, 95, 99]], np.int32)
      mlm_position_ids = np.array([[1, 2, 3, 4]], np.int32)
      input_dict = dict(
          input_word_ids=input_ids,
          input_mask=np.ones_like(input_ids),
          input_type_ids=np.zeros_like(input_ids),
          masked_lm_positions=mlm_position_ids)
      outputs = np.concatenate([
          hub_layer.resolved_object.mlm(input_dict,
                                        training=training)["pooled_output"]
          for _ in range(num_runs)
      ])
      return np.mean(np.std(outputs, axis=0))

    self.assertLess(_dropout_mean_stddev_mlm(training=False), 1e-6)
    self.assertGreater(_dropout_mean_stddev_mlm(training=True), 1e-3)

    # Test propagation of seq_length in shape inference.
    input_word_ids = tf_keras.layers.Input(shape=(seq_length,), dtype=tf.int32)
    input_mask = tf_keras.layers.Input(shape=(seq_length,), dtype=tf.int32)
    input_type_ids = tf_keras.layers.Input(shape=(seq_length,), dtype=tf.int32)
    input_dict = dict(
        input_word_ids=input_word_ids,
        input_mask=input_mask,
        input_type_ids=input_type_ids)
    hub_outputs_dict = hub_layer(input_dict)
    self.assertEqual(hub_outputs_dict["pooled_output"].shape.as_list(),
                     [None, hidden_size])
    self.assertEqual(hub_outputs_dict["sequence_output"].shape.as_list(),
                     [None, seq_length, hidden_size])


_STRING_NOT_TO_LEAK = "private_path_component_"


class ExportPreprocessingTest(tf.test.TestCase, parameterized.TestCase):

  def _make_vocab_file(self, vocab, filename="vocab.txt", add_mask_token=False):
    """Creates wordpiece vocab file with given words plus special tokens.

    The tokens of the resulting model are, in this order:
        [PAD], [UNK], [CLS], [SEP], [MASK]*, ...vocab...
    *=if requested by args.

    This function also accepts wordpieces that start with the ## continuation
    marker, but avoiding those makes this function interchangeable with
    _make_sp_model_file(), up to the extra dimension returned by BertTokenizer.

    Args:
      vocab: a list of strings with the words or wordpieces to put into the
        model's vocabulary. Do not include special tokens here.
      filename: Optionally, a filename (relative to the temporary directory
        created by this function).
      add_mask_token: an optional bool, whether to include a [MASK] token.

    Returns:
      The absolute filename of the created vocab file.
    """
    full_vocab = ["[PAD]", "[UNK]", "[CLS]", "[SEP]"
                 ] + ["[MASK]"] * add_mask_token + vocab
    path = os.path.join(
        tempfile.mkdtemp(
            dir=self.get_temp_dir(),  # New subdir each time.
            prefix=_STRING_NOT_TO_LEAK),
        filename)
    with tf.io.gfile.GFile(path, "w") as f:
      f.write("\n".join(full_vocab + [""]))
    return path

  def _make_sp_model_file(self, vocab, prefix="spm", add_mask_token=False):
    """Creates Sentencepiece word model with given words plus special tokens.

    The tokens of the resulting model are, in this order:
        <pad>, <unk>, [CLS], [SEP], [MASK]*, ...vocab..., <s>, </s>
    *=if requested by args.

    The words in the input vocab are plain text, without the whitespace marker.
    That makes this function interchangeable with _make_vocab_file().

    Args:
      vocab: a list of strings with the words to put into the model's
        vocabulary. Do not include special tokens here.
      prefix: an optional string, to change the filename prefix for the model
        (relative to the temporary directory created by this function).
      add_mask_token: an optional bool, whether to include a [MASK] token.

    Returns:
      The absolute filename of the created Sentencepiece model file.
    """
    model_prefix = os.path.join(
        tempfile.mkdtemp(dir=self.get_temp_dir()),  # New subdir each time.
        prefix)
    input_file = model_prefix + "_train_input.txt"
    # Create input text for training the sp model from the tokens provided.
    # Repeat tokens, the earlier the more, because they are sorted by frequency.
    input_text = []
    for i, token in enumerate(vocab):
      input_text.append(" ".join([token] * (len(vocab) - i)))
    with tf.io.gfile.GFile(input_file, "w") as f:
      f.write("\n".join(input_text + [""]))
    control_symbols = "[CLS],[SEP]"
    full_vocab_size = len(vocab) + 6  # <pad>, <unk>, [CLS], [SEP], <s>, </s>.
    if add_mask_token:
      control_symbols += ",[MASK]"
      full_vocab_size += 1
    flags = dict(
        model_prefix=model_prefix,
        model_type="word",
        input=input_file,
        pad_id=0,
        unk_id=1,
        control_symbols=control_symbols,
        vocab_size=full_vocab_size,
        bos_id=full_vocab_size - 2,
        eos_id=full_vocab_size - 1)
    SentencePieceTrainer.Train(" ".join(
        ["--{}={}".format(k, v) for k, v in flags.items()]))
    return model_prefix + ".model"

  def _do_export(self,
                 vocab,
                 do_lower_case,
                 default_seq_length=128,
                 tokenize_with_offsets=True,
                 use_sp_model=False,
                 experimental_disable_assert=False,
                 add_mask_token=False):
    """Runs SavedModel export and returns the export_path."""
    export_path = tempfile.mkdtemp(dir=self.get_temp_dir())
    vocab_file = sp_model_file = None
    if use_sp_model:
      sp_model_file = self._make_sp_model_file(
          vocab, add_mask_token=add_mask_token)
    else:
      vocab_file = self._make_vocab_file(vocab, add_mask_token=add_mask_token)
    export_tfhub_lib.export_preprocessing(
        export_path,
        vocab_file=vocab_file,
        sp_model_file=sp_model_file,
        do_lower_case=do_lower_case,
        tokenize_with_offsets=tokenize_with_offsets,
        default_seq_length=default_seq_length,
        experimental_disable_assert=experimental_disable_assert)
    # Invalidate the original filename to verify loading from the SavedModel.
    tf.io.gfile.remove(sp_model_file or vocab_file)
    return export_path

  def test_no_leaks(self):
    """Tests not leaking the path to the original vocab file."""
    path = self._do_export(["d", "ef", "abc", "xy"],
                           do_lower_case=True,
                           use_sp_model=False)
    with tf.io.gfile.GFile(os.path.join(path, "saved_model.pb"), "rb") as f:
      self.assertFalse(  # pylint: disable=g-generic-assert
          _STRING_NOT_TO_LEAK.encode("ascii") in f.read())

  @parameterized.named_parameters(("Bert", False), ("Sentencepiece", True))
  def test_exported_callables(self, use_sp_model):
    preprocess = tf.saved_model.load(
        self._do_export(
            ["d", "ef", "abc", "xy"],
            do_lower_case=True,
            # TODO(b/181866850): drop this.
            tokenize_with_offsets=not use_sp_model,
            # TODO(b/175369555): drop this.
            experimental_disable_assert=True,
            use_sp_model=use_sp_model))

    def fold_dim(rt):
      """Removes the word/subword distinction of BertTokenizer."""
      return rt if use_sp_model else rt.merge_dims(1, 2)

    # .tokenize()
    inputs = tf.constant(["abc d ef", "ABC D EF d"])
    token_ids = preprocess.tokenize(inputs)
    self.assertAllEqual(
        fold_dim(token_ids), tf.ragged.constant([[6, 4, 5], [6, 4, 5, 4]]))

    special_tokens_dict = {
        k: v.numpy().item()  # Expecting eager Tensor, converting to Python.
        for k, v in preprocess.tokenize.get_special_tokens_dict().items()
    }
    self.assertDictEqual(
        special_tokens_dict,
        dict(
            padding_id=0,
            start_of_sequence_id=2,
            end_of_segment_id=3,
            vocab_size=4 + 6 if use_sp_model else 4 + 4))

    # .tokenize_with_offsets()
    if use_sp_model:
      # TODO(b/181866850): Enable tokenize_with_offsets when it works and test.
      self.assertFalse(hasattr(preprocess, "tokenize_with_offsets"))
    else:
      token_ids, start_offsets, limit_offsets = (
          preprocess.tokenize_with_offsets(inputs))
      self.assertAllEqual(
          fold_dim(token_ids), tf.ragged.constant([[6, 4, 5], [6, 4, 5, 4]]))
      self.assertAllEqual(
          fold_dim(start_offsets), tf.ragged.constant([[0, 4, 6], [0, 4, 6,
                                                                   9]]))
      self.assertAllEqual(
          fold_dim(limit_offsets), tf.ragged.constant([[3, 5, 8], [3, 5, 8,
                                                                   10]]))
      self.assertIs(preprocess.tokenize.get_special_tokens_dict,
                    preprocess.tokenize_with_offsets.get_special_tokens_dict)

    # Root callable.
    bert_inputs = preprocess(inputs)
    self.assertAllEqual(bert_inputs["input_word_ids"].shape.as_list(), [2, 128])
    self.assertAllEqual(
        bert_inputs["input_word_ids"][:, :10],
        tf.constant([[2, 6, 4, 5, 3, 0, 0, 0, 0, 0],
                     [2, 6, 4, 5, 4, 3, 0, 0, 0, 0]]))
    self.assertAllEqual(bert_inputs["input_mask"].shape.as_list(), [2, 128])
    self.assertAllEqual(
        bert_inputs["input_mask"][:, :10],
        tf.constant([[1, 1, 1, 1, 1, 0, 0, 0, 0, 0],
                     [1, 1, 1, 1, 1, 1, 0, 0, 0, 0]]))
    self.assertAllEqual(bert_inputs["input_type_ids"].shape.as_list(), [2, 128])
    self.assertAllEqual(
        bert_inputs["input_type_ids"][:, :10],
        tf.constant([[0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                     [0, 0, 0, 0, 0, 0, 0, 0, 0, 0]]))

    # .bert_pack_inputs()
    inputs_2 = tf.constant(["d xy", "xy abc"])
    token_ids_2 = preprocess.tokenize(inputs_2)
    bert_inputs = preprocess.bert_pack_inputs([token_ids, token_ids_2],
                                              seq_length=256)
    self.assertAllEqual(bert_inputs["input_word_ids"].shape.as_list(), [2, 256])
    self.assertAllEqual(
        bert_inputs["input_word_ids"][:, :10],
        tf.constant([[2, 6, 4, 5, 3, 4, 7, 3, 0, 0],
                     [2, 6, 4, 5, 4, 3, 7, 6, 3, 0]]))
    self.assertAllEqual(bert_inputs["input_mask"].shape.as_list(), [2, 256])
    self.assertAllEqual(
        bert_inputs["input_mask"][:, :10],
        tf.constant([[1, 1, 1, 1, 1, 1, 1, 1, 0, 0],
                     [1, 1, 1, 1, 1, 1, 1, 1, 1, 0]]))
    self.assertAllEqual(bert_inputs["input_type_ids"].shape.as_list(), [2, 256])
    self.assertAllEqual(
        bert_inputs["input_type_ids"][:, :10],
        tf.constant([[0, 0, 0, 0, 0, 1, 1, 1, 0, 0],
                     [0, 0, 0, 0, 0, 0, 1, 1, 1, 0]]))

  # For BertTokenizer only: repeat relevant parts for do_lower_case=False,
  # default_seq_length=10, experimental_disable_assert=False,
  # tokenize_with_offsets=False, and without folding the word/subword dimension.
  def test_cased_length10(self):
    preprocess = tf.saved_model.load(
        self._do_export(["d", "##ef", "abc", "ABC"],
                        do_lower_case=False,
                        default_seq_length=10,
                        tokenize_with_offsets=False,
                        use_sp_model=False,
                        experimental_disable_assert=False))
    inputs = tf.constant(["abc def", "ABC DEF"])
    token_ids = preprocess.tokenize(inputs)
    self.assertAllEqual(token_ids,
                        tf.ragged.constant([[[6], [4, 5]], [[7], [1]]]))

    self.assertFalse(hasattr(preprocess, "tokenize_with_offsets"))

    bert_inputs = preprocess(inputs)
    self.assertAllEqual(
        bert_inputs["input_word_ids"],
        tf.constant([[2, 6, 4, 5, 3, 0, 0, 0, 0, 0],
                     [2, 7, 1, 3, 0, 0, 0, 0, 0, 0]]))
    self.assertAllEqual(
        bert_inputs["input_mask"],
        tf.constant([[1, 1, 1, 1, 1, 0, 0, 0, 0, 0],
                     [1, 1, 1, 1, 0, 0, 0, 0, 0, 0]]))
    self.assertAllEqual(
        bert_inputs["input_type_ids"],
        tf.constant([[0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                     [0, 0, 0, 0, 0, 0, 0, 0, 0, 0]]))

    inputs_2 = tf.constant(["d ABC", "ABC abc"])
    token_ids_2 = preprocess.tokenize(inputs_2)
    bert_inputs = preprocess.bert_pack_inputs([token_ids, token_ids_2])
    # Test default seq_length=10.
    self.assertAllEqual(
        bert_inputs["input_word_ids"],
        tf.constant([[2, 6, 4, 5, 3, 4, 7, 3, 0, 0],
                     [2, 7, 1, 3, 7, 6, 3, 0, 0, 0]]))
    self.assertAllEqual(
        bert_inputs["input_mask"],
        tf.constant([[1, 1, 1, 1, 1, 1, 1, 1, 0, 0],
                     [1, 1, 1, 1, 1, 1, 1, 0, 0, 0]]))
    self.assertAllEqual(
        bert_inputs["input_type_ids"],
        tf.constant([[0, 0, 0, 0, 0, 1, 1, 1, 0, 0],
                     [0, 0, 0, 0, 1, 1, 1, 0, 0, 0]]))

  # XLA requires fixed shapes for tensors found in graph mode.
  # Statically known shapes in Python are a particularly firm way to
  # guarantee that, and they are generally more convenient to work with.
  # We test that the exported SavedModel plays well with TF's shape
  # inference when applied to fully or partially known input shapes.
  @parameterized.named_parameters(("Bert", False), ("Sentencepiece", True))
  def test_shapes(self, use_sp_model):
    preprocess = tf.saved_model.load(
        self._do_export(
            ["abc", "def"],
            do_lower_case=True,
            # TODO(b/181866850): drop this.
            tokenize_with_offsets=not use_sp_model,
            # TODO(b/175369555): drop this.
            experimental_disable_assert=True,
            use_sp_model=use_sp_model))

    def expected_bert_input_shapes(batch_size, seq_length):
      return dict(
          input_word_ids=[batch_size, seq_length],
          input_mask=[batch_size, seq_length],
          input_type_ids=[batch_size, seq_length])

    for batch_size in [7, None]:
      if use_sp_model:
        token_out_shape = [batch_size, None]  # No word/subword distinction.
      else:
        token_out_shape = [batch_size, None, None]
      self.assertEqual(
          _result_shapes_in_tf_function(preprocess.tokenize,
                                        tf.TensorSpec([batch_size], tf.string)),
          token_out_shape, "with batch_size=%s" % batch_size)
      # TODO(b/181866850): Enable tokenize_with_offsets when it works and test.
      if use_sp_model:
        self.assertFalse(hasattr(preprocess, "tokenize_with_offsets"))
      else:
        self.assertEqual(
            _result_shapes_in_tf_function(
                preprocess.tokenize_with_offsets,
                tf.TensorSpec([batch_size], tf.string)), [token_out_shape] * 3,
            "with batch_size=%s" % batch_size)
      self.assertEqual(
          _result_shapes_in_tf_function(
              preprocess.bert_pack_inputs,
              [tf.RaggedTensorSpec([batch_size, None, None], tf.int32)] * 2,
              seq_length=256), expected_bert_input_shapes(batch_size, 256),
          "with batch_size=%s" % batch_size)
      self.assertEqual(
          _result_shapes_in_tf_function(preprocess,
                                        tf.TensorSpec([batch_size], tf.string)),
          expected_bert_input_shapes(batch_size, 128),
          "with batch_size=%s" % batch_size)

  @parameterized.named_parameters(("Bert", False), ("Sentencepiece", True))
  def test_reexport(self, use_sp_model):
    """Test that preprocess keeps working after another save/load cycle."""
    path1 = self._do_export(
        ["d", "ef", "abc", "xy"],
        do_lower_case=True,
        default_seq_length=10,
        tokenize_with_offsets=False,
        experimental_disable_assert=True,  # TODO(b/175369555): drop this.
        use_sp_model=use_sp_model)
    path2 = path1.rstrip("/") + ".2"
    model1 = tf.saved_model.load(path1)
    tf.saved_model.save(model1, path2)
    # Delete the first SavedModel to test that the sceond one loads by itself.
    # https://github.com/tensorflow/tensorflow/issues/46456 reports such a
    # failure case for BertTokenizer.
    tf.io.gfile.rmtree(path1)
    model2 = tf.saved_model.load(path2)

    inputs = tf.constant(["abc d ef", "ABC D EF d"])
    bert_inputs = model2(inputs)
    self.assertAllEqual(
        bert_inputs["input_word_ids"],
        tf.constant([[2, 6, 4, 5, 3, 0, 0, 0, 0, 0],
                     [2, 6, 4, 5, 4, 3, 0, 0, 0, 0]]))
    self.assertAllEqual(
        bert_inputs["input_mask"],
        tf.constant([[1, 1, 1, 1, 1, 0, 0, 0, 0, 0],
                     [1, 1, 1, 1, 1, 1, 0, 0, 0, 0]]))
    self.assertAllEqual(
        bert_inputs["input_type_ids"],
        tf.constant([[0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                     [0, 0, 0, 0, 0, 0, 0, 0, 0, 0]]))

  @parameterized.named_parameters(("Bert", True), ("Albert", False))
  def test_preprocessing_for_mlm(self, use_bert):
    """Combines both SavedModel types and TF.text helpers for MLM."""
    # Create the preprocessing SavedModel with a [MASK] token.
    non_special_tokens = [
        "hello", "world", "nice", "movie", "great", "actors", "quick", "fox",
        "lazy", "dog"
    ]

    preprocess = tf.saved_model.load(
        self._do_export(
            non_special_tokens,
            do_lower_case=True,
            tokenize_with_offsets=use_bert,  # TODO(b/181866850): drop this.
            experimental_disable_assert=True,  # TODO(b/175369555): drop this.
            add_mask_token=True,
            use_sp_model=not use_bert))
    vocab_size = len(non_special_tokens) + (5 if use_bert else 7)

    # Create the encoder SavedModel with an .mlm subobject.
    hidden_size = 16
    num_hidden_layers = 2
    bert_config, encoder_config = _get_bert_config_or_encoder_config(
        use_bert_config=use_bert,
        hidden_size=hidden_size,
        num_hidden_layers=num_hidden_layers,
        vocab_size=vocab_size)
    _, pretrainer = export_tfhub_lib._create_model(
        bert_config=bert_config, encoder_config=encoder_config, with_mlm=True)
    model_checkpoint_dir = os.path.join(self.get_temp_dir(), "checkpoint")
    checkpoint = tf.train.Checkpoint(**pretrainer.checkpoint_items)
    checkpoint.save(os.path.join(model_checkpoint_dir, "test"))
    model_checkpoint_path = tf.train.latest_checkpoint(model_checkpoint_dir)
    vocab_file, sp_model_file = _get_vocab_or_sp_model_dummy(  # Not used below.
        self.get_temp_dir(), use_sp_model=not use_bert)
    encoder_export_path = os.path.join(self.get_temp_dir(), "encoder_export")
    export_tfhub_lib.export_model(
        export_path=encoder_export_path,
        bert_config=bert_config,
        encoder_config=encoder_config,
        model_checkpoint_path=model_checkpoint_path,
        with_mlm=True,
        vocab_file=vocab_file,
        sp_model_file=sp_model_file,
        do_lower_case=True)
    encoder = tf.saved_model.load(encoder_export_path)

    # Get special tokens from the vocab (and vocab size).
    special_tokens_dict = preprocess.tokenize.get_special_tokens_dict()
    self.assertEqual(int(special_tokens_dict["vocab_size"]), vocab_size)
    padding_id = int(special_tokens_dict["padding_id"])
    self.assertEqual(padding_id, 0)
    start_of_sequence_id = int(special_tokens_dict["start_of_sequence_id"])
    self.assertEqual(start_of_sequence_id, 2)
    end_of_segment_id = int(special_tokens_dict["end_of_segment_id"])
    self.assertEqual(end_of_segment_id, 3)
    mask_id = int(special_tokens_dict["mask_id"])
    self.assertEqual(mask_id, 4)

    # A batch of 3 segment pairs.
    raw_segments = [
        tf.constant(["hello", "nice movie", "quick fox"]),
        tf.constant(["world", "great actors", "lazy dog"])
    ]
    batch_size = 3

    # Misc hyperparameters.
    seq_length = 10
    max_selections_per_seq = 2

    # Tokenize inputs.
    tokenized_segments = [preprocess.tokenize(s) for s in raw_segments]
    # Trim inputs to eventually fit seq_lentgh.
    num_special_tokens = len(raw_segments) + 1
    trimmed_segments = text.WaterfallTrimmer(
        seq_length - num_special_tokens).trim(tokenized_segments)
    # Combine input segments into one input sequence.
    input_ids, segment_ids = text.combine_segments(
        trimmed_segments,
        start_of_sequence_id=start_of_sequence_id,
        end_of_segment_id=end_of_segment_id)
    # Apply random masking controlled by policy objects.
    (masked_input_ids, masked_lm_positions,
     masked_ids) = text.mask_language_model(
         input_ids=input_ids,
         item_selector=text.RandomItemSelector(
             max_selections_per_seq,
             selection_rate=0.5,  # Adjusted for the short test examples.
             unselectable_ids=[start_of_sequence_id, end_of_segment_id]),
         mask_values_chooser=text.MaskValuesChooser(
             vocab_size=vocab_size,
             mask_token=mask_id,
             # Always put [MASK] to have a predictable result.
             mask_token_rate=1.0,
             random_token_rate=0.0))
    # Pad to fixed-length Transformer encoder inputs.
    input_word_ids, _ = text.pad_model_inputs(
        masked_input_ids, seq_length, pad_value=padding_id)
    input_type_ids, input_mask = text.pad_model_inputs(
        segment_ids, seq_length, pad_value=0)
    masked_lm_positions, _ = text.pad_model_inputs(
        masked_lm_positions, max_selections_per_seq, pad_value=0)
    masked_lm_positions = tf.cast(masked_lm_positions, tf.int32)
    num_predictions = int(tf.shape(masked_lm_positions)[1])

    # Test transformer inputs.
    self.assertEqual(num_predictions, max_selections_per_seq)
    expected_word_ids = np.array([
        # [CLS] hello [SEP] world [SEP]
        [2, 5, 3, 6, 3, 0, 0, 0, 0, 0],
        # [CLS] nice movie [SEP] great actors [SEP]
        [2, 7, 8, 3, 9, 10, 3, 0, 0, 0],
        # [CLS] brown fox [SEP] lazy dog [SEP]
        [2, 11, 12, 3, 13, 14, 3, 0, 0, 0]
    ])
    for i in range(batch_size):
      for j in range(num_predictions):
        k = int(masked_lm_positions[i, j])
        if k != 0:
          expected_word_ids[i, k] = 4  # [MASK]
    self.assertAllEqual(input_word_ids, expected_word_ids)

    # Call the MLM head of the Transformer encoder.
    mlm_inputs = dict(
        input_word_ids=input_word_ids,
        input_mask=input_mask,
        input_type_ids=input_type_ids,
        masked_lm_positions=masked_lm_positions,
    )
    mlm_outputs = encoder.mlm(mlm_inputs)
    self.assertEqual(mlm_outputs["pooled_output"].shape,
                     (batch_size, hidden_size))
    self.assertEqual(mlm_outputs["sequence_output"].shape,
                     (batch_size, seq_length, hidden_size))
    self.assertEqual(mlm_outputs["mlm_logits"].shape,
                     (batch_size, num_predictions, vocab_size))
    self.assertLen(mlm_outputs["encoder_outputs"], num_hidden_layers)

    # A real trainer would now compute the loss of mlm_logits
    # trying to predict the masked_ids.
    del masked_ids  # Unused.

  @parameterized.named_parameters(("Bert", False), ("Sentencepiece", True))
  def test_special_tokens_in_estimator(self, use_sp_model):
    """Tests getting special tokens without an Eager init context."""
    preprocess_export_path = self._do_export(["d", "ef", "abc", "xy"],
                                             do_lower_case=True,
                                             use_sp_model=use_sp_model,
                                             tokenize_with_offsets=False)

    def _get_special_tokens_dict(obj):
      """Returns special tokens of restored tokenizer as Python values."""
      if tf.executing_eagerly():
        special_tokens_numpy = {
            k: v.numpy() for k, v in obj.get_special_tokens_dict()
        }
      else:
        with tf.Graph().as_default():
          # This code expects `get_special_tokens_dict()` to be a tf.function
          # with no dependencies (bound args) from the context it was loaded in,
          # and boldly assumes that it can just be called in a dfferent context.
          special_tokens_tensors = obj.get_special_tokens_dict()
          with tf.compat.v1.Session() as sess:
            special_tokens_numpy = sess.run(special_tokens_tensors)
      return {
          k: v.item()  # Numpy to Python.
          for k, v in special_tokens_numpy.items()
      }

    def input_fn():
      self.assertFalse(tf.executing_eagerly())
      # Build a preprocessing Model.
      sentences = tf_keras.layers.Input(shape=[], dtype=tf.string)
      preprocess = tf.saved_model.load(preprocess_export_path)
      tokenize = hub.KerasLayer(preprocess.tokenize)
      special_tokens_dict = _get_special_tokens_dict(tokenize.resolved_object)
      for k, v in special_tokens_dict.items():
        self.assertIsInstance(v, int, "Unexpected type for {}".format(k))
      tokens = tokenize(sentences)
      packed_inputs = layers.BertPackInputs(
          4, special_tokens_dict=special_tokens_dict)(
              tokens)
      preprocessing = tf_keras.Model(sentences, packed_inputs)
      # Map the dataset.
      ds = tf.data.Dataset.from_tensors(
          (tf.constant(["abc", "D EF"]), tf.constant([0, 1])))
      ds = ds.map(lambda features, labels: (preprocessing(features), labels))
      return ds

    def model_fn(features, labels, mode):
      del labels  # Unused.
      return tf_estimator.EstimatorSpec(
          mode=mode, predictions=features["input_word_ids"])

    estimator = tf_estimator.Estimator(model_fn=model_fn)
    outputs = list(estimator.predict(input_fn))
    self.assertAllEqual(outputs, np.array([[2, 6, 3, 0], [2, 4, 5, 3]]))

  # TODO(b/175369555): Remove that code and its test.
  @parameterized.named_parameters(("Bert", False), ("Sentencepiece", True))
  def test_check_no_assert(self, use_sp_model):
    """Tests the self-check during export without assertions."""
    preprocess_export_path = self._do_export(["d", "ef", "abc", "xy"],
                                             do_lower_case=True,
                                             use_sp_model=use_sp_model,
                                             tokenize_with_offsets=False,
                                             experimental_disable_assert=False)
    with self.assertRaisesRegex(AssertionError,
                                r"failed to suppress \d+ Assert ops"):
      export_tfhub_lib._check_no_assert(preprocess_export_path)


def _result_shapes_in_tf_function(fn, *args, **kwargs):
  """Returns shapes (as lists) observed on the result of `fn`.

  Args:
    fn: A callable.
    *args: TensorSpecs for Tensor-valued arguments and actual values for
      Python-valued arguments to fn.
    **kwargs: Same for keyword arguments.

  Returns:
    The nest of partial tensor shapes (as lists) that is statically known inside
    tf.function(fn)(*args, **kwargs) for the nest of its results.
  """
  # Use a captured mutable container for a side outout from the wrapper.
  uninitialized = "uninitialized!"
  result_shapes_container = [uninitialized]
  assert result_shapes_container[0] is uninitialized

  @tf.function
  def shape_reporting_wrapper(*args, **kwargs):
    result = fn(*args, **kwargs)
    result_shapes_container[0] = tf.nest.map_structure(
        lambda x: x.shape.as_list(), result)
    return result

  shape_reporting_wrapper.get_concrete_function(*args, **kwargs)
  assert result_shapes_container[0] is not uninitialized
  return result_shapes_container[0]


if __name__ == "__main__":
  tf.test.main()
