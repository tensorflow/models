# Copyright 2023 The TensorFlow Authors. All Rights Reserved.
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

"""Tests for nlp.serving.serving_modules."""

import os

from absl.testing import parameterized
import tensorflow as tf

from sentencepiece import SentencePieceTrainer
from official.core import export_base
from official.nlp.configs import bert
from official.nlp.configs import encoders
from official.nlp.serving import serving_modules
from official.nlp.tasks import masked_lm
from official.nlp.tasks import question_answering
from official.nlp.tasks import sentence_prediction
from official.nlp.tasks import tagging
from official.nlp.tasks import translation


def _create_fake_serialized_examples(features_dict):
  """Creates a fake dataset."""

  def create_int_feature(values):
    f = tf.train.Feature(int64_list=tf.train.Int64List(value=list(values)))
    return f

  def create_str_feature(value):
    f = tf.train.Feature(bytes_list=tf.train.BytesList(value=[value]))
    return f

  examples = []
  for _ in range(10):
    features = {}
    for key, values in features_dict.items():
      if isinstance(values, bytes):
        features[key] = create_str_feature(values)
      else:
        features[key] = create_int_feature(values)
    tf_example = tf.train.Example(features=tf.train.Features(feature=features))
    examples.append(tf_example.SerializeToString())
  return tf.constant(examples)


def _create_fake_vocab_file(vocab_file_path):
  tokens = ["[PAD]"]
  for i in range(1, 100):
    tokens.append("[unused%d]" % i)
  tokens.extend(["[UNK]", "[CLS]", "[SEP]", "[MASK]", "hello", "world"])
  with tf.io.gfile.GFile(vocab_file_path, "w") as outfile:
    outfile.write("\n".join(tokens))


def _train_sentencepiece(input_path, vocab_size, model_path, eos_id=1):
  argstr = " ".join([
      f"--input={input_path}", f"--vocab_size={vocab_size}",
      "--character_coverage=0.995",
      f"--model_prefix={model_path}", "--model_type=bpe",
      "--bos_id=-1", "--pad_id=0", f"--eos_id={eos_id}", "--unk_id=2"
  ])
  SentencePieceTrainer.Train(argstr)


def _generate_line_file(filepath, lines):
  with tf.io.gfile.GFile(filepath, "w") as f:
    for l in lines:
      f.write("{}\n".format(l))


def _make_sentencepeice(output_dir):
  src_lines = ["abc ede fg", "bbcd ef a g", "de f a a g"]
  tgt_lines = ["dd cc a ef  g", "bcd ef a g", "gef cd ba"]
  sentencepeice_input_path = os.path.join(output_dir, "inputs.txt")
  _generate_line_file(sentencepeice_input_path, src_lines + tgt_lines)
  sentencepeice_model_prefix = os.path.join(output_dir, "sp")
  _train_sentencepiece(sentencepeice_input_path, 11, sentencepeice_model_prefix)
  sentencepeice_model_path = "{}.model".format(sentencepeice_model_prefix)
  return sentencepeice_model_path


class ServingModulesTest(tf.test.TestCase, parameterized.TestCase):

  @parameterized.parameters(
      # use_v2_feature_names
      True,
      False)
  def test_sentence_prediction(self, use_v2_feature_names):
    if use_v2_feature_names:
      input_word_ids_field = "input_word_ids"
      input_type_ids_field = "input_type_ids"
    else:
      input_word_ids_field = "input_ids"
      input_type_ids_field = "segment_ids"

    config = sentence_prediction.SentencePredictionConfig(
        model=sentence_prediction.ModelConfig(
            encoder=encoders.EncoderConfig(
                bert=encoders.BertEncoderConfig(vocab_size=30522,
                                                num_layers=1)),
            num_classes=2))
    task = sentence_prediction.SentencePredictionTask(config)
    model = task.build_model()
    params = serving_modules.SentencePrediction.Params(
        inputs_only=True,
        parse_sequence_length=10,
        use_v2_feature_names=use_v2_feature_names)
    export_module = serving_modules.SentencePrediction(
        params=params, model=model)
    functions = export_module.get_inference_signatures({
        "serve": "serving_default",
        "serve_examples": "serving_examples"
    })
    self.assertSameElements(functions.keys(),
                            ["serving_default", "serving_examples"])
    dummy_ids = tf.ones((10, 10), dtype=tf.int32)
    outputs = functions["serving_default"](dummy_ids)
    self.assertEqual(outputs["outputs"].shape, (10, 2))

    params = serving_modules.SentencePrediction.Params(
        inputs_only=False,
        parse_sequence_length=10,
        use_v2_feature_names=use_v2_feature_names)
    export_module = serving_modules.SentencePrediction(
        params=params, model=model)
    functions = export_module.get_inference_signatures({
        "serve": "serving_default",
        "serve_examples": "serving_examples"
    })
    outputs = functions["serving_default"](
        input_word_ids=dummy_ids,
        input_mask=dummy_ids,
        input_type_ids=dummy_ids)
    self.assertEqual(outputs["outputs"].shape, (10, 2))

    dummy_ids = tf.ones((10,), dtype=tf.int32)
    examples = _create_fake_serialized_examples({
        input_word_ids_field: dummy_ids,
        "input_mask": dummy_ids,
        input_type_ids_field: dummy_ids
    })
    outputs = functions["serving_examples"](examples)
    self.assertEqual(outputs["outputs"].shape, (10, 2))

    with self.assertRaises(ValueError):
      _ = export_module.get_inference_signatures({"foo": None})

  @parameterized.parameters(
      # inputs_only
      True,
      False)
  def test_sentence_prediction_text(self, inputs_only):
    vocab_file_path = os.path.join(self.get_temp_dir(), "vocab.txt")
    _create_fake_vocab_file(vocab_file_path)
    config = sentence_prediction.SentencePredictionConfig(
        model=sentence_prediction.ModelConfig(
            encoder=encoders.EncoderConfig(
                bert=encoders.BertEncoderConfig(vocab_size=30522,
                                                num_layers=1)),
            num_classes=2))
    task = sentence_prediction.SentencePredictionTask(config)
    model = task.build_model()
    params = serving_modules.SentencePrediction.Params(
        inputs_only=inputs_only,
        parse_sequence_length=10,
        text_fields=["foo", "bar"],
        vocab_file=vocab_file_path)
    export_module = serving_modules.SentencePrediction(
        params=params, model=model)
    examples = _create_fake_serialized_examples({
        "foo": b"hello world",
        "bar": b"hello world"
    })
    functions = export_module.get_inference_signatures({
        "serve_text_examples": "serving_default",
    })
    outputs = functions["serving_default"](examples)
    self.assertEqual(outputs["outputs"].shape, (10, 2))

  @parameterized.parameters(
      # use_v2_feature_names
      True,
      False)
  def test_masked_lm(self, use_v2_feature_names):
    if use_v2_feature_names:
      input_word_ids_field = "input_word_ids"
      input_type_ids_field = "input_type_ids"
    else:
      input_word_ids_field = "input_ids"
      input_type_ids_field = "segment_ids"
    config = masked_lm.MaskedLMConfig(
        model=bert.PretrainerConfig(
            encoder=encoders.EncoderConfig(
                bert=encoders.BertEncoderConfig(vocab_size=30522,
                                                num_layers=1)),
            cls_heads=[
                bert.ClsHeadConfig(
                    inner_dim=10, num_classes=2, name="next_sentence")
            ]))
    task = masked_lm.MaskedLMTask(config)
    model = task.build_model()
    params = serving_modules.MaskedLM.Params(
        parse_sequence_length=10,
        max_predictions_per_seq=5,
        use_v2_feature_names=use_v2_feature_names)
    export_module = serving_modules.MaskedLM(params=params, model=model)
    functions = export_module.get_inference_signatures({
        "serve": "serving_default",
        "serve_examples": "serving_examples"
    })
    self.assertSameElements(functions.keys(),
                            ["serving_default", "serving_examples"])
    dummy_ids = tf.ones((10, 10), dtype=tf.int32)
    dummy_pos = tf.ones((10, 5), dtype=tf.int32)
    outputs = functions["serving_default"](
        input_word_ids=dummy_ids,
        input_mask=dummy_ids,
        input_type_ids=dummy_ids,
        masked_lm_positions=dummy_pos)
    self.assertEqual(outputs["classification"].shape, (10, 2))

    dummy_ids = tf.ones((10,), dtype=tf.int32)
    dummy_pos = tf.ones((5,), dtype=tf.int32)
    examples = _create_fake_serialized_examples({
        input_word_ids_field: dummy_ids,
        "input_mask": dummy_ids,
        input_type_ids_field: dummy_ids,
        "masked_lm_positions": dummy_pos
    })
    outputs = functions["serving_examples"](examples)
    self.assertEqual(outputs["classification"].shape, (10, 2))

  @parameterized.parameters(
      # use_v2_feature_names
      True,
      False)
  def test_question_answering(self, use_v2_feature_names):
    if use_v2_feature_names:
      input_word_ids_field = "input_word_ids"
      input_type_ids_field = "input_type_ids"
    else:
      input_word_ids_field = "input_ids"
      input_type_ids_field = "segment_ids"

    config = question_answering.QuestionAnsweringConfig(
        model=question_answering.ModelConfig(
            encoder=encoders.EncoderConfig(
                bert=encoders.BertEncoderConfig(vocab_size=30522,
                                                num_layers=1))),
        validation_data=None)
    task = question_answering.QuestionAnsweringTask(config)
    model = task.build_model()
    params = serving_modules.QuestionAnswering.Params(
        parse_sequence_length=10, use_v2_feature_names=use_v2_feature_names)
    export_module = serving_modules.QuestionAnswering(
        params=params, model=model)
    functions = export_module.get_inference_signatures({
        "serve": "serving_default",
        "serve_examples": "serving_examples"
    })
    self.assertSameElements(functions.keys(),
                            ["serving_default", "serving_examples"])
    dummy_ids = tf.ones((10, 10), dtype=tf.int32)
    outputs = functions["serving_default"](
        input_word_ids=dummy_ids,
        input_mask=dummy_ids,
        input_type_ids=dummy_ids)
    self.assertEqual(outputs["start_logits"].shape, (10, 10))
    self.assertEqual(outputs["end_logits"].shape, (10, 10))
    dummy_ids = tf.ones((10,), dtype=tf.int32)
    examples = _create_fake_serialized_examples({
        input_word_ids_field: dummy_ids,
        "input_mask": dummy_ids,
        input_type_ids_field: dummy_ids
    })
    outputs = functions["serving_examples"](examples)
    self.assertEqual(outputs["start_logits"].shape, (10, 10))
    self.assertEqual(outputs["end_logits"].shape, (10, 10))

  @parameterized.parameters(
      # (use_v2_feature_names, output_encoder_outputs)
      (True, True),
      (False, False))
  def test_tagging(self, use_v2_feature_names, output_encoder_outputs):
    if use_v2_feature_names:
      input_word_ids_field = "input_word_ids"
      input_type_ids_field = "input_type_ids"
    else:
      input_word_ids_field = "input_ids"
      input_type_ids_field = "segment_ids"

    hidden_size = 768
    num_classes = 3
    config = tagging.TaggingConfig(
        model=tagging.ModelConfig(
            encoder=encoders.EncoderConfig(
                bert=encoders.BertEncoderConfig(
                    hidden_size=hidden_size, num_layers=1))),
        class_names=["class_0", "class_1", "class_2"])
    task = tagging.TaggingTask(config)
    model = task.build_model()

    params = serving_modules.Tagging.Params(
        parse_sequence_length=10,
        use_v2_feature_names=use_v2_feature_names,
        output_encoder_outputs=output_encoder_outputs)
    export_module = serving_modules.Tagging(params=params, model=model)
    functions = export_module.get_inference_signatures({
        "serve": "serving_default",
        "serve_examples": "serving_examples"
    })
    dummy_ids = tf.ones((10, 10), dtype=tf.int32)
    outputs = functions["serving_default"](
        input_word_ids=dummy_ids,
        input_mask=dummy_ids,
        input_type_ids=dummy_ids)
    self.assertEqual(outputs["logits"].shape, (10, 10, num_classes))
    if output_encoder_outputs:
      self.assertEqual(outputs["encoder_outputs"].shape, (10, 10, hidden_size))

    dummy_ids = tf.ones((10,), dtype=tf.int32)
    examples = _create_fake_serialized_examples({
        input_word_ids_field: dummy_ids,
        "input_mask": dummy_ids,
        input_type_ids_field: dummy_ids
    })
    outputs = functions["serving_examples"](examples)
    self.assertEqual(outputs["logits"].shape, (10, 10, num_classes))
    if output_encoder_outputs:
      self.assertEqual(outputs["encoder_outputs"].shape, (10, 10, hidden_size))

    with self.assertRaises(ValueError):
      _ = export_module.get_inference_signatures({"foo": None})

  @parameterized.parameters(
      (False, None),
      (True, 2))
  def test_translation(self, padded_decode, batch_size):
    sp_path = _make_sentencepeice(self.get_temp_dir())
    encdecoder = translation.EncDecoder(
        num_attention_heads=4, intermediate_size=256)
    config = translation.TranslationConfig(
        model=translation.ModelConfig(
            encoder=encdecoder,
            decoder=encdecoder,
            embedding_width=256,
            padded_decode=padded_decode,
            decode_max_length=100),
        sentencepiece_model_path=sp_path,
    )
    task = translation.TranslationTask(config)
    model = task.build_model()

    params = serving_modules.Translation.Params(
        sentencepiece_model_path=sp_path, batch_size=batch_size)
    export_module = serving_modules.Translation(params=params, model=model)
    functions = export_module.get_inference_signatures({
        "serve_text": "serving_default"
    })
    outputs = functions["serving_default"](tf.constant(["abcd", "ef gh"]))
    self.assertEqual(outputs.shape, (2,))
    self.assertEqual(outputs.dtype, tf.string)

    tmp_dir = self.get_temp_dir()
    tmp_dir = os.path.join(tmp_dir, "padded_decode", str(padded_decode))
    export_base_dir = os.path.join(tmp_dir, "export")
    ckpt_dir = os.path.join(tmp_dir, "ckpt")
    ckpt_path = tf.train.Checkpoint(model=model).save(ckpt_dir)
    export_dir = export_base.export(export_module,
                                    {"serve_text": "serving_default"},
                                    export_base_dir, ckpt_path)
    loaded = tf.saved_model.load(export_dir)
    infer = loaded.signatures["serving_default"]
    out = infer(text=tf.constant(["abcd", "ef gh"]))
    self.assertLen(out["output_0"], 2)


if __name__ == "__main__":
  tf.test.main()
