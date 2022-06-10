# Copyright 2022 The TensorFlow Authors. All Rights Reserved.
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

"""Serving export modules for TF Model Garden NLP models."""
# pylint:disable=missing-class-docstring
import dataclasses
from typing import Dict, List, Optional, Text

import tensorflow as tf
import tensorflow_text as tf_text

from official.core import export_base
from official.modeling.hyperparams import base_config
from official.nlp.data import sentence_prediction_dataloader


def features_to_int32(features: Dict[str, tf.Tensor]) -> Dict[str, tf.Tensor]:
  """Converts tf.int64 features to tf.int32, keep other features the same.

  tf.Example only supports tf.int64, but the TPU only supports tf.int32.

  Args:
    features: Input tensor dictionary.

  Returns:
    Features with tf.int64 converted to tf.int32.
  """
  converted_features = {}
  for name, tensor in features.items():
    if tensor.dtype == tf.int64:
      converted_features[name] = tf.cast(tensor, tf.int32)
    else:
      converted_features[name] = tensor
  return converted_features


class SentencePrediction(export_base.ExportModule):
  """The export module for the sentence prediction task."""

  @dataclasses.dataclass
  class Params(base_config.Config):
    inputs_only: bool = True
    parse_sequence_length: Optional[int] = None
    use_v2_feature_names: bool = True

    # For text input processing.
    text_fields: Optional[List[str]] = None
    # Either specify these values for preprocessing by Python code...
    tokenization: str = "WordPiece"  # WordPiece or SentencePiece
    # Text vocab file if tokenization is WordPiece, or sentencepiece.ModelProto
    # file if tokenization is SentencePiece.
    vocab_file: str = ""
    lower_case: bool = True
    # ...or load preprocessing from a SavedModel at this location.
    preprocessing_hub_module_url: str = ""

  def __init__(self, params, model: tf.keras.Model, inference_step=None):
    super().__init__(params, model, inference_step)
    if params.use_v2_feature_names:
      self.input_word_ids_field = "input_word_ids"
      self.input_type_ids_field = "input_type_ids"
    else:
      self.input_word_ids_field = "input_ids"
      self.input_type_ids_field = "segment_ids"

    if params.text_fields:
      self._text_processor = sentence_prediction_dataloader.TextProcessor(
          seq_length=params.parse_sequence_length,
          vocab_file=params.vocab_file,
          tokenization=params.tokenization,
          lower_case=params.lower_case,
          preprocessing_hub_module_url=params.preprocessing_hub_module_url)

  def _serve_tokenized_input(self,
                             input_word_ids,
                             input_mask=None,
                             input_type_ids=None) -> tf.Tensor:
    if input_type_ids is None:
      # Requires CLS token is the first token of inputs.
      input_type_ids = tf.zeros_like(input_word_ids)
    if input_mask is None:
      # The mask has 1 for real tokens and 0 for padding tokens.
      input_mask = tf.where(
          tf.equal(input_word_ids, 0), tf.zeros_like(input_word_ids),
          tf.ones_like(input_word_ids))
    inputs = dict(
        input_word_ids=input_word_ids,
        input_mask=input_mask,
        input_type_ids=input_type_ids)
    return self.inference_step(inputs)

  @tf.function
  def serve(self,
            input_word_ids,
            input_mask=None,
            input_type_ids=None) -> Dict[str, tf.Tensor]:
    return dict(
        outputs=self._serve_tokenized_input(input_word_ids, input_mask,
                                            input_type_ids))

  @tf.function
  def serve_probability(self,
                        input_word_ids,
                        input_mask=None,
                        input_type_ids=None) -> Dict[str, tf.Tensor]:
    return dict(
        outputs=tf.nn.softmax(
            self._serve_tokenized_input(input_word_ids, input_mask,
                                        input_type_ids)))

  @tf.function
  def serve_examples(self, inputs) -> Dict[str, tf.Tensor]:
    sequence_length = self.params.parse_sequence_length
    inputs_only = self.params.inputs_only
    name_to_features = {
        self.input_word_ids_field:
            tf.io.FixedLenFeature([sequence_length], tf.int64),
    }
    if not inputs_only:
      name_to_features.update({
          "input_mask":
              tf.io.FixedLenFeature([sequence_length], tf.int64),
          self.input_type_ids_field:
              tf.io.FixedLenFeature([sequence_length], tf.int64)
      })
    features = tf.io.parse_example(inputs, name_to_features)
    features = features_to_int32(features)
    return self.serve(
        features[self.input_word_ids_field],
        input_mask=None if inputs_only else features["input_mask"],
        input_type_ids=None
        if inputs_only else features[self.input_type_ids_field])

  @tf.function
  def serve_text_examples(self, inputs) -> Dict[str, tf.Tensor]:
    name_to_features = {}
    for text_field in self.params.text_fields:
      name_to_features[text_field] = tf.io.FixedLenFeature([], tf.string)
    features = tf.io.parse_example(inputs, name_to_features)
    segments = [features[x] for x in self.params.text_fields]
    model_inputs = self._text_processor(segments)
    if self.params.inputs_only:
      return self.serve(input_word_ids=model_inputs["input_word_ids"])
    return self.serve(**model_inputs)

  def get_inference_signatures(self, function_keys: Dict[Text, Text]):
    signatures = {}
    valid_keys = ("serve", "serve_examples", "serve_text_examples")
    for func_key, signature_key in function_keys.items():
      if func_key not in valid_keys:
        raise ValueError("Invalid function key for the module: %s with key %s. "
                         "Valid keys are: %s" %
                         (self.__class__, func_key, valid_keys))
      if func_key == "serve":
        if self.params.inputs_only:
          signatures[signature_key] = self.serve.get_concrete_function(
              input_word_ids=tf.TensorSpec(
                  shape=[None, None], dtype=tf.int32, name="input_word_ids"))
        else:
          signatures[signature_key] = self.serve.get_concrete_function(
              input_word_ids=tf.TensorSpec(
                  shape=[None, None], dtype=tf.int32, name="input_word_ids"),
              input_mask=tf.TensorSpec(
                  shape=[None, None], dtype=tf.int32, name="input_mask"),
              input_type_ids=tf.TensorSpec(
                  shape=[None, None], dtype=tf.int32, name="input_type_ids"))
      if func_key == "serve_examples":
        signatures[signature_key] = self.serve_examples.get_concrete_function(
            tf.TensorSpec(shape=[None], dtype=tf.string, name="examples"))
      if func_key == "serve_text_examples":
        signatures[
            signature_key] = self.serve_text_examples.get_concrete_function(
                tf.TensorSpec(shape=[None], dtype=tf.string, name="examples"))
    return signatures


class MaskedLM(export_base.ExportModule):
  """The export module for the Bert Pretrain (MaskedLM) task."""

  def __init__(self, params, model: tf.keras.Model, inference_step=None):
    super().__init__(params, model, inference_step)
    if params.use_v2_feature_names:
      self.input_word_ids_field = "input_word_ids"
      self.input_type_ids_field = "input_type_ids"
    else:
      self.input_word_ids_field = "input_ids"
      self.input_type_ids_field = "segment_ids"

  @dataclasses.dataclass
  class Params(base_config.Config):
    cls_head_name: str = "next_sentence"
    use_v2_feature_names: bool = True
    parse_sequence_length: Optional[int] = None
    max_predictions_per_seq: Optional[int] = None

  @tf.function
  def serve(self, input_word_ids, input_mask, input_type_ids,
            masked_lm_positions) -> Dict[str, tf.Tensor]:
    inputs = dict(
        input_word_ids=input_word_ids,
        input_mask=input_mask,
        input_type_ids=input_type_ids,
        masked_lm_positions=masked_lm_positions)
    outputs = self.inference_step(inputs)
    return dict(classification=outputs[self.params.cls_head_name])

  @tf.function
  def serve_examples(self, inputs) -> Dict[str, tf.Tensor]:
    sequence_length = self.params.parse_sequence_length
    max_predictions_per_seq = self.params.max_predictions_per_seq
    name_to_features = {
        self.input_word_ids_field:
            tf.io.FixedLenFeature([sequence_length], tf.int64),
        "input_mask":
            tf.io.FixedLenFeature([sequence_length], tf.int64),
        self.input_type_ids_field:
            tf.io.FixedLenFeature([sequence_length], tf.int64),
        "masked_lm_positions":
            tf.io.FixedLenFeature([max_predictions_per_seq], tf.int64)
    }
    features = tf.io.parse_example(inputs, name_to_features)
    features = features_to_int32(features)
    return self.serve(
        input_word_ids=features[self.input_word_ids_field],
        input_mask=features["input_mask"],
        input_type_ids=features[self.input_word_ids_field],
        masked_lm_positions=features["masked_lm_positions"])

  def get_inference_signatures(self, function_keys: Dict[Text, Text]):
    signatures = {}
    valid_keys = ("serve", "serve_examples")
    for func_key, signature_key in function_keys.items():
      if func_key not in valid_keys:
        raise ValueError("Invalid function key for the module: %s with key %s. "
                         "Valid keys are: %s" %
                         (self.__class__, func_key, valid_keys))
      if func_key == "serve":
        signatures[signature_key] = self.serve.get_concrete_function(
            input_word_ids=tf.TensorSpec(
                shape=[None, None], dtype=tf.int32, name="input_word_ids"),
            input_mask=tf.TensorSpec(
                shape=[None, None], dtype=tf.int32, name="input_mask"),
            input_type_ids=tf.TensorSpec(
                shape=[None, None], dtype=tf.int32, name="input_type_ids"),
            masked_lm_positions=tf.TensorSpec(
                shape=[None, None], dtype=tf.int32, name="masked_lm_positions"))
      if func_key == "serve_examples":
        signatures[signature_key] = self.serve_examples.get_concrete_function(
            tf.TensorSpec(shape=[None], dtype=tf.string, name="examples"))
    return signatures


class QuestionAnswering(export_base.ExportModule):
  """The export module for the question answering task."""

  @dataclasses.dataclass
  class Params(base_config.Config):
    parse_sequence_length: Optional[int] = None
    use_v2_feature_names: bool = True

  def __init__(self, params, model: tf.keras.Model, inference_step=None):
    super().__init__(params, model, inference_step)
    if params.use_v2_feature_names:
      self.input_word_ids_field = "input_word_ids"
      self.input_type_ids_field = "input_type_ids"
    else:
      self.input_word_ids_field = "input_ids"
      self.input_type_ids_field = "segment_ids"

  @tf.function
  def serve(self,
            input_word_ids,
            input_mask=None,
            input_type_ids=None) -> Dict[str, tf.Tensor]:
    if input_mask is None:
      # The mask has 1 for real tokens and 0 for padding tokens.
      input_mask = tf.where(
          tf.equal(input_word_ids, 0), tf.zeros_like(input_word_ids),
          tf.ones_like(input_word_ids))
    inputs = dict(
        input_word_ids=input_word_ids,
        input_mask=input_mask,
        input_type_ids=input_type_ids)
    outputs = self.inference_step(inputs)
    return dict(start_logits=outputs[0], end_logits=outputs[1])

  @tf.function
  def serve_examples(self, inputs) -> Dict[str, tf.Tensor]:
    sequence_length = self.params.parse_sequence_length
    name_to_features = {
        self.input_word_ids_field:
            tf.io.FixedLenFeature([sequence_length], tf.int64),
        "input_mask":
            tf.io.FixedLenFeature([sequence_length], tf.int64),
        self.input_type_ids_field:
            tf.io.FixedLenFeature([sequence_length], tf.int64)
    }
    features = tf.io.parse_example(inputs, name_to_features)
    features = features_to_int32(features)
    return self.serve(
        input_word_ids=features[self.input_word_ids_field],
        input_mask=features["input_mask"],
        input_type_ids=features[self.input_type_ids_field])

  def get_inference_signatures(self, function_keys: Dict[Text, Text]):
    signatures = {}
    valid_keys = ("serve", "serve_examples")
    for func_key, signature_key in function_keys.items():
      if func_key not in valid_keys:
        raise ValueError("Invalid function key for the module: %s with key %s. "
                         "Valid keys are: %s" %
                         (self.__class__, func_key, valid_keys))
      if func_key == "serve":
        signatures[signature_key] = self.serve.get_concrete_function(
            input_word_ids=tf.TensorSpec(
                shape=[None, None], dtype=tf.int32, name="input_word_ids"),
            input_mask=tf.TensorSpec(
                shape=[None, None], dtype=tf.int32, name="input_mask"),
            input_type_ids=tf.TensorSpec(
                shape=[None, None], dtype=tf.int32, name="input_type_ids"))
      if func_key == "serve_examples":
        signatures[signature_key] = self.serve_examples.get_concrete_function(
            tf.TensorSpec(shape=[None], dtype=tf.string, name="examples"))
    return signatures


class Tagging(export_base.ExportModule):
  """The export module for the tagging task."""

  @dataclasses.dataclass
  class Params(base_config.Config):
    parse_sequence_length: Optional[int] = None
    use_v2_feature_names: bool = True
    output_encoder_outputs: bool = False

  def __init__(self, params, model: tf.keras.Model, inference_step=None):
    super().__init__(params, model, inference_step)
    if params.use_v2_feature_names:
      self.input_word_ids_field = "input_word_ids"
      self.input_type_ids_field = "input_type_ids"
    else:
      self.input_word_ids_field = "input_ids"
      self.input_type_ids_field = "segment_ids"

  @tf.function
  def serve(self, input_word_ids, input_mask,
            input_type_ids) -> Dict[str, tf.Tensor]:
    inputs = dict(
        input_word_ids=input_word_ids,
        input_mask=input_mask,
        input_type_ids=input_type_ids)
    outputs = self.inference_step(inputs)
    if self.params.output_encoder_outputs:
      return dict(
          logits=outputs["logits"], encoder_outputs=outputs["encoder_outputs"])
    else:
      return dict(logits=outputs["logits"])

  @tf.function
  def serve_examples(self, inputs) -> Dict[str, tf.Tensor]:
    sequence_length = self.params.parse_sequence_length
    name_to_features = {
        self.input_word_ids_field:
            tf.io.FixedLenFeature([sequence_length], tf.int64),
        "input_mask":
            tf.io.FixedLenFeature([sequence_length], tf.int64),
        self.input_type_ids_field:
            tf.io.FixedLenFeature([sequence_length], tf.int64)
    }
    features = tf.io.parse_example(inputs, name_to_features)
    features = features_to_int32(features)
    return self.serve(
        input_word_ids=features[self.input_word_ids_field],
        input_mask=features["input_mask"],
        input_type_ids=features[self.input_type_ids_field])

  def get_inference_signatures(self, function_keys: Dict[Text, Text]):
    signatures = {}
    valid_keys = ("serve", "serve_examples")
    for func_key, signature_key in function_keys.items():
      if func_key not in valid_keys:
        raise ValueError("Invalid function key for the module: %s with key %s. "
                         "Valid keys are: %s" %
                         (self.__class__, func_key, valid_keys))
      if func_key == "serve":
        signatures[signature_key] = self.serve.get_concrete_function(
            input_word_ids=tf.TensorSpec(
                shape=[None, None],
                dtype=tf.int32,
                name=self.input_word_ids_field),
            input_mask=tf.TensorSpec(
                shape=[None, None], dtype=tf.int32, name="input_mask"),
            input_type_ids=tf.TensorSpec(
                shape=[None, None],
                dtype=tf.int32,
                name=self.input_type_ids_field))
      if func_key == "serve_examples":
        signatures[signature_key] = self.serve_examples.get_concrete_function(
            tf.TensorSpec(shape=[None], dtype=tf.string, name="examples"))
    return signatures


class Translation(export_base.ExportModule):
  """The export module for the translation task."""

  @dataclasses.dataclass
  class Params(base_config.Config):
    sentencepiece_model_path: str = ""
    # Needs to be specified if padded_decode is True/on TPUs.
    batch_size: Optional[int] = None

  def __init__(self, params, model: tf.keras.Model, inference_step=None):
    super().__init__(params, model, inference_step)
    self._sp_tokenizer = tf_text.SentencepieceTokenizer(
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
    self._batch_size = params.batch_size

  @tf.function
  def serve(self, inputs) -> Dict[str, tf.Tensor]:
    return self.inference_step(inputs)

  @tf.function
  def serve_text(self, text: tf.Tensor) -> Dict[str, tf.Tensor]:
    tokenized = self._sp_tokenizer.tokenize(text).to_tensor(0)
    return self._sp_tokenizer.detokenize(
        self.serve({"inputs": tokenized})["outputs"])

  def get_inference_signatures(self, function_keys: Dict[Text, Text]):
    signatures = {}
    valid_keys = ("serve_text")
    for func_key, signature_key in function_keys.items():
      if func_key not in valid_keys:
        raise ValueError("Invalid function key for the module: %s with key %s. "
                         "Valid keys are: %s" %
                         (self.__class__, func_key, valid_keys))
      if func_key == "serve_text":
        signatures[signature_key] = self.serve_text.get_concrete_function(
            tf.TensorSpec(shape=[self._batch_size],
                          dtype=tf.string, name="text"))
    return signatures
