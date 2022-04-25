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

"""Tests for nlp.serving.export_saved_model."""

from absl.testing import parameterized

import tensorflow as tf
from official.nlp.configs import bert
from official.nlp.configs import encoders
from official.nlp.serving import export_savedmodel
from official.nlp.serving import export_savedmodel_util
from official.nlp.tasks import masked_lm
from official.nlp.tasks import sentence_prediction
from official.nlp.tasks import tagging


class ExportSavedModelTest(tf.test.TestCase, parameterized.TestCase):

  def test_create_export_module(self):
    export_module = export_savedmodel.create_export_module(
        task_name="SentencePrediction",
        config_file=None,
        serving_params={
            "inputs_only": False,
            "parse_sequence_length": 10
        })
    self.assertEqual(export_module.name, "sentence_prediction")
    self.assertFalse(export_module.params.inputs_only)
    self.assertEqual(export_module.params.parse_sequence_length, 10)

  def test_sentence_prediction(self):
    config = sentence_prediction.SentencePredictionConfig(
        model=sentence_prediction.ModelConfig(
            encoder=encoders.EncoderConfig(
                bert=encoders.BertEncoderConfig(vocab_size=30522,
                                                num_layers=1)),
            num_classes=2))
    task = sentence_prediction.SentencePredictionTask(config)
    model = task.build_model()
    ckpt = tf.train.Checkpoint(model=model)
    ckpt_path = ckpt.save(self.get_temp_dir())
    export_module_cls = export_savedmodel.lookup_export_module(task)
    serving_params = {"inputs_only": False}
    params = export_module_cls.Params(**serving_params)
    export_module = export_module_cls(params=params, model=model)
    export_dir = export_savedmodel_util.export(
        export_module,
        function_keys=["serve"],
        checkpoint_path=ckpt_path,
        export_savedmodel_dir=self.get_temp_dir())
    imported = tf.saved_model.load(export_dir)
    serving_fn = imported.signatures["serving_default"]

    dummy_ids = tf.ones((1, 5), dtype=tf.int32)
    inputs = dict(
        input_word_ids=dummy_ids,
        input_mask=dummy_ids,
        input_type_ids=dummy_ids)
    ref_outputs = model(inputs)
    outputs = serving_fn(**inputs)
    self.assertAllClose(ref_outputs, outputs["outputs"])
    self.assertEqual(outputs["outputs"].shape, (1, 2))

  def test_masked_lm(self):
    config = masked_lm.MaskedLMConfig(
        model=bert.PretrainerConfig(
            encoder=encoders.EncoderConfig(
                bert=encoders.BertEncoderConfig(vocab_size=30522,
                                                num_layers=1)),
            cls_heads=[
                bert.ClsHeadConfig(inner_dim=10, num_classes=2, name="foo")
            ]))
    task = masked_lm.MaskedLMTask(config)
    model = task.build_model()
    ckpt = tf.train.Checkpoint(model=model)
    ckpt_path = ckpt.save(self.get_temp_dir())
    export_module_cls = export_savedmodel.lookup_export_module(task)
    serving_params = {
        "cls_head_name": "foo",
        "parse_sequence_length": 10,
        "max_predictions_per_seq": 5
    }
    params = export_module_cls.Params(**serving_params)
    export_module = export_module_cls(params=params, model=model)
    export_dir = export_savedmodel_util.export(
        export_module,
        function_keys={
            "serve": "serving_default",
            "serve_examples": "serving_examples"
        },
        checkpoint_path=ckpt_path,
        export_savedmodel_dir=self.get_temp_dir())
    imported = tf.saved_model.load(export_dir)
    self.assertSameElements(imported.signatures.keys(),
                            ["serving_default", "serving_examples"])
    serving_fn = imported.signatures["serving_default"]
    dummy_ids = tf.ones((1, 10), dtype=tf.int32)
    dummy_pos = tf.ones((1, 5), dtype=tf.int32)
    outputs = serving_fn(
        input_word_ids=dummy_ids,
        input_mask=dummy_ids,
        input_type_ids=dummy_ids,
        masked_lm_positions=dummy_pos)
    self.assertEqual(outputs["classification"].shape, (1, 2))

  @parameterized.parameters(True, False)
  def test_tagging(self, output_encoder_outputs):
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
    ckpt = tf.train.Checkpoint(model=model)
    ckpt_path = ckpt.save(self.get_temp_dir())
    export_module_cls = export_savedmodel.lookup_export_module(task)
    serving_params = {
        "parse_sequence_length": 10,
    }
    params = export_module_cls.Params(
        **serving_params, output_encoder_outputs=output_encoder_outputs)
    export_module = export_module_cls(params=params, model=model)
    export_dir = export_savedmodel_util.export(
        export_module,
        function_keys={
            "serve": "serving_default",
            "serve_examples": "serving_examples"
        },
        checkpoint_path=ckpt_path,
        export_savedmodel_dir=self.get_temp_dir())
    imported = tf.saved_model.load(export_dir)
    self.assertCountEqual(imported.signatures.keys(),
                          ["serving_default", "serving_examples"])

    serving_fn = imported.signatures["serving_default"]
    dummy_ids = tf.ones((1, 5), dtype=tf.int32)
    inputs = dict(
        input_word_ids=dummy_ids,
        input_mask=dummy_ids,
        input_type_ids=dummy_ids)
    outputs = serving_fn(**inputs)
    self.assertEqual(outputs["logits"].shape, (1, 5, num_classes))
    if output_encoder_outputs:
      self.assertEqual(outputs["encoder_outputs"].shape, (1, 5, hidden_size))


if __name__ == "__main__":
  tf.test.main()
