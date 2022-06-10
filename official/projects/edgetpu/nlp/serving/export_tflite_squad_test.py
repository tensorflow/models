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

"""Tests for export_tflite_squad."""

import tensorflow as tf

from official.nlp.modeling import models
from official.projects.edgetpu.nlp.configs import params
from official.projects.edgetpu.nlp.modeling import model_builder
from official.projects.edgetpu.nlp.serving import export_tflite_squad


class ExportTfliteSquadTest(tf.test.TestCase):

  def setUp(self):
    super(ExportTfliteSquadTest, self).setUp()
    experiment_params = params.EdgeTPUBERTCustomParams()
    pretrainer_model = model_builder.build_bert_pretrainer(
        experiment_params.student_model, name='pretrainer')
    encoder_network = pretrainer_model.encoder_network
    self.span_labeler = models.BertSpanLabeler(
        network=encoder_network,
        initializer=tf.keras.initializers.TruncatedNormal(stddev=0.01))

  def test_model_input_output(self):
    test_model = export_tflite_squad.build_model_for_serving(self.span_labeler)
    # Test model input order, names, and shape.
    self.assertEqual(test_model.input[0].name, 'input_word_ids')
    self.assertEqual(test_model.input[1].name, 'input_type_ids')
    self.assertEqual(test_model.input[2].name, 'input_mask')

    self.assertEqual(test_model.input[0].shape, (1, 384))
    self.assertEqual(test_model.input[1].shape, (1, 384))
    self.assertEqual(test_model.input[2].shape, (1, 384))

    # Test model output order, name, and shape.
    self.assertEqual(test_model.output[0].name, 'start_positions/Identity:0')
    self.assertEqual(test_model.output[1].name, 'end_positions/Identity:0')

    self.assertEqual(test_model.output[0].shape, (1, 384))
    self.assertEqual(test_model.output[1].shape, (1, 384))


if __name__ == '__main__':
  tf.test.main()
