# Copyright 2024 The TensorFlow Authors. All Rights Reserved.
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

"""Question/Answering configuration definition."""
import dataclasses

import tensorflow as tf, tf_keras

import tensorflow_model_optimization as tfmot
from official.core import task_factory
from official.nlp import modeling
from official.nlp.tasks import question_answering
from official.projects.qat.nlp.modeling.layers import mobile_bert_layers
from official.projects.qat.nlp.modeling.layers import transformer_encoder_block
from official.projects.qat.nlp.modeling.models import bert_span_labeler
from official.projects.qat.nlp.quantization import configs
from official.projects.qat.nlp.quantization import schemes


@dataclasses.dataclass
class QuantizedModelQAConfig(question_answering.QuestionAnsweringConfig):
  pass


@task_factory.register_task_cls(QuantizedModelQAConfig)
class QuantizedModelQATask(question_answering.QuestionAnsweringTask):
  """Task object for question answering with QAT."""

  def build_model(self):
    model = super(QuantizedModelQATask, self).build_model()
    # pylint: disable=protected-access
    encoder_network = model._network
    # pylint: enable=protected-access

    with tfmot.quantization.keras.quantize_scope({
        'TruncatedNormal':
            tf_keras.initializers.TruncatedNormal,
        'MobileBertTransformerQuantized':
            mobile_bert_layers.MobileBertTransformerQuantized,
        'MobileBertEmbeddingQuantized':
            mobile_bert_layers.MobileBertEmbeddingQuantized,
        'TransformerEncoderBlockQuantized':
            transformer_encoder_block.TransformerEncoderBlockQuantized,
        'NoQuantizeConfig':
            configs.NoQuantizeConfig,
    }):
      def quantize_annotate_layer(layer):
        if isinstance(layer, (tf_keras.layers.LayerNormalization)):
          return tfmot.quantization.keras.quantize_annotate_layer(
              layer, configs.Default8BitOutputQuantizeConfig())
        if isinstance(layer, (tf_keras.layers.Dense,
                              tf_keras.layers.Dropout)):
          return tfmot.quantization.keras.quantize_annotate_layer(layer)
        if isinstance(layer, (modeling.layers.TransformerEncoderBlock,
                              modeling.layers.MobileBertTransformer,
                              modeling.layers.MobileBertEmbedding)):
          return tfmot.quantization.keras.quantize_annotate_layer(
              layer, configs.NoQuantizeConfig())
        return layer

      annotated_encoder_network = tf_keras.models.clone_model(
          encoder_network,
          clone_function=quantize_annotate_layer,
      )
      quantized_encoder_network = tfmot.quantization.keras.quantize_apply(
          annotated_encoder_network, scheme=schemes.Default8BitQuantizeScheme())

    encoder_cfg = self.task_config.model.encoder.get()
    model = bert_span_labeler.BertSpanLabelerQuantized(
        network=quantized_encoder_network,
        initializer=tf_keras.initializers.TruncatedNormal(
            stddev=encoder_cfg.initializer_range))
    return model
