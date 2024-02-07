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

"""Sentence prediction (classification) task."""

from official.core import task_factory
from official.nlp.tasks import sentence_prediction
from official.projects.perceiver.configs import encoders
from official.projects.perceiver.configs import perceiver
from official.projects.perceiver.modeling.layers import decoder
from official.projects.perceiver.modeling.models import classifier
from official.projects.perceiver.modeling.networks import positional_decoder


@task_factory.register_task_cls(perceiver.SentencePredictionConfig)
class SentencePredictionTask(sentence_prediction.SentencePredictionTask):
  """Task object for sentence_prediction.

  Note: Making this similar to nlp.tasks.sentence_prediction.py to potentially
  merge.
  """

  def build_model(self):
    """Creates perceiver classification model architecture.

    Returns:
      A model instance.
    """
    encoder_network = encoders.build_encoder(self.task_config.model.encoder)
    decoder_config = self.task_config.model.decoder
    decoder_ = decoder.Decoder(decoder_config.decoder.as_dict())
    classification_decoder = positional_decoder.PositionalDecoder(
        decoder=decoder_,
        d_model=decoder_config.d_model,
        output_index_dim=decoder_config.output_index_dim,
        z_index_dim=decoder_config.z_index_dim,
        d_latents=decoder_config.d_latents,
        position_encoding_intializer_stddev=decoder_config
        .position_encoding_intializer_stddev)
    return classifier.Classifier(
        network=encoder_network,
        decoder=classification_decoder,
        num_classes=self.task_config.model.num_classes)
