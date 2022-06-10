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

"""Semantic segmentation task definition."""
import tensorflow as tf

from official.core import task_factory
from official.projects.qat.vision.configs import semantic_segmentation as exp_cfg
from official.projects.qat.vision.modeling import factory
from official.vision.tasks import semantic_segmentation


@task_factory.register_task_cls(exp_cfg.SemanticSegmentationTask)
class SemanticSegmentationTask(semantic_segmentation.SemanticSegmentationTask):
  """A task for semantic segmentation with QAT."""

  def build_model(self) -> tf.keras.Model:
    """Builds semantic segmentation model with QAT."""
    model = super().build_model()
    input_specs = tf.keras.layers.InputSpec(shape=[None] +
                                            self.task_config.model.input_size)
    if self.task_config.quantization:
      model = factory.build_qat_segmentation_model(
          model, self.task_config.quantization, input_specs)
    return model
