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

"""Semantic segmentation task definition."""
import tensorflow as tf, tf_keras

from official.core import task_factory
from official.projects.mosaic import mosaic_tasks
from official.projects.mosaic.qat.configs import mosaic_config as exp_cfg
from official.projects.mosaic.qat.modeling import factory


@task_factory.register_task_cls(exp_cfg.MosaicSemanticSegmentationTask)
class MosaicSemanticSegmentationTask(mosaic_tasks.MosaicSemanticSegmentationTask
                                    ):
  """A task for semantic segmentation with QAT."""

  def build_model(self, training=True) -> tf_keras.Model:
    """Builds semantic segmentation model with QAT."""
    model = super().build_model(training)
    if training:
      input_size = self.task_config.train_data.output_size
      crop_size = self.task_config.train_data.crop_size
      if crop_size:
        input_size = crop_size
    else:
      input_size = self.task_config.validation_data.output_size
    input_specs = tf_keras.layers.InputSpec(shape=[None] + input_size + [3])
    if self.task_config.quantization:
      model = factory.build_qat_mosaic_model(
          model, self.task_config.quantization, input_specs)
    return model
