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

"""RetinaNet task definition."""
import tensorflow as tf

from official.core import task_factory
from official.projects.qat.vision.configs import retinanet as exp_cfg
from official.projects.qat.vision.modeling import factory
from official.vision.tasks import retinanet


@task_factory.register_task_cls(exp_cfg.RetinaNetTask)
class RetinaNetTask(retinanet.RetinaNetTask):
  """A task for RetinaNet object detection with QAT."""

  def build_model(self) -> tf.keras.Model:
    """Builds RetinaNet model with QAT."""
    model = super(RetinaNetTask, self).build_model()
    if self.task_config.quantization:
      model = factory.build_qat_retinanet(
          model,
          self.task_config.quantization,
          model_config=self.task_config.model)
    return model
