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

"""Image classification task definition."""
import tensorflow as tf, tf_keras

from official.core import task_factory
from official.projects.qat.vision.configs import image_classification as exp_cfg
from official.projects.qat.vision.modeling import factory
from official.vision.tasks import image_classification


@task_factory.register_task_cls(exp_cfg.ImageClassificationTask)
class ImageClassificationTask(image_classification.ImageClassificationTask):
  """A task for image classification with QAT."""

  def build_model(self) -> tf_keras.Model:
    """Builds classification model with QAT."""
    input_specs = tf_keras.layers.InputSpec(
        shape=[None] + self.task_config.model.input_size
    )

    l2_weight_decay = self.task_config.losses.l2_weight_decay
    # Divide weight decay by 2.0 to match the implementation of tf.nn.l2_loss.
    # (https://www.tensorflow.org/api_docs/python/tf/keras/regularizers/l2)
    # (https://www.tensorflow.org/api_docs/python/tf/nn/l2_loss)
    l2_regularizer = (
        tf_keras.regularizers.l2(l2_weight_decay / 2.0)
        if l2_weight_decay
        else None
    )

    model = super().build_model()

    # Only build a QAT model when quantization version is v2; otherwise leave it
    # for outer quantization scope.
    if (
        self.task_config.quantization
        and hasattr(self.task_config.quantization, 'version')
        and self.task_config.quantization.version == 'v2'
    ):
      model = factory.build_qat_classification_model(
          model,
          self.task_config.quantization,
          input_specs=input_specs,
          model_config=self.task_config.model,
          l2_regularizer=l2_regularizer,
      )

    return model
