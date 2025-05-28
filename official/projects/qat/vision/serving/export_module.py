# Copyright 2025 The TensorFlow Authors. All Rights Reserved.
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

"""Export modules for QAT model serving/inference."""
import tensorflow as tf, tf_keras

from official.projects.qat.vision.modeling import factory as qat_factory
from official.vision import configs
from official.vision.serving import detection
from official.vision.serving import image_classification
from official.vision.serving import semantic_segmentation


class ClassificationModule(image_classification.ClassificationModule):
  """Classification Module."""

  def _build_model(self):
    model = super()._build_model()
    input_specs = tf_keras.layers.InputSpec(shape=[self._batch_size] +
                                            self._input_image_size + [3])
    return qat_factory.build_qat_classification_model(
        model, self.params.task.quantization, input_specs,
        self.params.task.model)


class SegmentationModule(semantic_segmentation.SegmentationModule):
  """Segmentation Module."""

  def _build_model(self):
    model = super()._build_model()
    input_specs = tf_keras.layers.InputSpec(shape=[self._batch_size] +
                                            self._input_image_size + [3])
    return qat_factory.build_qat_segmentation_model(
        model, self.params.task.quantization, input_specs)


class DetectionModule(detection.DetectionModule):
  """Detection Module."""

  def _build_model(self):
    model = super()._build_model()

    if isinstance(self.params.task.model, configs.retinanet.RetinaNet):
      model = qat_factory.build_qat_retinanet(model,
                                              self.params.task.quantization,
                                              self.params.task.model)
    else:
      raise ValueError('Detection module not implemented for {} model.'.format(
          type(self.params.task.model)))

    return model
