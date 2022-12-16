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

"""Export modules for QAT model serving/inference."""

import tensorflow as tf

from official.projects.mosaic.modeling import mosaic_model
from official.projects.mosaic.qat.modeling import factory as qat_factory
from official.vision.serving import semantic_segmentation


class MosaicModule(semantic_segmentation.SegmentationModule):
  """MOSAIC Module."""

  def _build_model(self) -> tf.keras.Model:
    input_specs = tf.keras.layers.InputSpec(shape=[1] +
                                            self._input_image_size + [3])

    model = mosaic_model.build_mosaic_segmentation_model(
        input_specs=input_specs,
        model_config=self.params.task.model,
        l2_regularizer=None)

    dummy_input = tf.ones(shape=input_specs.shape)
    model(dummy_input)
    # Check whether "quantization" is in task config to support both
    # `quantized` and `non-quantized` version of Mosaic.
    if hasattr(self.params.task, "quantization"):
      return qat_factory.build_qat_mosaic_model(
          model, self.params.task.quantization, input_specs)
    return model
