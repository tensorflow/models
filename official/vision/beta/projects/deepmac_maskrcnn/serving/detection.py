# Copyright 2021 The TensorFlow Authors. All Rights Reserved.
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

"""Detection input and model functions for serving/inference."""

import tensorflow as tf

from official.vision.beta.projects.deepmac_maskrcnn.configs import deep_mask_head_rcnn as cfg
from official.vision.beta.projects.deepmac_maskrcnn.tasks import deep_mask_head_rcnn
from official.vision.beta.serving import detection


class DetectionModule(detection.DetectionModule):
  """Detection Module."""

  def _build_model(self):

    if self._batch_size is None:
      ValueError("batch_size can't be None for detection models")
    if not self.params.task.model.detection_generator.use_batched_nms:
      ValueError('Only batched_nms is supported.')
    input_specs = tf.keras.layers.InputSpec(shape=[self._batch_size] +
                                            self._input_image_size + [3])

    if isinstance(self.params.task.model, cfg.DeepMaskHeadRCNN):
      model = deep_mask_head_rcnn.build_maskrcnn(
          input_specs=input_specs, model_config=self.params.task.model)
    else:
      raise ValueError('Detection module not implemented for {} model.'.format(
          type(self.params.task.model)))

    return model
