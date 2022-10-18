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

"""Export module for BASNet."""

import tensorflow as tf

from official.projects.basnet.tasks import basnet
from official.vision.serving import semantic_segmentation


class BASNetModule(semantic_segmentation.SegmentationModule):
  """BASNet Module."""

  def _build_model(self):
    input_specs = tf.keras.layers.InputSpec(
        shape=[self._batch_size] + self._input_image_size + [3])

    return basnet.build_basnet_model(
        input_specs=input_specs,
        model_config=self.params.task.model,
        l2_regularizer=None)

  def serve(self, images):
    """Cast image to float and run inference.

    Args:
      images: uint8 Tensor of shape [batch_size, None, None, 3]
    Returns:
      Tensor holding classification output logits.
    """
    with tf.device('cpu:0'):
      images = tf.cast(images, dtype=tf.float32)

      images = tf.nest.map_structure(
          tf.identity,
          tf.map_fn(
              self._build_inputs, elems=images,
              fn_output_signature=tf.TensorSpec(
                  shape=self._input_image_size + [3], dtype=tf.float32),
              parallel_iterations=32
              )
          )

    masks = self.inference_step(images)
    keys = sorted(masks.keys())
    output = tf.image.resize(
        masks[keys[-1]],
        self._input_image_size, method='bilinear')

    return dict(predicted_masks=output)
