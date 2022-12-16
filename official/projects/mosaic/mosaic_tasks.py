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

"""Task definition for image semantic segmentation with MOSAIC models."""

from absl import logging
import tensorflow as tf

from official.core import task_factory
from official.projects.mosaic.configs import mosaic_config
from official.projects.mosaic.modeling import mosaic_model
from official.vision.tasks import semantic_segmentation as seg_tasks


@task_factory.register_task_cls(mosaic_config.MosaicSemanticSegmentationTask)
class MosaicSemanticSegmentationTask(seg_tasks.SemanticSegmentationTask):
  """A task for semantic segmentation using MOSAIC model."""

  # Note: the `build_model` is overrided to add an additional `train` flag
  # for the purpose of indicating the model is built for performing `training`
  # or `eval`. This is to make sure the model is initialized with proper
  # `input_shape` if the model will be trained and evaluated in different
  # `input_shape`. For example, the model is trained with cropping but
  # evaluated with original shape.
  def build_model(self, training: bool = True) -> tf.keras.Model:
    """Builds MOSAIC segmentation model."""
    input_specs = tf.keras.layers.InputSpec(
        shape=[None] + self.task_config.model.input_size)

    l2_weight_decay = self.task_config.losses.l2_weight_decay
    # Divide weight decay by 2.0 to match the implementation of tf.nn.l2_loss.
    # (https://www.tensorflow.org/api_docs/python/tf/keras/regularizers/l2)
    # (https://www.tensorflow.org/api_docs/python/tf/nn/l2_loss)
    l2_regularizer = (tf.keras.regularizers.l2(
        l2_weight_decay / 2.0) if l2_weight_decay else None)

    model = mosaic_model.build_mosaic_segmentation_model(
        input_specs=input_specs,
        model_config=self.task_config.model,
        l2_regularizer=l2_regularizer)

    # Note: Create a dummy input and call model instance to initialize.
    # This ensures all the layers are built; otherwise some layers may be
    # missing from the model and cannot be associated with variables from
    # a loaded checkpoint. The input size is determined by whether the model
    # is built for performing training or eval.
    if training:
      input_size = self.task_config.train_data.output_size
      crop_size = self.task_config.train_data.crop_size
      if crop_size:
        input_size = crop_size
    else:
      input_size = self.task_config.validation_data.output_size
    dummy_input = tf.ones(shape=[1] + input_size + [3])
    model(dummy_input)

    return model

  def initialize(self, model: tf.keras.Model):
    """Loads pretrained checkpoint."""
    if not self.task_config.init_checkpoint:
      return

    ckpt_dir_or_file = self.task_config.init_checkpoint
    if tf.io.gfile.isdir(ckpt_dir_or_file):
      ckpt_dir_or_file = tf.train.latest_checkpoint(ckpt_dir_or_file)

    # Restoring checkpoint.
    if 'all' in self.task_config.init_checkpoint_modules:
      ckpt = tf.train.Checkpoint(**model.checkpoint_items)
      status = ckpt.read(ckpt_dir_or_file)
      status.expect_partial().assert_existing_objects_matched()
    else:
      ckpt_items = {}
      if 'backbone' in self.task_config.init_checkpoint_modules:
        ckpt_items.update(backbone=model.backbone)
      if 'neck' in self.task_config.init_checkpoint_modules:
        ckpt_items.update(neck=model.neck)

      ckpt = tf.train.Checkpoint(**ckpt_items)
      status = ckpt.read(ckpt_dir_or_file)
      status.expect_partial().assert_existing_objects_matched()

    logging.info('Finished loading pretrained checkpoint from %s',
                 ckpt_dir_or_file)
