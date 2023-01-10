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

"""Task for masked autoencoder pretraining."""

from typing import Optional
import tensorflow as tf

from official.core import base_task
from official.core import input_reader
from official.core import task_factory
from official.modeling import tf_utils
from official.projects.mae.configs import mae as mae_cfg
from official.projects.mae.modeling import masked_ae
from official.projects.mae.modeling import vit
from official.vision.dataloaders import classification_input
from official.vision.dataloaders import tfds_factory


@task_factory.register_task_cls(mae_cfg.MAEConfig)
class MaskedAETask(base_task.Task):
  """Task for masked autoencoder training."""

  def build_model(self) -> tf.keras.Model:
    encoder = vit.VisionTransformer(
        self.task_config.patch_h,
        self.task_config.patch_w,
        0.0)
    # trigger build to be called.
    input_size = self.task_config.input_size
    encoder({'images': tf.ones((1, input_size[0], input_size[1], 3))})
    model = masked_ae.MaskedAE(encoder)
    return model

  def build_inputs(self,
                   params,
                   input_context: Optional[tf.distribute.InputContext] = None):
    num_classes = self.task_config.num_classes
    input_size = self.task_config.input_size
    image_field_key = self.task_config.train_data.image_field_key
    label_field_key = self.task_config.train_data.label_field_key

    decoder = tfds_factory.get_classification_decoder(params.tfds_name)
    parser = classification_input.Parser(
        output_size=input_size[:2],
        num_classes=num_classes,
        image_field_key=image_field_key,
        label_field_key=label_field_key,
        decode_jpeg_only=params.decode_jpeg_only,
        aug_rand_hflip=params.aug_rand_hflip,
        aug_type=params.aug_type,
        color_jitter=params.color_jitter,
        random_erasing=params.random_erasing,
        dtype=params.dtype,
        crop_area_range=params.crop_area_range)

    def patch_and_mask(images, labels):
      del labels
      patches = vit.to_patch(
          images, self.task_config.patch_h, self.task_config.patch_w)
      batch_size, num_h_patches, num_w_patches = tf_utils.get_shape_list(
          patches)[:3]
      num_patches = num_h_patches * num_w_patches
      num_masked = tf.cast(
          self.task_config.masking_ratio * num_patches, dtype=tf.int32)
      r = tf.random.uniform((batch_size, num_patches))
      rand_indices = tf.argsort(r)

      masked_indices = rand_indices[:, :num_masked]
      unmasked_indices = rand_indices[:, num_masked:]
      patches_1d = tf.reshape(patches, (batch_size, num_patches, -1))
      masked_patches = tf.gather(patches_1d, masked_indices, batch_dims=1)

      if self.task_config.norm_target:
        mean = tf.reduce_mean(masked_patches, axis=-1, keepdims=True)
        var = tf.math.reduce_variance(masked_patches, axis=-1, keepdims=True)
        std = (var + 1.e-6)**.5
        masked_patches = (masked_patches - mean) / std

      return {'patches': patches,
              'masked_indices': masked_indices,
              'unmasked_indices': unmasked_indices}, masked_patches

    reader = input_reader.InputReader(
        params=params,
        decoder_fn=decoder.decode,
        parser_fn=parser.parse_fn(params.is_training),
        postprocess_fn=patch_and_mask)

    dataset = reader.read(input_context=input_context)
    return dataset

  def build_losses(self, labels, model_outputs, aux_losses=None) -> tf.Tensor:
    return tf.keras.metrics.mean_squared_error(
        labels, model_outputs)
