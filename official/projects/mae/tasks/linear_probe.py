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

"""Image classification task with ViT and linear probe."""
import dataclasses
from typing import Optional
import tensorflow as tf, tf_keras

from official.core import base_task
from official.core import input_reader
from official.core import task_factory
from official.projects.mae.modeling import vit
from official.projects.mae.tasks import image_classification
from official.vision.dataloaders import classification_input
from official.vision.dataloaders import tfds_factory


@dataclasses.dataclass
class ViTLinearProbeConfig(image_classification.ViTConfig):
  """The LinearProbe task config."""


@task_factory.register_task_cls(ViTLinearProbeConfig)
class ViTLinearProbeTask(base_task.Task):
  """Image classificaiton with ViT and load checkpoint if exists."""

  def build_model(self) -> tf_keras.Model:
    encoder = vit.VisionTransformer(
        self.task_config.patch_h,
        self.task_config.patch_w,
        self.task_config.init_stochastic_depth_rate,
    )
    # Freeze backbone.
    encoder.trainable = False
    model = vit.ViTLinearClassifier(encoder, self.task_config.num_classes)
    model(tf.ones((1, 224, 224, 3)))
    return model

  def build_inputs(
      self, params, input_context: Optional[tf.distribute.InputContext] = None
  ):
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
    )

    postprocess_fn = lambda images, labels: (  # pylint:disable=g-long-lambda
        images,
        tf.one_hot(labels, num_classes),
    )

    reader = input_reader.InputReader(
        params=params,
        decoder_fn=decoder.decode,
        parser_fn=parser.parse_fn(params.is_training),
        postprocess_fn=postprocess_fn,
    )

    dataset = reader.read(input_context=input_context)
    return dataset

  def initialize(self, model: tf_keras.Model):
    """Load encoder if checkpoint exists.

    Args:
      model: The keras.Model built or used by this task.
    """
    ckpt_dir_or_file = self.task_config.init_checkpoint
    if tf.io.gfile.isdir(ckpt_dir_or_file):
      ckpt_dir_or_file = tf.train.latest_checkpoint(ckpt_dir_or_file)
    if not ckpt_dir_or_file:
      return

    checkpoint_items = dict(encoder=model.encoder)
    ckpt = tf.train.Checkpoint(**checkpoint_items)
    status = ckpt.read(ckpt_dir_or_file)
    status.expect_partial().assert_existing_objects_matched()

  def build_metrics(self, training=None):
    del training
    metrics = [
        tf_keras.metrics.CategoricalAccuracy(name='accuracy'),
    ]
    return metrics

  def build_losses(self, labels, model_outputs, aux_losses=None) -> tf.Tensor:
    return tf_keras.losses.categorical_crossentropy(
        labels, model_outputs, from_logits=True
    )
