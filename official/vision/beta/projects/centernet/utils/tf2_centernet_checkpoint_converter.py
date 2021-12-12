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

"""A converter from a tf1 OD API checkpoint to a tf2 checkpoint."""

from absl import app
from absl import flags
from absl import logging
import tensorflow as tf

from official.vision.beta.modeling.backbones import factory
from official.vision.beta.projects.centernet.common import registry_imports  # pylint: disable=unused-import
from official.vision.beta.projects.centernet.configs import backbones
from official.vision.beta.projects.centernet.configs import centernet
from official.vision.beta.projects.centernet.modeling import centernet_model
from official.vision.beta.projects.centernet.modeling.heads import centernet_head
from official.vision.beta.projects.centernet.modeling.layers import detection_generator
from official.vision.beta.projects.centernet.utils.checkpoints import load_weights
from official.vision.beta.projects.centernet.utils.checkpoints import read_checkpoints

FLAGS = flags.FLAGS

flags.DEFINE_string("checkpoint_to_convert", None,
                    "Initial checkpoint from a pretrained model.")
flags.DEFINE_string("checkpoint_backbone_name", "hourglass104_512",
                    "IIndicate the desired backbone configuration.")
flags.DEFINE_string("checkpoint_head_name", "detection_2d",
                    "Indicate the desired head configuration.")
flags.DEFINE_string("converted_checkpoint_path", None,
                    "Output path of converted checkpoint.")
flags.DEFINE_integer("hourglass_id", 52,
                     "Model id of hourglass backbone.")
flags.DEFINE_integer("num_hourglasses", 2,
                     "Number of hourglass blocks in backbone.")


def _create_centernet_model(model_id: int = 52,
                            num_hourglasses: int = 2
                            ) -> centernet_model.CenterNetModel:
  """Create centernet model to load TF1 weights."""
  task_config = centernet.CenterNetTask(
      model=centernet.CenterNetModel(
          backbone=backbones.Backbone(
              type="hourglass",
              hourglass=backbones.Hourglass(
                  model_id=model_id, num_hourglasses=num_hourglasses))))
  model_config = task_config.model

  backbone = factory.build_backbone(
      input_specs=tf.keras.layers.InputSpec(shape=[1, 512, 512, 3]),
      backbone_config=model_config.backbone,
      norm_activation_config=model_config.norm_activation)

  task_outputs = task_config.get_output_length_dict()
  head = centernet_head.CenterNetHead(
      input_specs=backbone.output_specs,
      task_outputs=task_outputs,
      input_levels=model_config.head.input_levels)

  detect_generator_obj = detection_generator.CenterNetDetectionGenerator()

  model = centernet_model.CenterNetModel(
      backbone=backbone, head=head, detection_generator=detect_generator_obj)
  logging.info("Successfully created centernet model.")

  return model


def _load_weights(model: centernet_model.CenterNetModel,
                  ckpt_dir_or_file: str,
                  ckpt_backbone_name: str,
                  ckpt_head_name: str):
  """Read TF1 checkpoint and load the weights to centernet model."""
  weights_dict, _ = read_checkpoints.get_ckpt_weights_as_dict(
      ckpt_dir_or_file)
  load_weights.load_weights_model(
      model=model,
      weights_dict=weights_dict,
      backbone_name=ckpt_backbone_name,
      head_name=ckpt_head_name)


def _save_checkpoint(model: centernet_model.CenterNetModel,
                     ckpt_dir: str):
  """Save the TF2 centernet model checkpoint."""
  checkpoint = tf.train.Checkpoint(model=model, **model.checkpoint_items)
  manager = tf.train.CheckpointManager(checkpoint,
                                       directory=ckpt_dir,
                                       max_to_keep=3)
  manager.save()
  logging.info("Save checkpoint to %s.", ckpt_dir)


def convert_checkpoint(model_id: int,
                       num_hourglasses: int,
                       ckpt_dir_or_file: str,
                       ckpt_backbone_name: str,
                       ckpt_head_name: str,
                       output_ckpt_dir: str):
  """Convert the TF1 OD API checkpoint to a tf2 checkpoint."""
  model = _create_centernet_model(
      model_id=model_id,
      num_hourglasses=num_hourglasses)
  _load_weights(
      model=model,
      ckpt_dir_or_file=ckpt_dir_or_file,
      ckpt_backbone_name=ckpt_backbone_name,
      ckpt_head_name=ckpt_head_name)
  _save_checkpoint(
      model=model,
      ckpt_dir=output_ckpt_dir)


def main(_):
  convert_checkpoint(
      model_id=FLAGS.hourglass_id,
      num_hourglasses=FLAGS.num_hourglasses,
      ckpt_dir_or_file=FLAGS.checkpoint_to_convert,
      ckpt_backbone_name=FLAGS.checkpoint_backbone_name,
      ckpt_head_name=FLAGS.checkpoint_head_name,
      output_ckpt_dir=FLAGS.converted_checkpoint_path)


if __name__ == "__main__":
  app.run(main)
