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

"""Utility functions."""

import os
import pprint

from absl import logging
import tensorflow as tf, tf_keras

from official.modeling import hyperparams
from official.projects.edgetpu.nlp.configs import params


def serialize_config(experiment_params: params.EdgeTPUBERTCustomParams,
                     model_dir: str):
  """Serializes and saves the experiment config."""
  params_save_path = os.path.join(model_dir, 'params.yaml')
  logging.info('Saving experiment configuration to %s', params_save_path)
  tf.io.gfile.makedirs(model_dir)
  hyperparams.save_params_dict_to_yaml(experiment_params, params_save_path)


# Note: Do not call this utility function unless you load the `flags`
# module in your script.
def config_override(experiment_params, flags_obj):
  """Overrides ExperimentConfig according to flags."""
  if not hasattr(flags_obj, 'tpu'):
    raise ModuleNotFoundError(
        '`tpu` is not found in FLAGS. Need to load flags.py first.')
  # Change runtime.tpu to the real tpu.
  experiment_params.override({
      'runtime': {
          'tpu_address': flags_obj.tpu,
      }
  })

  # Get the first level of override from `--config_file`.
  #   `--config_file` is typically used as a template that specifies the common
  #   override for a particular experiment.
  for config_file in flags_obj.config_file or []:
    experiment_params = hyperparams.override_params_dict(
        experiment_params, config_file, is_strict=True)

  # Get the second level of override from `--params_override`.
  #   `--params_override` is typically used as a further override over the
  #   template. For example, one may define a particular template for training
  #   ResNet50 on ImageNet in a config file and pass it via `--config_file`,
  #   then define different learning rates and pass it via `--params_override`.
  if flags_obj.params_override:
    experiment_params = hyperparams.override_params_dict(
        experiment_params, flags_obj.params_override, is_strict=True)

  experiment_params.validate()
  experiment_params.lock()

  pp = pprint.PrettyPrinter()
  logging.info('Final experiment parameters: %s',
               pp.pformat(experiment_params.as_dict()))

  model_dir = get_model_dir(experiment_params, flags_obj)
  if flags_obj.mode is not None:
    if 'train' in flags_obj.mode:
      # Pure eval modes do not output yaml files. Otherwise continuous eval job
      # may race against the train job for writing the same file.
      serialize_config(experiment_params, model_dir)

  return experiment_params


def get_model_dir(experiment_params, flags_obj):
  """Gets model dir from Flags."""
  del experiment_params
  return flags_obj.model_dir


def load_checkpoint(model: tf_keras.Model, ckpt_path: str):
  """Initializes model with the checkpoint."""
  ckpt_dir_or_file = ckpt_path

  if tf.io.gfile.isdir(ckpt_dir_or_file):
    ckpt_dir_or_file = tf.train.latest_checkpoint(ckpt_dir_or_file)
  # Makes sure the pretrainer variables are created.
  _ = model(model.inputs)
  checkpoint = tf.train.Checkpoint(
      **model.checkpoint_items)
  checkpoint.read(ckpt_dir_or_file).expect_partial()
  logging.info('Successfully load parameters for %s model', model.name)
