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

"""Utilities to save models."""

import os
import typing
from absl import logging
import tensorflow as tf, tf_keras


def export_bert_model(model_export_path: typing.Text,
                      model: tf_keras.Model,
                      checkpoint_dir: typing.Optional[typing.Text] = None,
                      restore_model_using_load_weights: bool = False) -> None:
  """Export BERT model for serving which does not include the optimizer.

  Args:
      model_export_path: Path to which exported model will be saved.
      model: Keras model object to export.
      checkpoint_dir: Path from which model weights will be loaded, if
        specified.
      restore_model_using_load_weights: Whether to use checkpoint.restore() API
        for custom checkpoint or to use model.load_weights() API. There are 2
        different ways to save checkpoints. One is using tf.train.Checkpoint and
        another is using Keras model.save_weights(). Custom training loop
        implementation uses tf.train.Checkpoint API and Keras ModelCheckpoint
        callback internally uses model.save_weights() API. Since these two API's
        cannot be used toghether, model loading logic must be take into account
        how model checkpoint was saved.

  Raises:
    ValueError when either model_export_path or model is not specified.
  """
  if not model_export_path:
    raise ValueError('model_export_path must be specified.')
  if not isinstance(model, tf_keras.Model):
    raise ValueError('model must be a tf_keras.Model object.')

  if checkpoint_dir:
    if restore_model_using_load_weights:
      model_weight_path = os.path.join(checkpoint_dir, 'checkpoint')
      assert tf.io.gfile.exists(model_weight_path)
      model.load_weights(model_weight_path)
    else:
      checkpoint = tf.train.Checkpoint(model=model)

      # Restores the model from latest checkpoint.
      latest_checkpoint_file = tf.train.latest_checkpoint(checkpoint_dir)
      assert latest_checkpoint_file
      logging.info('Checkpoint file %s found and restoring from '
                   'checkpoint', latest_checkpoint_file)
      checkpoint.restore(
          latest_checkpoint_file).assert_existing_objects_matched()

  model.save(model_export_path, include_optimizer=False, save_format='tf')
