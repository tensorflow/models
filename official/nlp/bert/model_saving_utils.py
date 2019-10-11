# Copyright 2019 The TensorFlow Authors. All Rights Reserved.
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
# ==============================================================================
"""Utilities to save models."""

from __future__ import absolute_import
from __future__ import division
# from __future__ import google_type_annotations
from __future__ import print_function

import os

from absl import logging
import tensorflow as tf
import typing


def export_bert_model(model_export_path: typing.Text,
                      model: tf.keras.Model,
                      checkpoint_dir: typing.Optional[typing.Text] = None,
                      restore_model_using_load_weights: bool = False) -> None:
  """Export BERT model for serving which does not include the optimizer.

  Arguments:
      model_export_path: Path to which exported model will be saved.
      model: Keras model object to export.
      checkpoint_dir: Path from which model weights will be loaded, if
        specified.
      restore_model_using_load_weights: Whether to use checkpoint.restore() API
        for custom checkpoint or to use model.load_weights() API.
        There are 2 different ways to save checkpoints. One is using
        tf.train.Checkpoint and another is using Keras model.save_weights().
        Custom training loop implementation uses tf.train.Checkpoint API
        and Keras ModelCheckpoint callback internally uses model.save_weights()
        API. Since these two API's cannot be used toghether, model loading logic
        must be take into account how model checkpoint was saved.

  Raises:
    ValueError when either model_export_path or model is not specified.
  """
  if not model_export_path:
    raise ValueError('model_export_path must be specified.')
  if not isinstance(model, tf.keras.Model):
    raise ValueError('model must be a tf.keras.Model object.')

  if checkpoint_dir:
    # Keras compile/fit() was used to save checkpoint using
    # model.save_weights().
    if restore_model_using_load_weights:
      model_weight_path = os.path.join(checkpoint_dir, 'checkpoint')
      assert tf.io.gfile.exists(model_weight_path)
      model.load_weights(model_weight_path)

    # tf.train.Checkpoint API was used via custom training loop logic.
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


def export_pretraining_checkpoint(
    checkpoint_dir: typing.Text,
    model: tf.keras.Model,
    checkpoint_name: typing.Optional[
        typing.Text] = 'pretrained/bert_model.ckpt'):
  """Exports BERT model for as a checkpoint without optimizer.

  Arguments:
      checkpoint_dir: Path to where training model checkpoints are stored.
      model: Keras model object to export.
      checkpoint_name: File name or suffix path to export pretrained checkpoint.

  Raises:
    ValueError when either checkpoint_dir or model is not specified.
  """
  if not checkpoint_dir:
    raise ValueError('checkpoint_dir must be specified.')
  if not isinstance(model, tf.keras.Model):
    raise ValueError('model must be a tf.keras.Model object.')

  checkpoint = tf.train.Checkpoint(model=model)
  latest_checkpoint_file = tf.train.latest_checkpoint(checkpoint_dir)
  assert latest_checkpoint_file
  logging.info('Checkpoint file %s found and restoring from '
               'checkpoint', latest_checkpoint_file)
  status = checkpoint.restore(latest_checkpoint_file)
  status.assert_existing_objects_matched().expect_partial()
  saved_path = checkpoint.save(os.path.join(checkpoint_dir, checkpoint_name))
  logging.info('Exporting the model as a new TF checkpoint: %s', saved_path)


class BertModelCheckpoint(tf.keras.callbacks.Callback):
  """Keras callback that saves model at the end of every epoch."""

  def __init__(self, checkpoint_dir, checkpoint):
    """Initializes BertModelCheckpoint.

    Arguments:
      checkpoint_dir: Directory of the to be saved checkpoint file.
      checkpoint: tf.train.Checkpoint object.
    """
    super(BertModelCheckpoint, self).__init__()
    self.checkpoint_file_name = os.path.join(
        checkpoint_dir, 'bert_training_checkpoint_step_{global_step}.ckpt')
    assert isinstance(checkpoint, tf.train.Checkpoint)
    self.checkpoint = checkpoint

  def on_epoch_end(self, epoch, logs=None):
    global_step = tf.keras.backend.get_value(self.model.optimizer.iterations)
    formatted_file_name = self.checkpoint_file_name.format(
        global_step=global_step)
    saved_path = self.checkpoint.save(formatted_file_name)
    logging.info('Saving model TF checkpoint to : %s', saved_path)
