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
from __future__ import print_function

import os

from absl import logging
import tensorflow as tf


def export_bert_model(model_export_path,
                      model=None,
                      model_fn=None,
                      checkpoint_dir=None):
  """Export BERT model for serving which does not include the optimizer.

  Arguments:
      model_export_path: Path to which exported model will be saved.
      model: Keras model object to export. If none, new model is created via
        `model_fn`.
      model_fn: Function that returns a BERT model. Used when `model` is not
        provided.
      checkpoint_dir: Path from which model weights will be loaded.
  """
  if model:
    model.save(model_export_path, include_optimizer=False, save_format='tf')
    return

  assert model_fn and checkpoint_dir
  model_to_export = model_fn()
  checkpoint = tf.train.Checkpoint(model=model_to_export)
  latest_checkpoint_file = tf.train.latest_checkpoint(checkpoint_dir)
  assert latest_checkpoint_file
  logging.info('Checkpoint file %s found and restoring from '
               'checkpoint', latest_checkpoint_file)
  checkpoint.restore(latest_checkpoint_file).assert_existing_objects_matched()
  model_to_export.save(
      model_export_path, include_optimizer=False, save_format='tf')


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
