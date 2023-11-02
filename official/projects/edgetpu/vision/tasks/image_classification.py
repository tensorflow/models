# Copyright 2023 The TensorFlow Authors. All Rights Reserved.
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

"""Image classification task definition."""
import os
import tempfile
from typing import Any, List, Mapping, Optional, Tuple

from absl import logging
import tensorflow as tf, tf_keras

from official.common import dataset_fn
from official.core import base_task
from official.core import task_factory
from official.modeling import tf_utils
from official.projects.edgetpu.vision.configs import mobilenet_edgetpu_config as edgetpu_cfg
from official.projects.edgetpu.vision.dataloaders import classification_input
from official.projects.edgetpu.vision.modeling import mobilenet_edgetpu_v1_model
from official.projects.edgetpu.vision.modeling import mobilenet_edgetpu_v2_model
from official.vision.configs import image_classification as base_cfg
from official.vision.dataloaders import input_reader_factory


def _copy_recursively(src: str, dst: str) -> None:
  """Recursively copy directory."""
  for src_dir, _, src_files in tf.io.gfile.walk(src):
    dst_dir = os.path.join(dst, os.path.relpath(src_dir, src))
    if not tf.io.gfile.exists(dst_dir):
      tf.io.gfile.makedirs(dst_dir)
    for src_file in src_files:
      tf.io.gfile.copy(
          os.path.join(src_dir, src_file),
          os.path.join(dst_dir, src_file),
          overwrite=True)


def get_models() -> Mapping[str, tf_keras.Model]:
  """Returns the mapping from model type name to Keras model."""
  model_mapping = {}

  def add_models(name: str, constructor: Any):
    if name in model_mapping:
      raise ValueError(f'Model {name} already exists in the mapping.')
    model_mapping[name] = constructor

  for model in mobilenet_edgetpu_v1_model.MODEL_CONFIGS.keys():
    add_models(model, mobilenet_edgetpu_v1_model.MobilenetEdgeTPU.from_name)

  for model in mobilenet_edgetpu_v2_model.MODEL_CONFIGS.keys():
    add_models(model, mobilenet_edgetpu_v2_model.MobilenetEdgeTPUV2.from_name)

  return model_mapping


def load_searched_model(saved_model_path: str) -> tf_keras.Model:
  """Loads saved model from file.

  Excepting loading MobileNet-EdgeTPU-V1/V2 models, we can also load searched
  model directly from saved model path by changing the model path in
  mobilenet_edgetpu_search (defined in mobilenet_edgetpu_config.py)

  Args:
    saved_model_path: Directory path for the saved searched model.
  Returns:
    Loaded keras model.
  """
  with tempfile.TemporaryDirectory() as tmp_dir:
    if tf.io.gfile.isdir(saved_model_path):
      _copy_recursively(saved_model_path, tmp_dir)
      load_path = tmp_dir
    else:
      raise ValueError('Saved model path is invalid.')
    load_options = tf.saved_model.LoadOptions(
        experimental_io_device='/job:localhost')
    model = tf_keras.models.load_model(load_path, options=load_options)

  return model


@task_factory.register_task_cls(edgetpu_cfg.MobilenetEdgeTPUTaskConfig)
class EdgeTPUTask(base_task.Task):
  """A task for training MobileNet-EdgeTPU models."""

  def build_model(self):
    """Builds model for MobileNet-EdgeTPU Task."""
    model_config = self.task_config.model
    model_params = model_config.model_params.as_dict()
    model_name = model_params['model_name']
    registered_models = get_models()
    if model_name in registered_models:
      logging.info('Load MobileNet-EdgeTPU-V1/V2 model.')
      logging.info(model_params)
      model = registered_models[model_name](**model_params)
    elif model_name == 'mobilenet_edgetpu_search':
      if self.task_config.saved_model_path is None:
        raise ValueError('If using MobileNet-EdgeTPU-Search model, please'
                         'specify the saved model path via the'
                         '--params_override flag.')
      logging.info('Load saved model (model from search) directly.')
      model = load_searched_model(self.task_config.saved_model_path)
    else:
      raise ValueError('Model has to be mobilenet-edgetpu model or searched'
                       'model with given saved model path.')

    return model

  def initialize(self, model: tf_keras.Model):
    """Loads pretrained checkpoint."""
    if not self.task_config.init_checkpoint:
      return

    ckpt_dir_or_file = self.task_config.init_checkpoint
    if tf.io.gfile.isdir(ckpt_dir_or_file):
      ckpt_dir_or_file = tf.train.latest_checkpoint(ckpt_dir_or_file)

    # Restoring checkpoint.
    if self.task_config.init_checkpoint_modules == 'all':
      ckpt = tf.train.Checkpoint(**model.checkpoint_items)
      status = ckpt.read(ckpt_dir_or_file)
      status.expect_partial().assert_existing_objects_matched()
    elif self.task_config.init_checkpoint_modules == 'backbone':
      ckpt = tf.train.Checkpoint(backbone=model.backbone)
      status = ckpt.read(ckpt_dir_or_file)
      status.expect_partial().assert_existing_objects_matched()
    else:
      raise ValueError(
          "Only 'all' or 'backbone' can be used to initialize the model.")

    logging.info('Finished loading pretrained checkpoint from %s',
                 ckpt_dir_or_file)

  def build_inputs(
      self,
      params: base_cfg.DataConfig,
      input_context: Optional[tf.distribute.InputContext] = None
  ) -> tf.data.Dataset:
    """Builds classification input."""

    num_classes = self.task_config.model.num_classes
    input_size = self.task_config.model.input_size
    image_field_key = self.task_config.train_data.image_field_key
    label_field_key = self.task_config.train_data.label_field_key
    is_multilabel = self.task_config.train_data.is_multilabel

    if params.tfds_name:
      raise ValueError('TFDS {} is not supported'.format(params.tfds_name))
    else:
      decoder = classification_input.Decoder(
          image_field_key=image_field_key, label_field_key=label_field_key,
          is_multilabel=is_multilabel)

    parser = classification_input.Parser(
        output_size=input_size[:2],
        num_classes=num_classes,
        image_field_key=image_field_key,
        label_field_key=label_field_key,
        decode_jpeg_only=params.decode_jpeg_only,
        aug_rand_hflip=params.aug_rand_hflip,
        aug_type=params.aug_type,
        is_multilabel=is_multilabel,
        dtype=params.dtype)

    reader = input_reader_factory.input_reader_generator(
        params,
        dataset_fn=dataset_fn.pick_dataset_fn(params.file_type),
        decoder_fn=decoder.decode,
        parser_fn=parser.parse_fn(params.is_training))

    dataset = reader.read(input_context=input_context)

    return dataset

  def build_losses(self,
                   labels: tf.Tensor,
                   model_outputs: tf.Tensor,
                   aux_losses: Optional[Any] = None) -> tf.Tensor:
    """Builds sparse categorical cross entropy loss.

    Args:
      labels: Input groundtruth labels.
      model_outputs: Output logits of the classifier.
      aux_losses: The auxiliarly loss tensors, i.e. `losses` in tf_keras.Model.

    Returns:
      The total loss tensor.
    """
    losses_config = self.task_config.losses
    is_multilabel = self.task_config.train_data.is_multilabel

    if not is_multilabel:
      if losses_config.one_hot:
        total_loss = tf_keras.losses.categorical_crossentropy(
            labels,
            model_outputs,
            from_logits=False,
            label_smoothing=losses_config.label_smoothing)
      else:
        total_loss = tf_keras.losses.sparse_categorical_crossentropy(
            labels, model_outputs, from_logits=True)
    else:
      # Multi-label weighted binary cross entropy loss.
      total_loss = tf.nn.sigmoid_cross_entropy_with_logits(
          labels=labels, logits=model_outputs)
      total_loss = tf.reduce_sum(total_loss, axis=-1)

    total_loss = tf_utils.safe_mean(total_loss)
    if aux_losses:
      total_loss += tf.add_n(aux_losses)

    return total_loss

  def build_metrics(self,
                    training: bool = True) -> List[tf_keras.metrics.Metric]:
    """Gets streaming metrics for training/validation."""
    is_multilabel = self.task_config.train_data.is_multilabel
    if not is_multilabel:
      k = self.task_config.evaluation.top_k
      if self.task_config.losses.one_hot:
        metrics = [
            tf_keras.metrics.CategoricalAccuracy(name='accuracy'),
            tf_keras.metrics.TopKCategoricalAccuracy(
                k=k, name='top_{}_accuracy'.format(k))]
      else:
        metrics = [
            tf_keras.metrics.SparseCategoricalAccuracy(name='accuracy'),
            tf_keras.metrics.SparseTopKCategoricalAccuracy(
                k=k, name='top_{}_accuracy'.format(k))]
    else:
      metrics = []
      # These metrics destablize the training if included in training. The jobs
      # fail due to OOM.
      # TODO(arashwan): Investigate adding following metric to train.
      if not training:
        metrics = [
            tf_keras.metrics.AUC(
                name='globalPR-AUC',
                curve='PR',
                multi_label=False,
                from_logits=True),
            tf_keras.metrics.AUC(
                name='meanPR-AUC',
                curve='PR',
                multi_label=True,
                num_labels=self.task_config.model.num_classes,
                from_logits=True),
        ]
    return metrics

  def train_step(self,
                 inputs: Tuple[Any, Any],
                 model: tf_keras.Model,
                 optimizer: tf_keras.optimizers.Optimizer,
                 metrics: Optional[List[Any]] = None):
    """Does forward and backward.

    Args:
      inputs: A tuple of input tensors of (features, labels).
      model: A tf_keras.Model instance.
      optimizer: The optimizer for this training step.
      metrics: A nested structure of metrics objects.

    Returns:
      A dictionary of logs.
    """
    features, labels = inputs
    is_multilabel = self.task_config.train_data.is_multilabel
    if self.task_config.losses.one_hot and not is_multilabel:
      labels = tf.one_hot(labels, self.task_config.model.num_classes)

    num_replicas = tf.distribute.get_strategy().num_replicas_in_sync
    with tf.GradientTape() as tape:
      outputs = model(features, training=True)

      # Computes per-replica loss.
      loss = self.build_losses(
          model_outputs=outputs, labels=labels, aux_losses=model.losses)
      # Scales loss as the default gradients allreduce performs sum inside the
      # optimizer.
      scaled_loss = loss / num_replicas

      # For mixed_precision policy, when LossScaleOptimizer is used, loss is
      # scaled for numerical stability.
      if isinstance(
          optimizer, tf_keras.mixed_precision.LossScaleOptimizer):
        scaled_loss = optimizer.get_scaled_loss(scaled_loss)

    tvars = model.trainable_variables
    grads = tape.gradient(scaled_loss, tvars)
    # Scales back gradient before apply_gradients when LossScaleOptimizer is
    # used.
    if isinstance(
        optimizer, tf_keras.mixed_precision.LossScaleOptimizer):
      grads = optimizer.get_unscaled_gradients(grads)
    optimizer.apply_gradients(list(zip(grads, tvars)))

    logs = {self.loss: loss}
    if metrics:
      self.process_metrics(metrics, labels, outputs)
    elif model.compiled_metrics:
      self.process_compiled_metrics(model.compiled_metrics, labels, outputs)
      logs.update({m.name: m.result() for m in model.metrics})
    return logs

  def validation_step(self,
                      inputs: Tuple[Any, Any],
                      model: tf_keras.Model,
                      metrics: Optional[List[Any]] = None):
    """Runs validatation step.

    Args:
      inputs: A tuple of input tensors of (features, labels).
      model: A tf_keras.Model instance.
      metrics: A nested structure of metrics objects.

    Returns:
      A dictionary of logs.
    """
    features, labels = inputs
    is_multilabel = self.task_config.train_data.is_multilabel
    if self.task_config.losses.one_hot and not is_multilabel:
      labels = tf.one_hot(labels, self.task_config.model.num_classes)

    outputs = self.inference_step(features, model)
    outputs = tf.nest.map_structure(lambda x: tf.cast(x, tf.float32), outputs)
    loss = self.build_losses(model_outputs=outputs, labels=labels,
                             aux_losses=model.losses)

    logs = {self.loss: loss}
    if metrics:
      self.process_metrics(metrics, labels, outputs)
    elif model.compiled_metrics:
      self.process_compiled_metrics(model.compiled_metrics, labels, outputs)
      logs.update({m.name: m.result() for m in model.metrics})
    return logs

  def inference_step(self, inputs: tf.Tensor, model: tf_keras.Model):
    """Performs the forward step."""
    return model(inputs, training=False)
