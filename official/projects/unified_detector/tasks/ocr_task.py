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

"""Task definition for ocr."""

from typing import Callable, Dict, Optional, Sequence, Tuple, Union

import gin
import tensorflow as tf

from official.core import base_task
from official.core import config_definitions as cfg
from official.core import task_factory
from official.projects.unified_detector.configs import ocr_config
from official.projects.unified_detector.data_loaders import input_reader
from official.projects.unified_detector.tasks import all_models  # pylint: disable=unused-import
from official.projects.unified_detector.utils import typing

NestedTensorDict = typing.NestedTensorDict
ModelType = Union[tf.keras.layers.Layer, tf.keras.Model]


@task_factory.register_task_cls(ocr_config.OcrTaskConfig)
@gin.configurable
class OcrTask(base_task.Task):
  """Defining the OCR training task."""

  _loss_items = []

  def __init__(self,
               params: cfg.TaskConfig,
               logging_dir: Optional[str] = None,
               name: Optional[str] = None,
               model_fn: Callable[..., ModelType] = gin.REQUIRED):
    super().__init__(params, logging_dir, name)
    self._modef_fn = model_fn

  def build_model(self) -> ModelType:
    """Build and return the model, record the loss items as well."""
    model = self._modef_fn()
    self._loss_items.extend(model.loss_items)
    return model

  def build_inputs(
      self,
      params: cfg.DataConfig,
      input_context: Optional[tf.distribute.InputContext] = None
  ) -> tf.data.Dataset:
    """Build the tf.data.Dataset instance."""
    return input_reader.InputFn(is_training=params.is_training)({},
                                                                input_context)

  def build_metrics(self,
                    training: bool = True) -> Sequence[tf.keras.metrics.Metric]:
    """Build the metrics (currently, only for loss summaries in TensorBoard)."""
    del training
    metrics = []
    # Add loss items
    for name in self._loss_items:
      metrics.append(tf.keras.metrics.Mean(name, dtype=tf.float32))
    # TODO(longshangbang): add evaluation metrics
    return metrics

  def train_step(
      self,
      inputs: Tuple[NestedTensorDict, NestedTensorDict],
      model: ModelType,
      optimizer: tf.keras.optimizers.Optimizer,
      metrics: Optional[Sequence[tf.keras.metrics.Metric]] = None
  ) -> Dict[str, tf.Tensor]:
    features, labels = inputs
    input_dict = {"features": features}
    if self.task_config.model_call_needs_labels:
      input_dict["labels"] = labels

    is_mixed_precision = isinstance(optimizer,
                                    tf.keras.mixed_precision.LossScaleOptimizer)

    with tf.GradientTape() as tape:
      outputs = model(**input_dict, training=True)
      loss, loss_dict = model.compute_losses(labels=labels, outputs=outputs)
      loss = loss / tf.distribute.get_strategy().num_replicas_in_sync
      if is_mixed_precision:
        loss = optimizer.get_scaled_loss(loss)

    tvars = model.trainable_variables
    grads = tape.gradient(loss, tvars)
    if is_mixed_precision:
      grads = optimizer.get_unscaled_gradients(grads)

    optimizer.apply_gradients(list(zip(grads, tvars)))

    logs = {"loss": loss}
    if metrics:
      for m in metrics:
        m.update_state(loss_dict[m.name])
    return logs
