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

"""Image classification task definition."""
from absl import logging
import tensorflow as tf
import tensorflow_model_optimization as tfmot

from official.core import task_factory
from official.projects.pruning.configs import image_classification as exp_cfg
from official.vision.modeling.backbones import mobilenet
from official.vision.modeling.layers import nn_blocks
from official.vision.tasks import image_classification


@task_factory.register_task_cls(exp_cfg.ImageClassificationTask)
class ImageClassificationTask(image_classification.ImageClassificationTask):
  """A task for image classification with pruning."""
  _BLOCK_LAYER_SUFFIX_MAP = {
      mobilenet.Conv2DBNBlock: ('conv2d/kernel:0',),
      nn_blocks.BottleneckBlock: (
          'conv2d/kernel:0',
          'conv2d_1/kernel:0',
          'conv2d_2/kernel:0',
          'conv2d_3/kernel:0',
      ),
      nn_blocks.InvertedBottleneckBlock: (
          'conv2d/kernel:0',
          'conv2d_1/kernel:0',
          'conv2d_2/kernel:0',
          'conv2d_3/kernel:0',
          'depthwise_conv2d/depthwise_kernel:0',
      ),
      nn_blocks.ResidualBlock: (
          'conv2d/kernel:0',
          'conv2d_1/kernel:0',
          'conv2d_2/kernel:0',
      ),
  }

  def build_model(self) -> tf.keras.Model:
    """Builds classification model with pruning."""
    model = super(ImageClassificationTask, self).build_model()
    if self.task_config.pruning is None:
      return model

    pruning_cfg = self.task_config.pruning

    prunable_model = tf.keras.models.clone_model(
        model,
        clone_function=self._make_block_prunable,
    )

    original_checkpoint = pruning_cfg.pretrained_original_checkpoint
    if original_checkpoint is not None:
      ckpt = tf.train.Checkpoint(model=prunable_model, **model.checkpoint_items)
      status = ckpt.read(original_checkpoint)
      status.expect_partial().assert_existing_objects_matched()

    pruning_params = {}
    if pruning_cfg.sparsity_m_by_n is not None:
      pruning_params['sparsity_m_by_n'] = pruning_cfg.sparsity_m_by_n

    if pruning_cfg.pruning_schedule == 'PolynomialDecay':
      pruning_params['pruning_schedule'] = tfmot.sparsity.keras.PolynomialDecay(
          initial_sparsity=pruning_cfg.initial_sparsity,
          final_sparsity=pruning_cfg.final_sparsity,
          begin_step=pruning_cfg.begin_step,
          end_step=pruning_cfg.end_step,
          frequency=pruning_cfg.frequency)
    elif pruning_cfg.pruning_schedule == 'ConstantSparsity':
      pruning_params[
          'pruning_schedule'] = tfmot.sparsity.keras.ConstantSparsity(
              target_sparsity=pruning_cfg.final_sparsity,
              begin_step=pruning_cfg.begin_step,
              frequency=pruning_cfg.frequency)
    else:
      raise NotImplementedError(
          'Only PolynomialDecay and ConstantSparsity are currently supported. Not support %s'
          % pruning_cfg.pruning_schedule)

    pruned_model = tfmot.sparsity.keras.prune_low_magnitude(
        prunable_model, **pruning_params)

    # Print out prunable weights for debugging purpose.
    prunable_layers = collect_prunable_layers(pruned_model)
    pruned_weights = []
    for layer in prunable_layers:
      pruned_weights += [weight.name for weight, _, _ in layer.pruning_vars]
    unpruned_weights = [
        weight.name
        for weight in pruned_model.weights
        if weight.name not in pruned_weights
    ]

    logging.info(
        '%d / %d weights are pruned.\nPruned weights: [ \n%s \n],\n'
        'Unpruned weights: [ \n%s \n],',
        len(pruned_weights), len(model.weights), ', '.join(pruned_weights),
        ', '.join(unpruned_weights))

    return pruned_model

  def _make_block_prunable(
      self, layer: tf.keras.layers.Layer) -> tf.keras.layers.Layer:
    if isinstance(layer, tf.keras.Model):
      return tf.keras.models.clone_model(
          layer, input_tensors=None, clone_function=self._make_block_prunable)

    if layer.__class__ not in self._BLOCK_LAYER_SUFFIX_MAP:
      return layer

    prunable_weights = []
    for layer_suffix in self._BLOCK_LAYER_SUFFIX_MAP[layer.__class__]:
      for weight in layer.weights:
        if weight.name.endswith(layer_suffix):
          prunable_weights.append(weight)

    def get_prunable_weights():
      return prunable_weights

    layer.get_prunable_weights = get_prunable_weights

    return layer


def collect_prunable_layers(model):
  """Recursively collect the prunable layers in the model."""
  prunable_layers = []
  for layer in model.layers:
    if isinstance(layer, tf.keras.Model):
      prunable_layers += collect_prunable_layers(layer)
    if layer.__class__.__name__ == 'PruneLowMagnitude':
      prunable_layers.append(layer)

  return prunable_layers
