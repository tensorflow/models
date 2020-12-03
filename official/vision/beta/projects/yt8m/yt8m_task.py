# Lint as: python3
# Copyright 2020 The TensorFlow Authors. All Rights Reserved.
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
"""Video classification task definition."""
from absl import logging
import tensorflow as tf
from official.core import base_task
from official.core import input_reader
from official.core import task_factory
from official.modeling import tf_utils
# from official.vision.beta.dataloaders import video_input
from official.vision.beta.projects.yt8m.dataloaders import yt8m_input
from official.vision.beta.modeling import factory_3d
from yt8m_model import YT8MModel
import average_precision_calculator as ap_calculator
import mean_average_precision_calculator as map_calculator
from configs import yt8m as yt8m_cfg


@task_factory.register_task_cls(yt8m_cfg.YT8MTask)
class YT8MTask(base_task.Task):
  """A task for video classification."""

  def build_model(self, num_classes, num_frames):
    """Builds video classification model."""
    common_input_shape = [
        d1 if d1 == d2 else None
        for d1, d2 in zip(self.task_config.train_data.feature_shape,
                          self.task_config.validation_data.feature_shape)
    ]
    input_specs = tf.keras.layers.InputSpec(shape=[None] + common_input_shape)
    logging.info('Build model input %r', common_input_shape)

    #model configuration
    model_config = self.task_config.model
    model = YT8MModel(
              input_params=model_config,
              num_frames=num_frames,
              num_classes=num_classes,
              input_specs=input_specs
              )
    return model

  def build_inputs(self, params: yt8m_cfg.DataConfig, input_context=None):
    """Builds classification input."""

    decoder = yt8m_input.Decoder()
    decoder_fn = decoder.decode
    parser = yt8m_input.Parser(input_params=params)
    postprocess_fn = yt8m_input.PostBatchProcessor(params)

    reader = input_reader.InputReader(
        params,
        dataset_fn=tf.data.TFRecordDataset,
        decoder_fn=decoder_fn,
        parser_fn=parser.parse_fn(params.is_training),
        postprocess_fn=postprocess_fn)

    dataset = reader.read(input_context=input_context)

    return dataset

  def build_losses(self, labels, model_outputs, aux_losses=None):
    """Sigmoid Cross Entropy (should be replaced by Keras implementation)
    Args:
      labels: labels.
      model_outputs: Output logits of the classifier.
      aux_losses: auxiliarly loss tensors, i.e. `losses` in keras.Model.
    Returns:
      The total loss tensor.
    """
    losses_config = self.task_config.losses
    total_loss = tf.keras.losses.binary_crossentropy(
      labels,
      model_outputs,
      from_logits=losses_config.from_logits,
      label_smoothing=losses_config.label_smoothing)

    total_loss = tf_utils.safe_mean(total_loss)

    if aux_losses:
      total_loss += tf.add_n(aux_losses)

    return total_loss

  def build_metrics(self, training=True):
    """Gets streaming metrics for training/validation.
       metric: mAP, gAP
      Args:
      num_class: A positive integer specifying the number of classes.
      top_n: A positive Integer specifying the average precision at n, or None
        to use all provided data points.
    """

    metrics = {
      'map_calculator' : map_calculator.MeanAveragePrecisionCalculator(
        num_class=self.task_config.num_classes,
        top_n=self.task_config.top_n),
      'global_ap_calculator' : ap_calculator.AveragePrecisionCalculator()
    }
    return metrics

  def process_metrics(self, metrics, labels, outputs):
    '''Processes metrics'''
    metrics['map_calculator'].accumulate(outputs, labels)
    metrics['global_ap_calculator'].accumulate(outputs,labels)

    # aps = metrics['map_calculator'].peek_map_at_n()
    # gap = metrics['global_ap_calculator'].peek_ap_at_n()


  def train_step(self, inputs, model, optimizer, metrics=None):
    """Does forward and backward.
    Args:
      inputs: a dictionary of input tensors.
            output_dict = {
          "video_ids": batch_video_ids,
          "video_matrix": batch_video_matrix,
          "labels": batch_labels,
          "num_frames": batch_frames,
          }
      model: the model, forward pass definition.
      optimizer: the optimizer for this training step.
      metrics: a nested structure of metrics objects.
    Returns:
      A dictionary of logs.
    """
    features, labels = inputs['video_matrix'], inputs['labels']
    video_ids, num_frames = inputs['video_ids'], inputs['num_frames']

    num_replicas = tf.distribute.get_strategy().num_replicas_in_sync
    with tf.GradientTape() as tape:
      outputs = model(features, training=True)
      outputs =

      # Computes per-replica loss
      loss = self.build_losses(
        model_outputs=outputs,

      )


    #reimplement in tf2



    logs = {self.loss: loss}
    if metrics:
      self.process_metrics(metrics, labels, outputs)
      logs.update({m.name: m.result() for m in metrics})
    elif model.compiled_metrics:
      self.process_compiled_metrics(model.compiled_metrics, labels, outputs)
      logs.update({m.name: m.result() for m in model.metrics})
    return logs

  def validation_step(self, inputs, model, metrics=None):
    """Validatation step.
    Args:
      inputs: a dictionary of input tensors.
      model: the keras.Model.
      metrics: a nested structure of metrics objects.
    Returns:
      A dictionary of logs.
    """
    features, labels = inputs

    outputs = self.inference_step(features['image'], model)
    outputs = tf.nest.map_structure(lambda x: tf.cast(x, tf.float32), outputs) #TODO: check if necessary
    loss = self.build_losses(model_outputs=outputs, labels=labels,
                             aux_losses=model.losses)

    logs = {self.loss: loss}
    if metrics:
      self.process_metrics(metrics, labels, outputs)
      logs.update({m.name: m.result() for m in metrics})
    elif model.compiled_metrics:
      self.process_compiled_metrics(model.compiled_metrics, labels, outputs)
      logs.update({m.name: m.result() for m in model.metrics})
    return logs

  def inference_step(self, inputs, model):
    """Performs the forward step."""
    return model(inputs, training=False)