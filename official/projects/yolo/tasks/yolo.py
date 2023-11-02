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

"""Contains classes used to train Yolo."""

import collections
from typing import Optional

from absl import logging
import tensorflow as tf, tf_keras

from official.common import dataset_fn
from official.core import base_task
from official.core import config_definitions
from official.core import input_reader
from official.core import task_factory
from official.modeling import performance
from official.projects.yolo import optimization
from official.projects.yolo.configs import yolo as exp_cfg
from official.projects.yolo.dataloaders import tf_example_decoder
from official.projects.yolo.dataloaders import yolo_input
from official.projects.yolo.modeling import factory
from official.projects.yolo.ops import kmeans_anchors
from official.projects.yolo.ops import mosaic
from official.projects.yolo.ops import preprocessing_ops
from official.projects.yolo.tasks import task_utils
from official.vision.dataloaders import tfds_factory
from official.vision.dataloaders import tf_example_label_map_decoder
from official.vision.evaluation import coco_evaluator
from official.vision.ops import box_ops

OptimizationConfig = optimization.OptimizationConfig
RuntimeConfig = config_definitions.RuntimeConfig


@task_factory.register_task_cls(exp_cfg.YoloTask)
class YoloTask(base_task.Task):
  """A single-replica view of training procedure.

  YOLO task provides artifacts for training/evalution procedures, including
  loading/iterating over Datasets, initializing the model, calculating the loss,
  post-processing, and customized metrics with reduction.
  """

  def __init__(self, params, logging_dir: Optional[str] = None):
    super().__init__(params, logging_dir)
    self.coco_metric = None
    self._loss_fn = None
    self._model = None
    self._coco_91_to_80 = False
    self._metrics = []

    # globally set the random seed
    preprocessing_ops.set_random_seeds(seed=params.seed)

    if self.task_config.model.anchor_boxes.generate_anchors:
      self.generate_anchors()
    return

  def generate_anchors(self):
    """Generate Anchor boxes for an arbitrary object detection dataset."""
    input_size = self.task_config.model.input_size
    anchor_cfg = self.task_config.model.anchor_boxes
    backbone = self.task_config.model.backbone.get()

    dataset = self.task_config.train_data
    decoder = self._get_data_decoder(dataset)

    num_anchors = backbone.max_level - backbone.min_level + 1
    num_anchors *= anchor_cfg.anchors_per_scale

    gbs = dataset.global_batch_size
    dataset.global_batch_size = 1
    box_reader = kmeans_anchors.BoxGenInputReader(
        dataset,
        dataset_fn=dataset_fn.pick_dataset_fn(
            self.task_config.train_data.file_type),
        decoder_fn=decoder.decode)

    boxes = box_reader.read(
        k=num_anchors,
        anchors_per_scale=anchor_cfg.anchors_per_scale,
        image_resolution=input_size,
        scaling_mode=anchor_cfg.scaling_mode,
        box_generation_mode=anchor_cfg.box_generation_mode,
        num_samples=anchor_cfg.num_samples)

    dataset.global_batch_size = gbs

    with open('anchors.txt', 'w') as f:
      f.write(f'input resolution: {input_size} \n boxes: \n {boxes}')
      logging.info('INFO: boxes will be saved to anchors.txt, mack sure to save'
                   'them and update the boxes feild in you yaml config file.')

    anchor_cfg.set_boxes(boxes)
    return boxes

  def build_model(self):
    """Build an instance of Yolo."""

    model_base_cfg = self.task_config.model
    l2_weight_decay = self.task_config.weight_decay / 2.0

    input_size = model_base_cfg.input_size.copy()
    input_specs = tf_keras.layers.InputSpec(shape=[None] + input_size)
    l2_regularizer = (
        tf_keras.regularizers.l2(l2_weight_decay) if l2_weight_decay else None)
    model, losses = factory.build_yolo(
        input_specs, model_base_cfg, l2_regularizer)
    model.build(input_specs.shape)
    model.summary(print_fn=logging.info)

    # save for later usage within the task.
    self._loss_fn = losses
    self._model = model
    return model

  def _get_data_decoder(self, params):
    """Get a decoder object to decode the dataset."""
    if params.tfds_name:
      decoder = tfds_factory.get_detection_decoder(params.tfds_name)
    else:
      decoder_cfg = params.decoder.get()
      if params.decoder.type == 'simple_decoder':
        self._coco_91_to_80 = decoder_cfg.coco91_to_80
        decoder = tf_example_decoder.TfExampleDecoder(
            coco91_to_80=decoder_cfg.coco91_to_80,
            regenerate_source_id=decoder_cfg.regenerate_source_id)
      elif params.decoder.type == 'label_map_decoder':
        decoder = tf_example_label_map_decoder.TfExampleDecoderLabelMap(
            label_map=decoder_cfg.label_map,
            regenerate_source_id=decoder_cfg.regenerate_source_id)
      else:
        raise ValueError('Unknown decoder type: {}!'.format(
            params.decoder.type))
    return decoder

  def build_inputs(self, params, input_context=None):
    """Build input dataset."""
    model = self.task_config.model

    # get anchor boxes dict based on models min and max level
    backbone = model.backbone.get()
    anchor_dict, level_limits = model.anchor_boxes.get(backbone.min_level,
                                                       backbone.max_level)

    params.seed = self.task_config.seed
    # set shared patamters between mosaic and yolo_input
    base_config = dict(
        letter_box=params.parser.letter_box,
        aug_rand_translate=params.parser.aug_rand_translate,
        aug_rand_angle=params.parser.aug_rand_angle,
        aug_rand_perspective=params.parser.aug_rand_perspective,
        area_thresh=params.parser.area_thresh,
        random_flip=params.parser.random_flip,
        seed=params.seed,
    )

    # get the decoder
    decoder = self._get_data_decoder(params)

    # init Mosaic
    sample_fn = mosaic.Mosaic(
        output_size=model.input_size,
        mosaic_frequency=params.parser.mosaic.mosaic_frequency,
        mixup_frequency=params.parser.mosaic.mixup_frequency,
        jitter=params.parser.mosaic.jitter,
        mosaic_center=params.parser.mosaic.mosaic_center,
        mosaic_crop_mode=params.parser.mosaic.mosaic_crop_mode,
        aug_scale_min=params.parser.mosaic.aug_scale_min,
        aug_scale_max=params.parser.mosaic.aug_scale_max,
        **base_config)

    # init Parser
    parser = yolo_input.Parser(
        output_size=model.input_size,
        anchors=anchor_dict,
        use_tie_breaker=params.parser.use_tie_breaker,
        jitter=params.parser.jitter,
        aug_scale_min=params.parser.aug_scale_min,
        aug_scale_max=params.parser.aug_scale_max,
        aug_rand_hue=params.parser.aug_rand_hue,
        aug_rand_saturation=params.parser.aug_rand_saturation,
        aug_rand_brightness=params.parser.aug_rand_brightness,
        max_num_instances=params.parser.max_num_instances,
        scale_xy=model.detection_generator.scale_xy.get(),
        expanded_strides=model.detection_generator.path_scales.get(),
        darknet=model.darknet_based_model,
        best_match_only=params.parser.best_match_only,
        anchor_t=params.parser.anchor_thresh,
        random_pad=params.parser.random_pad,
        level_limits=level_limits,
        dtype=params.dtype,
        **base_config)

    # init the dataset reader
    reader = input_reader.InputReader(
        params,
        dataset_fn=dataset_fn.pick_dataset_fn(params.file_type),
        decoder_fn=decoder.decode,
        sample_fn=sample_fn.mosaic_fn(is_training=params.is_training),
        parser_fn=parser.parse_fn(params.is_training))
    dataset = reader.read(input_context=input_context)
    return dataset

  def build_metrics(self, training=True):
    """Build detection metrics."""
    metrics = []

    backbone = self.task_config.model.backbone.get()
    metric_names = collections.defaultdict(list)
    for key in range(backbone.min_level, backbone.max_level + 1):
      key = str(key)
      metric_names[key].append('loss')
      metric_names[key].append('avg_iou')
      metric_names[key].append('avg_obj')

    metric_names['net'].append('box')
    metric_names['net'].append('class')
    metric_names['net'].append('conf')

    for _, key in enumerate(metric_names.keys()):
      metrics.append(task_utils.ListMetrics(metric_names[key], name=key))

    self._metrics = metrics
    if not training:
      annotation_file = self.task_config.annotation_file
      if self._coco_91_to_80:
        annotation_file = None
      self.coco_metric = coco_evaluator.COCOEvaluator(
          annotation_file=annotation_file,
          include_mask=False,
          need_rescale_bboxes=False,
          per_category_metrics=self._task_config.per_category_metrics,
          max_num_eval_detections=self.task_config.max_num_eval_detections)

    return metrics

  def build_losses(self, outputs, labels, aux_losses=None):
    """Build YOLO losses."""
    return self._loss_fn(labels, outputs)

  def train_step(self, inputs, model, optimizer, metrics=None):
    """Train Step.

    Forward step and backwards propagate the model.

    Args:
      inputs: a dictionary of input tensors.
      model: the model, forward pass definition.
      optimizer: the optimizer for this training step.
      metrics: a nested structure of metrics objects.

    Returns:
      A dictionary of logs.
    """
    image, label = inputs

    with tf.GradientTape(persistent=False) as tape:
      # Compute a prediction
      y_pred = model(image, training=True)

      # Cast to float32 for gradietn computation
      y_pred = tf.nest.map_structure(lambda x: tf.cast(x, tf.float32), y_pred)

      # Get the total loss
      (scaled_loss, metric_loss,
       loss_metrics) = self.build_losses(y_pred['raw_output'], label)

      # Scale the loss for numerical stability
      if isinstance(optimizer, tf_keras.mixed_precision.LossScaleOptimizer):
        scaled_loss = optimizer.get_scaled_loss(scaled_loss)

    # Compute the gradient
    train_vars = model.trainable_variables
    gradients = tape.gradient(scaled_loss, train_vars)

    # Get unscaled loss if we are using the loss scale optimizer on fp16
    if isinstance(optimizer, tf_keras.mixed_precision.LossScaleOptimizer):
      gradients = optimizer.get_unscaled_gradients(gradients)

    # Apply gradients to the model
    optimizer.apply_gradients(zip(gradients, train_vars))
    logs = {self.loss: metric_loss}

    # Compute all metrics
    if metrics:
      for m in metrics:
        m.update_state(loss_metrics[m.name])
        logs.update({m.name: m.result()})
    return logs

  def _reorg_boxes(self, boxes, info, num_detections):
    """Scale and Clean boxes prior to Evaluation."""
    mask = tf.sequence_mask(num_detections, maxlen=tf.shape(boxes)[1])
    mask = tf.cast(tf.expand_dims(mask, axis=-1), boxes.dtype)

    # Denormalize the boxes by the shape of the image
    inshape = tf.expand_dims(info[:, 1, :], axis=1)
    ogshape = tf.expand_dims(info[:, 0, :], axis=1)
    scale = tf.expand_dims(info[:, 2, :], axis=1)
    offset = tf.expand_dims(info[:, 3, :], axis=1)

    boxes = box_ops.denormalize_boxes(boxes, inshape)
    boxes = box_ops.clip_boxes(boxes, inshape)
    boxes += tf.tile(offset, [1, 1, 2])
    boxes /= tf.tile(scale, [1, 1, 2])
    boxes = box_ops.clip_boxes(boxes, ogshape)

    # Mask the boxes for usage
    boxes *= mask
    boxes += (mask - 1)
    return boxes

  def validation_step(self, inputs, model, metrics=None):
    """Validatation step.

    Args:
      inputs: a dictionary of input tensors.
      model: the keras.Model.
      metrics: a nested structure of metrics objects.

    Returns:
      A dictionary of logs.
    """
    image, label = inputs

    # Step the model once
    y_pred = model(image, training=False)
    y_pred = tf.nest.map_structure(lambda x: tf.cast(x, tf.float32), y_pred)
    (_, metric_loss, loss_metrics) = self.build_losses(y_pred['raw_output'],
                                                       label)
    logs = {self.loss: metric_loss}

    # Reorganize and rescale the boxes
    info = label['groundtruths']['image_info']
    boxes = self._reorg_boxes(y_pred['bbox'], info, y_pred['num_detections'])

    # Build the input for the coc evaluation metric
    coco_model_outputs = {
        'detection_boxes': boxes,
        'detection_scores': y_pred['confidence'],
        'detection_classes': y_pred['classes'],
        'num_detections': y_pred['num_detections'],
        'source_id': label['groundtruths']['source_id'],
        'image_info': label['groundtruths']['image_info']
    }

    # Compute all metrics
    if metrics:
      logs.update(
          {self.coco_metric.name: (label['groundtruths'], coco_model_outputs)})
      for m in metrics:
        m.update_state(loss_metrics[m.name])
        logs.update({m.name: m.result()})
    return logs

  def aggregate_logs(self, state=None, step_outputs=None):
    """Get Metric Results."""
    if not state:
      self.coco_metric.reset_states()
      state = self.coco_metric
    self.coco_metric.update_state(step_outputs[self.coco_metric.name][0],
                                  step_outputs[self.coco_metric.name][1])
    return state

  def reduce_aggregated_logs(self, aggregated_logs, global_step=None):
    """Reduce logs and remove unneeded items. Update with COCO results."""
    res = self.coco_metric.result()
    return res

  def initialize(self, model: tf_keras.Model):
    """Loading pretrained checkpoint."""

    if not self.task_config.init_checkpoint:
      logging.info('Training from Scratch.')
      return

    ckpt_dir_or_file = self.task_config.init_checkpoint
    if tf.io.gfile.isdir(ckpt_dir_or_file):
      ckpt_dir_or_file = tf.train.latest_checkpoint(ckpt_dir_or_file)

    # Restoring checkpoint.
    if self.task_config.init_checkpoint_modules == 'all':
      ckpt = tf.train.Checkpoint(**model.checkpoint_items)
      status = ckpt.read(ckpt_dir_or_file)
      status.expect_partial().assert_existing_objects_matched()
    else:
      ckpt_items = {}
      if 'backbone' in self.task_config.init_checkpoint_modules:
        ckpt_items.update(backbone=model.backbone)
      if 'decoder' in self.task_config.init_checkpoint_modules:
        ckpt_items.update(decoder=model.decoder)

      ckpt = tf.train.Checkpoint(**ckpt_items)
      status = ckpt.read(ckpt_dir_or_file)
      status.expect_partial().assert_existing_objects_matched()

    logging.info('Finished loading pretrained checkpoint from %s',
                 ckpt_dir_or_file)

  def create_optimizer(self,
                       optimizer_config: OptimizationConfig,
                       runtime_config: Optional[RuntimeConfig] = None):
    """Creates an TF optimizer from configurations.

    Args:
      optimizer_config: the parameters of the Optimization settings.
      runtime_config: the parameters of the runtime.

    Returns:
      A tf.optimizers.Optimizer object.
    """
    opt_factory = optimization.YoloOptimizerFactory(optimizer_config)
    # pylint: disable=protected-access
    ema = opt_factory._use_ema
    opt_factory._use_ema = False

    opt_type = opt_factory._optimizer_type
    if opt_type == 'sgd_torch':
      optimizer = opt_factory.build_optimizer(opt_factory.build_learning_rate())
      optimizer.set_bias_lr(
          opt_factory.get_bias_lr_schedule(self._task_config.smart_bias_lr))
      optimizer.search_and_set_variable_groups(self._model.trainable_variables)
    else:
      optimizer = opt_factory.build_optimizer(opt_factory.build_learning_rate())
    opt_factory._use_ema = ema

    if ema:
      logging.info('EMA is enabled.')
    optimizer = opt_factory.add_ema(optimizer)

    # pylint: enable=protected-access

    if runtime_config and runtime_config.loss_scale:
      use_float16 = runtime_config.mixed_precision_dtype == 'float16'
      optimizer = performance.configure_optimizer(
          optimizer,
          use_float16=use_float16,
          loss_scale=runtime_config.loss_scale)

    return optimizer
