"""MeshRCNN task definition."""

from official.core import base_task
from official.core import task_factory
from official.vision.beta.projects.mesh_rcnn.configs import mesh_rcnn as exp_cfg
from official.vision.beta.projects.mesh_rcnn.modeling import factory
from official.vision.beta.projects.mesh_rcnn import optimization
from official.modeling.optimization import ema_optimizer
from official.modeling import performance
from official.vision.beta.projects.mesh_rcnn.ops.mesh_ops import compute_mesh_shape
from official.vision.beta.projects.mesh_rcnn.ops.mesh_ops import MeshSampler

from official.vision.beta.projects.mesh_rcnn.dataloaders import meshrcnn_input

import tensorflow as tf
from typing import Any, Optional, List, Tuple, Mapping

from official.vision.beta.projects.mesh_rcnn.configs.mesh_rcnn import Parser

from official.vision.dataloaders import input_reader_factory
from official.vision.dataloaders import tf_example_decoder
from official.vision.dataloaders import tf_example_label_map_decoder
from official.common import dataset_fn as dataset_fn_lib

@task_factory.register_task_cls(exp_cfg.MeshRCNNTask)
class MeshRCNNTask(base_task.Task):
    """A single-replica view of training procedure.
        MeshRCNN task provides artifacts for training/evalution procedures, including
    loading/iterating over Datasets, initializing the model, calculating the loss,
    post-processing, and customized metrics with reduction.
    """
    def __init__(self, params, logging_dir: Optional[str] = None):
        super().__init__(params, logging_dir)
        return
    def build_model(self):
        """Build Mesh R-CNN model."""

        input_specs = tf.keras.layers.InputSpec(
            shape=[None] + self.task_config.model.input_size)

        l2_weight_decay = self.task_config.losses.l2_weight_decay # either T or F
        
        # Divide weight decay by 2.0 to match the implementation of tf.nn.l2_loss.
        # (https://www.tensorflow.org/api_docs/python/tf/keras/regularizers/l2)
        # (https://www.tensorflow.org/api_docs/python/tf/nn/l2_loss)
        l2_regularizer = (tf.keras.regularizers.l2(
            l2_weight_decay / 2.0) if l2_weight_decay else None)

        model, losses = factory.build_mesh_rcnn(
            input_specs, model_base_cfg, l2_regularizer)

        if self.task_config.freeze_backbone:
            model.backbone.trainable = False

        return model

    def build_inputs(
        self, 
        params: exp_cfg.DataConfig, 
        input_context: Optional[tf.distribute.InputContext] = None,
        dataset_fn: Optional[dataset_fn_lib.PossibleDatasetType] = None
    ) -> tf.data.Dataset:
        """Build input dataset."""

        """Decoder Builder"""
        decoder_cfg = params.decoder.get()
        if params.decoder.type == 'simple_decoder':
            decoder = tf_example_decoder.TfExampleDecoder(
                include_mask=self._task_config.model.include_mask,
                regenerate_source_id=decoder_cfg.regenerate_source_id,
                mask_binarize_threshold=decoder_cfg.mask_binarize_threshold)
        elif params.decoder.type == 'label_map_decoder':
            decoder = tf_example_label_map_decoder.TfExampleDecoderLabelMap(
                label_map=decoder_cfg.label_map,
                include_mask=self._task_config.model.include_mask,
                regenerate_source_id=decoder_cfg.regenerate_source_id,
                mask_binarize_threshold=decoder_cfg.mask_binarize_threshold)
        else:
            raise ValueError('Unknown decoder type: {}!'.format(params.decoder.type))

        #Parser
        parser = meshrcnn_input.Parser(
            output_size=self.task_config.model.input_size[:2],
            min_level=self.task_config.model.min_level,
            max_level=self.task_config.model.max_level,
            num_scales=self.task_config.model.anchor.num_scales,
            aspect_ratios=self.task_config.model.anchor.aspect_ratios,
            anchor_size=self.task_config.model.anchor.anchor_size,
            dtype=params.dtype,
            rpn_match_threshold=Parser.rpn_match_threshold, 
            rpn_unmatched_threshold=Parser.rpn_unmatched_threshold, 
            rpn_batch_size_per_im=Parser.rpn_batch_size_per_im, 
            rpn_fg_fraction=Parser.rpn_fg_fraction, 
            aug_rand_hflip=Parser.aug_rand_hflip, 
            aug_scale_min=Parser.aug_scale_min, 
            aug_scale_max=Parser.aug_scale_max, 
            skip_crowd_during_training=Parser.skip_crowd_during_training, 
            max_num_instances=Parser.max_num_instances, 
            max_num_verts=Parser.max_num_verts, 
            max_num_faces=Parser.max_num_faces, 
            max_num_voxels=Parser.max_num_voxels, 
            include_mask=Parser.include_mask, 
            mask_crop_size=Parser.mask_crop_size
        )

        if not dataset_fn:
            dataset_fn = dataset_fn_lib.pick_dataset_fn(params.file_type)

        reader = input_reader_factory.input_reader_generator(
            params,
            dataset_fn=dataset_fn,
            decoder_fn=decoder.decode,
            parser_fn=parser.parse_fn(params.is_training))
        dataset = reader.read(input_context=input_context)
        return dataset


    def build_metrics(self, pred_num_samples, true_num_samples, training=True):
        """Build metrics."""
        pred_verts, pred_verts_mask, pred_faces, pred_faces_mask = \
        compute_mesh_shape(pred_num_samples,3)
        true_verts, true_verts_mask, true_faces, true_faces_mask = \
        compute_mesh_shape(true_num_samples,3)
        sampler = MeshSampler(true_num_samples)
        return MeshSampler.metric_evaluation(pred_verts, pred_verts_mask, pred_faces, pred_faces_mask,
                                         true_verts, true_verts_mask, true_faces, true_faces_mask)

    def build_losses(self, outputs, labels, aux_losses=None):
        """Build Mesh RCNN losses."""
        return

    def initialize(self, model: tf.keras.Model):
        """Loading pretrained checkpoint."""
        return

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
        return

    def validation_step(self, inputs, model, metrics=None):
        """Validatation step.
        Args:
        inputs: a dictionary of input tensors.
        model: the keras.Model.
        metrics: a nested structure of metrics objects.
        Returns:
        A dictionary of logs.
        """
        return

    def aggregate_logs(self, state=None, step_outputs=None):
        """Get Metric Results."""
        return

    def reduce_aggregated_logs(self, aggregated_logs, global_step=None):
        """Reduce logs and remove unneeded items. Update with results."""
        return

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
        opt_factory = optimization.MeshOptimizerFactory(optimizer_config)
        # pylint: disable=protected-access
        ema = opt_factory._use_ema
        opt_factory._use_ema = False

        opt_type = opt_factory._optimizer_type
        if opt_type == 'sgd_torch':
            optimizer = opt_factory.build_optimizer(opt_factory.build_learning_rate())
            optimizer.set_bias_lr(
                opt_factory.get_bias_lr_schedule(self._task_config.smart_bias_lr))
            optimizer.search_and_set_variable_groups(self._model.trainable_variables)
        optimizer = opt_factory.build_optimizer(opt_factory.build_learning_rate())
        opt_factory._use_ema = ema

        if ema:
            logging.info('EMA is enabled.')
            optimizer = ema_optimizer.ExponentialMovingAverage(optimizer, **self._ema_config.as_dict())
            #optimizer = opt_factory.add_ema(optimizer)

        # pylint: enable=protected-access

        if runtime_config and runtime_config.loss_scale:
            use_float16 = runtime_config.mixed_precision_dtype == 'float16'
            optimizer = performance.configure_optimizer(
                optimizer,
                use_float16=use_float16,
                loss_scale=runtime_config.loss_scale)

        return optimizer