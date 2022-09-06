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

"""Tests for Mask R-CNN variant with support for deep mask heads."""
import orbit
import tensorflow as tf

from official import vision  # pylint: disable=unused-import
from official.core import exp_factory
from official.modeling import optimization
from official.projects.deepmac_maskrcnn.tasks import deep_mask_head_rcnn


class DeepMaskHeadRCNNTaskTest(tf.test.TestCase):

  def _edit_config_for_testing(self, config):
    # modify config to suit local testing
    config.trainer.steps_per_loop = 1
    config.task.train_data.global_batch_size = 2
    config.task.model.backbone.resnet.model_id = 18
    config.task.model.decoder.fpn.num_filters = 32
    config.task.model.detection_generator.pre_nms_top_k = 500
    config.task.model.detection_head.fc_dims = 128
    if config.task.model.include_mask:
      config.task.model.mask_sampler.num_sampled_masks = 10
      config.task.model.mask_head.num_convs = 1
    config.task.model.roi_generator.num_proposals = 100
    config.task.model.roi_generator.pre_nms_top_k = 150
    config.task.model.roi_generator.test_pre_nms_top_k = 150
    config.task.model.roi_generator.test_num_proposals = 100
    config.task.model.rpn_head.num_filters = 32
    config.task.model.roi_sampler.num_sampled_rois = 200
    config.task.model.input_size = [128, 128, 3]
    config.trainer.train_steps = 2
    config.task.train_data.shuffle_buffer_size = 2
    config.task.train_data.input_path = "/readahead/200M/placer/prod/home/tensorflow-performance-data/datasets/coco/train-00000-of-00256.tfrecord"
    config.task.validation_data.global_batch_size = 2
    config.task.validation_data.input_path = "/readahead/200M/placer/prod/home/tensorflow-performance-data/datasets/coco/val-00000-of-00032.tfrecord"

  def _build_deep_mask_head_rcnn_from_config(self, config, is_training):
    task = deep_mask_head_rcnn.DeepMaskHeadRCNNTask(config.task)
    model = task.build_model()
    metrics = task.build_metrics(training=is_training)

    strategy = tf.distribute.get_strategy()

    data_config = config.task.train_data if is_training else config.task.validation_data
    dataset = orbit.utils.make_distributed_dataset(strategy, task.build_inputs,
                                                   data_config)
    iterator = iter(dataset)
    opt_factory = optimization.OptimizerFactory(config.trainer.optimizer_config)
    optimizer = opt_factory.build_optimizer(opt_factory.build_learning_rate())

    return task, iterator, model, optimizer, metrics

  def test_freeze_backbone_number_of_trainable_weights(self):
    config = exp_factory.get_exp_config("deep_mask_head_rcnn_resnetfpn_coco")
    self._edit_config_for_testing(config)

    # Get the number of weights from a regular model
    (task, iterator, model,
     optimizer, metrics) = self._build_deep_mask_head_rcnn_from_config(
         config, True)

    task.train_step(next(iterator), model, optimizer, metrics=metrics)
    default_weights = sum(len(l.trainable_weights) for l in model.layers)

    # Get the number of weights from a model with backbone frozen
    config.task.freeze_backbone = True
    (task, iterator, model,
     optimizer, metrics) = self._build_deep_mask_head_rcnn_from_config(
         config, True)

    task.train_step(next(iterator), model, optimizer, metrics=metrics)
    frozen_weights = sum(len(l.trainable_weights) for l in model.layers)

    # Make sure that the frozen model has less number of trainable weights
    self.assertLess(frozen_weights, default_weights,
                    msg="A frozen backbone should have fewer trainable weights")

  def test_freeze_backbone(self):
    config = exp_factory.get_exp_config("deep_mask_head_rcnn_resnetfpn_coco")
    self._edit_config_for_testing(config)
    config.task.freeze_backbone = True

    (task, iterator, model,
     optimizer, metrics) = self._build_deep_mask_head_rcnn_from_config(
         config, True)

    weights_before = [
        w.numpy().copy() for w in model.backbone.weights]

    self.assertNotEmpty(weights_before)
    task.train_step(next(iterator), model, optimizer, metrics=metrics)

    weights_after = [
        w.numpy().copy() for w in model.backbone.weights]

    self.assertAllClose(weights_after, weights_before,
                        msg="Backbone weights should not change when frozen")


if __name__ == "__main__":
  tf.test.main()
