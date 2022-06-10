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

"""Test that Mask RCNN is deterministic when TF determinism is enabled."""

# pylint: disable=unused-import
from absl.testing import parameterized
import orbit
import tensorflow as tf

from official.core import exp_factory
from official.modeling import optimization
from official.vision.tasks import maskrcnn


class MaskRcnnTaskTest(parameterized.TestCase, tf.test.TestCase):

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
    config.task.train_data.input_path = "coco/train-00000-of-00256.tfrecord"
    config.task.validation_data.global_batch_size = 2
    config.task.validation_data.input_path = "coco/val-00000-of-00032.tfrecord"

  def _build_and_run_model(self, config):
    task = maskrcnn.MaskRCNNTask(config.task)
    model = task.build_model()
    train_metrics = task.build_metrics(training=True)
    validation_metrics = task.build_metrics(training=False)

    strategy = tf.distribute.get_strategy()

    train_dataset = orbit.utils.make_distributed_dataset(
        strategy, task.build_inputs, config.task.train_data)
    train_iterator = iter(train_dataset)
    validation_dataset = orbit.utils.make_distributed_dataset(
        strategy, task.build_inputs, config.task.validation_data)
    validation_iterator = iter(validation_dataset)
    opt_factory = optimization.OptimizerFactory(config.trainer.optimizer_config)
    optimizer = opt_factory.build_optimizer(opt_factory.build_learning_rate())

    # Run training
    logs = task.train_step(next(train_iterator), model, optimizer,
                           metrics=train_metrics)
    for metric in train_metrics:
      logs[metric.name] = metric.result()

    # Run validation
    validation_logs = task.validation_step(next(validation_iterator), model,
                                           metrics=validation_metrics)
    for metric in validation_metrics:
      validation_logs[metric.name] = metric.result()

    return logs, validation_logs, model.weights

  @parameterized.parameters(
      "fasterrcnn_resnetfpn_coco",
      "maskrcnn_resnetfpn_coco",
      "maskrcnn_spinenet_coco",
      "cascadercnn_spinenet_coco",
  )
  def test_maskrcnn_task_train(self, test_config):
    """RetinaNet task test for training and val using toy configs."""
    config = exp_factory.get_exp_config(test_config)
    self._edit_config_for_testing(config)

    tf.keras.utils.set_random_seed(1)
    logs1, validation_logs1, weights1 = self._build_and_run_model(config)
    tf.keras.utils.set_random_seed(1)
    logs2, validation_logs2, weights2 = self._build_and_run_model(config)

    self.assertAllEqual(logs1["loss"], logs2["loss"])
    self.assertAllEqual(logs1["total_loss"], logs2["total_loss"])
    self.assertAllEqual(logs1["loss"], logs2["loss"])
    self.assertAllEqual(validation_logs1["coco_metric"][1]["detection_boxes"],
                        validation_logs2["coco_metric"][1]["detection_boxes"])
    self.assertAllEqual(validation_logs1["coco_metric"][1]["detection_scores"],
                        validation_logs2["coco_metric"][1]["detection_scores"])
    self.assertAllEqual(validation_logs1["coco_metric"][1]["detection_classes"],
                        validation_logs2["coco_metric"][1]["detection_classes"])
    for weight1, weight2 in zip(weights1, weights2):
      self.assertAllEqual(weight1, weight2)


if __name__ == "__main__":
  tf.config.experimental.enable_op_determinism()
  tf.test.main()
