# Copyright 2021 The TensorFlow Authors. All Rights Reserved.
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

"""Yolo Dataset Testing functions"""
from official.vision.beta.projects.yolo.common import registry_imports  # pylint: disable=unused-import
from official.vision.beta.projects.yolo.tasks import image_classification as imc
from official.vision.beta.projects.yolo.configs import darknet_classification as dcfg

import os
import tensorflow as tf
from official.core import train_utils
from official.core import task_factory
from absl.testing import parameterized

PATH_TO_COCO = '/media/vbanna/DATA_SHARE/CV/datasets/COCO_raw/records/'

def test_yolo_input_task(scaled_pipeline = True, batch_size = 1):
  if not scaled_pipeline:
    experiment = "yolo_darknet"
    config_path = [
      "official/vision/beta/projects/yolo/configs/experiments/yolov4/detection/yolov4_512_tpu.yaml"]
  else:
    experiment = "large_yolo"
    # config_path = [
    #   "official/vision/beta/projects/yolo/configs/experiments/scaled-yolo/detection/yolo_l_p6_1280_tpu.yaml"]
    config_path = [
      "official/vision/beta/projects/yolo/configs/experiments/scaled-yolo/detection/yolo_l_p7_1536_tpu.yaml"]

  config = train_utils.ParseConfigOptions(experiment=experiment, 
                                          config_file=config_path)
  params = train_utils.parse_configuration(config)
  config = params.task
  task = task_factory.get_task(params.task)

  config.train_data.global_batch_size = batch_size
  config.validation_data.global_batch_size = 1
  config.train_data.dtype = 'float32'
  config.validation_data.dtype = 'float32'
  config.validation_data.shuffle_buffer_size = 1
  config.train_data.shuffle_buffer_size = 1
  config.train_data.input_path = os.path.join(PATH_TO_COCO, 'train*')
  config.validation_data.input_path = os.path.join(PATH_TO_COCO, 'val*')

  with tf.device('/CPU:0'):
    train_data = task.build_inputs(config.train_data)
    test_data = task.build_inputs(config.validation_data)
  return train_data, test_data, config

def test_yolo_pipeline_visually(is_training=True, num=30):
  # visualize the datapipeline
  import matplotlib.pyplot as plt
  dataset, testing, _ = test_yolo_input_task()

  data = dataset if is_training else testing
  data = data.take(num)
  for l, (image, label) in enumerate(data):
    image = tf.image.draw_bounding_boxes(
        image, label['bbox'], [[1.0, 0.0, 1.0]])

    gt = label['true_conf']

    obj3 = tf.clip_by_value(gt['3'][..., 0], 0.0, 1.0)
    obj4 = tf.clip_by_value(gt['4'][..., 0], 0.0, 1.0)
    obj5 = tf.clip_by_value(gt['5'][..., 0], 0.0, 1.0)
    obj6 = tf.clip_by_value(gt['6'][..., 0], 0.0, 1.0)
    obj7 = tf.clip_by_value(gt['7'][..., 0], 0.0, 1.0)

    for shind in range(1):
      fig, axe = plt.subplots(2, 4)

      image = image[shind]

      axe[0, 0].imshow(image)
      axe[0, 1].imshow(obj3[shind, ..., :3].numpy())
      axe[0, 2].imshow(obj4[shind, ..., :3].numpy())
      axe[0, 3].imshow(obj5[shind, ..., :3].numpy())
      axe[1, 0].imshow(obj6[shind, ..., :3].numpy())
      axe[1, 2].imshow(obj7[shind, ..., :3].numpy())
      axe[1, 1].imshow(obj6[shind, ..., 3].numpy())
      axe[1, 3].imshow(obj7[shind, ..., 3].numpy())

      fig.set_size_inches(18.5, 6.5, forward=True)
      plt.tight_layout()
      plt.show()

class YoloDetectionInputTest(tf.test.TestCase, parameterized.TestCase):

  @parameterized.named_parameters(('scaled', True), ('darknet', False))
  def test_yolo_input(self, scaled_pipeline):
    # builds a pipline forom the config and tests the datapipline shapes
    # dataset, _, params = test_yolo_input_task(
    #     scaled_pipeline=scaled_pipeline, 
    #     batch_size=1)
    _, dataset, params = test_yolo_input_task(
        scaled_pipeline=scaled_pipeline, 
        batch_size=1)

    dataset = dataset.take(100)

    for image, label in dataset:
      self.assertAllEqual(image.shape, ([1] + params.model.input_size))
      self.assertTrue(
          tf.reduce_all(tf.math.logical_and(image >= 0, image <= 1)))


if __name__ == '__main__':
  # tf.test.main()
  test_yolo_pipeline_visually(is_training=True, num=20)