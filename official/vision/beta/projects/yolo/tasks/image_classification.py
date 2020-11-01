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
"""Image classification task definition."""
import tensorflow as tf
from official.core import base_task
from official.core import input_reader
from official.core import task_factory
from official.modeling import tf_utils
from official.vision.beta.projects.yolo.configs import darknet_classification as exp_cfg
from official.vision.beta.projects.yolo.dataloaders import classification_input as cli
from official.vision.beta.dataloaders import classification_input
from official.vision.beta.modeling import factory
from official.vision.beta.tasks import image_classification


@task_factory.register_task_cls(exp_cfg.ImageClassificationTask)
class ImageClassificationTask(image_classification.ImageClassificationTask):
  """A task for image classification."""
  def build_inputs(self, params, input_context=None):
    """Builds classification input."""

    num_classes = self.task_config.model.num_classes
    input_size = self.task_config.model.input_size

    if params.tfds_name != None: 
      decoder = cli.Decoder()
    else:
      decoder = classification_input.Decoder()
      
    parser = classification_input.Parser(
        output_size=input_size[:2],
        num_classes=num_classes,
        dtype=params.dtype)

    reader = input_reader.InputReader(
        params,
        dataset_fn=tf.data.TFRecordDataset,
        decoder_fn=decoder.decode,
        parser_fn=parser.parse_fn(params.is_training))

    dataset = reader.read(input_context=input_context)
    return dataset


