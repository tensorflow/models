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
from official.common import dataset_fn
from official.core import task_factory
from official.projects.yolo.configs import darknet_classification as exp_cfg
from official.projects.yolo.dataloaders import classification_input
from official.vision.dataloaders import classification_input as classification_input_base
from official.vision.dataloaders import input_reader_factory
from official.vision.dataloaders import tfds_factory
from official.vision.tasks import image_classification


@task_factory.register_task_cls(exp_cfg.ImageClassificationTask)
class ImageClassificationTask(image_classification.ImageClassificationTask):
  """A task for image classification."""

  def build_inputs(self, params, input_context=None):
    """Builds classification input."""

    num_classes = self.task_config.model.num_classes
    input_size = self.task_config.model.input_size
    image_field_key = self.task_config.train_data.image_field_key
    label_field_key = self.task_config.train_data.label_field_key
    is_multilabel = self.task_config.train_data.is_multilabel

    if params.tfds_name:
      decoder = tfds_factory.get_classification_decoder(params.tfds_name)
    else:
      decoder = classification_input_base.Decoder(
          image_field_key=image_field_key,
          label_field_key=label_field_key,
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
