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
"""All necessary imports for registration."""

# pylint: disable=unused-import
from official.nlp import tasks as nlp_task
from official.utils.testing import mock_task
from official.vision import beta

from official.vision.beta.projects import yolo
from official.vision.beta.projects.yolo.modeling.backbones import Darknet
from official.vision.beta.projects.yolo.configs import darknet_classification

from official.vision.beta.projects.yolo.configs.darknet_classification import image_classification
from official.vision.beta.projects.yolo.configs.darknet_classification import ImageClassificationTask

from official.vision.beta.projects.yolo.tasks.image_classification import ImageClassificationTask

# task_factory.register_task_cls(ImageClassificationTask)(ImageClassificationTask)
# print(task_factory._REGISTERED_TASK_CLS)