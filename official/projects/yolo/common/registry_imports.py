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

"""All necessary imports for registration."""

# pylint: disable=unused-import
# pylint: disable=g-bad-import-order
from official.vision import registry_imports

# import configs
from official.projects.yolo.configs import darknet_classification
from official.projects.yolo.configs import yolo as yolo_config

# import modeling components
from official.projects.yolo.modeling.backbones import darknet
from official.projects.yolo.modeling.decoders import yolo_decoder

# import tasks
from official.projects.yolo.tasks import image_classification
from official.projects.yolo.tasks import yolo as yolo_task

# import optimization packages
from official.projects.yolo.optimization import optimizer_factory
from official.projects.yolo.optimization.configs import optimizer_config
from official.projects.yolo.optimization.configs import optimization_config
