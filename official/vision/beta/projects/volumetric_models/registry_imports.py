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

"""All necessary imports for registration."""

# pylint: disable=unused-import
from official.common import registry_imports
from official.vision.beta.projects.volumetric_models.configs import semantic_segmentation_3d as semantic_segmentation_3d_cfg
from official.vision.beta.projects.volumetric_models.modeling import backbones
from official.vision.beta.projects.volumetric_models.modeling import decoders
from official.vision.beta.projects.volumetric_models.tasks import semantic_segmentation_3d
