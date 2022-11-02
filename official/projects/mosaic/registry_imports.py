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

"""All necessary imports for registration on MOSAIC project."""
# pylint: disable=unused-import
from official.projects.mosaic import mosaic_tasks
from official.projects.mosaic.configs import mosaic_config
from official.projects.mosaic.modeling import mosaic_model
from official.projects.mosaic.qat.configs import mosaic_config as mosaic_qat_config
from official.projects.mosaic.qat.tasks import mosaic_tasks as mosaic_qat_tasks
