# Copyright 2024 The TensorFlow Authors. All Rights Reserved.
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

"""Panoptic MaskRCNN trainer."""

from absl import app

from official.common import flags as tfm_flags
from official.projects.maskconver.configs import maskconver as maskconver_cfg  # pylint: disable=unused-import
from official.projects.maskconver.configs import multiscale_maskconver as multiscale_maskconver_cfg  # pylint: disable=unused-import
from official.projects.maskconver.modeling import fpn  # pylint: disable=unused-import
from official.projects.maskconver.tasks import maskconver as maskconver_task  # pylint: disable=unused-import
from official.projects.maskconver.tasks import multiscale_maskconver as multiscale_maskconver_task  # pylint: disable=unused-import
from official.vision import train


if __name__ == '__main__':
  tfm_flags.define_flags()
  app.run(train.main)
