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

"""Panoptic MaskRCNN trainer."""

from absl import app

from official.common import flags as tfm_flags
from official.vision import train
# pylint: disable=unused-import
from official.vision.beta.projects.panoptic_maskrcnn.configs import panoptic_deeplab
from official.vision.beta.projects.panoptic_maskrcnn.configs import panoptic_maskrcnn
from official.vision.beta.projects.panoptic_maskrcnn.tasks import panoptic_deeplab as panoptic_deeplab_task
from official.vision.beta.projects.panoptic_maskrcnn.tasks import panoptic_maskrcnn as panoptic_maskrcnn_task
# pylint: enable=unused-import

if __name__ == '__main__':
  tfm_flags.define_flags()
  app.run(train.main)
