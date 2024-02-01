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

"""TensorFlow Model Garden Vision training driver, register MAE configs."""

from absl import app

from official.common import flags as tfm_flags
# pylint: disable=unused-import
from official.projects.mae.configs import linear_probe
from official.projects.mae.configs import mae
from official.projects.mae.configs import vit
from official.projects.mae.tasks import image_classification
from official.projects.mae.tasks import linear_probe as linear_probe_task
from official.projects.mae.tasks import masked_ae
# pylint: enable=unused-import
from official.vision import train

if __name__ == '__main__':
  tfm_flags.define_flags()
  app.run(train.main)
