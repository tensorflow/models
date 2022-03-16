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

"""TensorFlow Model Garden Labse training driver, register labse configs."""

# pylint: disable=unused-import
from absl import app

from official.common import flags as tfm_flags
from official.nlp import tasks
from official.nlp import train
from official.projects.labse import config_labse

if __name__ == '__main__':
  tfm_flags.define_flags()
  app.run(train.main)
