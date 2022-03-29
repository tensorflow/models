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
"""Mesh R-CNN configuration definition."""

import dataclasses

from official.modeling import hyperparams  # type: ignore


@dataclasses.dataclass
class ZHead(hyperparams.Config):
    """Parameterization for the Mesh R-CNN Z Head."""
    num_fc: int = 2
    fc_dim: int = 1024
    cls_agnostic: bool = False
    num_classes: int = 9