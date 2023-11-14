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

"""Metric configurations for TF Model Garden."""

from collections.abc import Mapping
import dataclasses
from typing import Any

import tensorflow_models as tfm


@dataclasses.dataclass(kw_only=True)
class SlicedMetricConfig(tfm.core.config_definitions.base_config.Config):
  """Sliced metric configuration.

  Attributes:
    slicing_feature: The feature whose values to slice the metric on. Required.
    slicing_spec: A mapping from the name of the slice to the value to slice on.
      The name will be displayed on TB. Required.
    slicing_feature_dtype: Optional dtype to cast the slicing feature and the
      values to slice on.
  """

  slicing_feature: str | None = None
  slicing_spec: Mapping[str, int] | None = None
  slicing_feature_dtype: str | None = None

  def __post_init__(
      self, default_params: dict[str, Any], restrictions: list[str]
  ):
    if not restrictions:
      restrictions = ['slicing_feature != None', 'slicing_spec != None']
    super().__post_init__(
        default_params=default_params, restrictions=restrictions
    )
