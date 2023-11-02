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

"""Custom summary manager utilities."""
import os
from typing import Any, Callable, Dict, Optional

import orbit
import tensorflow as tf, tf_keras
from official.core import config_definitions


class ImageScalarSummaryManager(orbit.utils.SummaryManager):
  """Class of custom summary manager that creates scalar and image summary."""

  def __init__(
      self,
      summary_dir: str,
      scalar_summary_fn: Callable[..., Any],
      image_summary_fn: Optional[Callable[..., Any]],
      max_outputs: int = 20,
      global_step=None,
  ):
    """Initializes the `ImageScalarSummaryManager` instance."""
    self._enabled = summary_dir is not None
    self._summary_dir = summary_dir
    self._scalar_summary_fn = scalar_summary_fn
    self._image_summary_fn = image_summary_fn
    self._summary_writers = {}
    self._max_outputs = max_outputs

    if global_step is None:
      self._global_step = tf.summary.experimental.get_step()
    else:
      self._global_step = global_step

  def _write_summaries(
      self, summary_dict: Dict[str, Any], relative_path: str = ''
  ):
    for name, value in summary_dict.items():
      if isinstance(value, dict):
        self._write_summaries(
            value, relative_path=os.path.join(relative_path, name)
        )
      else:
        with self.summary_writer(relative_path).as_default():
          if name.startswith('image/'):
            self._image_summary_fn(
                name, value, self._global_step, max_outputs=self._max_outputs
            )
          else:
            self._scalar_summary_fn(name, value, self._global_step)


def maybe_build_eval_summary_manager(
    params: config_definitions.ExperimentConfig, model_dir: str
) -> Optional[orbit.utils.SummaryManager]:
  """Maybe creates a SummaryManager."""

  if (
      hasattr(params.task, 'allow_image_summary')
      and params.task.allow_image_summary
  ):
    eval_summary_dir = os.path.join(
        model_dir, params.trainer.validation_summary_subdir
    )

    return ImageScalarSummaryManager(
        eval_summary_dir,
        scalar_summary_fn=tf.summary.scalar,
        image_summary_fn=tf.summary.image,
    )
  return None
