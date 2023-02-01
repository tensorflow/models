# Copyright 2022 The Orbit Authors. All Rights Reserved.
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

"""Provides the `SaveCheckpointIfPreempted` action."""

from typing import Optional

import tensorflow as tf


class SaveCheckpointIfPreempted:
  """Action that saves on-demand checkpoints after a preemption."""

  def __init__(
      self,
      cluster_resolver: tf.distribute.cluster_resolver.ClusterResolver,
      checkpoint_manager: tf.train.CheckpointManager,
      checkpoint_number: Optional[tf.Variable] = None,
      keep_running_after_save: Optional[bool] = False,
  ):
    """Initializes the instance.

    Args:
      cluster_resolver: A `tf.distribute.cluster_resolver.ClusterResolver`
        object.
      checkpoint_manager: A `tf.train.CheckpointManager` object.
      checkpoint_number: A `tf.Variable` to indicate the checkpoint_number for
        checkpoint manager, usually it will be the global step.
      keep_running_after_save: Whether to keep the job running after the
        preemption on-demand checkpoint. Only set to True when in-process
        preemption recovery with tf.distribute.experimental.PreemptionWatcher is
        enabled.
    """
    self._checkpoint_number = checkpoint_number
    self._termination_config = None
    if keep_running_after_save:
      self._termination_config = tf.distribute.experimental.TerminationConfig(
          exit_fn=lambda: None
      )
    self._preemption_handler = (
        tf.distribute.experimental.PreemptionCheckpointHandler(
            cluster_resolver,
            checkpoint_manager,
            termination_config=self._termination_config,
        )
    )

  def __call__(self, _) -> None:
    self._preemption_handler._save_checkpoint_if_preempted(
        checkpoint_number=self._checkpoint_number, check_interval=False
    )
