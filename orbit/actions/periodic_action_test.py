# Copyright 2025 The Orbit Authors. All Rights Reserved.
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

"""Tests for orbit.actions.periodic_action."""

from orbit import actions
from orbit.utils import common
import tensorflow as tf


class PeriodicActionTest(tf.test.TestCase):

  def test_periodic_execution(self):
    global_step = common.create_global_step()
    call_count = 0

    def action(_):
      nonlocal call_count
      call_count += 1

    # Execute action every 10 steps.
    periodic_action = actions.PeriodicAction(
        action=action, interval=10, global_step=global_step)

    # Step 5: Should not execute.
    global_step.assign(5)
    periodic_action({})
    self.assertEqual(call_count, 0)

    # Step 10: Should execute.
    global_step.assign(10)
    periodic_action({})
    self.assertEqual(call_count, 1)

    # Step 15: Should not execute.
    global_step.assign(15)
    periodic_action({})
    self.assertEqual(call_count, 1)

    # Step 20: Should execute again.
    global_step.assign(20)
    periodic_action({})
    self.assertEqual(call_count, 2)


if __name__ == '__main__':
  tf.test.main()