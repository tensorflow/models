# Copyright 2021 The Orbit Authors. All Rights Reserved.
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

"""Tests for orbit.actions.conditional_action."""

from orbit import actions

import tensorflow as tf


class ConditionalActionTest(tf.test.TestCase):

  def test_conditional_action(self):
    # Define a function to raise an AssertionError, since we can't in a lambda.
    def raise_assertion(arg):
      raise AssertionError(str(arg))

    conditional_action = actions.ConditionalAction(
        condition=lambda x: x['value'], action=raise_assertion)

    conditional_action({'value': False})  # Nothing is raised.
    with self.assertRaises(AssertionError) as ctx:
      conditional_action({'value': True})
      self.assertEqual(ctx.exception.message, "{'value': True}")


if __name__ == '__main__':
  tf.test.main()
