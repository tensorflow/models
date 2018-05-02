# Copyright 2017 Google Inc. All Rights Reserved.
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
# ==============================================================================
"""Tests for dragnn.python.trainer_lib."""


from tensorflow.python.framework import test_util
from tensorflow.python.platform import googletest

from dragnn.python import trainer_lib


class TrainerLibTest(test_util.TensorFlowTestCase):

  def testImmutabilityOfArguments(self):
    """Tests that training schedule generation does not change its arguments."""
    pretrain_steps = [1, 2, 3]
    train_steps = [5, 5, 5]
    trainer_lib.generate_target_per_step_schedule(pretrain_steps, train_steps)
    self.assertEqual(pretrain_steps, [1, 2, 3])
    self.assertEqual(train_steps, [5, 5, 5])

  def testTrainingScheduleGenerationAndDeterminism(self):
    """Non-trivial schedule, check generation and determinism."""
    pretrain_steps = [1, 2, 3]
    train_steps = [5, 5, 5]
    generated_schedule = trainer_lib.generate_target_per_step_schedule(
        pretrain_steps, train_steps)
    expected_schedule = [
        0, 1, 1, 2, 2, 2, 1, 0, 2, 1, 0, 0, 0, 0, 1, 1, 1, 2, 2, 2, 2
    ]
    self.assertEqual(generated_schedule, expected_schedule)

  def testNoPretrainSteps(self):
    """Edge case, 1 target, no pretrain."""
    generated_schedule = trainer_lib.generate_target_per_step_schedule([0],
                                                                       [10])
    expected_schedule = [0] * 10
    self.assertEqual(generated_schedule, expected_schedule)

  def testNoTrainSteps(self):
    """Edge case, 1 target, only pretrain."""
    generated_schedule = trainer_lib.generate_target_per_step_schedule([10],
                                                                       [0])
    expected_schedule = [0] * 10
    self.assertEqual(generated_schedule, expected_schedule)


if __name__ == '__main__':
  googletest.main()
