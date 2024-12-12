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

"""Tests for factory.py."""

from absl.testing import parameterized
import tensorflow as tf, tf_keras


from official.projects.pointpillars.configs import pointpillars as cfg
from official.projects.pointpillars.modeling import factory
from official.projects.pointpillars.modeling import models


class PointPillarsBuilderTest(parameterized.TestCase, tf.test.TestCase):

  @parameterized.parameters(
      (4, 4),
      (1, 2),
      (2, 1),
  )
  def test_builder(self, train_batch_size, eval_batch_size):
    model_config = cfg.PointPillarsModel()
    model_config.anchors = [cfg.Anchor(length=1.0, width=1.0)]
    pillars_config = model_config.pillars
    input_specs = {
        'pillars':
            tf_keras.layers.InputSpec(
                shape=(None, pillars_config.num_pillars,
                       pillars_config.num_points_per_pillar,
                       pillars_config.num_features_per_point)),
        'indices':
            tf_keras.layers.InputSpec(
                shape=(None, pillars_config.num_pillars, 2), dtype='int32'),
    }
    model = factory.build_pointpillars(
        input_specs, model_config, train_batch_size, eval_batch_size
    )
    config = model.get_config()
    new_model = models.PointPillarsModel.from_config(config)
    _ = new_model.to_json()
    self.assertAllEqual(model.get_config(), new_model.get_config())


if __name__ == '__main__':
  tf.test.main()
