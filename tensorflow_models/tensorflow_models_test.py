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

"""Tests for tensorflow_models imports."""

import tensorflow as tf, tf_keras
import tensorflow_models as tfm


class TensorflowModelsTest(tf.test.TestCase):

  def testVisionImport(self):
    _ = tfm.vision.layers.SqueezeExcitation(
        in_filters=8, out_filters=4, se_ratio=1)
    _ = tfm.vision.configs.image_classification.Losses()

  def testNLPImport(self):
    _ = tfm.nlp.layers.TransformerEncoderBlock(
        num_attention_heads=2, inner_dim=10, inner_activation='relu')
    _ = tfm.nlp.tasks.TaggingTask(params=tfm.nlp.tasks.TaggingConfig())

  def testCommonImports(self):
    _ = tfm.hyperparams.Config()
    _ = tfm.optimization.LinearWarmup(
        after_warmup_lr_sched=0.0, warmup_steps=10, warmup_learning_rate=0.1)

  def testUpliftImports(self):
    _ = tfm.uplift.keys.TwoTowerOutputKeys.CONTROL_PREDICTIONS
    _ = tfm.uplift.types.TwoTowerNetworkOutputs(
        shared_embedding=tf.ones((10, 10)),
        control_logits=tf.ones((10, 1)),
        treatment_logits=tf.ones((10, 1)),
    )
    _ = tfm.uplift.layers.encoders.concat_features.ConcatFeatures(['feature'])
    _ = tfm.uplift.metrics.treatment_fraction.TreatmentFraction()
    _ = tfm.uplift.losses.true_logits_loss.TrueLogitsLoss(tf_keras.losses.mse)


if __name__ == '__main__':
  tf.test.main()
