# Lint as: python3
# Copyright 2020 The TensorFlow Authors. All Rights Reserved.
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
"""Tests for the DELF model."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from absl.testing import parameterized
import tensorflow as tf

from delf.python.training.model import delf_model


class DelfTest(tf.test.TestCase, parameterized.TestCase):

  @parameterized.named_parameters(
      ('block3_stridesTrue', True),
      ('block3_stridesFalse', False),
  )
  def test_build_model(self, block3_strides):
    image_size = 321
    num_classes = 1000
    batch_size = 2
    input_shape = (batch_size, image_size, image_size, 3)

    model = delf_model.Delf(block3_strides=block3_strides, name='DELF')
    model.init_classifiers(num_classes)

    images = tf.random.uniform(input_shape, minval=-1.0, maxval=1.0, seed=0)
    blocks = {}

    # Get global feature by pooling block4 features.
    desc_prelogits = model.backbone(
        images, intermediates_dict=blocks, training=False)
    desc_logits = model.desc_classification(desc_prelogits)
    self.assertAllEqual(desc_prelogits.shape, (batch_size, 2048))
    self.assertAllEqual(desc_logits.shape, (batch_size, num_classes))

    features = blocks['block3']
    attn_prelogits, _, _ = model.attention(features)
    attn_logits = model.attn_classification(attn_prelogits)
    self.assertAllEqual(attn_prelogits.shape, (batch_size, 1024))
    self.assertAllEqual(attn_logits.shape, (batch_size, num_classes))

  @parameterized.named_parameters(
      ('block3_stridesTrue', True),
      ('block3_stridesFalse', False),
  )
  def test_train_step(self, block3_strides):

    image_size = 321
    num_classes = 1000
    batch_size = 2
    clip_val = 10.0
    input_shape = (batch_size, image_size, image_size, 3)

    model = delf_model.Delf(block3_strides=block3_strides, name='DELF')
    model.init_classifiers(num_classes)

    optimizer = tf.keras.optimizers.SGD(learning_rate=0.001, momentum=0.9)

    images = tf.random.uniform(input_shape, minval=0.0, maxval=1.0, seed=0)
    labels = tf.random.uniform((batch_size,),
                               minval=0,
                               maxval=model.num_classes - 1,
                               dtype=tf.int64)

    loss_object = tf.keras.losses.SparseCategoricalCrossentropy(
        from_logits=True, reduction=tf.keras.losses.Reduction.NONE)

    def compute_loss(labels, predictions):
      per_example_loss = loss_object(labels, predictions)
      return tf.nn.compute_average_loss(
          per_example_loss, global_batch_size=batch_size)

    with tf.GradientTape() as gradient_tape:
      (desc_prelogits, attn_prelogits, _,
       _) = model.global_and_local_forward_pass(images)
      # Calculate global loss by applying the descriptor classifier.
      desc_logits = model.desc_classification(desc_prelogits)
      desc_loss = compute_loss(labels, desc_logits)
      # Calculate attention loss by applying the attention block classifier.
      attn_logits = model.attn_classification(attn_prelogits)
      attn_loss = compute_loss(labels, attn_logits)
      # Cumulate global loss and attention loss and backpropagate through the
      # descriptor layer and attention layer together.
      total_loss = desc_loss + attn_loss
    gradients = gradient_tape.gradient(total_loss, model.trainable_weights)
    clipped, _ = tf.clip_by_global_norm(gradients, clip_norm=clip_val)
    optimizer.apply_gradients(zip(clipped, model.trainable_weights))


if __name__ == '__main__':
  tf.test.main()
