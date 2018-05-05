# Copyright 2017 The TensorFlow Authors All Rights Reserved.
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

"""Tests for the model."""

import numpy as np
from six.moves import xrange
import string
import tensorflow as tf
from tensorflow.contrib import slim

import model
import data_provider


def create_fake_charset(num_char_classes):
  charset = {}
  for i in xrange(num_char_classes):
    charset[i] = string.printable[i % len(string.printable)]
  return charset


class ModelTest(tf.test.TestCase):
  def setUp(self):
    tf.test.TestCase.setUp(self)

    self.rng = np.random.RandomState([11, 23, 50])

    self.batch_size = 4
    self.image_width = 600
    self.image_height = 30
    self.seq_length = 40
    self.num_char_classes = 72
    self.null_code = 62
    self.num_views = 4

    feature_size = 288
    self.conv_tower_shape = (self.batch_size, 1, 72, feature_size)
    self.features_shape = (self.batch_size, self.seq_length, feature_size)
    self.chars_logit_shape = (self.batch_size, self.seq_length,
                              self.num_char_classes)
    self.length_logit_shape = (self.batch_size, self.seq_length + 1)

    self.initialize_fakes()

  def initialize_fakes(self):
    self.images_shape = (self.batch_size, self.image_height, self.image_width,
                         3)
    self.fake_images = tf.constant(
        self.rng.randint(low=0, high=255,
                         size=self.images_shape).astype('float32'),
        name='input_node')
    self.fake_conv_tower_np = self.rng.randn(
        *self.conv_tower_shape).astype('float32')
    self.fake_conv_tower = tf.constant(self.fake_conv_tower_np)
    self.fake_logits = tf.constant(
        self.rng.randn(*self.chars_logit_shape).astype('float32'))
    self.fake_labels = tf.constant(
        self.rng.randint(
            low=0,
            high=self.num_char_classes,
            size=(self.batch_size, self.seq_length)).astype('int64'))

  def create_model(self, charset=None):
    return model.Model(
        self.num_char_classes, self.seq_length, num_views=4, null_code=62,
        charset=charset)

  def test_char_related_shapes(self):
    ocr_model = self.create_model()
    with self.test_session() as sess:
      endpoints_tf = ocr_model.create_base(
          images=self.fake_images, labels_one_hot=None)

      sess.run(tf.global_variables_initializer())
      endpoints = sess.run(endpoints_tf)

      self.assertEqual((self.batch_size, self.seq_length,
                        self.num_char_classes), endpoints.chars_logit.shape)
      self.assertEqual((self.batch_size, self.seq_length,
                        self.num_char_classes), endpoints.chars_log_prob.shape)
      self.assertEqual((self.batch_size, self.seq_length),
                       endpoints.predicted_chars.shape)
      self.assertEqual((self.batch_size, self.seq_length),
                       endpoints.predicted_scores.shape)

  def test_predicted_scores_are_within_range(self):
    ocr_model = self.create_model()

    _, _, scores = ocr_model.char_predictions(self.fake_logits)
    with self.test_session() as sess:
      scores_np = sess.run(scores)

    values_in_range = (scores_np >= 0.0) & (scores_np <= 1.0)
    self.assertTrue(
        np.all(values_in_range),
        msg=('Scores contains out of the range values %s' %
             scores_np[np.logical_not(values_in_range)]))

  def test_conv_tower_shape(self):
    with self.test_session() as sess:
      ocr_model = self.create_model()
      conv_tower = ocr_model.conv_tower_fn(self.fake_images)

      sess.run(tf.global_variables_initializer())
      conv_tower_np = sess.run(conv_tower)

      self.assertEqual(self.conv_tower_shape, conv_tower_np.shape)

  def test_model_size_less_then1_gb(self):
    # NOTE: Actual amount of memory occupied my TF during training will be at
    # least 4X times bigger because of space need to store original weights,
    # updates, gradients and variances. It also depends on the type of used
    # optimizer.
    ocr_model = self.create_model()
    ocr_model.create_base(images=self.fake_images, labels_one_hot=None)
    with self.test_session() as sess:
      tfprof_root = tf.profiler.profile(
          sess.graph,
          options=tf.profiler.ProfileOptionBuilder.trainable_variables_parameter())

      model_size_bytes = 4 * tfprof_root.total_parameters
      self.assertLess(model_size_bytes, 1 * 2**30)

  def test_create_summaries_is_runnable(self):
    ocr_model = self.create_model()
    data = data_provider.InputEndpoints(
        images=self.fake_images,
        images_orig=self.fake_images,
        labels=self.fake_labels,
        labels_one_hot=slim.one_hot_encoding(self.fake_labels,
                                             self.num_char_classes))
    endpoints = ocr_model.create_base(
        images=self.fake_images, labels_one_hot=None)
    charset = create_fake_charset(self.num_char_classes)
    summaries = ocr_model.create_summaries(
        data, endpoints, charset, is_training=False)
    with self.test_session() as sess:
      sess.run(tf.global_variables_initializer())
      sess.run(tf.local_variables_initializer())
      tf.tables_initializer().run()
      sess.run(summaries)  # just check it is runnable

  def test_sequence_loss_function_without_label_smoothing(self):
    model = self.create_model()
    model.set_mparam('sequence_loss_fn', label_smoothing=0)

    loss = model.sequence_loss_fn(self.fake_logits, self.fake_labels)
    with self.test_session() as sess:
      loss_np = sess.run(loss)

    # This test checks that the loss function is 'runnable'.
    self.assertEqual(loss_np.shape, tuple())

  def encode_coordinates_alt(self, net):
    """An alternative implemenation for the encoding coordinates.

    Args:
      net: a tensor of shape=[batch_size, height, width, num_features]

    Returns:
      a list of tensors with encoded image coordinates in them.
    """
    batch_size, h, w, _ = net.shape.as_list()
    h_loc = [
      tf.tile(
          tf.reshape(
              tf.contrib.layers.one_hot_encoding(
                  tf.constant([i]), num_classes=h), [h, 1]), [1, w])
      for i in xrange(h)
    ]
    h_loc = tf.concat([tf.expand_dims(t, 2) for t in h_loc], 2)
    w_loc = [
      tf.tile(
          tf.contrib.layers.one_hot_encoding(tf.constant([i]), num_classes=w),
          [h, 1]) for i in xrange(w)
    ]
    w_loc = tf.concat([tf.expand_dims(t, 2) for t in w_loc], 2)
    loc = tf.concat([h_loc, w_loc], 2)
    loc = tf.tile(tf.expand_dims(loc, 0), [batch_size, 1, 1, 1])
    return tf.concat([net, loc], 3)

  def test_encoded_coordinates_have_correct_shape(self):
    model = self.create_model()
    model.set_mparam('encode_coordinates_fn', enabled=True)
    conv_w_coords_tf = model.encode_coordinates_fn(self.fake_conv_tower)

    with self.test_session() as sess:
      conv_w_coords = sess.run(conv_w_coords_tf)

    batch_size, height, width, feature_size = self.conv_tower_shape
    self.assertEqual(conv_w_coords.shape, (batch_size, height, width,
                                           feature_size + height + width))

  def test_disabled_coordinate_encoding_returns_features_unchanged(self):
    model = self.create_model()
    model.set_mparam('encode_coordinates_fn', enabled=False)
    conv_w_coords_tf = model.encode_coordinates_fn(self.fake_conv_tower)

    with self.test_session() as sess:
      conv_w_coords = sess.run(conv_w_coords_tf)

    self.assertAllEqual(conv_w_coords, self.fake_conv_tower_np)

  def test_coordinate_encoding_is_correct_for_simple_example(self):
    shape = (1, 2, 3, 4)  # batch_size, height, width, feature_size
    fake_conv_tower = tf.constant(2 * np.ones(shape), dtype=tf.float32)
    model = self.create_model()
    model.set_mparam('encode_coordinates_fn', enabled=True)
    conv_w_coords_tf = model.encode_coordinates_fn(fake_conv_tower)

    with self.test_session() as sess:
      conv_w_coords = sess.run(conv_w_coords_tf)

    # Original features
    self.assertAllEqual(conv_w_coords[0, :, :, :4],
                        [[[2, 2, 2, 2], [2, 2, 2, 2], [2, 2, 2, 2]],
                         [[2, 2, 2, 2], [2, 2, 2, 2], [2, 2, 2, 2]]])
    # Encoded coordinates
    self.assertAllEqual(conv_w_coords[0, :, :, 4:],
                        [[[1, 0, 1, 0, 0], [1, 0, 0, 1, 0], [1, 0, 0, 0, 1]],
                         [[0, 1, 1, 0, 0], [0, 1, 0, 1, 0], [0, 1, 0, 0, 1]]])

  def test_alt_implementation_of_coordinate_encoding_returns_same_values(self):
    model = self.create_model()
    model.set_mparam('encode_coordinates_fn', enabled=True)
    conv_w_coords_tf = model.encode_coordinates_fn(self.fake_conv_tower)
    conv_w_coords_alt_tf = self.encode_coordinates_alt(self.fake_conv_tower)

    with self.test_session() as sess:
      conv_w_coords_tf, conv_w_coords_alt_tf = sess.run(
          [conv_w_coords_tf, conv_w_coords_alt_tf])

    self.assertAllEqual(conv_w_coords_tf, conv_w_coords_alt_tf)

  def test_predicted_text_has_correct_shape_w_charset(self):
    charset = create_fake_charset(self.num_char_classes)
    ocr_model = self.create_model(charset=charset)

    with self.test_session() as sess:
      endpoints_tf = ocr_model.create_base(
          images=self.fake_images, labels_one_hot=None)

      sess.run(tf.global_variables_initializer())
      tf.tables_initializer().run()
      endpoints = sess.run(endpoints_tf)

      self.assertEqual(endpoints.predicted_text.shape, (self.batch_size,))
      self.assertEqual(len(endpoints.predicted_text[0]), self.seq_length)


class CharsetMapperTest(tf.test.TestCase):
  def test_text_corresponds_to_ids(self):
    charset = create_fake_charset(36)
    ids = tf.constant(
        [[17, 14, 21, 21, 24], [32, 24, 27, 21, 13]], dtype=tf.int64)
    charset_mapper = model.CharsetMapper(charset)

    with self.test_session() as sess:
      tf.tables_initializer().run()
      text = sess.run(charset_mapper.get_text(ids))

    self.assertAllEqual(text, ['hello', 'world'])


if __name__ == '__main__':
  tf.test.main()
