# Copyright 2017 The TensorFlow Authors. All Rights Reserved.
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

"""Tests for anchor_generators.multiple_grid_anchor_generator_test.py."""

import numpy as np

import tensorflow.compat.v1 as tf

from object_detection.anchor_generators import multiple_grid_anchor_generator as ag
from object_detection.utils import test_case


class MultipleGridAnchorGeneratorTest(test_case.TestCase):

  def test_construct_single_anchor_grid(self):
    """Builds a 1x1 anchor grid to test the size of the output boxes."""
    def graph_fn():

      box_specs_list = [[(.5, .25), (1.0, .25), (2.0, .25),
                         (.5, 1.0), (1.0, 1.0), (2.0, 1.0),
                         (.5, 4.0), (1.0, 4.0), (2.0, 4.0)]]
      anchor_generator = ag.MultipleGridAnchorGenerator(
          box_specs_list,
          base_anchor_size=tf.constant([256, 256], dtype=tf.float32),
          anchor_strides=[(16, 16)],
          anchor_offsets=[(7, -3)])
      anchors_list = anchor_generator.generate(feature_map_shape_list=[(1, 1)])
      return anchors_list[0].get()
    exp_anchor_corners = [[-121, -35, 135, 29], [-249, -67, 263, 61],
                          [-505, -131, 519, 125], [-57, -67, 71, 61],
                          [-121, -131, 135, 125], [-249, -259, 263, 253],
                          [-25, -131, 39, 125], [-57, -259, 71, 253],
                          [-121, -515, 135, 509]]

    anchor_corners_out = self.execute(graph_fn, [])
    self.assertAllClose(anchor_corners_out, exp_anchor_corners)

  def test_construct_anchor_grid(self):
    def graph_fn():
      box_specs_list = [[(0.5, 1.0), (1.0, 1.0), (2.0, 1.0)]]

      anchor_generator = ag.MultipleGridAnchorGenerator(
          box_specs_list,
          base_anchor_size=tf.constant([10, 10], dtype=tf.float32),
          anchor_strides=[(19, 19)],
          anchor_offsets=[(0, 0)])
      anchors_list = anchor_generator.generate(feature_map_shape_list=[(2, 2)])
      return anchors_list[0].get()
    exp_anchor_corners = [[-2.5, -2.5, 2.5, 2.5], [-5., -5., 5., 5.],
                          [-10., -10., 10., 10.], [-2.5, 16.5, 2.5, 21.5],
                          [-5., 14., 5, 24], [-10., 9., 10, 29],
                          [16.5, -2.5, 21.5, 2.5], [14., -5., 24, 5],
                          [9., -10., 29, 10], [16.5, 16.5, 21.5, 21.5],
                          [14., 14., 24, 24], [9., 9., 29, 29]]

    anchor_corners_out = self.execute(graph_fn, [])
    self.assertAllClose(anchor_corners_out, exp_anchor_corners)

  def test_construct_anchor_grid_non_square(self):

    def graph_fn():
      box_specs_list = [[(1.0, 1.0)]]
      anchor_generator = ag.MultipleGridAnchorGenerator(
          box_specs_list, base_anchor_size=tf.constant([1, 1],
                                                       dtype=tf.float32))
      anchors_list = anchor_generator.generate(feature_map_shape_list=[(
          tf.constant(1, dtype=tf.int32), tf.constant(2, dtype=tf.int32))])
      return anchors_list[0].get()

    exp_anchor_corners = [[0., -0.25, 1., 0.75], [0., 0.25, 1., 1.25]]
    anchor_corners_out = self.execute(graph_fn, [])
    self.assertAllClose(anchor_corners_out, exp_anchor_corners)

  def test_construct_dynamic_size_anchor_grid(self):

    def graph_fn(height, width):
      box_specs_list = [[(1.0, 1.0)]]
      anchor_generator = ag.MultipleGridAnchorGenerator(
          box_specs_list, base_anchor_size=tf.constant([1, 1],
                                                       dtype=tf.float32))
      anchors_list = anchor_generator.generate(feature_map_shape_list=[(height,
                                                                        width)])
      return anchors_list[0].get()

    exp_anchor_corners = [[0., -0.25, 1., 0.75], [0., 0.25, 1., 1.25]]

    anchor_corners_out = self.execute_cpu(graph_fn,
                                          [np.array(1, dtype=np.int32),
                                           np.array(2, dtype=np.int32)])
    self.assertAllClose(anchor_corners_out, exp_anchor_corners)

  def test_construct_anchor_grid_normalized(self):
    def graph_fn():
      box_specs_list = [[(1.0, 1.0)]]

      anchor_generator = ag.MultipleGridAnchorGenerator(
          box_specs_list, base_anchor_size=tf.constant([1, 1],
                                                       dtype=tf.float32))
      anchors_list = anchor_generator.generate(
          feature_map_shape_list=[(tf.constant(1, dtype=tf.int32), tf.constant(
              2, dtype=tf.int32))],
          im_height=320,
          im_width=640)
      return anchors_list[0].get()

    exp_anchor_corners = [[0., 0., 1., 0.5], [0., 0.5, 1., 1.]]
    anchor_corners_out = self.execute(graph_fn, [])
    self.assertAllClose(anchor_corners_out, exp_anchor_corners)

  def test_construct_multiple_grids(self):

    def graph_fn():
      box_specs_list = [[(1.0, 1.0), (2.0, 1.0), (1.0, 0.5)],
                        [(1.0, 1.0), (1.0, 0.5)]]

      anchor_generator = ag.MultipleGridAnchorGenerator(
          box_specs_list,
          base_anchor_size=tf.constant([1.0, 1.0], dtype=tf.float32),
          anchor_strides=[(.25, .25), (.5, .5)],
          anchor_offsets=[(.125, .125), (.25, .25)])
      anchors_list = anchor_generator.generate(feature_map_shape_list=[(4, 4), (
          2, 2)])
      return [anchors.get() for anchors in anchors_list]
    # height and width of box with .5 aspect ratio
    h = np.sqrt(2)
    w = 1.0/np.sqrt(2)
    exp_small_grid_corners = [[-.25, -.25, .75, .75],
                              [.25-.5*h, .25-.5*w, .25+.5*h, .25+.5*w],
                              [-.25, .25, .75, 1.25],
                              [.25-.5*h, .75-.5*w, .25+.5*h, .75+.5*w],
                              [.25, -.25, 1.25, .75],
                              [.75-.5*h, .25-.5*w, .75+.5*h, .25+.5*w],
                              [.25, .25, 1.25, 1.25],
                              [.75-.5*h, .75-.5*w, .75+.5*h, .75+.5*w]]
    # only test first entry of larger set of anchors
    exp_big_grid_corners = [[.125-.5, .125-.5, .125+.5, .125+.5],
                            [.125-1.0, .125-1.0, .125+1.0, .125+1.0],
                            [.125-.5*h, .125-.5*w, .125+.5*h, .125+.5*w],]

    anchor_corners_out = np.concatenate(self.execute(graph_fn, []), axis=0)
    self.assertEquals(anchor_corners_out.shape, (56, 4))
    big_grid_corners = anchor_corners_out[0:3, :]
    small_grid_corners = anchor_corners_out[48:, :]
    self.assertAllClose(small_grid_corners, exp_small_grid_corners)
    self.assertAllClose(big_grid_corners, exp_big_grid_corners)

  def test_construct_multiple_grids_with_clipping(self):

    def graph_fn():
      box_specs_list = [[(1.0, 1.0), (2.0, 1.0), (1.0, 0.5)],
                        [(1.0, 1.0), (1.0, 0.5)]]

      clip_window = tf.constant([0, 0, 1, 1], dtype=tf.float32)
      anchor_generator = ag.MultipleGridAnchorGenerator(
          box_specs_list,
          base_anchor_size=tf.constant([1.0, 1.0], dtype=tf.float32),
          clip_window=clip_window)
      anchors_list = anchor_generator.generate(feature_map_shape_list=[(4, 4), (
          2, 2)])
      return [anchors.get() for anchors in anchors_list]
    # height and width of box with .5 aspect ratio
    h = np.sqrt(2)
    w = 1.0/np.sqrt(2)
    exp_small_grid_corners = [[0, 0, .75, .75],
                              [0, 0, .25+.5*h, .25+.5*w],
                              [0, .25, .75, 1],
                              [0, .75-.5*w, .25+.5*h, 1],
                              [.25, 0, 1, .75],
                              [.75-.5*h, 0, 1, .25+.5*w],
                              [.25, .25, 1, 1],
                              [.75-.5*h, .75-.5*w, 1, 1]]

    anchor_corners_out = np.concatenate(self.execute(graph_fn, []), axis=0)
    small_grid_corners = anchor_corners_out[48:, :]
    self.assertAllClose(small_grid_corners, exp_small_grid_corners)

  def test_invalid_box_specs(self):
    # not all box specs are pairs
    box_specs_list = [[(1.0, 1.0), (2.0, 1.0), (1.0, 0.5)],
                      [(1.0, 1.0), (1.0, 0.5, .3)]]
    with self.assertRaises(ValueError):
      ag.MultipleGridAnchorGenerator(box_specs_list)

    # box_specs_list is not a list of lists
    box_specs_list = [(1.0, 1.0), (2.0, 1.0), (1.0, 0.5)]
    with self.assertRaises(ValueError):
      ag.MultipleGridAnchorGenerator(box_specs_list)

  def test_invalid_generate_arguments(self):
    box_specs_list = [[(1.0, 1.0), (2.0, 1.0), (1.0, 0.5)],
                      [(1.0, 1.0), (1.0, 0.5)]]

    # incompatible lengths with box_specs_list
    with self.assertRaises(ValueError):
      anchor_generator = ag.MultipleGridAnchorGenerator(
          box_specs_list,
          base_anchor_size=tf.constant([1.0, 1.0], dtype=tf.float32),
          anchor_strides=[(.25, .25)],
          anchor_offsets=[(.125, .125), (.25, .25)])
      anchor_generator.generate(feature_map_shape_list=[(4, 4), (2, 2)])
    with self.assertRaises(ValueError):
      anchor_generator = ag.MultipleGridAnchorGenerator(
          box_specs_list,
          base_anchor_size=tf.constant([1.0, 1.0], dtype=tf.float32),
          anchor_strides=[(.25, .25), (.5, .5)],
          anchor_offsets=[(.125, .125), (.25, .25)])
      anchor_generator.generate(feature_map_shape_list=[(4, 4), (2, 2), (1, 1)])
    with self.assertRaises(ValueError):
      anchor_generator = ag.MultipleGridAnchorGenerator(
          box_specs_list,
          base_anchor_size=tf.constant([1.0, 1.0], dtype=tf.float32),
          anchor_strides=[(.5, .5)],
          anchor_offsets=[(.25, .25)])
      anchor_generator.generate(feature_map_shape_list=[(4, 4), (2, 2)])

    # not pairs
    with self.assertRaises(ValueError):
      anchor_generator = ag.MultipleGridAnchorGenerator(
          box_specs_list,
          base_anchor_size=tf.constant([1.0, 1.0], dtype=tf.float32),
          anchor_strides=[(.25, .25), (.5, .5)],
          anchor_offsets=[(.125, .125), (.25, .25)])
      anchor_generator.generate(feature_map_shape_list=[(4, 4, 4), (2, 2)])
    with self.assertRaises(ValueError):
      anchor_generator = ag.MultipleGridAnchorGenerator(
          box_specs_list,
          base_anchor_size=tf.constant([1.0, 1.0], dtype=tf.float32),
          anchor_strides=[(.25, .25, .1), (.5, .5)],
          anchor_offsets=[(.125, .125), (.25, .25)])
      anchor_generator.generate(feature_map_shape_list=[(4, 4), (2, 2)])
    with self.assertRaises(ValueError):
      anchor_generator = ag.MultipleGridAnchorGenerator(
          box_specs_list,
          base_anchor_size=tf.constant([1.0, 1.0], dtype=tf.float32),
          anchor_strides=[(.25, .25), (.5, .5)],
          anchor_offsets=[(.125, .125), (.25, .25)])
      anchor_generator.generate(feature_map_shape_list=[(4), (2, 2)])


class CreateSSDAnchorsTest(test_case.TestCase):

  def test_create_ssd_anchors_returns_correct_shape(self):

    def graph_fn1():
      anchor_generator = ag.create_ssd_anchors(
          num_layers=6,
          min_scale=0.2,
          max_scale=0.95,
          aspect_ratios=(1.0, 2.0, 3.0, 1.0 / 2, 1.0 / 3),
          reduce_boxes_in_lowest_layer=True)

      feature_map_shape_list = [(38, 38), (19, 19), (10, 10),
                                (5, 5), (3, 3), (1, 1)]
      anchors_list = anchor_generator.generate(
          feature_map_shape_list=feature_map_shape_list)
      return [anchors.get() for anchors in anchors_list]
    anchor_corners_out = np.concatenate(self.execute(graph_fn1, []), axis=0)
    self.assertEquals(anchor_corners_out.shape, (7308, 4))

    def graph_fn2():
      anchor_generator = ag.create_ssd_anchors(
          num_layers=6, min_scale=0.2, max_scale=0.95,
          aspect_ratios=(1.0, 2.0, 3.0, 1.0/2, 1.0/3),
          reduce_boxes_in_lowest_layer=False)

      feature_map_shape_list = [(38, 38), (19, 19), (10, 10),
                                (5, 5), (3, 3), (1, 1)]
      anchors_list = anchor_generator.generate(
          feature_map_shape_list=feature_map_shape_list)
      return [anchors.get() for anchors in anchors_list]
    anchor_corners_out = np.concatenate(self.execute(graph_fn2, []), axis=0)
    self.assertEquals(anchor_corners_out.shape, (11640, 4))


if __name__ == '__main__':
  tf.test.main()
