# Lint as: python2, python3
# Copyright 2019 The TensorFlow Authors All Rights Reserved.
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
"""Tests for segmentation "streaming" metrics."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import collections



import numpy as np
import six
import tensorflow as tf

from deeplab.evaluation import streaming_metrics
from deeplab.evaluation import test_utils

# See the definition of the color names at:
#   https://en.wikipedia.org/wiki/Web_colors.
_CLASS_COLOR_MAP = {
    (0, 0, 0): 0,
    (0, 0, 255): 1,  # Person (blue).
    (255, 0, 0): 2,  # Bear (red).
    (0, 255, 0): 3,  # Tree (lime).
    (255, 0, 255): 4,  # Bird (fuchsia).
    (0, 255, 255): 5,  # Sky (aqua).
    (255, 255, 0): 6,  # Cat (yellow).
}


class StreamingPanopticQualityTest(tf.test.TestCase):

  def test_streaming_metric_on_single_image(self):
    offset = 256 * 256

    instance_class_map = {
        0: 0,
        47: 1,
        97: 1,
        133: 1,
        150: 1,
        174: 1,
        198: 2,
        215: 1,
        244: 1,
        255: 1,
    }
    gt_instances, gt_classes = test_utils.panoptic_segmentation_with_class_map(
        'team_gt_instance.png', instance_class_map)

    pred_classes = test_utils.read_segmentation_with_rgb_color_map(
        'team_pred_class.png', _CLASS_COLOR_MAP)
    pred_instances = test_utils.read_test_image(
        'team_pred_instance.png', mode='L')

    gt_class_tensor = tf.placeholder(tf.uint16)
    gt_instance_tensor = tf.placeholder(tf.uint16)
    pred_class_tensor = tf.placeholder(tf.uint16)
    pred_instance_tensor = tf.placeholder(tf.uint16)
    qualities, update_pq = streaming_metrics.streaming_panoptic_quality(
        gt_class_tensor,
        gt_instance_tensor,
        pred_class_tensor,
        pred_instance_tensor,
        num_classes=3,
        max_instances_per_category=256,
        ignored_label=0,
        offset=offset)
    pq, sq, rq, total_tp, total_fn, total_fp = tf.unstack(qualities, 6, axis=0)
    feed_dict = {
        gt_class_tensor: gt_classes,
        gt_instance_tensor: gt_instances,
        pred_class_tensor: pred_classes,
        pred_instance_tensor: pred_instances
    }

    with self.session() as sess:
      sess.run(tf.local_variables_initializer())
      sess.run(update_pq, feed_dict=feed_dict)
      (result_pq, result_sq, result_rq, result_total_tp, result_total_fn,
       result_total_fp) = sess.run([pq, sq, rq, total_tp, total_fn, total_fp],
                                   feed_dict=feed_dict)
    np.testing.assert_array_almost_equal(
        result_pq, [2.06104, 0.7024, 0.54069], decimal=4)
    np.testing.assert_array_almost_equal(
        result_sq, [2.06104, 0.7526, 0.54069], decimal=4)
    np.testing.assert_array_almost_equal(result_rq, [1., 0.9333, 1.], decimal=4)
    np.testing.assert_array_almost_equal(
        result_total_tp, [1., 7., 1.], decimal=4)
    np.testing.assert_array_almost_equal(
        result_total_fn, [0., 1., 0.], decimal=4)
    np.testing.assert_array_almost_equal(
        result_total_fp, [0., 0., 0.], decimal=4)

  def test_streaming_metric_on_multiple_images(self):
    num_classes = 7
    offset = 256 * 256

    bird_gt_instance_class_map = {
        92: 5,
        176: 3,
        255: 4,
    }
    cat_gt_instance_class_map = {
        0: 0,
        255: 6,
    }
    team_gt_instance_class_map = {
        0: 0,
        47: 1,
        97: 1,
        133: 1,
        150: 1,
        174: 1,
        198: 2,
        215: 1,
        244: 1,
        255: 1,
    }
    test_image = collections.namedtuple(
        'TestImage',
        ['gt_class_map', 'gt_path', 'pred_inst_path', 'pred_class_path'])
    test_images = [
        test_image(bird_gt_instance_class_map, 'bird_gt.png',
                   'bird_pred_instance.png', 'bird_pred_class.png'),
        test_image(cat_gt_instance_class_map, 'cat_gt.png',
                   'cat_pred_instance.png', 'cat_pred_class.png'),
        test_image(team_gt_instance_class_map, 'team_gt_instance.png',
                   'team_pred_instance.png', 'team_pred_class.png'),
    ]

    gt_classes = []
    gt_instances = []
    pred_classes = []
    pred_instances = []
    for test_image in test_images:
      (image_gt_instances,
       image_gt_classes) = test_utils.panoptic_segmentation_with_class_map(
           test_image.gt_path, test_image.gt_class_map)
      gt_classes.append(image_gt_classes)
      gt_instances.append(image_gt_instances)

      pred_classes.append(
          test_utils.read_segmentation_with_rgb_color_map(
              test_image.pred_class_path, _CLASS_COLOR_MAP))
      pred_instances.append(
          test_utils.read_test_image(test_image.pred_inst_path, mode='L'))

    gt_class_tensor = tf.placeholder(tf.uint16)
    gt_instance_tensor = tf.placeholder(tf.uint16)
    pred_class_tensor = tf.placeholder(tf.uint16)
    pred_instance_tensor = tf.placeholder(tf.uint16)
    qualities, update_pq = streaming_metrics.streaming_panoptic_quality(
        gt_class_tensor,
        gt_instance_tensor,
        pred_class_tensor,
        pred_instance_tensor,
        num_classes=num_classes,
        max_instances_per_category=256,
        ignored_label=0,
        offset=offset)
    pq, sq, rq, total_tp, total_fn, total_fp = tf.unstack(qualities, 6, axis=0)
    with self.session() as sess:
      sess.run(tf.local_variables_initializer())
      for pred_class, pred_instance, gt_class, gt_instance in six.moves.zip(
          pred_classes, pred_instances, gt_classes, gt_instances):
        sess.run(
            update_pq,
            feed_dict={
                gt_class_tensor: gt_class,
                gt_instance_tensor: gt_instance,
                pred_class_tensor: pred_class,
                pred_instance_tensor: pred_instance
            })
      (result_pq, result_sq, result_rq, result_total_tp, result_total_fn,
       result_total_fp) = sess.run(
           [pq, sq, rq, total_tp, total_fn, total_fp],
           feed_dict={
               gt_class_tensor: 0,
               gt_instance_tensor: 0,
               pred_class_tensor: 0,
               pred_instance_tensor: 0
           })
    np.testing.assert_array_almost_equal(
        result_pq,
        [4.3107, 0.7024, 0.54069, 0.745353, 0.85768, 0.99107, 0.77410],
        decimal=4)
    np.testing.assert_array_almost_equal(
        result_sq, [5.3883, 0.7526, 0.5407, 0.7454, 0.8577, 0.9911, 0.7741],
        decimal=4)
    np.testing.assert_array_almost_equal(
        result_rq, [0.8, 0.9333, 1., 1., 1., 1., 1.], decimal=4)
    np.testing.assert_array_almost_equal(
        result_total_tp, [2., 7., 1., 1., 1., 1., 1.], decimal=4)
    np.testing.assert_array_almost_equal(
        result_total_fn, [0., 1., 0., 0., 0., 0., 0.], decimal=4)
    np.testing.assert_array_almost_equal(
        result_total_fp, [1., 0., 0., 0., 0., 0., 0.], decimal=4)


class StreamingParsingCoveringTest(tf.test.TestCase):

  def test_streaming_metric_on_single_image(self):
    offset = 256 * 256

    instance_class_map = {
        0: 0,
        47: 1,
        97: 1,
        133: 1,
        150: 1,
        174: 1,
        198: 2,
        215: 1,
        244: 1,
        255: 1,
    }
    gt_instances, gt_classes = test_utils.panoptic_segmentation_with_class_map(
        'team_gt_instance.png', instance_class_map)

    pred_classes = test_utils.read_segmentation_with_rgb_color_map(
        'team_pred_class.png', _CLASS_COLOR_MAP)
    pred_instances = test_utils.read_test_image(
        'team_pred_instance.png', mode='L')

    gt_class_tensor = tf.placeholder(tf.uint16)
    gt_instance_tensor = tf.placeholder(tf.uint16)
    pred_class_tensor = tf.placeholder(tf.uint16)
    pred_instance_tensor = tf.placeholder(tf.uint16)
    coverings, update_ops = streaming_metrics.streaming_parsing_covering(
        gt_class_tensor,
        gt_instance_tensor,
        pred_class_tensor,
        pred_instance_tensor,
        num_classes=3,
        max_instances_per_category=256,
        ignored_label=0,
        offset=offset,
        normalize_by_image_size=False)
    (per_class_coverings, per_class_weighted_ious, per_class_gt_areas) = (
        tf.unstack(coverings, num=3, axis=0))
    feed_dict = {
        gt_class_tensor: gt_classes,
        gt_instance_tensor: gt_instances,
        pred_class_tensor: pred_classes,
        pred_instance_tensor: pred_instances
    }

    with self.session() as sess:
      sess.run(tf.local_variables_initializer())
      sess.run(update_ops, feed_dict=feed_dict)
      (result_per_class_coverings, result_per_class_weighted_ious,
       result_per_class_gt_areas) = (
           sess.run([
               per_class_coverings,
               per_class_weighted_ious,
               per_class_gt_areas,
           ],
                    feed_dict=feed_dict))

    np.testing.assert_array_almost_equal(
        result_per_class_coverings, [0.0, 0.7009696912, 0.5406896552],
        decimal=4)
    np.testing.assert_array_almost_equal(
        result_per_class_weighted_ious, [0.0, 39864.14634, 3136], decimal=4)
    np.testing.assert_array_equal(result_per_class_gt_areas, [0, 56870, 5800])

  def test_streaming_metric_on_multiple_images(self):
    """Tests streaming parsing covering metric."""
    num_classes = 7
    offset = 256 * 256

    bird_gt_instance_class_map = {
        92: 5,
        176: 3,
        255: 4,
    }
    cat_gt_instance_class_map = {
        0: 0,
        255: 6,
    }
    team_gt_instance_class_map = {
        0: 0,
        47: 1,
        97: 1,
        133: 1,
        150: 1,
        174: 1,
        198: 2,
        215: 1,
        244: 1,
        255: 1,
    }
    test_image = collections.namedtuple(
        'TestImage',
        ['gt_class_map', 'gt_path', 'pred_inst_path', 'pred_class_path'])
    test_images = [
        test_image(bird_gt_instance_class_map, 'bird_gt.png',
                   'bird_pred_instance.png', 'bird_pred_class.png'),
        test_image(cat_gt_instance_class_map, 'cat_gt.png',
                   'cat_pred_instance.png', 'cat_pred_class.png'),
        test_image(team_gt_instance_class_map, 'team_gt_instance.png',
                   'team_pred_instance.png', 'team_pred_class.png'),
    ]

    gt_classes = []
    gt_instances = []
    pred_classes = []
    pred_instances = []
    for test_image in test_images:
      (image_gt_instances,
       image_gt_classes) = test_utils.panoptic_segmentation_with_class_map(
           test_image.gt_path, test_image.gt_class_map)
      gt_classes.append(image_gt_classes)
      gt_instances.append(image_gt_instances)

      pred_instances.append(
          test_utils.read_test_image(test_image.pred_inst_path, mode='L'))
      pred_classes.append(
          test_utils.read_segmentation_with_rgb_color_map(
              test_image.pred_class_path, _CLASS_COLOR_MAP))

    gt_class_tensor = tf.placeholder(tf.uint16)
    gt_instance_tensor = tf.placeholder(tf.uint16)
    pred_class_tensor = tf.placeholder(tf.uint16)
    pred_instance_tensor = tf.placeholder(tf.uint16)
    coverings, update_ops = streaming_metrics.streaming_parsing_covering(
        gt_class_tensor,
        gt_instance_tensor,
        pred_class_tensor,
        pred_instance_tensor,
        num_classes=num_classes,
        max_instances_per_category=256,
        ignored_label=0,
        offset=offset,
        normalize_by_image_size=False)
    (per_class_coverings, per_class_weighted_ious, per_class_gt_areas) = (
        tf.unstack(coverings, num=3, axis=0))

    with self.session() as sess:
      sess.run(tf.local_variables_initializer())
      for pred_class, pred_instance, gt_class, gt_instance in six.moves.zip(
          pred_classes, pred_instances, gt_classes, gt_instances):
        sess.run(
            update_ops,
            feed_dict={
                gt_class_tensor: gt_class,
                gt_instance_tensor: gt_instance,
                pred_class_tensor: pred_class,
                pred_instance_tensor: pred_instance
            })
        (result_per_class_coverings, result_per_class_weighted_ious,
         result_per_class_gt_areas) = (
             sess.run(
                 [
                     per_class_coverings,
                     per_class_weighted_ious,
                     per_class_gt_areas,
                 ],
                 feed_dict={
                     gt_class_tensor: 0,
                     gt_instance_tensor: 0,
                     pred_class_tensor: 0,
                     pred_instance_tensor: 0
                 }))

    np.testing.assert_array_almost_equal(
        result_per_class_coverings, [
            0.0,
            0.7009696912,
            0.5406896552,
            0.7453531599,
            0.8576779026,
            0.9910687881,
            0.7741046032,
        ],
        decimal=4)
    np.testing.assert_array_almost_equal(
        result_per_class_weighted_ious, [
            0.0,
            39864.14634,
            3136,
            1177.657993,
            2498.41573,
            33366.31289,
            26671,
        ],
        decimal=4)
    np.testing.assert_array_equal(result_per_class_gt_areas, [
        0.0,
        56870,
        5800,
        1580,
        2913,
        33667,
        34454,
    ])

  def test_streaming_metric_on_multiple_images_normalize_by_size(self):
    """Tests streaming parsing covering metric with image size normalization."""
    num_classes = 7
    offset = 256 * 256

    bird_gt_instance_class_map = {
        92: 5,
        176: 3,
        255: 4,
    }
    cat_gt_instance_class_map = {
        0: 0,
        255: 6,
    }
    team_gt_instance_class_map = {
        0: 0,
        47: 1,
        97: 1,
        133: 1,
        150: 1,
        174: 1,
        198: 2,
        215: 1,
        244: 1,
        255: 1,
    }
    test_image = collections.namedtuple(
        'TestImage',
        ['gt_class_map', 'gt_path', 'pred_inst_path', 'pred_class_path'])
    test_images = [
        test_image(bird_gt_instance_class_map, 'bird_gt.png',
                   'bird_pred_instance.png', 'bird_pred_class.png'),
        test_image(cat_gt_instance_class_map, 'cat_gt.png',
                   'cat_pred_instance.png', 'cat_pred_class.png'),
        test_image(team_gt_instance_class_map, 'team_gt_instance.png',
                   'team_pred_instance.png', 'team_pred_class.png'),
    ]

    gt_classes = []
    gt_instances = []
    pred_classes = []
    pred_instances = []
    for test_image in test_images:
      (image_gt_instances,
       image_gt_classes) = test_utils.panoptic_segmentation_with_class_map(
           test_image.gt_path, test_image.gt_class_map)
      gt_classes.append(image_gt_classes)
      gt_instances.append(image_gt_instances)

      pred_instances.append(
          test_utils.read_test_image(test_image.pred_inst_path, mode='L'))
      pred_classes.append(
          test_utils.read_segmentation_with_rgb_color_map(
              test_image.pred_class_path, _CLASS_COLOR_MAP))

    gt_class_tensor = tf.placeholder(tf.uint16)
    gt_instance_tensor = tf.placeholder(tf.uint16)
    pred_class_tensor = tf.placeholder(tf.uint16)
    pred_instance_tensor = tf.placeholder(tf.uint16)
    coverings, update_ops = streaming_metrics.streaming_parsing_covering(
        gt_class_tensor,
        gt_instance_tensor,
        pred_class_tensor,
        pred_instance_tensor,
        num_classes=num_classes,
        max_instances_per_category=256,
        ignored_label=0,
        offset=offset,
        normalize_by_image_size=True)
    (per_class_coverings, per_class_weighted_ious, per_class_gt_areas) = (
        tf.unstack(coverings, num=3, axis=0))

    with self.session() as sess:
      sess.run(tf.local_variables_initializer())
      for pred_class, pred_instance, gt_class, gt_instance in six.moves.zip(
          pred_classes, pred_instances, gt_classes, gt_instances):
        sess.run(
            update_ops,
            feed_dict={
                gt_class_tensor: gt_class,
                gt_instance_tensor: gt_instance,
                pred_class_tensor: pred_class,
                pred_instance_tensor: pred_instance
            })
        (result_per_class_coverings, result_per_class_weighted_ious,
         result_per_class_gt_areas) = (
             sess.run(
                 [
                     per_class_coverings,
                     per_class_weighted_ious,
                     per_class_gt_areas,
                 ],
                 feed_dict={
                     gt_class_tensor: 0,
                     gt_instance_tensor: 0,
                     pred_class_tensor: 0,
                     pred_instance_tensor: 0
                 }))

    np.testing.assert_array_almost_equal(
        result_per_class_coverings, [
            0.0,
            0.7009696912,
            0.5406896552,
            0.7453531599,
            0.8576779026,
            0.9910687881,
            0.7741046032,
        ],
        decimal=4)
    np.testing.assert_array_almost_equal(
        result_per_class_weighted_ious, [
            0.0,
            0.5002088756,
            0.03935002196,
            0.03086105851,
            0.06547211033,
            0.8743792686,
            0.2549565051,
        ],
        decimal=4)
    np.testing.assert_array_almost_equal(
        result_per_class_gt_areas, [
            0.0,
            0.7135955832,
            0.07277746408,
            0.04140461216,
            0.07633647799,
            0.8822589099,
            0.3293566581,
        ],
        decimal=4)


if __name__ == '__main__':
  tf.test.main()
