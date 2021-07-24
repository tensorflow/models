# Copyright 2021 The TensorFlow Authors. All Rights Reserved.
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

import tensorflow as tf
from absl.testing import parameterized

from official.vision.beta.projects.centernet.dataloaders.centernet_input import \
  CenterNetParser
from official.vision.beta.projects.centernet.ops import gt_builder
from official.vision.beta.projects.centernet.ops import preprocess_ops


class CenterNetInputTest(tf.test.TestCase, parameterized.TestCase):
  def check_labels_correct(self, boxes, classes, output_size, input_size):
    parser = CenterNetParser(dtype=tf.bfloat16)
    num_detections = len(boxes)
    boxes = tf.constant(boxes, dtype=tf.float32)
    classes = tf.constant(classes, dtype=tf.float32)
    
    boxes = preprocess_ops.pad_max_instances(boxes, 128, 0)
    classes = preprocess_ops.pad_max_instances(classes, 128, -1)
    
    labels = gt_builder.build_heatmap_and_regressed_features(
        labels={
            'bbox': boxes,
            'num_detections': num_detections,
            'classes': classes
        },
        output_size=output_size,
        input_size=input_size)
    
    ct_heatmaps = labels['ct_heatmaps']
    ct_offset = labels['ct_offset']
    size = labels['size']
    box_mask = labels['box_mask']
    box_indices = labels['box_indices']
    
    boxes = tf.cast(boxes, tf.float32)
    classes = tf.cast(classes, tf.float32)
    height_ratio = output_size[0] / input_size[0]
    width_ratio = output_size[1] / input_size[1]
    
    # Shape checks
    self.assertEqual(ct_heatmaps.shape, (output_size[0], output_size[1], 90))
    
    self.assertEqual(ct_offset.shape, (parser._max_num_instances, 2))
    
    self.assertEqual(size.shape, (parser._max_num_instances, 2))
    self.assertEqual(box_mask.shape, (parser._max_num_instances))
    self.assertEqual(box_indices.shape, (parser._max_num_instances, 2))
    
    self.assertAllInRange(ct_heatmaps, 0, 1)
    
    for i in range(len(boxes)):
      # Check sizes
      self.assertAllEqual(size[i],
                          [(boxes[i][3] - boxes[i][1]) * width_ratio,
                           (boxes[i][2] - boxes[i][0]) * height_ratio])
      
      # Check box indices
      y = tf.math.floor((boxes[i][0] + boxes[i][2]) / 2 * height_ratio)
      x = tf.math.floor((boxes[i][1] + boxes[i][3]) / 2 * width_ratio)
      self.assertAllEqual(box_indices[i], [y, x])
      
      # check offsets
      true_y = (boxes[i][0] + boxes[i][2]) / 2 * height_ratio
      true_x = (boxes[i][1] + boxes[i][3]) / 2 * width_ratio
      self.assertAllEqual(ct_offset[i], [true_x - x, true_y - y])
    
    for i in range(len(boxes), parser._max_num_instances):
      # Make sure rest are zero
      self.assertAllEqual(size[i], [0, 0])
      self.assertAllEqual(box_indices[i], [0, 0])
      self.assertAllEqual(ct_offset[i], [0, 0])
    
    # Check mask indices
    self.assertAllEqual(tf.cast(box_mask[3:], tf.int32),
                        tf.repeat(0, repeats=parser._max_num_instances - 3))
    self.assertAllEqual(tf.cast(box_mask[:3], tf.int32),
                        tf.repeat(1, repeats=3))
  
  def test_generate_heatmap_no_scale(self):
    boxes = [
        (10, 300, 15, 370),
        (100, 300, 150, 370),
        (15, 100, 200, 170),
    ]
    classes = (1, 2, 3)
    sizes = [512, 512]
    
    self.check_labels_correct(boxes=boxes,
                              classes=classes,
                              output_size=sizes,
                              input_size=sizes)
  
  def test_generate_heatmap_scale_1(self):
    boxes = [
        (10, 300, 15, 370),
        (100, 300, 150, 370),
        (15, 100, 200, 170),
    ]
    classes = (1, 2, 3)
    output_size = [128, 128]
    input_size = [512, 512]
    
    self.check_labels_correct(boxes=boxes,
                              classes=classes,
                              output_size=output_size,
                              input_size=input_size)
  
  def test_generate_heatmap_scale_2(self):
    boxes = [
        (10, 300, 15, 370),
        (100, 300, 150, 370),
        (15, 100, 200, 170),
    ]
    classes = (1, 2, 3)
    output_size = [128, 128]
    input_size = [1024, 1024]
    
    self.check_labels_correct(boxes=boxes,
                              classes=classes,
                              output_size=output_size,
                              input_size=input_size)


if __name__ == '__main__':
  tf.test.main()
