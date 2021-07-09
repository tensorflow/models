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

"""Tests for panoptic_maskrcnn_input.py."""
from absl.testing import parameterized
import tensorflow as tf

from official.core import input_reader
from official.vision.beta.dataloaders import tf_example_decoder
from official.vision.beta.projects.panoptic_maskrcnn.configs \
    import panoptic_maskrcnn
from official.vision.beta.projects.panoptic_maskrcnn.dataloaders\
    import panoptic_maskrcnn_input

class PanopticMaskRCNNInputTest(tf.test.TestCase, parameterized.TestCase):

  @parameterized.parameters(
      (True, True),
      (True, False),
      (False, False)
    )
  def test_panoptic_maskrcnn_input(self, is_training, aug_rand_hflip):
    params = panoptic_maskrcnn.DataConfig(
        is_training=is_training,
        global_batch_size=4,
        input_path='coco/train*')

    batch_size = params.global_batch_size
    output_size = [640, 640]

    parser = panoptic_maskrcnn_input.Parser(
        output_size=output_size,
        min_level=2,
        max_level=6,
        num_scales=1,
        aspect_ratios=[0.5, 1.0, 2.0],
        anchor_size=8.0,
        rpn_match_threshold=0.5,
        rpn_unmatched_threshold=0.3,
        rpn_batch_size_per_im=256,
        rpn_fg_fraction=0.5,
        aug_rand_hflip=aug_rand_hflip,
        aug_scale_min=0.8,
        aug_scale_max=1.25,
        skip_crowd_during_training=True,
        max_num_instances=100,
        mask_crop_size=112,
        resize_eval_segmentation_groundtruth=True,
        segmentation_groundtruth_padded_size=None,
        segmentation_ignore_label=255,
        dtype='float32')
  
    decoder = tf_example_decoder.TfExampleDecoder(
        include_mask=True,
        regenerate_source_id=False,
        mask_binarize_threshold=None)
  
    reader = input_reader.InputReader(
        params=params,
        dataset_fn=tf.data.TFRecordDataset,
        decoder_fn=decoder.decode,
        parser_fn=parser.parse_fn(
            is_training=is_training))

    dataset = reader.read(input_context=None).take(1)
    image, labels = tf.data.Dataset.get_single_element(dataset)

    if is_training:
      segmentation_mask = labels['gt_segmentation_mask']
      segmentation_valid_mask = labels['gt_segmentation_valid_mask']
    else:
      segmentation_mask = labels['groundtruths']['gt_segmentation_mask']
      segmentation_valid_mask = \
          labels['groundtruths']['gt_segmentation_valid_mask']
    
    self.assertAllEqual(image.shape, (batch_size, *output_size, 3))
    self.assertAllEqual(segmentation_mask.shape, (batch_size, *output_size, 1))
    self.assertAllEqual(
        segmentation_valid_mask.shape,
        (batch_size, *output_size, 1))

if __name__ == '__main__':
  tf.test.main()
