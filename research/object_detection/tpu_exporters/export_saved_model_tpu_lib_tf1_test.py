# Copyright 2019 The TensorFlow Authors. All Rights Reserved.
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
"""Test for object detection's TPU exporter."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os
import unittest

from absl.testing import parameterized
import numpy as np
import tensorflow.compat.v1 as tf

from object_detection.tpu_exporters import export_saved_model_tpu_lib
from object_detection.utils import tf_version

flags = tf.app.flags
FLAGS = flags.FLAGS


def get_path(path_suffix):
  return os.path.join(tf.resource_loader.get_data_files_path(), 'testdata',
                      path_suffix)


@unittest.skipIf(tf_version.is_tf2(), 'Skipping TF1.X only test.')
class ExportSavedModelTPUTest(tf.test.TestCase, parameterized.TestCase):

  @parameterized.named_parameters(
      ('ssd', get_path('ssd/ssd_pipeline.config'), 'image_tensor', True, 20),
      ('faster_rcnn',
       get_path('faster_rcnn/faster_rcnn_resnet101_atrous_coco.config'),
       'image_tensor', True, 20))
  def testExportAndLoad(self,
                        pipeline_config_file,
                        input_type='image_tensor',
                        use_bfloat16=False,
                        repeat=1):

    input_placeholder_name = 'placeholder_tensor'
    export_dir = os.path.join(FLAGS.test_tmpdir, 'tpu_saved_model')
    if tf.gfile.Exists(export_dir):
      tf.gfile.DeleteRecursively(export_dir)
    ckpt_path = None
    export_saved_model_tpu_lib.export(pipeline_config_file, ckpt_path,
                                      export_dir, input_placeholder_name,
                                      input_type, use_bfloat16)

    inputs = np.random.rand(256, 256, 3)
    tensor_dict_out = export_saved_model_tpu_lib.run_inference_from_saved_model(
        inputs, export_dir, input_placeholder_name, repeat)
    for k, v in tensor_dict_out.items():
      tf.logging.info('{}: {}'.format(k, v))


if __name__ == '__main__':
  tf.test.main()
