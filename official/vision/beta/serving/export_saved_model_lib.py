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
r"""Vision models export utility function for serving/inference."""

import os

import tensorflow as tf

from official.core import train_utils
from official.vision.beta import configs
from official.vision.beta.serving import detection
from official.vision.beta.serving import image_classification
from official.vision.beta.serving import semantic_segmentation


def export_inference_graph(input_type, batch_size, input_image_size, params,
                           checkpoint_path, export_dir,
                           export_checkpoint_subdir=None,
                           export_saved_model_subdir=None):
  """Exports inference graph for the model specified in the exp config.

  Saved model is stored at export_dir/saved_model, checkpoint is saved
  at export_dir/checkpoint, and params is saved at export_dir/params.yaml.

  Args:
    input_type: One of `image_tensor`, `image_bytes`, `tf_example`.
    batch_size: 'int', or None.
    input_image_size: List or Tuple of height and width.
    params: Experiment params.
    checkpoint_path: Trained checkpoint path or directory.
    export_dir: Export directory path.
    export_checkpoint_subdir: Optional subdirectory under export_dir
      to store checkpoint.
    export_saved_model_subdir: Optional subdirectory under export_dir
      to store saved model.
  """

  if export_checkpoint_subdir:
    output_checkpoint_directory = os.path.join(
        export_dir, export_checkpoint_subdir)
  else:
    output_checkpoint_directory = export_dir

  if export_saved_model_subdir:
    output_saved_model_directory = os.path.join(
        export_dir, export_saved_model_subdir)
  else:
    output_saved_model_directory = export_dir

  if isinstance(params.task,
                configs.image_classification.ImageClassificationTask):
    export_module = image_classification.ClassificationModule(
        params=params, batch_size=batch_size, input_image_size=input_image_size)
  elif isinstance(params.task, configs.retinanet.RetinaNetTask) or isinstance(
      params.task, configs.maskrcnn.MaskRCNNTask):
    export_module = detection.DetectionModule(
        params=params, batch_size=batch_size, input_image_size=input_image_size)
  elif isinstance(params.task,
                  configs.semantic_segmentation.SemanticSegmentationTask):
    export_module = semantic_segmentation.SegmentationModule(
        params=params, batch_size=batch_size, input_image_size=input_image_size)
  else:
    raise ValueError('Export module not implemented for {} task.'.format(
        type(params.task)))

  model = export_module.build_model()

  ckpt = tf.train.Checkpoint(model=model)

  ckpt_dir_or_file = checkpoint_path
  if tf.io.gfile.isdir(ckpt_dir_or_file):
    ckpt_dir_or_file = tf.train.latest_checkpoint(ckpt_dir_or_file)
  status = ckpt.restore(ckpt_dir_or_file).expect_partial()

  if input_type == 'image_tensor':
    input_signature = tf.TensorSpec(
        shape=[batch_size, None, None, 3],
        dtype=tf.uint8)
    signatures = {
        'serving_default':
            export_module.inference_from_image_tensors.get_concrete_function(
                input_signature)
    }
  elif input_type == 'image_bytes':
    input_signature = tf.TensorSpec(shape=[batch_size], dtype=tf.string)
    signatures = {
        'serving_default':
            export_module.inference_from_image_bytes.get_concrete_function(
                input_signature)
    }
  elif input_type == 'tf_example':
    input_signature = tf.TensorSpec(shape=[batch_size], dtype=tf.string)
    signatures = {
        'serving_default':
            export_module.inference_from_tf_example.get_concrete_function(
                input_signature)
    }
  else:
    raise ValueError('Unrecognized `input_type`')

  status.assert_existing_objects_matched()

  ckpt.save(os.path.join(output_checkpoint_directory, 'ckpt'))

  tf.saved_model.save(export_module,
                      output_saved_model_directory,
                      signatures=signatures)

  train_utils.serialize_config(params, export_dir)
