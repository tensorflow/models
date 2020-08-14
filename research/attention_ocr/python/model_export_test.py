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
"""Tests for model_export."""
import os

import numpy as np
from absl.testing import flagsaver
import tensorflow as tf

import common_flags
import model_export

_CHECKPOINT = 'model.ckpt-399731'
_CHECKPOINT_URL = (
    'http://download.tensorflow.org/models/attention_ocr_2017_08_09.tar.gz')


def _clean_up():
  tf.io.gfile.rmtree(tf.compat.v1.test.get_temp_dir())


def _create_tf_example_string(image):
  """Create a serialized tf.Example proto for feeding the model."""
  example = tf.train.Example()
  example.features.feature['image/encoded'].float_list.value.extend(
      list(np.reshape(image, (-1))))
  return example.SerializeToString()


class AttentionOcrExportTest(tf.test.TestCase):
  """Tests for model_export.export_model."""

  def setUp(self):
    for suffix in ['.meta', '.index', '.data-00000-of-00001']:
      filename = _CHECKPOINT + suffix
      self.assertTrue(
          tf.io.gfile.exists(filename),
          msg='Missing checkpoint file %s. '
          'Please download and extract it from %s' %
          (filename, _CHECKPOINT_URL))
    tf.flags.FLAGS.dataset_name = 'fsns'
    tf.flags.FLAGS.checkpoint = _CHECKPOINT
    tf.flags.FLAGS.dataset_dir = os.path.join(
        os.path.dirname(__file__), 'datasets/testdata/fsns')
    tf.test.TestCase.setUp(self)
    _clean_up()
    self.export_dir = os.path.join(
        tf.compat.v1.test.get_temp_dir(), 'exported_model')
    self.minimal_output_signature = {
        'predictions': 'AttentionOcr_v1/predicted_chars:0',
        'scores': 'AttentionOcr_v1/predicted_scores:0',
        'predicted_length': 'AttentionOcr_v1/predicted_length:0',
        'predicted_text': 'AttentionOcr_v1/predicted_text:0',
        'predicted_conf': 'AttentionOcr_v1/predicted_conf:0',
        'normalized_seq_conf': 'AttentionOcr_v1/normalized_seq_conf:0'
    }

  def create_input_feed(self, graph_def, serving):
    """Returns the input feed for the model.

    Creates random images, according to the size specified by dataset_name,
    format it in the correct way depending on whether the model was exported
    for serving, and return the correctly keyed feed_dict for inference.

    Args:
      graph_def: Graph definition of the loaded model.
      serving: Whether the model was exported for Serving.

    Returns:
      The feed_dict suitable for model inference.
    """
    # Creates a dataset based on FLAGS.dataset_name.
    self.dataset = common_flags.create_dataset('test')
    # Create some random images to test inference for any dataset.
    self.images = {
        'img1':
            np.random.uniform(low=64, high=192,
                              size=self.dataset.image_shape).astype('uint8'),
        'img2':
            np.random.uniform(low=32, high=224,
                              size=self.dataset.image_shape).astype('uint8'),
    }
    signature_def = graph_def.signature_def[
        tf.saved_model.DEFAULT_SERVING_SIGNATURE_DEF_KEY]
    if serving:
      input_name = signature_def.inputs[
          tf.saved_model.CLASSIFY_INPUTS].name
      # Model for serving takes input: inputs['inputs'] = 'tf_example:0'
      feed_dict = {
          input_name: [
              _create_tf_example_string(self.images['img1']),
              _create_tf_example_string(self.images['img2'])
          ]
      }
    else:
      input_name = signature_def.inputs['images'].name
      # Model for direct use takes input: inputs['images'] = 'original_image:0'
      feed_dict = {
          input_name: np.stack([self.images['img1'], self.images['img2']])
      }
    return feed_dict

  def verify_export_load_and_inference(self, export_for_serving=False):
    """Verify exported model can be loaded and inference can run successfully.

    This function will load the exported model in self.export_dir, then create
    some fake images according to the specification of FLAGS.dataset_name.
    It then feeds the input through the model, and verify the minimal set of
    output signatures are present.
    Note: Model and dataset creation in the underlying library depends on the
          following commandline flags:
            FLAGS.dataset_name
    Args:
      export_for_serving: True if the model was exported for Serving. This
        affects how input is fed into the model.
    """
    tf.compat.v1.reset_default_graph()
    sess = tf.compat.v1.Session()
    graph_def = tf.compat.v1.saved_model.loader.load(
        sess=sess,
        tags=[tf.saved_model.SERVING],
        export_dir=self.export_dir)
    feed_dict = self.create_input_feed(graph_def, export_for_serving)
    results = sess.run(self.minimal_output_signature, feed_dict=feed_dict)

    out_shape = (2,)
    self.assertEqual(np.shape(results['predicted_conf']), out_shape)
    self.assertEqual(np.shape(results['predicted_text']), out_shape)
    self.assertEqual(np.shape(results['predicted_length']), out_shape)
    self.assertEqual(np.shape(results['normalized_seq_conf']), out_shape)
    out_shape = (2, self.dataset.max_sequence_length)
    self.assertEqual(np.shape(results['scores']), out_shape)
    self.assertEqual(np.shape(results['predictions']), out_shape)

  @flagsaver.flagsaver
  def test_fsns_export_for_serving_and_load_inference(self):
    model_export.export_model(self.export_dir, True)
    self.verify_export_load_and_inference(True)

  @flagsaver.flagsaver
  def test_fsns_export_and_load_inference(self):
    model_export.export_model(self.export_dir, False, batch_size=2)
    self.verify_export_load_and_inference(False)


if __name__ == '__main__':
  tf.test.main()
