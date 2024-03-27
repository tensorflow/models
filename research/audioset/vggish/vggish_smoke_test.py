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

"""A smoke test for VGGish.

This is a simple smoke test of a local install of VGGish and its associated
downloaded files. We create a synthetic sound, extract log mel spectrogram
features, run them through VGGish, post-process the embedding outputs, and
check some simple statistics of the results, allowing for variations that
might occur due to platform/version differences in the libraries we use.

Usage:
- Download the VGGish checkpoint and PCA parameters into the same directory as
  the VGGish source code. If you keep them elsewhere, update the checkpoint_path
  and pca_params_path variables below.
- Run:
  $ python vggish_smoke_test.py
"""

from __future__ import print_function

import numpy as np
import resampy  # pylint: disable=import-error
import tensorflow.compat.v1 as tf

import vggish_input
import vggish_params
import vggish_postprocess
import vggish_slim

print('\nTesting your install of VGGish\n')

# Paths to downloaded VGGish files.
checkpoint_path = 'vggish_model.ckpt'
pca_params_path = 'vggish_pca_params.npz'

# Relative tolerance of errors in mean and standard deviation of embeddings.
rel_error = 0.1  # Up to 10%

# Generate a 1 kHz sine wave at 16 kHz, the preferred sample rate of VGGish.
num_secs = 3
freq = 1000
sr = 16000
t = np.arange(0, num_secs, 1 / sr)
x = np.sin(2 * np.pi * freq * t)

# Check that we can resample a signal. Don't use the resampled signal to
# produce an embedding where we check the results because we don't want
# to depend on the resampler never changing too much.
resampled_x = resampy.resample(x, sr, sr * 0.75)
print('Resampling via resampy works!')

# Produce a batch of log mel spectrogram examples.
input_batch = vggish_input.waveform_to_examples(x, sr)
print('Log Mel Spectrogram example: ', input_batch[0])
np.testing.assert_equal(
    input_batch.shape,
    [num_secs, vggish_params.NUM_FRAMES, vggish_params.NUM_BANDS])

# Define VGGish, load the checkpoint, and run the batch through the model to
# produce embeddings.
with tf.Graph().as_default(), tf.Session() as sess:
  vggish_slim.define_vggish_slim()
  vggish_slim.load_vggish_slim_checkpoint(sess, checkpoint_path)

  features_tensor = sess.graph.get_tensor_by_name(
      vggish_params.INPUT_TENSOR_NAME)
  embedding_tensor = sess.graph.get_tensor_by_name(
      vggish_params.OUTPUT_TENSOR_NAME)
  [embedding_batch] = sess.run([embedding_tensor],
                               feed_dict={features_tensor: input_batch})
  print('VGGish embedding: ', embedding_batch[0])
  print('embedding mean/stddev', np.mean(embedding_batch),
        np.std(embedding_batch))

# Postprocess the results to produce whitened quantized embeddings.
pproc = vggish_postprocess.Postprocessor(pca_params_path)
postprocessed_batch = pproc.postprocess(embedding_batch)
print('Postprocessed VGGish embedding: ', postprocessed_batch[0])
print('postproc embedding mean/stddev', np.mean(postprocessed_batch),
      np.std(postprocessed_batch))

# Expected mean/stddev were measured to 3 significant places on 07/25/23 with
# NumPy 1.21.6 / TF 2.8.2 (dating to Apr-May 2022)
# NumPy 1.24.3 / TF 2.13.0 (representative of July 2023)
# with Python 3.10 on a Debian-like Linux system. Both configs produced
# identical results.

expected_embedding_mean = 0.000657
expected_embedding_std = 0.343
np.testing.assert_allclose(
    [np.mean(embedding_batch), np.std(embedding_batch)],
    [expected_embedding_mean, expected_embedding_std],
    rtol=rel_error)

expected_postprocessed_mean = 126.0
expected_postprocessed_std = 89.3
np.testing.assert_allclose(
    [np.mean(postprocessed_batch), np.std(postprocessed_batch)],
    [expected_postprocessed_mean, expected_postprocessed_std],
    rtol=rel_error)

print('\nLooks Good To Me!\n')
