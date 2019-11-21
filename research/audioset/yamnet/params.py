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

"""Hyperparameters for YAMNet."""

# The following hyperparameters (except PATCH_HOP_SECONDS) were used to train YAMNet,
# so expect some variability in performance if you change these. The patch hop can
# be changed arbitrarily: a smaller hop should give you more patches from the same
# clip and possibly better performance at a larger computational cost.
SAMPLE_RATE = 16000
STFT_WINDOW_SECONDS = 0.025
STFT_HOP_SECONDS = 0.010
MEL_BANDS = 64
MEL_MIN_HZ = 125
MEL_MAX_HZ = 7500
LOG_OFFSET = 0.001
PATCH_WINDOW_SECONDS = 0.96
PATCH_HOP_SECONDS = 0.48

PATCH_FRAMES = int(round(PATCH_WINDOW_SECONDS / STFT_HOP_SECONDS))
PATCH_BANDS = MEL_BANDS
NUM_CLASSES = 521
CONV_PADDING = 'same'
BATCHNORM_CENTER = True
BATCHNORM_SCALE = False
BATCHNORM_EPSILON = 1e-4
CLASSIFIER_ACTIVATION = 'sigmoid'

FEATURES_LAYER_NAME = 'features'
EXAMPLE_PREDICTIONS_LAYER_NAME = 'predictions'
