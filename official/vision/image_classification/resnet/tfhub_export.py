# Copyright 2018 The TensorFlow Authors. All Rights Reserved.
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
"""A script to export TF-Hub SavedModel."""

from __future__ import absolute_import
from __future__ import division
# from __future__ import google_type_annotations
from __future__ import print_function

import os

from absl import app
from absl import flags

import tensorflow as tf

from official.vision.image_classification.resnet import imagenet_preprocessing
from official.vision.image_classification.resnet import resnet_model

FLAGS = flags.FLAGS

flags.DEFINE_string("model_path", None,
                    "File path to TF model checkpoint or H5 file.")
flags.DEFINE_string("export_path", None,
                    "TF-Hub SavedModel destination path to export.")


def export_tfhub(model_path, hub_destination):
  """Restores a tf.keras.Model and saves for TF-Hub."""
  model = resnet_model.resnet50(
      num_classes=imagenet_preprocessing.NUM_CLASSES, rescale_inputs=True)
  model.load_weights(model_path)
  model.save(
      os.path.join(hub_destination, "classification"), include_optimizer=False)

  # Extracts a sub-model to use pooling feature vector as model output.
  image_input = model.get_layer(index=0).get_output_at(0)
  feature_vector_output = model.get_layer(name="reduce_mean").get_output_at(0)
  hub_model = tf.keras.Model(image_input, feature_vector_output)

  # Exports a SavedModel.
  hub_model.save(
      os.path.join(hub_destination, "feature-vector"), include_optimizer=False)


def main(argv):
  if len(argv) > 1:
    raise app.UsageError("Too many command-line arguments.")

  export_tfhub(FLAGS.model_path, FLAGS.export_path)


if __name__ == "__main__":
  app.run(main)
