# Copyright 2022 The TensorFlow Authors. All Rights Reserved.
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

"""A script to export TF-Hub SavedModel."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os

from absl import app
from absl import flags

import tensorflow as tf

from official.legacy.image_classification.efficientnet import efficientnet_model

FLAGS = flags.FLAGS

flags.DEFINE_string("model_name", None, "EfficientNet model name.")
flags.DEFINE_string("model_path", None, "File path to TF model checkpoint.")
flags.DEFINE_string("export_path", None,
                    "TF-Hub SavedModel destination path to export.")


def export_tfhub(model_path, hub_destination, model_name):
  """Restores a tf.keras.Model and saves for TF-Hub."""
  model_configs = dict(efficientnet_model.MODEL_CONFIGS)
  config = model_configs[model_name]

  image_input = tf.keras.layers.Input(
      shape=(None, None, 3), name="image_input", dtype=tf.float32)
  x = image_input * 255.0
  ouputs = efficientnet_model.efficientnet(x, config)
  hub_model = tf.keras.Model(image_input, ouputs)
  ckpt = tf.train.Checkpoint(model=hub_model)
  ckpt.restore(model_path).assert_existing_objects_matched()
  hub_model.save(
      os.path.join(hub_destination, "classification"), include_optimizer=False)

  feature_vector_output = hub_model.get_layer(name="top_pool").get_output_at(0)
  hub_model2 = tf.keras.Model(image_input, feature_vector_output)
  hub_model2.save(
      os.path.join(hub_destination, "feature-vector"), include_optimizer=False)


def main(argv):
  if len(argv) > 1:
    raise app.UsageError("Too many command-line arguments.")

  export_tfhub(FLAGS.model_path, FLAGS.export_path, FLAGS.model_name)


if __name__ == "__main__":
  app.run(main)
