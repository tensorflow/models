# Copyright 2020 The TensorFlow Authors All Rights Reserved.
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
"""A tool to export TFLite model."""

import importlib
import json
import os

from absl import app
from absl import flags
import tensorflow.compat.v1 as tf
import tensorflow_text as tftext
from layers import base_layers # import seq_flow_lite module
from layers import projection_layers # import seq_flow_lite module
from utils import tflite_utils # import seq_flow_lite module

FLAGS = flags.FLAGS

flags.DEFINE_string("output_dir", None, "The output or model directory.")
flags.DEFINE_enum("output", "sigmoid", ["logits", "sigmoid", "softmax"],
                  "Specification of the output tensor.")


def load_runner_config():
  config = os.path.join(FLAGS.output_dir, "runner_config.txt")
  with tf.gfile.Open(config, "r") as f:
    return json.loads(f.read())


def main(_):
  runner_config = load_runner_config()
  model_config = runner_config["model_config"]
  rel_module_path = "" # empty base dir
  model = importlib.import_module(rel_module_path + runner_config["name"])
  with tf.Graph().as_default() as graph:
    with tf.Session(graph=graph) as session:
      text = tf.placeholder(tf.string, shape=[1], name="Input")
      inputs = [text]
      if "pqrnn" in runner_config["name"]:
        prxlayer = projection_layers.ProjectionLayer(model_config,
                                                     base_layers.TFLITE)
        encoder = model.Encoder(model_config, base_layers.TFLITE)
        projection, seq_length = prxlayer(text)
        logits = encoder(projection, seq_length)
      else:
        byte_int = tftext.ByteSplitter().split(text)
        token_ids = tf.cast(byte_int, tf.int32).to_tensor()
        token_ids = tf.reshape(token_ids, [1, -1])
        token_ids += 3
        encoder = model.Encoder(model_config, base_layers.TFLITE)
        logits = encoder(token_ids, None)
      if FLAGS.output == "logits":
        outputs = [logits]
      elif FLAGS.output == "sigmoid":
        outputs = [tf.math.sigmoid(logits)]
      else:
        assert FLAGS.output == "softmax", "Unexpected output"
        outputs = [tf.nn.softmax(logits)]

      session.run(tf.global_variables_initializer())
      session.run(tf.local_variables_initializer())
      saver = tf.train.Saver()
      saver.restore(session, tf.train.latest_checkpoint(FLAGS.output_dir))
      tflite_fb = tflite_utils.generate_tflite(session, graph, inputs, outputs)
      output_file_name = os.path.join(FLAGS.output_dir, "tflite.fb")
      with tf.gfile.Open(output_file_name, "wb") as f:
        f.write(tflite_fb)


if __name__ == "__main__":
  app.run(main)
