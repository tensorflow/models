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
# Lint as: python3
"""A utility for PRADO model to do train, eval, inference and model export."""

import json
import os
from typing import Any, Mapping, Optional, Sequence, Tuple, Dict

from absl import logging
import tensorflow.compat.v1 as tf

from tensorflow.core.framework import types_pb2 as tf_types
from tensorflow.python.tools import optimize_for_inference_lib  # pylint: disable=g-direct-tensorflow-import
from prado import input_fn_reader # import sequence_projection module
from prado import metric_functions # import sequence_projection module
from prado import prado_model as model # import sequence_projection module

tf.disable_v2_behavior()


FLAGS = tf.flags.FLAGS

tf.flags.DEFINE_string("config_path", None, "Path to a RunnerConfig.")
tf.flags.DEFINE_enum("runner_mode", None,
                     ["train", "train_and_eval", "eval", "export"],
                     "Runner mode.")
tf.flags.DEFINE_string("master", None, "TensorFlow master URL.")
tf.flags.DEFINE_string(
    "output_dir", None,
    "The output directory where the model checkpoints will be written.")
tf.flags.DEFINE_bool("use_tpu", False, "Whether to use TPU or GPU/CPU.")
tf.flags.DEFINE_integer(
    "num_tpu_cores", 8,
    "Only used if `use_tpu` is True. Total number of TPU cores to use.")


def load_runner_config() -> Dict[str, Any]:
  with tf.gfile.GFile(FLAGS.config_path, "r") as f:
    return json.loads(f.read())


def create_model(
    model_config: Dict[str, Any], projection: tf.Tensor, seq_length: tf.Tensor,
    mode: tf.estimator.ModeKeys, label_ids: tf.Tensor
) -> Tuple[tf.Tensor, tf.Tensor, Mapping[str, tf.Tensor]]:
  """Creates a sequence labeling model."""
  outputs = model.create_encoder(model_config, projection, seq_length, mode)
  with tf.variable_scope("loss"):
    loss = None
    per_example_loss = None
    if mode != tf.estimator.ModeKeys.PREDICT:
      if not model_config["multilabel"]:
        per_example_loss = tf.nn.sparse_softmax_cross_entropy_with_logits(
            labels=label_ids, logits=outputs["logits"])
      else:
        per_label_loss = tf.nn.sigmoid_cross_entropy_with_logits(
            labels=label_ids, logits=outputs["logits"])
        per_example_loss = tf.reduce_mean(per_label_loss, axis=1)
      loss = tf.reduce_mean(per_example_loss)
      loss += tf.add_n(tf.compat.v1.losses.get_regularization_losses())

  return (loss, per_example_loss, outputs)


def create_optimizer(loss: tf.Tensor, runner_config: Dict[str,
                                                          Any]) -> tf.Operation:
  """Returns a train_op using Adam optimizer."""
  learning_rate = tf.train.exponential_decay(
      learning_rate=runner_config["learning_rate"],
      global_step=tf.train.get_global_step(),
      decay_steps=runner_config["learning_rate_decay_steps"],
      decay_rate=runner_config["learning_rate_decay_rate"],
      staircase=True)
  optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate)
  if FLAGS.use_tpu:
    optimizer = tf.tpu.CrossShardOptimizer(optimizer)
  else:
    tf.compat.v1.summary.scalar("learning_rate", learning_rate)

  tvars = tf.trainable_variables()
  grads = tf.gradients(loss, tvars)
  train_op = optimizer.apply_gradients(
      zip(grads, tvars), global_step=tf.train.get_global_step())
  return train_op


def model_fn_builder(runner_config: Dict[str, Any]):
  """Returns `model_fn` closure for TPUEstimator."""

  def model_fn(
      features: Mapping[str, tf.Tensor],
      mode: tf.estimator.ModeKeys,
      params: Optional[Mapping[str, Any]]  # pylint: disable=unused-argument
  ) -> tf.compat.v1.estimator.tpu.TPUEstimatorSpec:
    """The `model_fn` for TPUEstimator."""

    projection = features["projection"]
    seq_length = features["seq_length"]
    label_ids = None
    if mode != tf.estimator.ModeKeys.PREDICT:
      label_ids = features["label"]

    (total_loss, per_example_loss,
     model_outputs) = create_model(runner_config["model_config"], projection,
                                   seq_length, mode, label_ids)

    if mode == tf.estimator.ModeKeys.TRAIN:
      train_op = create_optimizer(total_loss, runner_config)
      return tf.compat.v1.estimator.tpu.TPUEstimatorSpec(
          mode=mode, loss=total_loss, train_op=train_op)

    if mode == tf.estimator.ModeKeys.EVAL:

      if not runner_config["model_config"]["multilabel"]:
        metric_fn = metric_functions.classification_metric
      else:
        metric_fn = metric_functions.labeling_metric

      eval_metrics = (metric_fn,
                      [per_example_loss, label_ids, model_outputs["logits"]])
      return tf.compat.v1.estimator.tpu.TPUEstimatorSpec(
          mode=mode, loss=total_loss, eval_metrics=eval_metrics)

    # Prediction mode
    return tf.compat.v1.estimator.tpu.TPUEstimatorSpec(
        mode=mode, predictions=model_outputs)

  return model_fn


def set_output_types_and_quantized(graph_def, quantize):
  """Set _output_types and _output_quantized for custom ops."""
  for node in graph_def.node:
    if node.op == "SequenceStringProjection":
      node.attr["_output_quantized"].b = quantize
      node.attr["_output_types"].list.type[:] = [tf_types.DT_FLOAT]
      node.op = "SEQUENCE_STRING_PROJECTION"
    elif node.op == "SequenceStringProjectionV2":
      node.attr["_output_quantized"].b = quantize
      node.attr["_output_types"].list.type[:] = [tf_types.DT_FLOAT]
      node.op = "SEQUENCE_STRING_PROJECTION_V2"


def export_frozen_graph_def(
    session: tf.compat.v1.Session, model_config: Dict[str, Any],
    input_tensors: Sequence[tf.Tensor],
    output_tensors: Sequence[tf.Tensor]) -> tf.compat.v1.GraphDef:
  """Returns a GraphDef object holding a processed network ready for exporting.

  Args:
    session: Active TensorFlow session containing the variables.
    model_config: `ModelConfig` of the exported model.
    input_tensors: A list of input tensors.
    output_tensors: A list of output tensors.

  Returns:
    A frozen GraphDef object holding a processed network ready for exporting.
  """
  graph_def = session.graph_def

  input_node_names = [tensor.op.name for tensor in input_tensors]
  output_node_names = [tensor.op.name for tensor in output_tensors]
  input_node_types = [tensor.dtype.as_datatype_enum for tensor in input_tensors]

  graph_def = tf.compat.v1.graph_util.convert_variables_to_constants(
      session, graph_def, output_node_names)
  set_output_types_and_quantized(
      graph_def, quantize=model_config["quantize"])

  # Optimize the graph for inference by removing unused nodes. Also removes
  # nodes related to training, which are not going to be used for inference.
  graph_def = optimize_for_inference_lib.optimize_for_inference(
      graph_def, input_node_names, output_node_names, input_node_types)

  return graph_def


def convert_frozen_graph_def_to_tflite(
    graph_def: tf.compat.v1.GraphDef, model_config: Dict[str, Any],
    input_tensors: Sequence[tf.Tensor],
    output_tensors: Sequence[tf.Tensor]) -> bytes:
  """Converts a TensorFlow GraphDef into a serialized TFLite Flatbuffer."""
  converter = tf.lite.TFLiteConverter(graph_def, input_tensors, output_tensors)
  if model_config["quantize"]:
    converter.inference_type = tf.uint8
    converter.inference_input_type = tf.uint8
    converter.default_ranges_stats = (0., 1.)
    converter.quantized_input_stats = {
        tensor.op.name: (0., 1.) for tensor in input_tensors
    }
  # Custom ops 'PoolingOp' and 'SequenceStringProjection' are used.
  converter.allow_custom_ops = True
  converter.experimental_new_converter = False
  return converter.convert()


def export_tflite_model(model_config: Dict[str, Any], saved_model_dir: str,
                        export_dir: str) -> None:
  """Exports a saved_model into a tflite format."""
  graph = tf.Graph()
  with graph.as_default():
    with tf.Session(graph=graph) as session:
      metagraph_def = tf.compat.v1.saved_model.loader.load(
          session, [tf.saved_model.tag_constants.SERVING], saved_model_dir)

      serving_signature_key = tf.saved_model.DEFAULT_SERVING_SIGNATURE_DEF_KEY
      signature_def = metagraph_def.signature_def[serving_signature_key]

      def _get_tensors(tensor_infos):
        tensor_names = [tensor_info.name for tensor_info in tensor_infos]
        # Always use reverse lexicographic order for consistency and
        # compatibility with PoD inference libraries.
        tensor_names.sort(reverse=True)
        return [graph.get_tensor_by_name(name) for name in tensor_names]

      input_tensors = _get_tensors(signature_def.inputs.values())
      output_tensors = _get_tensors(signature_def.outputs.values())

      graph_def = export_frozen_graph_def(session, model_config, input_tensors,
                                          output_tensors)
      tflite_model = convert_frozen_graph_def_to_tflite(graph_def, model_config,
                                                        input_tensors,
                                                        output_tensors)

      export_path = os.path.join(export_dir, "model.tflite")
      with tf.gfile.GFile(export_path, "wb") as handle:
        handle.write(tflite_model)
      logging.info("TFLite model written to: %s", export_path)


def main(_):
  runner_config = load_runner_config()

  if FLAGS.output_dir:
    tf.gfile.MakeDirs(FLAGS.output_dir)

  is_per_host = tf.estimator.tpu.InputPipelineConfig.PER_HOST_V2
  run_config = tf.estimator.tpu.RunConfig(
      master=FLAGS.master,
      model_dir=FLAGS.output_dir,
      save_checkpoints_steps=runner_config["save_checkpoints_steps"],
      keep_checkpoint_max=20,
      tpu_config=tf.estimator.tpu.TPUConfig(
          iterations_per_loop=runner_config["iterations_per_loop"],
          num_shards=FLAGS.num_tpu_cores,
          per_host_input_for_training=is_per_host))

  model_fn = model_fn_builder(runner_config=runner_config)

  # If TPU is not available, this will fall back to normal Estimator on CPU
  # or GPU.
  estimator = tf.estimator.tpu.TPUEstimator(
      use_tpu=FLAGS.use_tpu,
      model_fn=model_fn,
      config=run_config,
      train_batch_size=runner_config["batch_size"],
      eval_batch_size=runner_config["batch_size"],
      predict_batch_size=runner_config["batch_size"])

  if FLAGS.runner_mode == "train":
    train_input_fn = input_fn_reader.create_input_fn(
        runner_config=runner_config,
        create_projection=model.create_projection,
        mode=tf.estimator.ModeKeys.TRAIN,
        drop_remainder=True)
    estimator.train(
        input_fn=train_input_fn, max_steps=runner_config["train_steps"])

  if FLAGS.runner_mode == "eval":

    # TPU needs fixed shapes, so if the last batch is smaller, we drop it.
    eval_input_fn = input_fn_reader.create_input_fn(
        runner_config=runner_config,
        create_projection=model.create_projection,
        mode=tf.estimator.ModeKeys.EVAL,
        drop_remainder=True)

    for _ in tf.train.checkpoints_iterator(FLAGS.output_dir):
      result = estimator.evaluate(input_fn=eval_input_fn)
      for key in sorted(result):
        logging.info("  %s = %s", key, str(result[key]))

  if FLAGS.runner_mode == "export":
    logging.info("Exporting the model...")

    def serving_input_fn():
      """Input function of the exported model."""

      def _input_fn():
        text = tf.placeholder(tf.string, shape=[1], name="Input")
        projection, seq_length = model.create_projection(
            model_config=runner_config["model_config"],
            mode=tf.estimator.ModeKeys.PREDICT,
            inputs=text)
        features = {"projection": projection, "seq_length": seq_length}
        return tf.estimator.export.ServingInputReceiver(
            features=features, receiver_tensors=features)

      return _input_fn

    saved_model_dir = estimator.export_saved_model(FLAGS.output_dir,
                                                   serving_input_fn())

    export_tflite_model(runner_config["model_config"], saved_model_dir,
                        FLAGS.output_dir)


if __name__ == "__main__":
  tf.app.run()
