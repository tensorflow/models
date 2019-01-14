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
"""Methods for running the Official Models with TensorRT.

Please note that all of these methods are in development, and subject to
rapid change.
"""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import argparse
import imghdr
import json
import os
import sys
import time

import numpy as np
import tensorflow as tf
from tensorflow.contrib.saved_model.python.saved_model import reader
import tensorflow.contrib.tensorrt as trt

from official.resnet import imagenet_preprocessing  # pylint: disable=g-bad-import-order

_GPU_MEM_FRACTION = 0.50
_WARMUP_NUM_LOOPS = 5
_LOG_FILE = "log.txt"
_LABELS_FILE = "labellist.json"
_GRAPH_FILE = "frozen_graph.pb"


################################################################################
# Prep the image input to the graph.
################################################################################
def preprocess_image(file_name, output_height=224, output_width=224,
                     num_channels=3):
  """Run standard ImageNet preprocessing on the passed image file.

  Args:
    file_name: string, path to file containing a JPEG image
    output_height: int, final height of image
    output_width: int, final width of image
    num_channels: int, depth of input image

  Returns:
    Float array representing processed image with shape
      [output_height, output_width, num_channels]

  Raises:
    ValueError: if image is not a JPEG.
  """
  if imghdr.what(file_name) != "jpeg":
    raise ValueError("At this time, only JPEG images are supported. "
                     "Please try another image.")

  image_buffer = tf.read_file(file_name)
  normalized = imagenet_preprocessing.preprocess_image(
      image_buffer=image_buffer,
      bbox=None,
      output_height=output_height,
      output_width=output_width,
      num_channels=num_channels,
      is_training=False)

  with tf.Session(config=get_gpu_config()) as sess:
    result = sess.run([normalized])

  return result[0]


def batch_from_image(file_name, batch_size, output_height=224, output_width=224,
                     num_channels=3):
  """Produce a batch of data from the passed image file.

  Args:
    file_name: string, path to file containing a JPEG image
    batch_size: int, the size of the desired batch of data
    output_height: int, final height of data
    output_width: int, final width of data
    num_channels: int, depth of input data

  Returns:
    Float array representing copies of the image with shape
      [batch_size, output_height, output_width, num_channels]
  """
  image_array = preprocess_image(
      file_name, output_height, output_width, num_channels)

  tiled_array = np.tile(image_array, [batch_size, 1, 1, 1])
  return tiled_array


def batch_from_random(batch_size, output_height=224, output_width=224,
                      num_channels=3):
  """Produce a batch of random data.

  Args:
    batch_size: int, the size of the desired batch of data
    output_height: int, final height of data
    output_width: int, final width of data
    num_channels: int, depth of output data

  Returns:
    Float array containing random numbers with shape
      [batch_size, output_height, output_width, num_channels]
  """
  shape = [batch_size, output_height, output_width, num_channels]
  # Make sure we return float32, as float64 will not get cast automatically.
  return np.random.random_sample(shape).astype(np.float32)


################################################################################
# Utils for handling Frozen Graphs.
################################################################################
def get_serving_meta_graph_def(savedmodel_dir):
  """Extract the SERVING MetaGraphDef from a SavedModel directory.

  Args:
    savedmodel_dir: the string path to the directory containing the .pb
      and variables for a SavedModel. This is equivalent to the subdirectory
      that is created under the directory specified by --export_dir when
      running an Official Model.

  Returns:
    MetaGraphDef that should be used for tag_constants.SERVING mode.

  Raises:
    ValueError: if a MetaGraphDef matching tag_constants.SERVING is not found.
  """
  # We only care about the serving graph def
  tag_set = set([tf.saved_model.tag_constants.SERVING])
  serving_graph_def = None
  saved_model = reader.read_saved_model(savedmodel_dir)
  for meta_graph_def in saved_model.meta_graphs:
    if set(meta_graph_def.meta_info_def.tags) == tag_set:
      serving_graph_def = meta_graph_def
  if not serving_graph_def:
    raise ValueError("No MetaGraphDef found for tag_constants.SERVING. "
                     "Please make sure the SavedModel includes a SERVING def.")

  return serving_graph_def


def write_graph_to_file(graph_name, graph_def, output_dir):
  """Write Frozen Graph file to disk."""
  output_path = os.path.join(output_dir, graph_name)
  with tf.gfile.GFile(output_path, "wb") as f:
    f.write(graph_def.SerializeToString())


def convert_savedmodel_to_frozen_graph(savedmodel_dir, output_dir):
  """Convert a SavedModel to a Frozen Graph.

  A SavedModel includes a `variables` directory with variable values,
  and a specification of the MetaGraph in a ProtoBuffer file. A Frozen Graph
  takes the variable values and inserts them into the graph, such that the
  SavedModel is all bundled into a single file. TensorRT and TFLite both
  leverage Frozen Graphs. Here, we provide a simple utility for converting
  a SavedModel into a frozen graph for use with these other tools.

  Args:
    savedmodel_dir: the string path to the directory containing the .pb
      and variables for a SavedModel. This is equivalent to the subdirectory
      that is created under the directory specified by --export_dir when
      running an Official Model.
    output_dir: string representing path to the output directory for saving
      the frozen graph.

  Returns:
    Frozen Graph definition for use.
  """
  meta_graph_def = get_serving_meta_graph_def(savedmodel_dir)
  signature_def = meta_graph_def.signature_def[
      tf.saved_model.signature_constants.DEFAULT_SERVING_SIGNATURE_DEF_KEY]

  outputs = [v.name for v in signature_def.outputs.itervalues()]
  output_names = [node.split(":")[0] for node in outputs]

  graph = tf.Graph()
  with tf.Session(graph=graph) as sess:
    tf.saved_model.loader.load(
        sess, meta_graph_def.meta_info_def.tags, savedmodel_dir)
    frozen_graph_def = tf.graph_util.convert_variables_to_constants(
        sess, graph.as_graph_def(), output_names)

  write_graph_to_file(_GRAPH_FILE, frozen_graph_def, output_dir)

  return frozen_graph_def


def get_frozen_graph(graph_file):
  """Read Frozen Graph file from disk."""
  with tf.gfile.FastGFile(graph_file, "rb") as f:
    graph_def = tf.GraphDef()
    graph_def.ParseFromString(f.read())
  return graph_def


def get_tftrt_name(graph_name, precision_string):
  return "tftrt_{}_{}".format(precision_string.lower(), graph_name)


def get_trt_graph(graph_name, graph_def, precision_mode, output_dir,
                  output_node, batch_size=128, workspace_size=2<<10):
  """Create and save inference graph using the TensorRT library.

  Args:
    graph_name: string, name of the graph to be used for saving.
    graph_def: GraphDef, the Frozen Graph to be converted.
    precision_mode: string, the precision that TensorRT should convert into.
      Options- FP32, FP16, INT8.
    output_dir: string, the path to where files should be written.
    output_node: string, the names of the output node that will
      be returned during inference.
    batch_size: int, the number of examples that will be predicted at a time.
    workspace_size: int, size in megabytes that can be used during conversion.

  Returns:
    GraphDef for the TensorRT inference graph.
  """
  trt_graph = trt.create_inference_graph(
      graph_def, [output_node], max_batch_size=batch_size,
      max_workspace_size_bytes=workspace_size<<20,
      precision_mode=precision_mode)

  write_graph_to_file(graph_name, trt_graph, output_dir)

  return trt_graph


def get_trt_graph_from_calib(graph_name, calib_graph_def, output_dir):
  """Convert a TensorRT graph used for calibration to an inference graph."""
  trt_graph = trt.calib_graph_to_infer_graph(calib_graph_def)
  write_graph_to_file(graph_name, trt_graph, output_dir)
  return trt_graph


################################################################################
# Run the graph in various precision modes.
################################################################################
def get_gpu_config():
  """Share GPU memory between image preprocessing and inference."""
  gpu_options = tf.GPUOptions(per_process_gpu_memory_fraction=_GPU_MEM_FRACTION)
  return tf.ConfigProto(gpu_options=gpu_options)


def get_iterator(data):
  """Wrap numpy data in a dataset."""
  dataset = tf.data.Dataset.from_tensors(data).repeat()
  return dataset.make_one_shot_iterator()


def time_graph(graph_def, data, input_node, output_node, num_loops=100):
  """Run and time the inference graph.

  This function sets up the input and outputs for inference, warms up by
  running inference for _WARMUP_NUM_LOOPS, then times inference for num_loops
  loops.

  Args:
    graph_def: GraphDef, the graph to be timed.
    data: ndarray of shape [batch_size, height, width, depth], data to be
      predicted.
    input_node: string, the label of the input node where data will enter the
      graph.
    output_node: string, the names of the output node that will
      be returned during inference.
    num_loops: int, number of batches that should run through for timing.

  Returns:
    A tuple consisting of a list of num_loops inference times, and the
    predictions that were output for the batch.
  """
  tf.logging.info("Starting execution")

  tf.reset_default_graph()
  g = tf.Graph()

  with g.as_default():
    iterator = get_iterator(data)
    return_tensors = tf.import_graph_def(
        graph_def=graph_def,
        input_map={input_node: iterator.get_next()},
        return_elements=[output_node]
    )
    # Unwrap the returned output node. For now, we assume we only
    # want the tensor with index `:0`, which is the 0th element of the
    # `.outputs` list.
    output = return_tensors[0].outputs[0]

  timings = []
  with tf.Session(graph=g, config=get_gpu_config()) as sess:
    tf.logging.info("Starting Warmup cycle")

    for _ in range(_WARMUP_NUM_LOOPS):
      sess.run([output])

    tf.logging.info("Starting timing.")

    for _ in range(num_loops):
      tstart = time.time()
      val = sess.run([output])
      timings.append(time.time() - tstart)

    tf.logging.info("Timing loop done!")

  return timings, val[0]


def log_stats(graph_name, log_buffer, timings, batch_size):
  """Write stats to the passed log_buffer.

  Args:
    graph_name: string, name of the graph to be used for reporting.
    log_buffer: filehandle, log file opened for appending.
    timings: list of floats, times produced for multiple runs that will be
      used for statistic calculation
    batch_size: int, number of examples per batch
  """
  times = np.array(timings)
  steps = len(times)
  speeds = batch_size / times
  time_mean = np.mean(times)
  time_med = np.median(times)
  time_99th = np.percentile(times, 99)
  time_99th_uncertainty = np.abs(np.percentile(times[0::2], 99) -
                                 np.percentile(times[1::2], 99))
  speed_mean = np.mean(speeds)
  speed_med = np.median(speeds)
  speed_uncertainty = np.std(speeds, ddof=1) / np.sqrt(float(steps))
  speed_jitter = 1.4826 * np.median(np.abs(speeds - np.median(speeds)))

  msg = ("\n==========================\n"
         "network: %s,\t batchsize %d, steps %d\n"
         "  fps \tmedian: %.1f, \tmean: %.1f, \tuncertainty: %.1f, \tjitter: %.1f\n"  # pylint: disable=line-too-long
         "  latency \tmedian: %.5f, \tmean: %.5f, \t99th_p: %.5f, \t99th_uncertainty: %.5f\n"  # pylint: disable=line-too-long
        ) % (graph_name, batch_size, steps,
             speed_med, speed_mean, speed_uncertainty, speed_jitter,
             time_med, time_mean, time_99th, time_99th_uncertainty)

  log_buffer.write(msg)


def time_and_log_graph(graph_name, graph_def, data, log_buffer, flags):
  timings, result = time_graph(
      graph_def, data, flags.input_node, flags.output_node, flags.num_loops)
  log_stats(graph_name, log_buffer, timings, flags.batch_size)

  return result


def run_trt_graph_for_mode(
    graph_name, graph_def, mode, data, log_buffer, flags):
  """Convert, time, and log the graph at `mode` precision using TensorRT."""
  g_name = get_tftrt_name(graph_name, mode)
  graph = get_trt_graph(
      g_name, graph_def, mode, flags.output_dir, flags.output_node,
      flags.batch_size, flags.workspace_size)
  result = time_and_log_graph(g_name, graph, data, log_buffer, flags)
  return result


################################################################################
# Parse predictions
################################################################################
def get_labels():
  """Get the set of possible labels for classification."""
  with open(_LABELS_FILE, "r") as labels_file:
    labels = json.load(labels_file)

  return labels


def top_predictions(result, n):
  """Get the top n predictions given the array of softmax results."""
  # We only care about the first example.
  probabilities = result[0]
  # Get the ids of most probable labels. Reverse order to get greatest first.
  ids = np.argsort(probabilities)[::-1]
  return ids[:n]


def get_labels_for_ids(labels, ids, ids_are_one_indexed=False):
  """Get the human-readable labels for given ids.

  Args:
    labels: dict, string-ID to label mapping from ImageNet.
    ids: list of ints, IDs to return labels for.
    ids_are_one_indexed: whether to increment passed IDs by 1 to account for
      the background category. See ArgParser `--ids_are_one_indexed`
      for details.

  Returns:
    list of category labels
  """
  return [labels[str(x + int(ids_are_one_indexed))] for x in ids]


def print_predictions(results, ids_are_one_indexed=False, preds_to_print=5):
  """Given an array of mode, graph_name, predicted_ID, print labels."""
  labels = get_labels()

  print("Predictions:")
  for mode, result in results:
    pred_ids = top_predictions(result, preds_to_print)
    pred_labels = get_labels_for_ids(labels, pred_ids, ids_are_one_indexed)
    print("Precision: ", mode, pred_labels)


################################################################################
# Run this script
################################################################################
def main(argv):
  parser = TensorRTParser()
  flags = parser.parse_args(args=argv[1:])

  # Load the data.
  if flags.image_file:
    data = batch_from_image(flags.image_file, flags.batch_size)
  else:
    data = batch_from_random(flags.batch_size)

  # Load the graph def
  if flags.frozen_graph:
    frozen_graph_def = get_frozen_graph(flags.frozen_graph)
  elif flags.savedmodel_dir:
    frozen_graph_def = convert_savedmodel_to_frozen_graph(
        flags.savedmodel_dir, flags.output_dir)
  else:
    raise ValueError(
        "Either a Frozen Graph file or a SavedModel must be provided.")

  # Get a name for saving TensorRT versions of the graph.
  graph_name = os.path.basename(flags.frozen_graph or _GRAPH_FILE)

  # Write to a single file for all tests, continuing from previous logs.
  log_buffer = open(os.path.join(flags.output_dir, _LOG_FILE), "a")

  # Run inference in all desired modes.
  results = []
  if flags.native:
    mode = "native"
    print("Running {} graph".format(mode))
    g_name = "{}_{}".format(mode, graph_name)
    result = time_and_log_graph(
        g_name, frozen_graph_def, data, log_buffer, flags)
    results.append((mode, result))

  if flags.fp32:
    mode = "FP32"
    print("Running {} graph".format(mode))
    result = run_trt_graph_for_mode(
        graph_name, frozen_graph_def, mode, data, log_buffer, flags)
    results.append((mode, result))

  if flags.fp16:
    mode = "FP16"
    print("Running {} graph".format(mode))
    result = run_trt_graph_for_mode(
        graph_name, frozen_graph_def, mode, data, log_buffer, flags)
    results.append((mode, result))

  if flags.int8:
    mode = "INT8"
    print("Running {} graph".format(mode))
    save_name = get_tftrt_name(graph_name, "INT8_calib")
    calib_graph = get_trt_graph(
        save_name, frozen_graph_def, mode, flags.output_dir, flags.output_node,
        flags.batch_size, flags.workspace_size)
    time_graph(calib_graph, data, flags.input_node, flags.output_node,
               num_loops=1)

    g_name = get_tftrt_name(graph_name, mode)
    int8_graph = get_trt_graph_from_calib(g_name, calib_graph, flags.output_dir)
    result = time_and_log_graph(g_name, int8_graph, data, log_buffer, flags)
    results.append((mode, result))

  # Print prediction results to the command line.
  print_predictions(
      results, flags.ids_are_one_indexed, flags.predictions_to_print)


class TensorRTParser(argparse.ArgumentParser):
  """Parser to contain flags for running the TensorRT timers."""

  def __init__(self):
    super(TensorRTParser, self).__init__()

    self.add_argument(
        "--frozen_graph", "-fg", default=None,
        help="[default: %(default)s] The location of a Frozen Graph "
        "protobuf file that will be used for inference. Note that either "
        "savedmodel_dir or frozen_graph should be passed in, and "
        "frozen_graph will take precedence.",
        metavar="<FG>",
    )

    self.add_argument(
        "--savedmodel_dir", "-sd", default=None,
        help="[default: %(default)s] The location of a SavedModel directory "
        "to be converted into a Frozen Graph. This is equivalent to the "
        "subdirectory that is created under the directory specified by "
        "--export_dir when running an Official Model. Note that either "
        "savedmodel_dir or frozen_graph should be passed in, and "
        "frozen_graph will take precedence.",
        metavar="<SD>",
    )

    self.add_argument(
        "--output_dir", "-od", default="/tmp",
        help="[default: %(default)s] The location where output files will "
        "be saved.",
        metavar="<OD>",
    )

    self.add_argument(
        "--output_node", "-on", default="softmax_tensor",
        help="[default: %(default)s] The names of the graph output node "
        "that should be used when retrieving results. Assumed to be a softmax.",
        metavar="<ON>",
    )

    self.add_argument(
        "--input_node", "-in", default="input_tensor",
        help="[default: %(default)s] The name of the graph input node where "
        "the float image array should be fed for prediction.",
        metavar="<ON>",
    )

    self.add_argument(
        "--batch_size", "-bs", type=int, default=128,
        help="[default: %(default)s] Batch size for inference. If an "
        "image file is passed, it will be copied batch_size times to "
        "imitate a batch.",
        metavar="<BS>"
    )

    self.add_argument(
        "--image_file", "-if", default=None,
        help="[default: %(default)s] The location of a JPEG image that will "
        "be passed in for inference. This will be copied batch_size times to "
        "imitate a batch. If not passed, random data will be used.",
        metavar="<IF>",
    )

    self.add_argument(
        "--native", action="store_true",
        help="[default: %(default)s] If set, benchmark the model "
        "with it's native precision and without TensorRT."
    )

    self.add_argument(
        "--fp32", action="store_true",
        help="[default: %(default)s] If set, benchmark the model with TensorRT "
        "using fp32 precision."
    )

    self.add_argument(
        "--fp16", action="store_true",
        help="[default: %(default)s] If set, benchmark the model with TensorRT "
        "using fp16 precision."
    )

    self.add_argument(
        "--int8", action="store_true",
        help="[default: %(default)s] If set, benchmark the model with TensorRT "
        "using int8 precision."
    )

    self.add_argument(
        "--num_loops", "-nl", type=int, default=100,
        help="[default: %(default)s] Number of inferences to time per "
        "benchmarked model.",
        metavar="<NL>"
    )

    self.add_argument(
        "--workspace_size", "-ws", type=int, default=2<<10,
        help="[default: %(default)s] Workspace size in megabytes.",
        metavar="<WS>"
    )

    self.add_argument(
        "--ids_are_one_indexed", action="store_true",
        help="[default: %(default)s] Some ResNet models include a `background` "
        "category, and others do not. If the model used includes `background` "
        "at index 0 in the output and represents all 1001 categories, "
        "this should be False. If the model used omits the `background` label "
        "and has only 1000 categories, this should be True."
    )

    self.add_argument(
        "--predictions_to_print", "-pp", type=int, default=5,
        help="[default: %(default)s] Number of predicted labels to predict.",
        metavar="<PP>"
    )


if __name__ == "__main__":
  tf.logging.set_verbosity(tf.logging.INFO)
  main(argv=sys.argv)
