# Copyright 2017 Google Inc. All Rights Reserved.
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

r"""Runs a DRAGNN model on a given set of CoNLL-formatted sentences.

Sample invocation:
  bazel run -c opt <...>:dragnn_eval -- \
    --master_spec="/path/to/master-spec" \
    --resource_dir="/path/to/resources/"
    --checkpoint_file="/path/to/model/name.checkpoint" \
    --input_file="/path/to/input/documents/test.connlu"
"""

import os
import re
import time

import tensorflow as tf

from google.protobuf import text_format
from tensorflow.python.client import timeline
from tensorflow.python.platform import gfile

from dragnn.protos import spec_pb2
from dragnn.python import evaluation
from dragnn.python import graph_builder
from dragnn.python import sentence_io
from dragnn.python import spec_builder
from syntaxnet import sentence_pb2
from syntaxnet.ops import gen_parser_ops
from syntaxnet.util import check

flags = tf.app.flags
FLAGS = flags.FLAGS

flags.DEFINE_string('master_spec', '',
                    'Path to text file containing a DRAGNN master spec to run.')
flags.DEFINE_string('resource_dir', '',
                    'Optional base directory for resources in the master spec.')
flags.DEFINE_bool('complete_master_spec', False, 'Whether the master_spec '
                  'needs the lexicon and other resources added to it.')
flags.DEFINE_string('checkpoint_file', '', 'Path to trained model checkpoint.')
flags.DEFINE_string('input_file', '',
                    'File of CoNLL-formatted sentences to read from.')
flags.DEFINE_string('output_file', '',
                    'File path to write annotated sentences to.')
flags.DEFINE_integer('max_batch_size', 2048, 'Maximum batch size to support.')
flags.DEFINE_string('inference_beam_size', '', 'Comma separated list of '
                    'component_name=beam_size pairs.')
flags.DEFINE_string('locally_normalize', '', 'Comma separated list of '
                    'component names to do local normalization on.')
flags.DEFINE_integer('threads', 10, 'Number of threads used for intra- and '
                     'inter-op parallelism.')
flags.DEFINE_string('timeline_output_file', '', 'Path to save timeline to. '
                    'If specified, the final iteration of the evaluation loop '
                    'will capture and save a TensorFlow timeline.')


def main(unused_argv):

  # Parse the flags containint lists, using regular expressions.
  # This matches and extracts key=value pairs.
  component_beam_sizes = re.findall(r'([^=,]+)=(\d+)',
                                    FLAGS.inference_beam_size)
  # This matches strings separated by a comma. Does not return any empty
  # strings.
  components_to_locally_normalize = re.findall(r'[^,]+',
                                               FLAGS.locally_normalize)

  # Reads master spec.
  master_spec = spec_pb2.MasterSpec()
  with gfile.FastGFile(FLAGS.master_spec) as fin:
    text_format.Parse(fin.read(), master_spec)

  # Rewrite resource locations.
  if FLAGS.resource_dir:
    for component in master_spec.component:
      for resource in component.resource:
        for part in resource.part:
          part.file_pattern = os.path.join(FLAGS.resource_dir,
                                           part.file_pattern)

  if FLAGS.complete_master_spec:
    spec_builder.complete_master_spec(master_spec, None, FLAGS.resource_dir)

  # Graph building.
  tf.logging.info('Building the graph')
  g = tf.Graph()
  with g.as_default(), tf.device('/device:CPU:0'):
    hyperparam_config = spec_pb2.GridPoint()
    hyperparam_config.use_moving_average = True
    builder = graph_builder.MasterBuilder(master_spec, hyperparam_config)
    annotator = builder.add_annotation()
    builder.add_saver()

  tf.logging.info('Reading documents...')
  input_corpus = sentence_io.ConllSentenceReader(FLAGS.input_file).corpus()
  with tf.Session(graph=tf.Graph()) as tmp_session:
    char_input = gen_parser_ops.char_token_generator(input_corpus)
    char_corpus = tmp_session.run(char_input)
  check.Eq(len(input_corpus), len(char_corpus))

  session_config = tf.ConfigProto(
      log_device_placement=False,
      intra_op_parallelism_threads=FLAGS.threads,
      inter_op_parallelism_threads=FLAGS.threads)

  with tf.Session(graph=g, config=session_config) as sess:
    tf.logging.info('Initializing variables...')
    sess.run(tf.global_variables_initializer())

    tf.logging.info('Loading from checkpoint...')
    sess.run('save/restore_all', {'save/Const:0': FLAGS.checkpoint_file})

    tf.logging.info('Processing sentences...')

    processed = []
    start_time = time.time()
    run_metadata = tf.RunMetadata()
    for start in range(0, len(char_corpus), FLAGS.max_batch_size):
      end = min(start + FLAGS.max_batch_size, len(char_corpus))
      feed_dict = {annotator['input_batch']: char_corpus[start:end]}
      for comp, beam_size in component_beam_sizes:
        feed_dict['%s/InferenceBeamSize:0' % comp] = beam_size
      for comp in components_to_locally_normalize:
        feed_dict['%s/LocallyNormalize:0' % comp] = True
      if FLAGS.timeline_output_file and end == len(char_corpus):
        serialized_annotations = sess.run(
            annotator['annotations'], feed_dict=feed_dict,
            options=tf.RunOptions(trace_level=tf.RunOptions.FULL_TRACE),
            run_metadata=run_metadata)
        trace = timeline.Timeline(step_stats=run_metadata.step_stats)
        with open(FLAGS.timeline_output_file, 'w') as trace_file:
          trace_file.write(trace.generate_chrome_trace_format())
      else:
        serialized_annotations = sess.run(
            annotator['annotations'], feed_dict=feed_dict)
      processed.extend(serialized_annotations)

    tf.logging.info('Processed %d documents in %.2f seconds.',
                    len(char_corpus), time.time() - start_time)
    evaluation.calculate_segmentation_metrics(input_corpus, processed)

    if FLAGS.output_file:
      with gfile.GFile(FLAGS.output_file, 'w') as f:
        for serialized_sentence in processed:
          sentence = sentence_pb2.Sentence()
          sentence.ParseFromString(serialized_sentence)
          f.write(text_format.MessageToString(sentence) + '\n\n')


if __name__ == '__main__':
  tf.app.run()
