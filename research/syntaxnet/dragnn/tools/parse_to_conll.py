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
r"""Runs a both a segmentation and parsing model on a CoNLL dataset.
"""

import re
import time
import tensorflow as tf

from tensorflow.python.client import timeline
from tensorflow.python.platform import gfile

# The following line is necessary to load custom ops into the library.
from dragnn.python import dragnn_ops

from dragnn.python import evaluation
from dragnn.python import sentence_io
from syntaxnet import sentence_pb2

# The following line is necessary to load custom ops into the library.
from syntaxnet import syntaxnet_ops

from syntaxnet.ops import gen_parser_ops
from syntaxnet.util import check

flags = tf.app.flags
FLAGS = flags.FLAGS

flags.DEFINE_string(
    'segmenter_saved_model', None,
    'Path to segmenter saved model. If not provided, gold segmentation is used.'
)
flags.DEFINE_string('parser_saved_model', None, 'Path to parser saved model.')

flags.DEFINE_string('input_file', '',
                    'File of CoNLL-formatted sentences to read from.')
flags.DEFINE_string('output_file', '',
                    'File path to write annotated sentences to.')
flags.DEFINE_bool('text_format', False, '')

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


def get_segmenter_corpus(input_data_path, use_text_format):
  """Reads in a character corpus for segmenting."""
  # Read in the documents.
  tf.logging.info('Reading documents...')
  if use_text_format:
    char_corpus = sentence_io.FormatSentenceReader(input_data_path,
                                                   'untokenized-text').corpus()
  else:
    input_corpus = sentence_io.ConllSentenceReader(input_data_path).corpus()
    with tf.Session(graph=tf.Graph()) as tmp_session:
      char_input = gen_parser_ops.char_token_generator(input_corpus)
      char_corpus = tmp_session.run(char_input)
    check.Eq(len(input_corpus), len(char_corpus))

  return char_corpus


def run_segmenter(input_data, segmenter_model, session_config, max_batch_size,
                  timeline_output_file=None):
  """Runs the provided segmenter model on the provided character corpus.

  Args:
    input_data: Character input corpus to segment.
    segmenter_model: Path to a SavedModel file containing the segmenter graph.
    session_config: A session configuration object.
    max_batch_size: The maximum batch size to use.
    timeline_output_file: Filepath for timeline export. Does not export if None.

  Returns:
    A list of segmented sentences suitable for parsing.
  """
  # Create the session and graph, and load the SavedModel.
  g = tf.Graph()
  with tf.Session(graph=g, config=session_config) as sess:
    tf.logging.info('Initializing segmentation model...')
    tf.saved_model.loader.load(sess, [tf.saved_model.tag_constants.SERVING],
                               segmenter_model)

    # Use the graph to segment the sentences.
    tf.logging.info('Segmenting sentences...')
    processed = []
    start_time = time.time()
    run_metadata = tf.RunMetadata()
    for start in range(0, len(input_data), max_batch_size):
      # Prepare the inputs.
      end = min(start + max_batch_size, len(input_data))
      feed_dict = {
          'annotation/ComputeSession/InputBatch:0': input_data[start:end]
      }
      output_node = 'annotation/annotations:0'

      # Process.
      tf.logging.info('Processing examples %d to %d' % (start, end))
      if timeline_output_file and end == len(input_data):
        serialized_annotations = sess.run(
            output_node,
            feed_dict=feed_dict,
            options=tf.RunOptions(trace_level=tf.RunOptions.FULL_TRACE),
            run_metadata=run_metadata)
        trace = timeline.Timeline(step_stats=run_metadata.step_stats)
        with open(timeline_output_file, 'w') as trace_file:
          trace_file.write(trace.generate_chrome_trace_format())
      else:
        serialized_annotations = sess.run(output_node, feed_dict=feed_dict)

      # Save the outputs.
      processed.extend(serialized_annotations)

  # Report statistics.
  tf.logging.info('Segmented %d documents in %.2f seconds.',
                  len(input_data), time.time() - start_time)

  # Once all sentences are segmented, the processed data can be used in the
  # parsers.
  return processed


def run_parser(input_data, parser_model, session_config, beam_sizes,
               locally_normalized_components, max_batch_size,
               timeline_output_file):
  """Runs the provided segmenter model on the provided character corpus.

  Args:
    input_data: Input corpus to parse.
    parser_model: Path to a SavedModel file containing the parser graph.
    session_config: A session configuration object.
    beam_sizes: A dict of component names : beam sizes (optional).
    locally_normalized_components: A list of components to normalize (optional).
    max_batch_size: The maximum batch size to use.
    timeline_output_file: Filepath for timeline export. Does not export if None.

  Returns:
    A list of parsed sentences.
  """
  parser_graph = tf.Graph()
  with tf.Session(graph=parser_graph, config=session_config) as sess:
    tf.logging.info('Initializing parser model...')
    tf.saved_model.loader.load(sess, [tf.saved_model.tag_constants.SERVING],
                               parser_model)

    tf.logging.info('Parsing sentences...')

    processed = []
    start_time = time.time()
    run_metadata = tf.RunMetadata()
    tf.logging.info('Corpus length is %d' % len(input_data))
    for start in range(0, len(input_data), max_batch_size):
      # Set up the input and output.
      end = min(start + max_batch_size, len(input_data))
      feed_dict = {
          'annotation/ComputeSession/InputBatch:0': input_data[start:end]
      }
      for comp, beam_size in beam_sizes:
        feed_dict['%s/InferenceBeamSize:0' % comp] = beam_size
      for comp in locally_normalized_components:
        feed_dict['%s/LocallyNormalize:0' % comp] = True
      output_node = 'annotation/annotations:0'

      # Process.
      tf.logging.info('Processing examples %d to %d' % (start, end))
      if timeline_output_file and end == len(input_data):
        serialized_annotations = sess.run(
            output_node,
            feed_dict=feed_dict,
            options=tf.RunOptions(trace_level=tf.RunOptions.FULL_TRACE),
            run_metadata=run_metadata)
        trace = timeline.Timeline(step_stats=run_metadata.step_stats)
        with open(timeline_output_file, 'w') as trace_file:
          trace_file.write(trace.generate_chrome_trace_format())
      else:
        serialized_annotations = sess.run(output_node, feed_dict=feed_dict)

      processed.extend(serialized_annotations)

    tf.logging.info('Processed %d documents in %.2f seconds.',
                    len(input_data), time.time() - start_time)
    _, uas, las = evaluation.calculate_parse_metrics(input_data, processed)
    tf.logging.info('UAS: %.2f', uas)
    tf.logging.info('LAS: %.2f', las)

  return processed


def print_output(output_file, use_text_format, use_gold_segmentation, output):
  """Writes a set of sentences in CoNLL format.

  Args:
    output_file: The file to write to.
    use_text_format: Whether this computation used text-format input.
    use_gold_segmentation: Whether this computation used gold segmentation.
    output: A list of sentences to write to the output file.
  """
  with gfile.GFile(output_file, 'w') as f:
    f.write('## tf:{}\n'.format(use_text_format))
    f.write('## gs:{}\n'.format(use_gold_segmentation))
    for serialized_sentence in output:
      sentence = sentence_pb2.Sentence()
      sentence.ParseFromString(serialized_sentence)
      f.write('# text = {}\n'.format(sentence.text.encode('utf-8')))
      for i, token in enumerate(sentence.token):
        head = token.head + 1
        f.write('%s\t%s\t_\t_\t_\t_\t%d\t%s\t_\t_\n' %
                (i + 1, token.word.encode('utf-8'), head,
                 token.label.encode('utf-8')))
      f.write('\n')


def main(unused_argv):
  # Validate that we have a parser saved model passed to this script.
  if FLAGS.parser_saved_model is None:
    tf.logging.fatal('A parser saved model must be provided.')

  # Parse the flags containint lists, using regular expressions.
  # This matches and extracts key=value pairs.
  component_beam_sizes = re.findall(r'([^=,]+)=(\d+)',
                                    FLAGS.inference_beam_size)
  tf.logging.info('Found beam size dict %s' % component_beam_sizes)

  # This matches strings separated by a comma. Does not return any empty
  # strings.
  components_to_locally_normalize = re.findall(r'[^,]+',
                                               FLAGS.locally_normalize)
  tf.logging.info(
      'Found local normalization dict %s' % components_to_locally_normalize)

  # Create a session config with the requested number of threads.
  session_config = tf.ConfigProto(
      log_device_placement=False,
      intra_op_parallelism_threads=FLAGS.threads,
      inter_op_parallelism_threads=FLAGS.threads)

  # Get the segmented input data for the parser, either by running the
  # segmenter ourselves or by simply reading it from the CoNLL file.
  if FLAGS.segmenter_saved_model is None:
    # If no segmenter was provided, we must use the data from the CONLL file.
    input_file = FLAGS.input_file
    parser_input = sentence_io.ConllSentenceReader(input_file).corpus()
    use_gold_segmentation = True
  else:
    # If the segmenter was provided, use it.
    segmenter_input = get_segmenter_corpus(FLAGS.input_file, FLAGS.text_format)
    parser_input = run_segmenter(segmenter_input, FLAGS.segmenter_saved_model,
                                 session_config, FLAGS.max_batch_size,
                                 FLAGS.timeline_output_file)
    use_gold_segmentation = False

  # Now that we have parser input data, parse.
  processed = run_parser(parser_input, FLAGS.parser_saved_model, session_config,
                         component_beam_sizes, components_to_locally_normalize,
                         FLAGS.max_batch_size, FLAGS.timeline_output_file)

  if FLAGS.output_file:
    print_output(FLAGS.output_file, FLAGS.text_format, use_gold_segmentation,
                 processed)


if __name__ == '__main__':
  tf.app.run()
