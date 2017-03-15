r"""Runs a both a segmentation and parsing model on a CoNLL dataset.
"""

import re
import time

import tensorflow as tf

from google.protobuf import text_format
from tensorflow.python.client import timeline
from tensorflow.python.platform import gfile

from dragnn.protos import spec_pb2
from dragnn.python import graph_builder
from dragnn.python import sentence_io
from dragnn.python import spec_builder
from syntaxnet import sentence_pb2
from syntaxnet.ops import gen_parser_ops
from syntaxnet.util import check

import dragnn.python.load_dragnn_cc_impl
import syntaxnet.load_parser_ops

flags = tf.app.flags
FLAGS = flags.FLAGS


flags.DEFINE_string('parser_master_spec', '',
                    'Path to text file containing a DRAGNN master spec to run.')
flags.DEFINE_string('parser_checkpoint_file', '',
                    'Path to trained model checkpoint.')
flags.DEFINE_string('parser_resource_dir', '',
                    'Optional base directory for resources in the master spec.')
flags.DEFINE_string('segmenter_master_spec', '',
                    'Path to text file containing a DRAGNN master spec to run.')
flags.DEFINE_string('segmenter_checkpoint_file', '',
                    'Path to trained model checkpoint.')
flags.DEFINE_string('segmenter_resource_dir', '',
                    'Optional base directory for resources in the master spec.')
flags.DEFINE_bool('complete_master_spec', True, 'Whether the master_specs '
                  'needs the lexicon and other resources added to them.')
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
flags.DEFINE_bool('use_gold_segmentation', False,
                  'Whether or not to use gold segmentation.')


def main(unused_argv):

  # Parse the flags containint lists, using regular expressions.
  # This matches and extracts key=value pairs.
  component_beam_sizes = re.findall(r'([^=,]+)=(\d+)',
                                    FLAGS.inference_beam_size)
  # This matches strings separated by a comma. Does not return any empty
  # strings.
  components_to_locally_normalize = re.findall(r'[^,]+',
                                               FLAGS.locally_normalize)

  ## SEGMENTATION ##

  if not FLAGS.use_gold_segmentation:

    # Reads master spec.
    master_spec = spec_pb2.MasterSpec()
    with gfile.FastGFile(FLAGS.segmenter_master_spec) as fin:
      text_format.Parse(fin.read(), master_spec)

    if FLAGS.complete_master_spec:
      spec_builder.complete_master_spec(
          master_spec, None, FLAGS.segmenter_resource_dir)

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
      sess.run('save/restore_all',
               {'save/Const:0': FLAGS.segmenter_checkpoint_file})

      tf.logging.info('Processing sentences...')

      processed = []
      start_time = time.time()
      run_metadata = tf.RunMetadata()
      for start in range(0, len(char_corpus), FLAGS.max_batch_size):
        end = min(start + FLAGS.max_batch_size, len(char_corpus))
        feed_dict = {annotator['input_batch']: char_corpus[start:end]}
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

    input_corpus = processed
  else:
    input_corpus = sentence_io.ConllSentenceReader(FLAGS.input_file).corpus()

  ## PARSING

  # Reads master spec.
  master_spec = spec_pb2.MasterSpec()
  with gfile.FastGFile(FLAGS.parser_master_spec) as fin:
    text_format.Parse(fin.read(), master_spec)

  if FLAGS.complete_master_spec:
    spec_builder.complete_master_spec(
        master_spec, None, FLAGS.parser_resource_dir)

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

  session_config = tf.ConfigProto(
      log_device_placement=False,
      intra_op_parallelism_threads=FLAGS.threads,
      inter_op_parallelism_threads=FLAGS.threads)

  with tf.Session(graph=g, config=session_config) as sess:
    tf.logging.info('Initializing variables...')
    sess.run(tf.global_variables_initializer())

    tf.logging.info('Loading from checkpoint...')
    sess.run('save/restore_all', {'save/Const:0': FLAGS.parser_checkpoint_file})

    tf.logging.info('Processing sentences...')

    processed = []
    start_time = time.time()
    run_metadata = tf.RunMetadata()
    for start in range(0, len(input_corpus), FLAGS.max_batch_size):
      end = min(start + FLAGS.max_batch_size, len(input_corpus))
      feed_dict = {annotator['input_batch']: input_corpus[start:end]}
      for comp, beam_size in component_beam_sizes:
        feed_dict['%s/InferenceBeamSize:0' % comp] = beam_size
      for comp in components_to_locally_normalize:
        feed_dict['%s/LocallyNormalize:0' % comp] = True
      if FLAGS.timeline_output_file and end == len(input_corpus):
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
                    len(input_corpus), time.time() - start_time)

    if FLAGS.output_file:
      with gfile.GFile(FLAGS.output_file, 'w') as f:
        for serialized_sentence in processed:
          sentence = sentence_pb2.Sentence()
          sentence.ParseFromString(serialized_sentence)
          f.write('#' + sentence.text.encode('utf-8') + '\n')
          for i, token in enumerate(sentence.token):
            head = token.head + 1
            f.write('%s\t%s\t_\t_\t_\t_\t%d\t%s\t_\t_\n'%(
                i + 1,
                token.word.encode('utf-8'), head,
                token.label.encode('utf-8')))
          f.write('\n\n')


if __name__ == '__main__':
  tf.app.run()
