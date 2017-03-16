"""Utilities for reading and writing sentences in dragnn."""
import tensorflow as tf
from syntaxnet.ops import gen_parser_ops


class ConllSentenceReader(object):
  """A reader for conll files, with optional projectivizing."""

  def __init__(self, filepath, batch_size=32,
               projectivize=False, morph_to_pos=False):
    self._graph = tf.Graph()
    self._session = tf.Session(graph=self._graph)
    task_context_str = """
          input {
            name: 'documents'
            record_format: 'conll-sentence'
            Part {
             file_pattern: '%s'
            }
          }""" % filepath
    if morph_to_pos:
      task_context_str += """
          Parameter {
            name: "join_category_to_pos"
            value: "true"
          }
          Parameter {
            name: "add_pos_as_attribute"
            value: "true"
          }
          Parameter {
            name: "serialize_morph_to_pos"
            value: "true"
          }
          """
    with self._graph.as_default():
      self._source, self._is_last = gen_parser_ops.document_source(
          task_context_str=task_context_str, batch_size=batch_size)
      self._source = gen_parser_ops.well_formed_filter(self._source)
      if projectivize:
        self._source = gen_parser_ops.projectivize_filter(self._source)

  def read(self):
    """Reads a single batch of sentences."""
    if self._session:
      sentences, is_last = self._session.run([self._source, self._is_last])
      if is_last:
        self._session.close()
        self._session = None
    else:
      sentences, is_last = [], True
    return sentences, is_last

  def corpus(self):
    """Reads the entire corpus, and returns in a list."""
    tf.logging.info('Reading corpus...')
    corpus = []
    while True:
      sentences, is_last = self.read()
      corpus.extend(sentences)
      if is_last:
        break
    tf.logging.info('Read %d sentences.' % len(corpus))
    return corpus
