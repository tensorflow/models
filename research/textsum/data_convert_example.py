"""Example of Converting TextSum model data.
Usage:
python data_convert_example.py --command binary_to_text --in_file data/data --out_file data/text_data
python data_convert_example.py --command text_to_binary --in_file data/text_data --out_file data/binary_data
python data_convert_example.py --command binary_to_text --in_file data/binary_data --out_file data/text_data2
diff data/text_data2 data/text_data
"""

import struct
import sys

import six
import tensorflow as tf
from tensorflow.core.example import example_pb2

FLAGS = tf.app.flags.FLAGS
tf.app.flags.DEFINE_string('command', 'binary_to_text',
                           'Either binary_to_text or text_to_binary.'
                           'Specify FLAGS.in_file accordingly.')
tf.app.flags.DEFINE_string('in_file', '', 'path to file')
tf.app.flags.DEFINE_string('out_file', '', 'path to file')

def _binary_to_text():
  reader = open(FLAGS.in_file, 'rb')
  writer = open(FLAGS.out_file, 'w')
  while True:
    len_bytes = reader.read(8)
    if not len_bytes:
      sys.stderr.write('Done reading\n')
      return
    str_len = struct.unpack('q', len_bytes)[0]
    tf_example_str = struct.unpack('%ds' % str_len, reader.read(str_len))[0]
    tf_example = example_pb2.Example.FromString(tf_example_str)
    examples = []
    for key in tf_example.features.feature:
      value = tf_example.features.feature[key].bytes_list.value[0]

      # Convert to byte strings (PY2) or unicode strings (PY3)
      if six.PY2:
        key = key.encode('utf-8')
      else:
        value = value.decode('utf-8')

      examples.append('%s=%s' % (key, value))
    writer.write('%s\n' % '\t'.join(examples))
  reader.close()
  writer.close()


def _text_to_binary():
  inputs = open(FLAGS.in_file, 'r').readlines()
  writer = open(FLAGS.out_file, 'wb')
  for inp in inputs:
    tf_example = example_pb2.Example()
    for feature in inp.strip().split('\t'):
      (k, v) = feature.split('=')
      if six.PY3:
        v = v.encode('utf-8')
      tf_example.features.feature[k].bytes_list.value.extend([v])
    tf_example_str = tf_example.SerializeToString()
    str_len = len(tf_example_str)
    writer.write(struct.pack('q', str_len))
    writer.write(struct.pack('%ds' % str_len, tf_example_str))
  writer.close()


def main(unused_argv):
  assert FLAGS.command and FLAGS.in_file and FLAGS.out_file
  if FLAGS.command == 'binary_to_text':
    _binary_to_text()
  elif FLAGS.command == 'text_to_binary':
    _text_to_binary()


if __name__ == '__main__':
  tf.app.run()
