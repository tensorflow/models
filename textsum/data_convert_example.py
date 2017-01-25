"""Example of Converting TextSum model data.
Usage:
python data_convert_example.py --command binary_to_text --in_file data/binary_data --out_file data/text_data
python data_convert_example.py --command text_to_binary --in_file data/text_data --out_file data/binary_data
python data_convert_example.py --command binary_to_text --in_file data/binary_data --out_file data/text_data2
diff data/text_data2 data/text_data
python data_convert_example.py --command text_to_binary_multi --in_file srcArticlesTextDirectory --out_file srcArticlesBinaryDirectory

Below logic has not been tested but seems it should owrk in theory -- Have to still test
python data_convert_example.py --command binary_to_text_multi --in_file srcArticlesBinaryDirectory --out_file srcArticlesTextDirectory

"""

import struct
import sys
import os

import tensorflow as tf
from tensorflow.core.example import example_pb2

FLAGS = tf.app.flags.FLAGS
tf.app.flags.DEFINE_string('command', 'binary_to_text',
                           'Either binary_to_text or text_to_binary.'
                           'Specify FLAGS.in_file accordingly.')
tf.app.flags.DEFINE_string('in_file', '', 'path to file')
tf.app.flags.DEFINE_string('out_file', '', 'path to file')
fileNdx = 0

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
      examples.append('%s=%s' % (key, tf_example.features.feature[key].bytes_list.value[0]))
    writer.write('%s\n' % '\t'.join(examples))
  reader.close()
  writer.close()


def _text_to_binary(file=None):
  locFileIn = FLAGS.in_file
  locFileOut = FLAGS.out_file
  if file != None:
    locFileIn = file
    locFileOut = FLAGS.out_file + '/' + file.split('/',1)[-1]
  
  inputs = open(locFileIn, 'r').readlines()
  writer = open(locFileOut, 'wb')
  for inp in inputs:
    tf_example = example_pb2.Example()
    for feature in inp.strip().split('\t'):
      (k, v) = feature.split('=')
      tf_example.features.feature[k].bytes_list.value.extend([v])
    tf_example_str = tf_example.SerializeToString()
    str_len = len(tf_example_str)

    writer.write((struct.pack('q', str_len)))
    writer.write((struct.pack('%ds' % str_len, tf_example_str)))
  writer.close()

#Have to test below logic eventually after working out some format issues
#def _binary_to_text_multi():
#  try:
#    for path, dirs, files in os.walk(FLAGS.in_file):
#        for fn in files:
#            fullpath = os.path.join(path, fn)
#            if os.path.isfile(fullpath): 
#              try:
#                  _binary_to_text(fullpath)
#              except RuntimeError as e:
#                  print "Runtime Error: {0} : {1}".format(e.errno, e.strerror)
#  except RuntimeError as e:
#    print "Runtime Error: {0} : {1}".format(e.errno, e.strerror)


def _text_to_binary_multi():
  try:
    for path, dirs, files in os.walk(FLAGS.in_file):
        for fn in files:
            fullpath = os.path.join(path, fn)
            if os.path.isfile(fullpath): 
              try:
                  _text_to_binary(fullpath)
              except RuntimeError as e:
                  print "Runtime Error: {0} : {1}".format(e.errno, e.strerror)
  except RuntimeError as e:
    print "Runtime Error: {0} : {1}".format(e.errno, e.strerror)


def main(unused_argv):
  assert FLAGS.command and FLAGS.in_file and FLAGS.out_file
  if FLAGS.command == 'binary_to_text':
    _binary_to_text()
#  elif FLAGS.command == 'binary_to_text_multi':
#    _binary_to_text_binary()
  elif FLAGS.command == 'text_to_binary':
    _text_to_binary()
  elif FLAGS.command == 'text_to_binary_multi':
    _text_to_binary_multi()


if __name__ == '__main__':
    tf.app.run()
