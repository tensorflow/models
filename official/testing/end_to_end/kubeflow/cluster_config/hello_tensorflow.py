from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import tensorflow as tf

def main():
  s = tf.constant("Hello, Tensorflow!")
  tf.Session().run(s)


if __name__ == '__main__':
  main()
