"""Parser Ops Python library."""

import os.path
import tensorflow as tf

parser_ops_module = tf.load_op_library(
    os.path.join(tf.resource_loader.get_data_files_path(),
                 'parser_ops.so'))
parser_ops = _parser_ops_module.parser_ops
