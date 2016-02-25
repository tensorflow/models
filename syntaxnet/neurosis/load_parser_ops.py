"""Parser Ops Python library."""

import os.path
import tensorflow as tf

tf.load_op_library(
    os.path.join(tf.resource_loader.get_data_files_path(),
                 'parser_ops.so'))
