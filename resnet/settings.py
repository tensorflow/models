# encoding: utf-8

import tensorflow as tf

FLAGS = tf.app.flags.FLAGS

# Network Settings
tf.app.flags.DEFINE_integer('num_layers', 56, 'Number of network layers(num_layers-2 must be diviced by 9)')

# Data Settings
tf.app.flags.DEFINE_integer('image_height', 128, 'Image height.')
tf.app.flags.DEFINE_integer('image_width', 128, 'Image width.')
tf.app.flags.DEFINE_integer('image_depth', 3, 'Image depth.')

tf.app.flags.DEFINE_integer('num_classes', 97435, "Number of classes")

# Train Settings
tf.app.flags.DEFINE_string('optimizer', 'Momentum', "Type of the optimizer for training")

tf.app.flags.DEFINE_integer('batch_size', 64, 'the number of images in a batch.')
tf.app.flags.DEFINE_integer('num_gpus', 4, 'Number of gpus used for training. ')
tf.app.flags.DEFINE_float('initial_learning_rate', 0.1, "Initial learning rate.")
tf.app.flags.DEFINE_float('lr_decay_factor', 0.1, "Learning rate decay factor.")
tf.app.flags.DEFINE_integer('lr_decay_steps', 10000, "Epochs after which learning rate decays.")
tf.app.flags.DEFINE_integer('max_steps', 1000000, "Number of batches to run.")

tf.app.flags.DEFINE_integer('save_interval_secs', 15*60,
                            'Save interval seconds.')
tf.app.flags.DEFINE_integer('save_summaries_secs', 5*60,
                            'Save summaries interval seconds.')

# Eval Settings
tf.app.flags.DEFINE_integer('eval_batch_count', 50,
                            'Number of batches to eval.')
tf.app.flags.DEFINE_bool('eval_once', False,
                         'Whether evaluate the model only once.')

# Path Settings
tf.app.flags.DEFINE_string('summary_dir', '/home/wangguanshuo/resnet/resnet_summary/google_stoc', 
	                       'Directory to keep event log')

tf.app.flags.DEFINE_string('delimeter', '\t', "Delimeter of the list")
tf.app.flags.DEFINE_string('train_dir', '/home/wangguanshuo/resnet/resnet_model/google_stoc',
                           'Directory to keep trained model.')
tf.app.flags.DEFINE_string('train_list_path', '/home/wangguanshuo/lists/msra/training.npy', 
	                       'Filename for training data list.')

tf.app.flags.DEFINE_string('eval_dir', '',
	                       'Directory to keep eval outputs.')
tf.app.flags.DEFINE_string('eval_list_path', '', 
	                       'Filename for eval data')
