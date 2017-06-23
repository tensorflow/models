import tensorflow as tf
FLAGS = tf.app.flags.FLAGS

tf.flags.DEFINE_string("checkpoint_dir", "",
                       "checkpoint_file created by tf 2?")

OLD_CHECKPOINT_FILE = FLAGS.checkpoint_dir
NEW_CHECKPOINT_FILE = FLAGS.checkpoint_dir


vars_to_rename = {
    "lstm/BasicLSTMCell/Linear/Matrix": "lstm/basic_lstm_cell/weights",
    "lstm/BasicLSTMCell/Linear/Bias": "lstm/basic_lstm_cell/biases",
}
new_checkpoint_vars = {}
reader = tf.train.NewCheckpointReader(OLD_CHECKPOINT_FILE)
for old_name in reader.get_variable_to_shape_map():
  if old_name in vars_to_rename:
    new_name = vars_to_rename[old_name]
  else:
    new_name = old_name
  new_checkpoint_vars[new_name] = tf.Variable(reader.get_tensor(old_name))

init = tf.global_variables_initializer()
saver = tf.train.Saver(new_checkpoint_vars)

with tf.Session() as sess:
  sess.run(init)
  saver.save(sess, NEW_CHECKPOINT_FILE)