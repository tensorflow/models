import numpy as np
import tensorflow as tf

from cwrnn import CWRNNCell


input_size = 12 # % 3 == 0
T = 6

inputs = tf.placeholder(tf.float32, [T, input_size])

cell_1 = tf.nn.rnn_cell.BasicRNNCell(input_size // 3, activation=tf.identity)
cell_2 = tf.nn.rnn_cell.BasicRNNCell(input_size // 3, activation=tf.identity)
cell_3 = tf.nn.rnn_cell.BasicRNNCell(input_size // 3, activation=tf.identity)

cwrnn_cell = CWRNNCell([cell_1, cell_2, cell_3], [1, 2, 3], state_is_tuple=True)

t_major_inputs = [tf.reshape(ts, [1, input_size]) for ts in tf.split(0, T, inputs)]
outputs, _ = tf.nn.rnn(cwrnn_cell, t_major_inputs, dtype=tf.float32)

make_parameter_assignments = []
with tf.variable_scope("RNN/CWRNNCell"): # These scopes may need changing
    for i in range(3):
        with tf.variable_scope("SubCell%d/BasicRNNCell/Linear" % i, reuse=True):
            w_var = tf.get_variable('Matrix')
            b_var = tf.get_variable('Bias')

            assignment_w = tf.assign(w_var, tf.ones_like(w_var))
            assignment_b = tf.assign(b_var, tf.zeros_like(b_var))
            make_parameter_assignments += [assignment_w, assignment_b]

init = tf.global_variables_initializer()

with tf.Session() as sess:
    sess.run(init)
    sess.run(make_parameter_assignments)

    np_inputs = np.ones([T, input_size])
    expected_outputs = np.array([
        [12, 12, 12, 12, 12, 12, 12, 12, 12, 12, 12, 12], # cell_1, cell_2, cell_3 fire
        [60, 60, 60, 60, 12, 12, 12, 12, 12, 12, 12, 12], # cell_1 fire
        [252, 252, 252, 252, 60, 60, 60, 60, 12, 12, 12, 12], # cell_1, cell_2 fire
        [1020, 1020, 1020, 1020, 60, 60, 60, 60, 60, 60, 60, 60], # cell_1, cell_3 fire
        [4092, 4092, 4092, 4092, 252, 252, 252, 252, 60, 60, 60, 60], # cell_1, cell_2 fire
        [16380, 16380, 16380, 16380, 252, 252, 252, 252, 60, 60, 60, 60], # cell_1 fire
        ])

    np_outputs = sess.run(outputs, feed_dict={inputs: np_inputs})
    for i, output in enumerate(np_outputs):
        assert (output.squeeze() == expected_outputs[i]).all()

print("All tests passed.")
