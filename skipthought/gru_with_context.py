'''
Implements a new RNN cell used in the SkipThoughtModel.
'''
import tensorflow as tf


class GRUCellWithContext(tf.nn.rnn_cell.RNNCell):
    """Gated Recurrent Unit cell with Context (cf. http://arxiv.org/pdf/1506.06726v1.pdf)."""

    def __init__(self, num_units, context):
        self._num_units = num_units
        self._context = context

    @property
    def state_size(self):
        return self._num_units

    @property
    def context(self):
        return self._context

    @property
    def output_size(self):
        return self._num_units

    def __call__(self, inputs, state, scope=None):
        """Gated recurrent unit (GRU) with nunits cells."""
        with tf.variable_scope(scope or type(self).__name__):  # "GRUCellWithContext"
            with tf.variable_scope("Gates"):  # Reset gate and update gate.
                # We start with bias of 1.0 to not reset and not update.
                r, u = tf.split(1, 2, tf.nn.rnn_cell.linear([inputs, state, self._context],
                                                            2 * self._num_units, True, 1.0))
                r, u = tf.sigmoid(r), tf.sigmoid(u)
            with tf.variable_scope("Candidate"):
                c = tf.tanh(tf.nn.rnn_cell.linear(
                    [inputs, r * state, self._context], self._num_units, True))
            new_h = u * state + (1 - u) * c
        return new_h, new_h
