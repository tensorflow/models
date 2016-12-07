import collections
import six
import tensorflow as tf


def _is_sequence(seq):
  return (isinstance(seq, collections.Sequence)
          and not isinstance(seq, six.string_types))

class CWRNNCell(tf.nn.rnn_cell.RNNCell):
  """Multiple RNNCells called at distinct intervals."""

  def __init__(self, cells, intervals, state_is_tuple=False):
    """Create a RNN cell composed sequentially of a number of RNNCells.
    Args:
      cells: list of RNNCells.
      intervals: wait `interval - 1` timesteps in between calls to cell.
    Raises:
      ValueError: if cells is empty (not allowed), or at least one of the cells
        returns a state tuple but the flag `state_is_tuple` is `False`.
    """
    if len(cells) < 2 or len(intervals) < 2:
      raise ValueError("Must specify at least two cells and intervals.")
    if len(cells) != len(intervals):
      raise ValueError("Cell/interval count mismatch.")
    self._cells = cells
    self._intervals = intervals
    self._state_is_tuple = state_is_tuple
    if not state_is_tuple:
      if any(_is_sequence(c.state_size) for c in self._cells):
        raise ValueError("Some cells return tuples of states, but the flag "
                         "state_is_tuple is not set.  State sizes are: %s"
                         % str([c.state_size for c in self._cells]))
    self._t = 0
    self._last_output = None

  @property
  def state_size(self):
    if self._state_is_tuple:
      return tuple(cell.state_size for cell in self._cells)
    else:
      return sum([cell.state_size for cell in self._cells])

  @property
  def output_size(self):
      return sum([cell.output_size for cell in self._cells])

  def __call__(self, inputs, state, scope=None):
    """Run cells on inputs, starting from state."""
    with tf.variable_scope(scope or type(self).__name__):  # "CWRNNCell"
      batch_size = inputs.get_shape().as_list()[0]
      cur_state_pos = 0
      cur_outp_pos = 0
      outputs = []
      new_states = []

      if self._last_output is None:
        self._last_output = tf.zeros([batch_size, self.output_size])

      for i, cell in enumerate(self._cells):
        with tf.variable_scope("SubCell%d" % i):
          if self._state_is_tuple:
            if not _is_sequence(state):
              raise ValueError(
                  "Expected state to be a tuple of length %d, but received: %s"
                  % (len(self.state_size), state))
            cur_state = state[i]
          else:
            cur_state = tf.slice(
                state, [0, cur_state_pos], [-1, cell.state_size])
            cur_state_pos += cell.state_size
          if self._t % self._intervals[i] == 0:
            cur_outp, new_state = cell(inputs, cur_state)
            new_states.append(new_state)
            outputs.append(cur_outp)
          else:
            last_outp = tf.slice(self._last_output, [0, cur_outp_pos],
                                    [-1, cell.output_size])
            new_states.append(cur_state)
            outputs.append(last_outp)

          cur_outp_pos += cell.output_size

    new_states = (tuple(new_states) if self._state_is_tuple
                  else tf.concat(1, new_states))
    outputs = tf.concat(1, outputs)
    self._last_output = outputs
    self._t += 1
    return outputs, new_states
