"""An optimizer that switches between several methods."""

import tensorflow as tf
from tensorflow.python.training import optimizer


class CompositeOptimizer(optimizer.Optimizer):
  """Optimizer that switches between several methods.
  """

  def __init__(self,
               optimizer1,
               optimizer2,
               switch,
               use_locking=False,
               name='Composite'):
    """Construct a new Composite optimizer.

    Args:
      optimizer1: A tf.python.training.optimizer.Optimizer object.
      optimizer2: A tf.python.training.optimizer.Optimizer object.
      switch: A tf.bool Tensor, selecting whether to use the first or the second
        optimizer.
      use_locking: Bool. If True apply use locks to prevent concurrent updates
        to variables.
      name: Optional name prefix for the operations created when applying
        gradients.  Defaults to "Composite".
    """
    super(CompositeOptimizer, self).__init__(use_locking, name)
    self._optimizer1 = optimizer1
    self._optimizer2 = optimizer2
    self._switch = switch

  def apply_gradients(self, grads_and_vars, global_step=None, name=None):

    return tf.cond(
        self._switch,
        lambda: self._optimizer1.apply_gradients(grads_and_vars,
                                                 global_step, name),
        lambda: self._optimizer2.apply_gradients(grads_and_vars,
                                                 global_step, name)
    )


  def get_slot(self, var, name):
    slot1 = self._optimizer1.get_slot(var, name)
    slot2 = self._optimizer2.get_slot(var, name)
    if slot1 and slot2:
      raise LookupError('Slot named %s for variable %s populated for both '
                        'optimizers' % (name, var.name))
    return slot1 or slot2

  def get_slot_names(self):
    return sorted(self._optimizer1.get_slot_names() +
                  self._optimizer2.get_slot_names())
