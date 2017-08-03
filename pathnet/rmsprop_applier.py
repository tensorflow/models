# -*- coding: utf-8 -*-
import tensorflow as tf

from tensorflow.python.training import training_ops
from tensorflow.python.training import slot_creator

class RMSPropApplier(object):

  def __init__(self,
               learning_rate,
               decay=0.9,
               momentum=0.0,
               epsilon=1e-10,
               clip_norm=40.0,
               device="/cpu:0",
               name="RMSPropApplier"):

    self._name = name
    self._learning_rate = learning_rate
    self._decay = decay
    self._momentum = momentum
    self._epsilon = epsilon
    self._clip_norm = clip_norm
    self._device = device

    # Tensors for learning rate and momentum.  Created in _prepare.
    self._learning_rate_tensor = None
    self._decay_tensor = None
    self._momentum_tensor = None
    self._epsilon_tensor = None

    self._slots = {}

  def _create_slots(self, var_list):
    for v in var_list:
      # 'val' is Variable's intial value tensor.
      val = tf.constant(1.0, dtype=v.dtype, shape=v.get_shape())
      self._get_or_make_slot(v, val, "rms", self._name)
      self._zeros_slot(v, "momentum", self._name)

  def _prepare(self):
      self._learning_rate_tensor = tf.convert_to_tensor(self._learning_rate,
                                                      name="learning_rate")
      self._decay_tensor = tf.convert_to_tensor(self._decay, name="decay")
      self._momentum_tensor = tf.convert_to_tensor(self._momentum,
                                                 name="momentum")
      self._epsilon_tensor = tf.convert_to_tensor(self._epsilon,
                                                name="epsilon")

  def _slot_dict(self, slot_name):
    named_slots = self._slots.get(slot_name, None)
    if named_slots is None:
      named_slots = {}
      self._slots[slot_name] = named_slots
    return named_slots

  def _get_or_make_slot(self, var, val, slot_name, op_name):
    named_slots = self._slot_dict(slot_name)
    if var not in named_slots:
      named_slots[var] = slot_creator.create_slot(var, val, op_name)
    return named_slots[var]

  def get_slot(self, var, name):
    named_slots = self._slots.get(name, None)
    if not named_slots:
      return None
    return named_slots.get(var, None)

  def _zeros_slot(self, var, slot_name, op_name):
    named_slots = self._slot_dict(slot_name)
    if var not in named_slots:
      named_slots[var] = slot_creator.create_zeros_slot(var, op_name)
    return named_slots[var]

  # TODO: in RMSProp native code, memcpy() (for CPU) and
  # cudaMemcpyAsync() (for GPU) are used when updating values,
  # and values might tend to be overwritten with results from other threads.
  # (Need to check the learning performance with replacing it)  
  def _apply_dense(self, grad, var):
    rms = self.get_slot(var, "rms")
    mom = self.get_slot(var, "momentum")
    return training_ops.apply_rms_prop(
      var, rms, mom,
      self._learning_rate_tensor,
      self._decay_tensor,
      self._momentum_tensor,
      self._epsilon_tensor,
      grad,
      use_locking=False).op

  # Apply accumulated gradients to var.
  def apply_gradients(self, var_list, accum_grad_list, name=None):
    
    update_ops = []

    with tf.device(self._device):
      with tf.control_dependencies(None):
        self._create_slots(var_list)
      
      with tf.name_scope(name, self._name, []) as name:
        self._prepare()
        for var, accum_grad in zip(var_list, accum_grad_list):
          with tf.name_scope("update_" + var.op.name), tf.device(var.device):
            clipped_accum_grad = tf.clip_by_norm(accum_grad, self._clip_norm)
            update_ops.append(self._apply_dense(clipped_accum_grad, var))
        return update_ops;
        #return tf.group(*update_ops, name=name)
