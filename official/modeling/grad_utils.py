# Copyright 2022 The TensorFlow Authors. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Some gradient util functions to help users writing custom training loop."""

from absl import logging

import tensorflow as tf


def _filter_grads(grads_and_vars):
  """Filter out iterable with grad equal to None."""
  grads_and_vars = tuple(grads_and_vars)
  if not grads_and_vars:
    return grads_and_vars
  filtered = []
  vars_with_empty_grads = []
  for grad, var in grads_and_vars:
    if grad is None:
      vars_with_empty_grads.append(var)
    else:
      filtered.append((grad, var))
  filtered = tuple(filtered)
  if not filtered:
    raise ValueError("No gradients provided for any variable: %s." %
                     ([v.name for _, v in grads_and_vars],))
  if vars_with_empty_grads:
    logging.warning(
        ("Gradients do not exist for variables %s when minimizing the loss."),
        ([v.name for v in vars_with_empty_grads]))
  return filtered


def _filter_and_allreduce_gradients(grads_and_vars,
                                    allreduce_precision="float32",
                                    bytes_per_pack=0):
  """Filter None grads and then allreduce gradients in specified precision.

  This utils function is used when users intent to explicitly allreduce
  gradients and customize gradients operations before and after allreduce.
  The allreduced gradients are then passed to optimizer.apply_gradients(
  experimental_aggregate_gradients=False).

  Args:
      grads_and_vars: gradients and variables pairs.
      allreduce_precision: Whether to allreduce gradients in float32 or float16.
      bytes_per_pack: A non-negative integer. Breaks collective operations into
        packs of certain size. If it's zero, all gradients are in one pack.

  Returns:
      pairs of allreduced non-None gradients and variables.
  """
  filtered_grads_and_vars = _filter_grads(grads_and_vars)
  (grads, variables) = zip(*filtered_grads_and_vars)
  if allreduce_precision == "float16":
    grads = [tf.cast(grad, "float16") for grad in grads]
  hints = tf.distribute.experimental.CommunicationOptions(
      bytes_per_pack=bytes_per_pack)
  allreduced_grads = tf.distribute.get_strategy(  # pylint: disable=protected-access
  ).extended._replica_ctx_all_reduce(tf.distribute.ReduceOp.SUM, grads, hints)
  if allreduce_precision == "float16":
    allreduced_grads = [tf.cast(grad, "float32") for grad in allreduced_grads]
  return allreduced_grads, variables


def _run_callbacks(callbacks, grads_and_vars):
  for callback in callbacks:
    grads_and_vars = callback(grads_and_vars)
  return grads_and_vars


def minimize_using_explicit_allreduce(tape,
                                      optimizer,
                                      loss,
                                      trainable_variables,
                                      pre_allreduce_callbacks=None,
                                      post_allreduce_callbacks=None,
                                      allreduce_bytes_per_pack=0):
  """Minimizes loss for one step by updating `trainable_variables`.

  Minimizes loss for one step by updating `trainable_variables`.
  This explicitly performs gradient allreduce, instead of relying on implicit
  allreduce in optimizer.apply_gradients(). If training using FP16 mixed
  precision, explicit allreduce will aggregate gradients in FP16 format.
  For TPU and GPU training using FP32, explicit allreduce will aggregate
  gradients in FP32 format.

  Args:
      tape: An instance of `tf.GradientTape`.
      optimizer: An instance of `tf.keras.optimizers.Optimizer`.
      loss: the loss tensor.
      trainable_variables: A list of model Variables.
      pre_allreduce_callbacks: A list of callback functions that takes gradients
        and model variables pairs as input, manipulate them, and returns a new
        gradients and model variables pairs. The callback functions will be
        invoked in the list order and before gradients are allreduced. With
        mixed precision training, the pre_allreduce_allbacks will be applied on
        scaled_gradients. Default is no callbacks.
      post_allreduce_callbacks: A list of callback functions that takes
        gradients and model variables pairs as input, manipulate them, and
        returns a new gradients and model variables paris. The callback
        functions will be invoked in the list order and right before gradients
        are applied to variables for updates. Default is no callbacks.
      allreduce_bytes_per_pack: A non-negative integer. Breaks collective
        operations into packs of certain size. If it's zero, all gradients are
        in one pack.
  """
  if isinstance(optimizer,
                tf.keras.mixed_precision.LossScaleOptimizer):
    # FP16 GPU code path
    with tape:
      scaled_loss = optimizer.get_scaled_loss(loss)
    scaled_grads = tape.gradient(scaled_loss, trainable_variables)
    grads_and_vars = zip(scaled_grads, trainable_variables)
    if pre_allreduce_callbacks:
      grads_and_vars = _run_callbacks(pre_allreduce_callbacks, grads_and_vars)
    (allreduced_scaled_grads,
     filtered_training_vars) = _filter_and_allreduce_gradients(
         grads_and_vars,
         allreduce_precision="float16",
         bytes_per_pack=allreduce_bytes_per_pack)
    allreduced_unscaled_grads = optimizer.get_unscaled_gradients(
        allreduced_scaled_grads)
    grads_and_vars = zip(allreduced_unscaled_grads, filtered_training_vars)
  else:
    # TPU or FP32 GPU code path
    grads = tape.gradient(loss, trainable_variables)
    grads_and_vars = zip(grads, trainable_variables)
    if pre_allreduce_callbacks:
      grads_and_vars = _run_callbacks(pre_allreduce_callbacks, grads_and_vars)
    (allreduced_grads,
     filtered_training_vars) = _filter_and_allreduce_gradients(
         grads_and_vars,
         allreduce_precision="float32",
         bytes_per_pack=allreduce_bytes_per_pack)
    grads_and_vars = zip(allreduced_grads, filtered_training_vars)
  if post_allreduce_callbacks:
    grads_and_vars = _run_callbacks(post_allreduce_callbacks, grads_and_vars)
  optimizer.apply_gradients(
      grads_and_vars, experimental_aggregate_gradients=False)
