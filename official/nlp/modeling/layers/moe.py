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

"""Mixture of Experts layers and their routing mechanisms."""

import dataclasses
from typing import Any, Callable, Optional, Tuple

from absl import logging
import numpy as np
import tensorflow as tf

from official.modeling import tf_utils


_InitializerType = tf.keras.initializers.Initializer


_DEFAULT_KERNEL_INITIALIZER = tf.keras.initializers.TruncatedNormal(stddev=2e-2)
_DEFAULT_BIAS_INITIALIZER = tf.keras.initializers.Zeros()


################## Routers (gating functions) ##################


def _router_z_loss(router_logits: tf.Tensor) -> float:
  """Computes router z-loss.

   The router z-loss was introduced in Designing Effective Sparse Expert Models
   (https://arxiv.org/abs/2202.08906). It encourages router logits to remain
   small in an effort to improve stability.

  Args:
    router_logits: <float32>[num_groups, tokens_per_group, num_experts] router
      logits.

  Returns:
    Scalar router z-loss <float32>.
  """
  num_groups, tokens_per_group, _ = router_logits.shape

  log_z = tf.math.reduce_logsumexp(router_logits, axis=-1)
  z_loss = log_z**2
  return tf.math.reduce_sum(z_loss) / (num_groups * tokens_per_group)


@dataclasses.dataclass
class RouterMask:
  """Dispatch and combine arrays for expert routing with masked matmuls.

  Attributes:
    dispatch_mask:
      <float>[num_groups, tokens_per_group, num_experts, expert_capacity]
      dispatch array that is 1 if the token gets routed to the
      corresponding expert, and 0 otherwise.
    combine_array:
      <float>[num_groups, tokens_per_group, num_experts, expert_capacity]
      combine array used for combining expert outputs and
      scaling with router probability.
  """
  dispatch_mask: tf.Tensor
  combine_array: tf.Tensor

RouterOutput = RouterMask


class Router(tf.keras.layers.Layer):
  """Abstract base router class, defining router API and inner workings.

  Computations are performed in float32 for stability, and returned after
  conversion according to the precision policy. See the discussion of
  "selective precision" in https://arxiv.org/abs/2101.03961.

  Uses Keras add_loss() and add_metric() APIs.

  Attributes:
    num_experts: Number of experts, used to check consistency with
      FeedForwardExperts.
    jitter_noise: Amplitude of jitter noise applied to router logits.
    router_weights: Dense layer that computes logits for all tokens, which are
      then used as expert or token weights.
  """

  def __init__(
      self,
      num_experts: int,
      *,
      jitter_noise: float = 0.0,
      use_bias: bool = True,
      kernel_initializer: _InitializerType = _DEFAULT_KERNEL_INITIALIZER,
      bias_initializer: _InitializerType = _DEFAULT_BIAS_INITIALIZER,
      name: str = "router",
      dtype: Any = tf.float32,
      **kwargs):
    """Init.

    Args:
      num_experts: Number of experts.
      jitter_noise: Amplitude of jitter noise applied to router logits.
      use_bias: Whether or not to use the bias term in computing the router
        weights.
      kernel_initializer: Kernel initializer for router weights.
      bias_initializer: Bias initializer for router weights.
      name: Layer name.
      dtype: The dtype of the layer's computations and weights. tf.float32 is
        recommended for stability.
      **kwargs: Forwarded to super.
    """
    super().__init__(name=name, dtype=dtype, **kwargs)

    self.num_experts = num_experts  # Used to check consistency with
                                    # FeedForwardExperts.
    self.jitter_noise = jitter_noise

    self.router_weights = tf.keras.layers.Dense(
        num_experts,
        use_bias=use_bias,
        kernel_initializer=tf_utils.clone_initializer(kernel_initializer),
        bias_initializer=tf_utils.clone_initializer(bias_initializer),
        name="router_weights",
        dtype=dtype)

  def call(self,
           inputs: tf.Tensor,
           *,
           expert_capacity: int,
           training: Optional[bool] = None) -> RouterOutput:
    """Computes dispatch and combine arrays for routing to experts.

    Args:
      inputs: Inputs to send to experts of shape
        <float>[num_groups, tokens_per_group, hidden_dim].
      expert_capacity: Each group will send this many tokens to each expert.
      training: If true, apply jitter noise during routing. If not provided
        taken from tf.keras.backend.

    Returns:
      Router indices or mask arrays (depending on router type).
    """
    if training is None:
      training = tf.keras.backend.learning_phase()

    # inputs shape <float>[num_groups, tokens_per_group, hidden_dim]
    router_probs, router_logits = self._compute_router_probabilities(
        inputs, apply_jitter=training)
    # router_probs <float32>[num_groups, tokens_per_group, num_experts]
    # router_logits <float>[num_groups, tokens_per_group, num_experts]
    router_z_loss = _router_z_loss(router_logits)
    self.add_loss(router_z_loss)
    self.add_metric(router_z_loss, name="router_z_loss")

    routing_instructions = self._compute_routing_instructions(
        router_probs, expert_capacity)
    return routing_instructions

  def _compute_router_probabilities(
      self, inputs: tf.Tensor,
      apply_jitter: bool) -> Tuple[tf.Tensor, tf.Tensor]:
    """Computes router probabilities from input tokens.

    Args:
      inputs: Inputs from which router probabilities are computed, shape
        <float>[num_groups, tokens_per_group, hidden_dim].
      apply_jitter: If true, apply jitter noise.

    Returns:
      - <float32>[num_groups, tokens_per_group, num_experts] probabilities for
        each token and expert. Used for routing tokens to experts.
      - <float32>[num_groups, tokens_per_group, num_experts] raw router logits.
        Used for computing router z-loss.
    """
    if apply_jitter and self.jitter_noise > 0:
      inputs *= tf.random.uniform(
          inputs.shape,
          minval=1.0 - self.jitter_noise,
          maxval=1.0 + self.jitter_noise,
          dtype=inputs.dtype)
    # inputs <float>, router_logits <float32>
    router_logits = self.router_weights(inputs)
    router_probs = tf.keras.activations.softmax(router_logits, axis=-1)
    return router_probs, router_logits

  def _compute_routing_instructions(self, router_probs: tf.Tensor,
                                    expert_capacity: int) -> RouterOutput:
    """Computes instructions for routing inputs to experts."""
    raise NotImplementedError(
        "Router is an abstract class that should be subclassed.")


class MaskedRouter(Router):
  """Abstract base router class for masked matmul dispatch routers.

  MaskedRouter(s) return RouterMask(s) containing a dispatch mask and combine
  array for sending and receiving (via masked matmuls) inputs and outputs to and
  from experts.

  Routing using masked matmuls is generally faster than scatter-based routing on
  TPUs.

  Uses Keras add_loss() and add_metric() APIs.
  """

  def _compute_routing_instructions(self, router_probs: tf.Tensor,
                                    expert_capacity: int) -> RouterMask:
    """Computes masks for the top-k experts per token.

    Args:
      router_probs: <float32>[num_groups, tokens_per_group, num_experts]
        probabilities used to determine the routing of tokens to the experts.
      expert_capacity: Each group will send this many tokens to each expert.

    Returns:
      Router mask arrays.
    """
    raise NotImplementedError(
        "MaskedRouter is an abstract class that should be subclassed.")


class ExpertsChooseMaskedRouter(MaskedRouter):
  """Masked matmul router using experts choose tokens assignment.

  This router uses the same mechanism as in Mixture-of-Experts with Expert
  Choice (https://arxiv.org/abs/2202.09368): each expert selects its top
  expert_capacity tokens. An individual token may be processed by multiple
  experts or none at all.

  Note: "experts choose routing" should not be used in decoder blocks because it
  breaks the autoregressive behavior, leading to a mismatch between training
  (teacher forcing) and inference (autoregressive decoding).

  Uses Keras add_loss() and add_metric() APIs.
  """

  def _compute_routing_instructions(self, router_probs: tf.Tensor,
                                    expert_capacity: int) -> RouterMask:
    """Computes masks for the highest probability token per expert.

    Args:
      router_probs: <float32>[num_groups, tokens_per_group, num_experts]
        probabilities used to determine the routing of tokens to the experts.
      expert_capacity: Each group will send this many tokens to each expert.

    Returns:
      Dispatch and combine arrays for routing with masked matmuls.
    """
    num_groups, tokens_per_group, _ = router_probs.shape
    router_probs_t = tf.transpose(router_probs, perm=[0, 2, 1])
    # router_probs_t: <float32>[num_groups, num_experts, tokens_per_group]

    # Top expert_capacity router probability and corresponding token indices for
    # each expert.
    # Shapes [num_groups, num_experts, expert_capacity]
    expert_gate, expert_index = tf.math.top_k(
        router_probs_t, k=expert_capacity, sorted=False)

    # Convert to one-hot mask of expert indices for each token in each group.
    # Shape: [num_groups, num_experts, expert_capacity, tokens_per_group].
    dispatch_mask = tf.one_hot(
        expert_index, tokens_per_group, dtype=router_probs.dtype)

    # Move axes to conform with shape expected by MoeLayer API.
    # Shape: [num_groups, tokens_per_group, num_experts, expert_capacity]
    dispatch_mask = tf.transpose(dispatch_mask, perm=[0, 3, 1, 2])

    # The combine array will be used for combining expert outputs, scaled by the
    # router probabilities.
    # Shape: [num_groups, num_experts, tokens_per_group, expert_capacity]
    combine_array = tf.einsum(
        "...ec,...tec->...tec",
        expert_gate,
        dispatch_mask)

    # Add load balancing loss.
    # Each expert is choosing tokens until it reaches full capacity, so we don't
    # need an auxiliary loading balancing loss for expert choice routing.
    self.add_metric(0.0, name="load_balancing_loss")

    # Gather expert metrics.
    # Number of tokens that were dispatched to at least one expert.
    num_tokens = num_groups * tokens_per_group
    num_tokens_dispatched_somewhere = tf.math.reduce_sum(tf.math.reduce_max(
        dispatch_mask, axis=(-1, -2)))
    fraction_tokens_left_behind = 1.0 - num_tokens_dispatched_somewhere / float(
        num_tokens)
    # Total number of tokens that were dispatched (one token could be
    # dispatched to multiple experts).
    num_tokens_dispatched = tf.math.reduce_sum(dispatch_mask)
    # Of the tokens dispatched, how confident was the router in its routing?
    router_confidence = tf.math.reduce_sum(
        combine_array) / num_tokens_dispatched

    expert_usage = 1.0  # Experts fully utilized when "expert choose tokens"

    self.add_metric(fraction_tokens_left_behind,
                    name="fraction_tokens_left_behind")
    self.add_metric(router_confidence, name="router_confidence")
    self.add_metric(expert_usage, name="expert_usage")

    # Return to default dtype now that router computation is complete.
    dtype = tf.keras.mixed_precision.global_policy().compute_dtype
    dispatch_mask = tf.cast(dispatch_mask, dtype)
    combine_array = tf.cast(combine_array, dtype)
    output = RouterMask(dispatch_mask, combine_array)
    return output


################## Model layers ##################


class FeedForward(tf.keras.layers.Layer):
  """Feed-forward layer - position independent, dense, nonlinear transformation.

  Typically used in an MLP Transformer block.
  """

  def __init__(
      self,
      d_ff: int,
      *,
      dropout_rate: float = 0.1,
      activation: Callable[[tf.Tensor],
                           tf.Tensor] = tf.keras.activations.gelu,
      kernel_initializer: _InitializerType = _DEFAULT_KERNEL_INITIALIZER,
      bias_initializer: _InitializerType = _DEFAULT_BIAS_INITIALIZER,
      name: str = "feed_forward",
      **kwargs):
    """Initializes layer.

    Args:
      d_ff: Dimension of feed-forward layer.
      dropout_rate: The dropout probability.
      activation: (Nonlinear) transform applied in layer.
      kernel_initializer: Initialization scheme for kernel.
      bias_initializer: Initialization scheme for bias.
      name: Layer name.
      **kwargs: Forwarded to super.
    """
    super().__init__(name=name, **kwargs)
    self.activation = activation
    self.kernel_initializer = kernel_initializer
    self.bias_initializer = bias_initializer

    self.intermediate_layer = tf.keras.layers.Dense(
        d_ff,
        kernel_initializer=tf_utils.clone_initializer(self.kernel_initializer),
        bias_initializer=tf_utils.clone_initializer(self.bias_initializer),
        name="intermediate")
    self.dropout_layer = tf.keras.layers.Dropout(dropout_rate)

  def build(self, input_shape: Tuple[int, int, int]):
    """Creates the input shape dependent output weight variables."""
    self.output_layer = tf.keras.layers.Dense(
        input_shape[-1],
        kernel_initializer=tf_utils.clone_initializer(self.kernel_initializer),
        bias_initializer=tf_utils.clone_initializer(self.bias_initializer),
        name="output")

  def call(self,
           inputs: tf.Tensor,
           *,
           training: Optional[bool] = None) -> tf.Tensor:
    """Applies layer to inputs.

    Args:
      inputs: Batch of input embeddings, of shape
        <float>[batch_size, seq_len, hidden_dim].
      training: Only apply dropout during training.

    Returns:
      Transformed inputs with the same shape as inputs
        <float>[batch_size, seq_len, hidden_dim].
    """
    x = self.intermediate_layer(inputs)
    x = self.activation(x)
    x = self.output_layer(x)
    x = self.dropout_layer(x, training=training)
    return x


class FeedForwardExperts(tf.keras.layers.Layer):
  """Feed-forward layer with multiple experts.

  Note that call() takes inputs with shape
  [num_groups, num_experts, expert_capacity, hidden_dim]
  which is different from the usual [batch_size, seq_len, hidden_dim] used by
  the FeedForward layer.

  The experts are independent FeedForward layers of the
  same shape, i.e. the kernel doesn't have shape [hidden_dim, out_dim], but
  [num_experts, hidden_dim, out_dim].
  """

  def __init__(
      self,
      num_experts: int,
      d_ff: int,
      *,
      dropout_rate: float = 0.1,
      activation: Callable[[tf.Tensor],
                           tf.Tensor] = tf.keras.activations.gelu,
      kernel_initializer: _InitializerType = _DEFAULT_KERNEL_INITIALIZER,
      bias_initializer: _InitializerType = _DEFAULT_BIAS_INITIALIZER,
      name: str = "experts",
      **kwargs):
    """Initializes layer.

    Args:
      num_experts: Number of experts (i.e. number of independent feed-forward
        blocks).
      d_ff: Dimension of feed-forward layer of each expert.
      dropout_rate: The dropout probability (expert_dropout_rate).
      activation: (Nonlinear) transform applied in layer.
      kernel_initializer: Initialization scheme for kernel.
      bias_initializer: Initialization scheme for bias.
      name: Layer name.
      **kwargs: Forwarded to super.
    """
    super().__init__(name=name, **kwargs)
    self.num_experts = num_experts
    self.activation = activation
    self.kernel_initializer = kernel_initializer
    self.bias_initializer = bias_initializer

    self.intermediate_layer = tf.keras.layers.EinsumDense(
        "gech,ehf->gecf",
        output_shape=(self.num_experts, None, d_ff),
        bias_axes="ef",
        kernel_initializer=tf_utils.clone_initializer(self.kernel_initializer),
        bias_initializer=tf_utils.clone_initializer(self.bias_initializer),
        name="intermediate")
    self.dropout_layer = tf.keras.layers.Dropout(dropout_rate)

  def build(self, input_shape: Tuple[int, int, int, int]):
    """Creates the input shape dependent output weight variables."""
    if input_shape[1] != self.num_experts:
      raise ValueError(
          f"Input shape {input_shape} is inconsistent with num_experts "
          f"{self.num_experts}.")

    self.output_layer = tf.keras.layers.EinsumDense(
        "gecf,efh->gech",
        output_shape=(self.num_experts, None, input_shape[-1]),
        bias_axes="eh",
        kernel_initializer=tf_utils.clone_initializer(self.kernel_initializer),
        bias_initializer=tf_utils.clone_initializer(self.bias_initializer),
        name="output")

  def call(self,
           inputs: tf.Tensor,
           *,
           training: Optional[bool] = None) -> tf.Tensor:
    """Applies layer to inputs.

    Args:
      inputs: Inputs of shape
        <float>[num_groups, num_experts, expert_capacity, hidden_dim].
      training: Only apply dropout during training.

    Returns:
      Transformed inputs with the same shape as inputs
        <float>[num_groups, num_experts, expert_capacity, hidden_dim].
    """
    x = self.intermediate_layer(inputs)
    x = self.activation(x)
    x = self.output_layer(x)
    x = self.dropout_layer(x, training=training)
    return x


class MoeLayer(tf.keras.layers.Layer):
  """Sparse MoE layer with per-token routing.

  In this TF implementation, all experts need to fit onto a single device
  allowing for batch parallelism only.

  Uses Keras add_loss() and add_metric() APIs.

  Attributes:
    num_experts: Number of experts (i.e. number of independent feed-forward
      blocks).
  """

  def __init__(
      self,
      experts: FeedForwardExperts,
      router: MaskedRouter,
      *,
      train_capacity_factor: float = 1.0,
      eval_capacity_factor: float = 1.0,
      min_expert_capacity: int = 4,
      max_group_size: int = 4096,
      strict_group_size: bool = False,
      name: str = "moe",
      **kwargs):
    """Init.

    Args:
      experts: Instance of FeedForwardExperts. Needs to have the same
        num_experts as the router.
      router: Instance of MaskedRouter to route the tokens to
        the different experts.
      train_capacity_factor: Scaling factor to increase the expert token
        capacity during training. This factor plays an analogous, but slightly
        different, role depending on the routing assignment algorithm:
        - For "tokens choose" routing, the capacity factor only affects the
          maximum number of tokens that an expert will process. It does not
          affect how many experts a given token is routed to; see the
          num_selected_experts attributes of "tokens choose" routers.
        - For "experts choose" routing, because experts always fill their
          buffer, increasing the capacity factor will increase the number of
          tokens that an expert will process AND will indirectly increase the
          number of experts that a given token is routed to.
      eval_capacity_factor: As above, but used during evaluation.
      min_expert_capacity: Minimum token processing capacity for each expert.
      max_group_size: The total number of tokens on each device is subdivided
        into groups of this size. Router computations are then performed on a
        per-group basis. A larger group size will result in slower but more
        accurate top-k and sorting computations, whereas a smaller group size
        will result in faster but more approximate (and potentially less stable)
        routing choices. Note that actual group size may be smaller than
        max_group_size for consistency with the number of experts and tokens;
        see also `strict_group_size` attribute. In practice,
        we find that imperfect routing choices are tolerable and recommend
        choosing a group size on the order of 4096 tokens, although this number
        will vary based on model configuration and size.
      strict_group_size: If True, fail if unable to set the token group size
        equal to max_group_size. If False (default), the actual group size may
        be smaller than max_group_size for consistency with the number of
        experts and tokens.
      name: Layer name.
      **kwargs: Forwarded to super.
    """
    super().__init__(name=name, **kwargs)
    self._experts = experts
    self._router = router

    self.num_experts = experts.num_experts
    assert experts.num_experts == router.num_experts

    self._train_capacity_factor = train_capacity_factor
    self._eval_capacity_factor = eval_capacity_factor
    self._max_group_size = max_group_size
    self._min_expert_capacity = min_expert_capacity
    self._strict_group_size = strict_group_size

  def call(self,
           inputs: tf.Tensor,
           *,
           training: Optional[bool] = None) -> tf.Tensor:
    """Applies MoeLayer.

    Args:
      inputs: Batch of input embeddings of shape
        <float>[batch_size, seq_length, hidden_dim].
      training: Only apply dropout and jitter noise during training. If not
        provided taken from tf.keras.backend.

    Returns:
      Transformed inputs with same shape as inputs:
        <float>[batch_size, seq_length, hidden_dim].

    Raises:
      ValueError if we cannot find a group_size satisfying given requirements.
    """
    if training is None:
      training = tf.keras.backend.learning_phase()

    # inputs shape [batch_size, seq_length, hidden_dim]
    per_device_batch_size, seq_length, hidden_dim = inputs.shape
    num_tokens = per_device_batch_size * seq_length
    num_groups = self._num_groups(num_tokens, self._max_group_size)
    tokens_per_group = num_tokens // num_groups

    if training:
      capacity_factor = self._train_capacity_factor
    else:
      capacity_factor = self._eval_capacity_factor
    # Each group will send expert_capacity tokens to each expert.
    expert_capacity = int(
        round(capacity_factor * tokens_per_group / self.num_experts))
    expert_capacity = max(expert_capacity, self._min_expert_capacity)
    logging.info(
        "Selected expert_capacity=%d for num_experts=%d and training=%r.",
        expert_capacity, self.num_experts, training)

    # Reshape batch and sequence/token dimensions for expert routing.
    x = tf.reshape(inputs, (num_groups, tokens_per_group, hidden_dim))

    x = self._mask_and_dispatch_to_experts(x, expert_capacity, training)

    # Return to original input shape.
    x = tf.reshape(x, (per_device_batch_size, seq_length, hidden_dim))
    return x

  def _num_groups(self, num_tokens: int, max_group_size: int) -> int:
    """Returns the number of token routing groups.

    Note that the quantities are local to the device.

    We select the smallest num_groups such that:
    - num_groups >= num_tokens / max_group_size (ensuring the group size is no
      larger than max_group_size),
    - num_tokens % num_groups = 0 (ensuring that the group size evenly divides
      into the num_tokens),

    Args:
      num_tokens: Number of tokens from input batch.
      max_group_size: Maximum size of each token routing group. Actual group
        size may end up being smaller unless strict_group_size==True.

    Returns:
      Number of token routing groups.

    Raises:
      ValueError if we cannot find a group_size satisfying the above
        requirements.
    """
    # Increase the number of groups (and decrease the group size) until we have
    # a viable number of groups.
    min_num_groups = int(np.ceil(num_tokens / max_group_size))
    num_groups = min_num_groups
    while num_groups < num_tokens and num_tokens % num_groups != 0:
      num_groups += 1

    group_size = num_tokens // num_groups
    logging.info(
        "Selected group_size=%d and num_groups=%d for input num_tokens=%d, "
        "max_group_size=%d, num_experts=%d.",
        group_size, num_groups, num_tokens, max_group_size, self.num_experts)

    if group_size < self._min_expert_capacity:
      raise ValueError(
          f"Local (per-device) group_size {group_size} is smaller than "
          f"min_expert_capacity {self._min_expert_capacity}, which is probably "
          "not intended. Please increase max_group_size {max_group_size} to"
          " seq_length or increase batch_size or decrease min_expert_capacity.")

    if self._strict_group_size and group_size != self._max_group_size:
      raise ValueError(
          f"Selected group_size={group_size} is less than the "
          f"max_group_size={max_group_size}. Exiting because strict mode is "
          "active (strict_group_size=True)")

    return num_groups

  def _mask_and_dispatch_to_experts(self, inputs: tf.Tensor,
                                    expert_capacity: int,
                                    training: bool) -> tf.Tensor:
    """Wraps expert masked routing and dispatching algorithm.

    This algorithm takes the following steps:
    (1) Compute dispatch mask and combine array using self._router.
    (2) Dispatch inputs to experts based on dispatch mask.
    (3) Recombine individual expert outputs using combine array.

    Args:
      inputs: <float>[num_groups, tokens_per_group, hidden_dim] inputs to
        send to experts.
      expert_capacity: Each group will send this many tokens to each expert.
      training: If true, apply jitter noise during routing and dropout
        during expert computation.

    Returns:
      <float>[num_groups, num_tokens_per_group, hidden_dim] outputs from
        experts.
    """
    # Shape [num_groups, tokens_per_group, num_experts, expert_capacity]
    router_mask = self._router(
        inputs,
        expert_capacity=expert_capacity,
        training=training)

    # Shape [num_groups, num_experts, expert_capacity, hidden_dim]
    expert_inputs = tf.einsum(
        "gth,gtec->gech",
        inputs,
        router_mask.dispatch_mask)

    expert_outputs = self._experts(expert_inputs, training=training)

    # Shape [num_groups, tokens_per_group, hidden_dim]
    combined_outputs = tf.einsum(
        "gech,gtec->gth",
        expert_outputs,
        router_mask.combine_array)

    return combined_outputs


class MoeLayerWithBackbone(tf.keras.layers.Layer):
  """Sparse MoE layer plus a FeedForward layer evaluated for all tokens.

  Uses Keras add_loss() and add_metric() APIs.
  """

  def __init__(
      self,
      moe: MoeLayer,
      backbone_d_ff: int,
      *,
      dropout_rate: float = 0.1,
      activation: Callable[[tf.Tensor],
                           tf.Tensor] = tf.keras.activations.gelu,
      kernel_initializer: _InitializerType = _DEFAULT_KERNEL_INITIALIZER,
      bias_initializer: _InitializerType = _DEFAULT_BIAS_INITIALIZER,
      name: str = "moe_with_backbone",
      **kwargs):
    """Init.

    Args:
      moe: Instance of MoeLayer with experts and router.
      backbone_d_ff: Dimension of feed-forward layer of a lightweight backbone,
        which is evaluated for all tokens.
      dropout_rate: Dropout rate for the backbone.
      activation: (Nonlinear) transform applied in the backbone.
      kernel_initializer: Initialization scheme for kernels in the backbone.
      bias_initializer: Initialization scheme for biases in the backbone.
      name: Layer name.
      **kwargs: Forwarded to super.
    """
    super().__init__(name=name, **kwargs)
    self._moe = moe

    self._backbone = FeedForward(
        backbone_d_ff,
        dropout_rate=dropout_rate,
        activation=activation,
        kernel_initializer=tf_utils.clone_initializer(kernel_initializer),
        bias_initializer=tf_utils.clone_initializer(bias_initializer),
        name="backbone")

  def call(self,
           inputs: tf.Tensor,
           *,
           training: Optional[bool] = None) -> tf.Tensor:
    """Applies MoeLayerWithBackbone layer.

    Args:
      inputs: Batch of input embeddings of shape
        <float>[batch_size, seq_length, hidden_dim].
      training: Only apply dropout and jitter noise during training. If not
        provided taken from tf.keras.backend.

    Returns:
      Transformed inputs with same shape as inputs:
        <float>[batch_size, seq_length, hidden_dim].
    """
    return self._backbone(
        inputs, training=training) + self._moe(
            inputs, training=training)
