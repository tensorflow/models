# Copyright 2025 The TensorFlow Authors. All Rights Reserved.
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
from typing import Callable, Optional, Tuple

import tensorflow as tf, tf_keras

from official.modeling import tf_utils


_InitializerType = tf_keras.initializers.Initializer


_DEFAULT_KERNEL_INITIALIZER = tf_keras.initializers.TruncatedNormal(stddev=2e-2)
_DEFAULT_BIAS_INITIALIZER = tf_keras.initializers.Zeros()


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
  num_groups = tf.shape(router_logits)[0]
  tokens_per_group = router_logits.shape[1]

  log_z = tf.math.reduce_logsumexp(router_logits, axis=-1)
  z_loss = log_z**2
  return tf.math.reduce_sum(z_loss) / tf.cast(
      num_groups * tokens_per_group, tf.float32)


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


class Router(tf_keras.layers.Layer):
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
      router_z_loss_weight: float = 0.0,
      export_metrics: bool = True,
      name: str = "router",
      **kwargs):
    """Init.

    Args:
      num_experts: Number of experts.
      jitter_noise: Amplitude of jitter noise applied to router logits.
      use_bias: Whether or not to use the bias term in computing the router
        weights.
      kernel_initializer: Kernel initializer for router weights.
      bias_initializer: Bias initializer for router weights.
      router_z_loss_weight: Weight for router_z_loss. Use non-zero values if
        running into training instability (esp. with dtype 'bfloat16' or lower).
      export_metrics: Whether to export metrics using Keras add_metric API.
      name: Layer name.
      **kwargs: Forwarded to super.
    """
    super().__init__(name=name, **kwargs)

    self.num_experts = num_experts  # Used to check consistency with
                                    # FeedForwardExperts.
    self.jitter_noise = jitter_noise
    self.router_z_loss_weight = router_z_loss_weight
    self._export_metrics = export_metrics

    self.router_weights = tf_keras.layers.Dense(
        num_experts,
        use_bias=use_bias,
        kernel_initializer=tf_utils.clone_initializer(kernel_initializer),
        bias_initializer=tf_utils.clone_initializer(bias_initializer),
        name="router_weights",
        dtype=tf.float32)

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
        taken from tf_keras.backend.

    Returns:
      Router indices or mask arrays (depending on router type).
    """
    if training is None:
      training = tf_keras.backend.learning_phase()

    # inputs shape <float>[num_groups, tokens_per_group, hidden_dim]
    router_probs, router_logits = self._compute_router_probabilities(
        inputs, apply_jitter=training)
    # router_probs <float32>[num_groups, tokens_per_group, num_experts]
    # router_logits <float>[num_groups, tokens_per_group, num_experts]
    unscaled_router_z_loss = _router_z_loss(router_logits)
    router_z_loss = self.router_z_loss_weight * unscaled_router_z_loss
    self.add_loss(router_z_loss)
    if self._export_metrics:
      self.add_metric(unscaled_router_z_loss, name="unscaled_router_z_loss")
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
          tf.shape(inputs),
          minval=1.0 - self.jitter_noise,
          maxval=1.0 + self.jitter_noise,
          dtype=inputs.dtype)
    # inputs <float>, router_logits <float32>
    router_logits = self.router_weights(inputs)
    router_probs = tf_keras.activations.softmax(router_logits, axis=-1)
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
    num_groups = tf.shape(router_probs)[0]
    tokens_per_group = router_probs.shape[1]

    router_probs_t = tf.transpose(router_probs, perm=[0, 2, 1])
    # router_probs_t: <float32>[num_groups, num_experts, tokens_per_group]
    # Top expert_capacity router probability and corresponding token indices for
    # each expert.
    # Shapes [num_groups, num_experts, expert_capacity]
    _, expert_index = tf.math.top_k(
        router_probs_t, k=expert_capacity, sorted=False)

    # Convert to one-hot mask of expert indices for each token in each group.
    # Shape: [num_groups, tokens_per_group, num_experts, expert_capacity].
    dispatch_mask = tf.one_hot(
        expert_index, tokens_per_group, axis=1, dtype=router_probs.dtype)

    # The combine array will be used for combining expert outputs, scaled by the
    # router probabilities.
    # Shape: [num_groups, num_experts, tokens_per_group, expert_capacity]
    combine_array = tf.expand_dims(router_probs, axis=3) * dispatch_mask

    # Add load balancing loss.
    # Each expert is choosing tokens until it reaches full capacity, so we don't
    # need an auxiliary loading balancing loss for expert choice routing.
    if self._export_metrics:
      self.add_metric(0.0, name="load_balancing_loss")

      # Gather expert metrics.
      # Number of tokens that were dispatched to at least one expert.
      num_tokens = num_groups * tokens_per_group
      num_tokens_dispatched_somewhere = tf.math.reduce_sum(tf.math.reduce_max(
          dispatch_mask, axis=(-1, -2)))
      fraction_tokens_left_behind = 1.0 - tf.cast(
          num_tokens_dispatched_somewhere, tf.float32) / tf.cast(
              num_tokens, tf.float32)

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
    dispatch_mask = tf.cast(dispatch_mask, self.compute_dtype)
    combine_array = tf.cast(combine_array, self.compute_dtype)
    output = RouterMask(dispatch_mask, combine_array)
    return output


################## Model layers ##################


class FeedForward(tf_keras.layers.Layer):
  """Feed-forward layer - position independent, dense, nonlinear transformation.

  Typically used in an MLP Transformer block.
  """

  def __init__(
      self,
      d_ff: int,
      *,
      inner_dropout: float = 0.0,
      output_dropout: float = 0.0,
      activation: Callable[[tf.Tensor], tf.Tensor] = tf_keras.activations.gelu,
      kernel_initializer: _InitializerType = _DEFAULT_KERNEL_INITIALIZER,
      bias_initializer: _InitializerType = _DEFAULT_BIAS_INITIALIZER,
      name: str = "feed_forward",
      **kwargs):
    """Initializes layer.

    Args:
      d_ff: Dimension of feed-forward layer.
      inner_dropout: The dropout probability to be applied after intermediate
        activations.
      output_dropout: The dropout probability to be applied after output layer.
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

    self.intermediate_layer = tf_keras.layers.Dense(
        d_ff,
        kernel_initializer=tf_utils.clone_initializer(self.kernel_initializer),
        bias_initializer=tf_utils.clone_initializer(self.bias_initializer),
        name="intermediate")
    self.inner_dropout_layer = tf_keras.layers.Dropout(
        inner_dropout)
    self.output_dropout_layer = tf_keras.layers.Dropout(output_dropout)

  def build(self, input_shape: Tuple[int, int, int]):
    """Creates the input shape dependent output weight variables."""
    self.output_layer = tf_keras.layers.Dense(
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
    x = self.inner_dropout_layer(x, training=training)
    x = self.output_layer(x)
    x = self.output_dropout_layer(x, training=training)
    return x


class FeedForwardExperts(tf_keras.layers.Layer):
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
      inner_dropout: float = 0.0,
      output_dropout: float = 0.0,
      activation: Callable[[tf.Tensor], tf.Tensor] = tf_keras.activations.gelu,
      kernel_initializer: _InitializerType = _DEFAULT_KERNEL_INITIALIZER,
      bias_initializer: _InitializerType = _DEFAULT_BIAS_INITIALIZER,
      name: str = "experts",
      **kwargs):
    """Initializes layer.

    Args:
      num_experts: Number of experts (i.e. number of independent feed-forward
        blocks).
      d_ff: Dimension of feed-forward layer of each expert.
      inner_dropout: The dropout probability to be applied after intermediate
        activations.
      output_dropout: The dropout probability to be applied after output layer.
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

    self.intermediate_layer = tf_keras.layers.EinsumDense(
        "gech,ehf->gecf",
        output_shape=(self.num_experts, None, d_ff),
        bias_axes="ef",
        kernel_initializer=tf_utils.clone_initializer(self.kernel_initializer),
        bias_initializer=tf_utils.clone_initializer(self.bias_initializer),
        name="intermediate")
    self.inner_dropout_layer = tf_keras.layers.Dropout(
        inner_dropout)
    self.output_dropout_layer = tf_keras.layers.Dropout(output_dropout)

  def build(self, input_shape: Tuple[int, int, int, int]):
    """Creates the input shape dependent output weight variables."""
    if input_shape[1] != self.num_experts:
      raise ValueError(
          f"Input shape {input_shape} is inconsistent with num_experts "
          f"{self.num_experts}.")

    self.output_layer = tf_keras.layers.EinsumDense(
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
    x = self.inner_dropout_layer(x, training=training)
    x = self.output_layer(x)
    x = self.output_dropout_layer(x, training=training)
    return x


class MoeLayer(tf_keras.layers.Layer):
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
      examples_per_group: float = 1.0,
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
      examples_per_group: Number of examples to form a group. Router then
        performs top_k token selection for each expert on a per group basis.
        E.g. when `examples_per_group=4.0`, tokens are assigned to experts in
        groups formed from 4 examples. When `examples_per_group=0.5`,
        each example is split into 2 groups.
        `examples_per_group` must divide the local batch size.
        A larger group size will result in slower but more accurate top-k and
        sorting computations, whereas a smaller group size will result in faster
        but more approximate (and potentially less stable) routing choices.
        In practice, we find that imperfect routing choices are tolerable and
        recommend choosing a group size on the order of 4096 tokens, although
        this number will vary based on model configuration and size.
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
    self._examples_per_group = examples_per_group

  def call(self,
           inputs: tf.Tensor,
           *,
           training: Optional[bool] = None) -> tf.Tensor:
    """Applies MoeLayer.

    Args:
      inputs: Batch of input embeddings of shape
        <float>[batch_size, seq_length, hidden_dim].
      training: Only apply dropout and jitter noise during training. If not
        provided taken from tf_keras.backend.

    Returns:
      Transformed inputs with same shape as inputs:
        <float>[batch_size, seq_length, hidden_dim].

    Raises:
      ValueError if we cannot find a group_size satisfying given requirements.
    """
    if training is None:
      training = tf_keras.backend.learning_phase()

    # inputs shape [batch_size, seq_length, hidden_dim]
    batch_size, seq_length, hidden_dim = inputs.shape
    if batch_size is not None:
      if self._examples_per_group > batch_size:
        raise ValueError(
            f"examples_per_group={self._examples_per_group} is larger than the "
            "number of examples available in the local (per-device) batch_size="
            f"{batch_size}. Either decrease examples_per_group or increase the "
            "batch_size.")
    tokens_per_group = int(seq_length * self._examples_per_group)

    if training:
      capacity_factor = self._train_capacity_factor
    else:
      capacity_factor = self._eval_capacity_factor
    # Each group will send expert_capacity tokens to each expert.
    expert_capacity = int(
        round(capacity_factor * tokens_per_group / self.num_experts))

    # Reshape batch and sequence/token dimensions for expert routing.
    x = tf.reshape(inputs, (-1, tokens_per_group, hidden_dim))

    x = self._mask_and_dispatch_to_experts(x, expert_capacity, training)

    # Return to original input shape.
    x = tf.reshape(x, (-1, seq_length, hidden_dim))
    return x

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
        "gtec,gth->gech",
        router_mask.dispatch_mask,
        inputs)

    expert_outputs = self._experts(expert_inputs, training=training)

    # Shape [num_groups, tokens_per_group, hidden_dim]
    combined_outputs = tf.einsum(
        "gtec,gech->gth",
        router_mask.combine_array,
        expert_outputs)

    return combined_outputs


class MoeLayerWithBackbone(tf_keras.layers.Layer):
  """Sparse MoE layer plus a FeedForward layer evaluated for all tokens.

  Uses Keras add_loss() and add_metric() APIs.
  """

  def __init__(
      self,
      moe: MoeLayer,
      backbone_d_ff: int,
      *,
      inner_dropout: float = 0.0,
      output_dropout: float = 0.0,
      activation: Callable[[tf.Tensor],
                           tf.Tensor] = tf_keras.activations.gelu,
      kernel_initializer: _InitializerType = _DEFAULT_KERNEL_INITIALIZER,
      bias_initializer: _InitializerType = _DEFAULT_BIAS_INITIALIZER,
      name: str = "moe_with_backbone",
      **kwargs):
    """Init.

    Args:
      moe: Instance of MoeLayer with experts and router.
      backbone_d_ff: Dimension of feed-forward layer of a lightweight backbone,
        which is evaluated for all tokens.
      inner_dropout: The dropout probability to be applied after intermediate
        activations for the backbone.
      output_dropout: The dropout probability to be applied after the output
        of the backbone.
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
        inner_dropout=inner_dropout,
        output_dropout=output_dropout,
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
        provided taken from tf_keras.backend.

    Returns:
      Transformed inputs with same shape as inputs:
        <float>[batch_size, seq_length, hidden_dim].
    """
    return self._backbone(
        inputs, training=training) + self._moe(
            inputs, training=training)
