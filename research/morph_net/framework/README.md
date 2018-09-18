# Regularizers Framework


[TOC]


## Goal

The goal of this framework is to facilitate building sparsifying regularizers
for deep networks. A regularizer targets a certain cost (***targeted
cost***), such as the FLOP cost of inference, model size, latency, memory
footprint, etc.

In order to form such a regularizer, we traverse the TensorFlow graph and find
the ops that contribute to the *targeted cost*. For each op, we apply a
sparsifying regularizer that induces sparsity *among the activations*. The
sparsifying regularizer of each activation is weighted by its marginal
contribution to the *targeted cost*.

Calculating this weight may be a nontrivial task. For example, for a fully
connected layer the FLOP cost is proportional to the number of inputs times
the number of outputs, which means that the marginal cost of each output
is proportional to the number of inputs. Some of the inputs may have been
already regularized away, which means that the calculation of one op's FLOP
regularizer depends on the regularization of the output of other ops. Moreover,
if an op receives its input a concatenation or a sum of several other ops,
figuring out the regularizer requires some bookkeeping.

The goal of this framework is to take care of this bookkeeping in a general way,
to facilitate building a wide variety of regularizers, targeting a wide variety
of *targeted costs*, with little effort and less opportunities to err. In
what follows we outline the framework, building it from the bottom up: From a
single activation all the way to a full complex network.

## `OpRegularizers` and how they are assigned

### The `OpRegularizer` interface

`OpRegularizer` is the most primitive element in the framework. An
`OpRegularizer` refers to TensorFlow op, and has two methods,
`regularization_vector` and `alive_vector`, both return `tf.Tensor`s or rank 1
(vectors). `regularization_vector` is of type float, and its `i`-th entry is the
regularizer of the `i`-th activation of the op the `OpRegularizer` refers to.
In order to regularize away that activation, one would need to add the `i`-th
entry of `regularization_vector`, multiplied by some coefficient, to the
training loss. The stronger we want to penalize it, the larger the coefficient
is. Assuming that the regularizer is of sparsifying nature (e.g. L1 norm), with
a large enough coefficient, the `i`-th activation will eventually vanish.
Loosely speaking, if we were to target the total number of activations in the
network, we would add the sum of all `regularization_vector`s from all
`OpRegularizer` to the training loss.

Since `OpRegularizer` is an abstract interface, with no awareness of the nature
of regularization used, the decision when an activation can be considered alive
is also deferred to `OpRegularizer`, via the `alive_vector` method. The `i`-th
entry evaluates to a boolean that indicates whether the activation is alive.

```python
class OpRegularizer(object):

  @abc.abstractproperty
  def regularization_vector(self):
    """Returns a vector of floats with a regularizer for each activation."""
    pass

  @abc.abstractproperty
  def alive_vector(self):
    """Returns a bool vector indicating which activations are alive."""
    pass
```

As an example, we can consider a fully connected layer that has `m` inputs and
`n` outputs. The layer is represented by an `m * n` matrix, and one way to
impose sparsifying regularizer on the `i`-th output is by grouping all weights
associated with it into a group LASSO regularizer, such as the L2 norm of the
`i`-th row of the matrix. That would therefore be the `i`-th entry of the
`regularization_vector`.

When such a regularization is added to the training loss, the L2 norms of the
rows of the matrix tend to form a bimodal distribution with one peak near "zero"
(up to numerical noise), another peak away from zero, and a void in between. A
natural way to detemine whether the `i`-th activation is alive is thus by
comparing the `i`-th entry of the `regularization_vector` to some threshold that
lies in that void: If it's above the threshold, it's alive.

![HistogramOfActivationSTDs](../g3doc/histogram.png "Typical bimodal distribution of
the standatd deviations of the activations of a convolutional layer when a
sparsifying regularizer is applied.")

There are ops that are not regularized, such as constants, or the input to the
network. For an un-regularized op, the `OpRegularizer` is set to `None`, which
implies an all-zero `regularization_vector` and an all-True `alive_vector`.

### Rules for assigning `OpRegularizer`s to ops

As we traverse the TensorFlow graph, we assign an `OpRegularizer` to each op we
encounter according to the set of rules outlined in this section. We first
explain "default rules", rules that address propagating `OpRegularizers` across
connections in the TensorFlow graph. Then we discuss client-specified rules,
which can augment and override the default rules.

#### Pass-through ops

Many TensorFlow ops inherit the `OpRegularizer` of their input. These are ops
that:

* Don't change the alive status of activations.
* The only way an activation can be eliminated form their output is if
it's eliminated from their input.

An example is adding a bias to the output of a convolution. After adding a bias
to it, an activation will be alive (that is, have nonzero variance) if and only
if was alive before adding the bias. If we want to regularize away an activation
at the output of a `BiasAdd` op, the only way to do so is to penalize the same
activation in the preceding convolution.

Since both the `regularization_vector` and the `alive_vector` of such an op is
identical to those of its input, so is the entire `OpRegularizer`. We refer to
such ops as *pass-through* ops. Shape-preserving unary ops (e.g. ReLU) are
generally *pass-through*, but some binary ops are too. In our framework ops are
assumed to be *pass-through* by default. Exceptions to this rule are discussed
below.

#### Grouping

When learning the number of outputs of ops in a TensorFlow graph, some ops are
constrained to maintain the same number of outputs as others. Elementwise
ops that are performed on two (or more) tensors, such as addition,
multiplication, or maximum, constrain their input tensors to have the same size.

Common use cases are attention maps, recurrent models, and residual connections.
An example of a residual connection is illustrated in the diagram below. It
would be problematic if the activations of op1 and op2 didn't live or die
together. For example, if the `i`-th activation of op1 is alive but for op2 it's
dead, we still cannot eliminate the `i`-th activation from op2 without breaking
the topology of the network.

![ResidualConnections](../g3doc/grouping.png "Ops with residual connections"
)

In our framework we choose to impose preservation of the topology. That is, ops
that are connected with addition (or other elementwise binary ops) are
constrained to have their activations live and die together. The `i`-th
activations of each of those ops are grouped together in a single LASSO group.
The default grouping mechanism is maximum for the `regularization_vector` and
elementwise logical OR for the `alive_vector`. To regularize away the `i`-th
element of the group one needs to penalize the maximum of `i`-th regularization
terms of all ops comprising the group, and to declare the entire `i`-th group
dead, the `i`-th element in all ops comprising the group must be dead. However
the framework admits other forms of grouping, and user-defined grouping methods
can be easily plugged into it.

One property of the grouping, which may seem confusing initially, is that once
two (or more) `OpRegularizer`s are grouped, and the `OpRegularizer` of the
group is formed, the `OpRegularizer`s comprising the group are all 'replaced' by
the `OpRegularizer` of the group. For example, in the diagram above, the
`OpRegularizer`s of op1 and op2 have to be grouped. Therefore if the `i`-th
output of op1 is alive and that of op2 is dead, and we use the default grouping
described above, the `i`-th output of the group is *alive*.

Now, consider op4, which receives only op2 as input. From the point of view of
op4, the `i`-th activation of op2 must be considered *alive*, even though the
original op2 regularizer deemed it *dead*. This is because we already know that
we won't be able to do away with the `i`-th activation of op2 - it is tied to
the one of op1, which is alive. Therefore after the grouping, the
`OpRegularizer`s of all constituents of the group are henceforth *replaced* by
the `OpRegularizer` of the group.

#### Concatenation

Often outputs of several ops are concatenated to a single tensor. For example,
in Inception networks, the outputs of various convolutional 'towers' are
concatenated along the channels dimension. In such a case, it is obvious that
the `regularization_vector` (`alive_vector`) of the concatenation is a
concatenation of the `regularization_vector` (`alive_vector`) of the
concatenated ops.

Similarly to the logic of grouping, once the concatenation of the
`OpRegularizer`s has happened, the concatenated `OpRegularizer`s cease to exist
and are replaced by slices of their concatenation. For example if op1 has 3
outputs and op2 has 4, and op3 is their concatenation, op3 has 7 outputs. After
the concatenation, the `alive_vector` of op1 will be a slice (from index 0 to
index 2) of the `alive_vector` of op3, whereas for op2 it will be another slice
(index from 3 to index 6).

If op3 is later grouped with op4, as happens in Inception ResNet architectures,
a group will be formed, and the `alive_vector` of op1 will henceforth be a slice
(index from 0 to index 2) of the `alive_vector` of *the new group*. This is for
the same reasons as the ones described in the section above.

#### Client-specified rules

The client code of the framework has the opportunity to specify rules for
creating `OpRegularizers`. For example, for ops of type `MatMul`, which are the
common implementation of fully-connected layers, the client can choose to assign
group LASSO regularizers similar to the one described above. Typically the
client code would choose to do that for 'interesting' ops, like convolutions and
fully-connected layers, but the choice of rules is ultimately deferred to the
client code.

The client code may also choose to override the *default rules*. Ops are
considered *pass-through* by default, and obviously there are cases where this
is not true, such as reshaping, slicing, sparse maxtrix operations etc.
TensorFlow is much too expressive for us to be able to anticipate every usage
pattern of its ops and to properly regularize them. The set of default rules
cover most of the common published convolutional networks, but we do not presume
to cover *all* networks. More complex networks may require adding some custom
rules.


### OpRegularizerManager

`OpRegularizerManager` is the class responsible for assigning an `OpRegularizer`
to each op in the TensorFlow graph. Its constructor crawls the TensorFlow graph,
starting from the ops listed in the `ops` argument (typically the output of the
network), recursively, and assigns `OpRegularizer`s to each op encountered. Once
the object is constructed, it provides read-only methods that allow querying the
`OpRegularizer` for any op that was encountered during construction, and a list
of the latter ops for convenience.

```python
class OpRegularizerManager(object):
  """Assigns OpRegularizers to ops in a graph and bookkeeps the mapping."""

  def __init__(self, ops, op_regularizer_factory_dict,
               create_grouping_regularizer=None):
    """Creates an instance.

    Args:
      ops: A list of tf.Operation. An OpRegularizer will be created for all the
        ops in `ops`, and recursively for all ops they depend on via data
        dependency. Typically `ops` would contain a single tf.Operation, which
        is the output of the network.
      op_regularizer_factory_dict: A dictionary, where the keys are strings
        representing TensorFlow Op types, and the values are callables that
        create the respective OpRegularizers. For every op encountered during
        the recursion, if op.type is in op_regularizer_factory_dict, the
        respective callable will be used to create an OpRegularizer. The
        signature of the callables is the following args;
          op; a tf.Operation for which to create a regularizer.
          opreg_manager; A reference to an OpRegularizerManager object. Can be
            None if the callable does not need access to OpRegularizerManager.
      create_grouping_regularizer: A callable that has the signature of
        grouping_regularizers.MaxGroupingRegularizer's constructor. Will be
        called whenever a grouping op (see _GROUPING_OPS) is encountered.
        Defaults to MaxGroupingRegularizer if None.

    Raises:
      ValueError: If ops is not a list.
    """
    ...

  def get_regularizer(self, op):
    """Returns the OpRegularizer object pertaining to `op`.

    Args:
      op: a tf.Operation object.

    Returns:
      An OpRegularizer object, or None if `op` does not have one.

    Raises:
      ValueError: The OpRegularizerManager object did not encounter `op` when
        it was constructed and the grpah was traversed, and thus does not know
        the answer.
    """
    ...


  @property
  def ops(self):
    """Returns all tf.Operations for which `get_regularizer` is known."""
    ...

```
As the constructor crawls the graph, it invokes the following set of rules, for
any op encountered:

* If `op_regularizer_factory_dict` has a rule on how to create an
`OpRegularizer` for the type of the op encountered, invoke the rule. These
are the user-specified rules. Otherwise:
* If the op has no inputs, return `None`. Examples are constants and variables.
Otherwise:
* If the ops is concatenation, invoke the rule for concatenation decribed above.
Otherwise:
* If the op has more than one regularized input (that is, input that has a non-
`None` `OpRegularizer`, perform grouping. Being conservative, we first check if
the op is whitelisted for being a grouping op (elemetwise addition, subtraction
etc). Otherwise:
* The op is a *pass-through*. That is, its OpRegularizer is the same as of its
input.

The implementaiton is recursive: We start from the output nodes(s) of the graph.
To build an `OpRegularizer` for each op, we need to know the `OpRegularizer` of
its inputs, so we make a recursive call to find out those, and so on.

<!-- TODO: Explain how to change the grouping mechanism. -->

## Network Regularizers

A `NetworkRegularizer` object targets a certain *targeted cost* of an entire
network. Its interface is:

```python
class NetworkRegularizer(object):
  """An interface for Network Regularizers."""

  @abc.abstractmethod
  def get_regularization_term(self, ops=None):
    """Compute the FluidNet regularization term.

    Args:
      ops: A list of tf.Operation. If specified, only the regularization term
        associated with the ops in `ops` will be returned. Otherwise, all
        relevant ops in the default TensorFlow graph will be included.

    Returns:
      A tf.Tensor scalar of floating point type that evaluates to the
      regularization term.
    """
    pass

  @abc.abstractmethod
  def get_cost(self, ops=None):
    """Calculates the cost targeted by the Regularizer.

    Args:
      ops: A list of tf.Operation. If specified, only the cost pertaining to the
      ops in the `ops` will be returned. Otherwise, all relevant ops in the
      default TensorFlow graph will be included.

    Returns:
      A tf.Tensor scalar that evaluates to the cost.
    """
    pass
```

The TensorFlow scalar returned by `get_cost` evaluates to the *targeted
cost*, and is typically used for monitoring (e.g. displaying it in
TensorBoard). The scalar returned by `get_regularization_term` is the one that
has to be added to the training loss, multiplied by a coefficient controlling
its strength.

`OpRegularizerManager` and the `OpRegularizer`s it provides for ops in the graph
are intended to facilitate easy implementation of `NetworkRegularizer`s. We
exemplify it here in the context of targeting FLOPs for a convolutional network,
but the same principles apply for other *targeted costs*.

Most of the consumption of FLOPs in convolutional networks happens in the
convolutions. As a first approximation, we can neglect the FLOP impact of the
other ops in the graph, even though the framework readily allows including the
FLOP contribution of all ops, even the ones that have negligible cost.

Within this approximation, in order to build the FLOP `NetworkRegularizer`, its
constructor needs to:

* Crawl the graph, starting from the output of the network, and find all
convolution ops on which the output depends.
* For each of these convolution ops, create an `OpRegularizer`.
* Find the `OpRegularizer` of the *input* of each convolution op.
* Implement Eq. (6) in the [MorphNet paper](https://arxiv.org/abs/1711.06798) to
calculate the total FLOP cost of all convolutions, and an equation similar to
Eq. (9) to calcluate the respective regularization term. We say 'similar'
because Eq. (9) refers to a specific type of regularization, where the
`regularization_vector` of a convolution is the absolute value of the respective
batch-norm gamma vector. However the exact nature of the `regularization_vector`
is delegated to the `OpRegularizer`.
