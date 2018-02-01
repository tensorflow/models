# Creating a DRAGNN Component

[TOC]

## Why Create A Component?

A Component is the logic that performs actions based on Tensorflow inference
results. Because DRAGNN allows downstream components to access data from
Components that have already run - both during training and during inference -
wrapping your inference-using logic in a Component can allow it to be trained
and used with inferred data as inputs and have downstream units use its data as
an input during training (and inference). This doesn't even need to be a
linguistic task - any neural-net computation could be wrapped as a Component.

In addition, Components can be created to very efficiently perform a small set
of actions - for instance, computation with a Beam can be expensive, so if your
application does not need a Beam, you could create a component without one and
save its computational overhead.

## The Component Interface

All Components must implement the Component interface (located in
dragnn/core/interfaces/component.h). Of special note are the translator access
functions, which must be implemented for downstream components to correctly
examine the component's data.

These functions are:

```
int StepsTaken(int batch_index)
```

This function should return the number of steps taken by the Component as it has
operated on batch index "batch_index". Since each Component can operate on many
data items in parallel, some of which may become final earlier than others, it
is necessary to specify the batch index here. Also note that batch indices are
assumed to be constant throughout the DRAGNN system and between components -
batch i is the computation that corresponds to element i of the vector of input
data, always.

```
int GetBeamIndexAtStep(int step, int current_index, int batch)
```

This function should look up the TransitionState that is currently at element
'index' in the state's beam for batch element 'batch', then determine where that
element was in that batch element's beam at step 'step'. If it's out of bounds,
then return -1.

```
int GetSourceBeamIndex(int current_index, int batch)
```

This function should return the 'source beam index' of the TransitionState
currently at element 'current index' for batch element 'batch'. To find the
source beam index, first determine what beam index the element was in at time 0
(at initialization). Then, determine the element *from the previous component*
was used to initialize that element; that element's index is the source beam
index. (This is used to maintain beam history throughout DRAGNN).

### When Is A Component Terminal?

One of the important concepts for a Components is the idea of "being terminal".
A Component is terminal when all of its batch elements are completely analyzed-
that is, no computation is left to perform for any element. For a parser, for
instance, this occurs when all tokens have been examined.

Components must always become terminal after some number of steps; if not,
DRAGNN will become caught in an infinite loop. The number of steps does not have
to be deterministic, however.

### Defining Component Input

A DRAGNN graph is fed with strings, and each Component determines how to
interpret them. For example, the SyntaxNetComponent expects each string to be a
serialized Sentence protocol buffer. (If you want to operate on Sentence
objects, it probably makes sense to add more `syntaxnet::TransitionSystem`
classes than write a new Component from scratch.)

If you want to to read other types of data, you will need to sub-class
InputBatchCache, a container which holds the strings and deserializes them into
typed data on demand. Usage is fairly straightforward; see the SyntaxNet
implementation for how to use it.

**Note: all Component's in the graph should use a single InputBatchCache
sub-class. If you need multiple data types, you'll want to have a single
InputBatchCache that has fields for each of your data types.**

## Using Transition States

Each Component is expected to intialize itself from a beam of TransitionStates
(which may be empty, if the component is the first one in a DRAGNN computation)
and is expected to emit a set of TransitionStates when it is complete (via a
call to GetBeam). There is no requirement that TransitionStates be used
internally, but it does make things easier (for one thing, if you use
CloneableTransitionStates, you can use the provided Beam class to track your
component history and beam state).

### Initializing From TransitionStates

When your component is initialized, it will receive a vector of vectors of
TransitionStates. The external vector is the batch index and the internal vector
is in beam state order - so you should use the internal vector at index i to
initialize the beam for batch element's i. When initializing new
TransitionStates from the internal vector's states, be sure to record the index
of that element in the input beam somewhere; you'll need to be able to return it
for the GetSourceBeamIndex function. If you don't, translations won't work
properly.

### Finalizing & Emitting TransitionState Data

When your component is terminal, the ComputeSession will call FinalizeData() and
GetBeam() in that order. These steps "lock" the component computation and return
pointers (note the use of raw pointers - components always retain ownership of
their own transition states!) to the final state of the component as captured in
its TransitionStates.

When data is finalized, the best (in our case, the highest scoring, but you may
have a different metric) result is written to the underlying data that was
passed to the component. Writing the result back to the data will ensure that
components later in the pipeline can use that result in their computations.

Emitting data is straightforward: create a vector of vector of pointers to the
beam states in your component, and return it.

### Basic Testing

If you would like to validate that your TransitionState meets the DRAGNN
contract expectations, you can use the transition_state_starter_test.cc, and
adapt it to use your transition state. If all tests pass, your TransitionState
should work with the rest of DRAGNN.

## Creating Translators

One of the key features of DRAGNN is "translators" - functions that allow
components executing later in a DRAGNN pipeline to access data from earlier
components. There are two types of translators - universal and backend-specific.

### Maintaining History For Universal Translators

In order to support translation, your component must be able to report history
via the Component interface. (This is required whether you want to directly
support translation in your component or not - in order for translation to work
at all in the DRAGNN pipeline, all components must implement these methods).

### Defining Backend Specific Translators

If you would like downstream components to be able to access your component's
data in a more complex manner than the universal translators allow, you can
define your own translation functions. This is done via the
`GetStepLookupFunction` method in the component, which returns an arbitrary
function when given a string name. To create a translator with a specific name,
make the `GetStepLookupFunction` method return it when queried; once done, that
name can be specified as a LinkedFeature for components downstream.

Note that, when defining a backend-specific translator, the arguments to the
returned std::function are (int batch_index, int beam_index, int value), or,
"for the transition state at the given beam_index of the beam corresponding to
batch batch_index, return a function mapping value to a location in the input
tensor for this step."

## Beams & The Beam Class

If you would like a helper function to keep track of beams and history for your
component, you can use core/beam.h - make sure your TransitionState implements
the CloneableTransitionState interface, instantiate a beam (templated to your
TransitionState type) for each batch element, and you're good to go - all you
need to do is plumb the relevant methods from the Component interface to the
Beam. (For an example, you can examine syntaxnet_component.cc, which uses a Beam
of SyntaxNetTransitionState objects.
