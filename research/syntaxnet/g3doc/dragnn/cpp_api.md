# The DRAGNN C++ API & Multithreading Model

[TOC]

## Elements of DRAGNN

The DRAGNN framework allows its users to easily create, train, and use sets of
machine learning subsystems in a cohesive manner. To this end, the DRAGNN C++
code is centered around three main types of objects: Components, which represent
ML subsystems; TransitionStates, which hold the state of a single inference; and
ComputeSessions, which represent the overall state of a DRAGNN computation.

### Components and TransitionStates

The fundamental concept behind DRAGNN is that it allows users to compose sets of
ML subsystems in a single computation. Each of those subsystems is represented
in the DRAGNN framweork by a pair of classes - a Component subclass and a
TransitionState subclass. While the Component will generally contain the logic
code for the subsystem, the state of a computation should be held in a
TransitionState instance. This way, a batch of inferences can be represented by
a set of TransitionStates that are acted on by the Component. As an example, the
DRAGNN SyntaxNet backend has two parts - a SyntaxNetComponent and a
SyntaxNetTransitionState. Here, the SyntaxNetComponent owns the transition
system, feature extractor, and so forth, but the actual parser state for each
sentence being examined is contained in the SyntaxNetTransitionState.

Note that the actual /inference/ is not done by the Component or
TransitionState - Tensorflow handles that. The Component and TransitionState are
instead responsible for holding state that would otherwise be difficult or
impossible to represent with TensorFlow idioms.

Components also are responsible for keeping beam history (if applicable) - that
is, the location of a given state in the beam at a given step - but a helper
Beam class is provided if a beam is needed.

Finally, Components can also publish a set of Component-specific translators
(see [TODO: link]), which can provide additional data to Components that execute
later in the computation. For the SyntaxNetComponent, for instance, we provide
translators like 'reduce-step', which returns the data associated with the
parent of the requested token.

### Translators

One of the most powerful and flexible features of DRAGNN is the ability to use
activations from previous components as inputs to the current inference. This is
done through Translator functions. Translator functions take a desired value -
for example, a token index in a previous component - and convert that value into
the batch, beam, and step indices that correspond to the location of the
Tensorflow data for that value in that component These translations can be
straightforward - the 'identity(N)' translator, for instance, returns the inputs
to the inference performed on that transition state's data in the previous
component at step N - or complex, like the
'parent-shift-reduce-step(N)'translator provided by the SyntaxNet component,
which returns the data from the reduction step performed on the parent token of
the token at step N.

Some translators are 'universal' - that is, they apply to any component.
However, Components may declare a set of translators that they can provide that
go beyond the universal translators. These backend-specific translators often
provide richer access to data, or access in more meaningful ways.

For a complete list of translators provided in the DRAGNN baseline, please see
[TODO: LINK ME].

### ComputeSession

The ComputeSession contains the state of a single DRAGNN computation. It holds
local, independent copies of Component objects, TransitionStates, and input
data, making it a completely independent container for that computation. The
ComputeSession object is also the basic API layer for DRAGNN - most external
computation should use its interface rather than diving deeper into Components
and TransitionStates.

ComputeSessions are created by ComputeSessionPools, which will handle all the
relevant initialization and setup tasks for that ComputeSession. After a
computation is complete, the ComputeSession should be returned to the Pool that
created it; failing to do this will not leak resources, but will also cause the
ComputeSessionPool to allocate more ComputeSessions than are necessary.

### ComputeSessionPool

The ComputeSessionPool is a constructor, initializer, and storage object for
ComputeSessions. When a pool is created, it is passed a MasterSpec specification
that describes the DRAGNN graph that will be computed; when a ComputeSession is
requested, the ComputeSessionPool will return one that is set up to compute
based on the pool's MasterSpec.

When a DRAGNN computation is complete and the ComputeSession is no longer
needed, the calling code should return the ComputeSession to the
ComputeSessionPool; the pool will reset its internal state and reuse it. If this
is complete, it limits the number of extant ComputeSessions to the number of
parallell computations.

The ComputeSessionPool is threadsafe; DRAGNN multithreading generally should
take place at the ComputeSession level (as discussed below).

## The ComputeSession API

NOTE: All of the functionality described below is already wrapped in Tensorflow
ops. This is for developers who want C++ access to the DRAGNN framework.

The core elements of the ComputeSession API are intended to allow calling code
to collate a set of inputs for a Tensorflow inference using the internal state
of a Component, take the result of that inference and feed it back into the
Component so that the Component can advance its internal state, and then repeat
this process until the Component indicates that it no longer needs to advance.
This process is repeated for all Components, and when no Components are left the
computation is complete.

The key features of the API are as follows. Note that most functions are keyed
on the component name - this is the string 'ComponentSpec.name' from the
ComponentSpec message that describes the component in question in the
MasterSpec.

### Data Extraction Functions

The following functions allow the caller to extract data from the internal state
of a Component.

```
int GetInputFeatures(
      const string &component_name,
      std::function<int32 *(int num_items)> allocate_indices,
      std::function<int64 *(int num_items)> allocate_ids,
      std::function<float *(int num_items)> allocate_weights,
      int channel_id) const
```

GetInputFeatures extracts the fixed features from the given component for the
given channel ID and places them in the memory allocated by the allocator
functions. These allocator functions should take an int representing how many
elements will be extracted, allocate backing memory to hold that many elements,
and return a pointer. (This is used to wrap Tensorflow tensor allocation code
and efficiently extract large amounts of data in the current op kernels).

```
int BulkGetInputFeatures(const string &component_name,
                         const BulkFeatureExtractor &extractor);
```

BulkGetInputFeatures extracts all fixed features, advances the Component via the
oracle, and repeats the process until the component is terminal. This is
intended to efficiently extract features (like, for instance, word embeddings).
The passed BulkFeatureExtractor object contains allocator functions and
formatting functions required to correctly lay out the data that is being
extracted.

```
std::vector<LinkFeatures> GetTranslatedLinkFeatures(
      const string &component_name, int channel_id)
```

GetTranslatedLinkFeatures extracts /linked/ features for the given component and
channel ID. Linked features are indices into previous components' data tensors;
this function call will extract the raw data, translate it via the relevant
Translator call, and return a set of filled-out LinkFeatures protos that
indicate how to access the relevant data.

### Component Advancement Functions

The following functions allow their caller to advance the state of a given
Component.

```
void AdvanceFromOracle(const string &component_name)

```

The AdvanceFromOracle function advances the given component one step, according
to whatever oracle it has.

```
AdvanceFromPrediction(const string &component_name,
                                     const float score_matrix[],
                                     int score_matrix_length)
```

Advances the given component based on the given matrix of scores. The scores are
generally the outputs of a neural net, and should be padded to (batch size)x
(max beam size), ordered so that each beam element's scores are contiguous.

```
bool IsTerminal(const string &component_name)
```

Returns true if all batch items in the given component report that they are
final.

### Computation Advancement Functions

```
void SetInputData(const std::vector<string> &data)
```

Passes a set of input data to the ComputeSession. Input data should be in the
form of serialized protobuf messages (as can be seen in the graph builder test).
The components that were used to construct the graph will have a certain type of
proto that they expect - for instance, the SyntaxNetComponent expects Sentence
protos - and will attempt to deserialize the input data into that form. Each
element in the vector will become a batch element; batch sizes should be
controlled by limiting the number of items passed into this function.

```
void InitializeComponentData(const string &component_name,
                                       int max_beam_size)
```

This function performs setup tasks on the given component, and requests that it
set itself up using the given beam size. If there is a previous component, the
previous component's final transition states will be passed to the requested
component at this time.

```
void FinalizeData(const string &component_name)
```

This function essentially "completes" a component, forcing it to write out its
current best prediction to the backing data for other components to use. This
function should always be called on a component before calling
InitializeComponentData on the next component in the sequence.

```
std::vector<string> GetSerializedPredictions()
```

This function completes a computation, taking the current set of predictions and
data, re-serializing them, and returning them as output. Be sure that
FinalizeData has been called on all components before this function is called,
or the output predictions will be incomplete. Once this function has been
called, the ComputeSession is done and can be returned to the pool - it has no
further use. (ComputeSessions can be returned early, as well, but they will be
reset and their internal state will be lost.)

## The ComputeSessionPool API

NOTE: All of the functionality described below is already wrapped in Tensorflow
ops. This is for developers who want C++ access to the DRAGNN framework.

The ComputeSessionPool creates and manages completed ComputeSessions. It has
functions to provide and take ownership of ComputeSession objects.

```
ComputeSessionPool(const MasterSpec &master_spec,
                     const GridPoint &hyperparams)
```

This constructs a CommputeSessionPool with the associated MasterSpec and
hyperparameter grid point. All ComputeSessions that this pool will create will
perform computations according to this master spec. If the master spec must be
changed, then destroy this pool and create another one.
`std::unique_ptr<ComputeSession> GetSession();` Creates (or reuses, if possible)
a ComputeSession based on the pool's master spec. If the MasterSpec is
ill-formed, this method will CHECK-fail. Ownership of the compute session is
passed to the caller; for efficiency, the owned pointer should be returned to
the pool via ReturnSession().

```
void ReturnSession(std::unique_ptr<ComputeSession> session);
```

Returns a ComputeSession object to the pool. This ComputeSession object will be
reset and re-used.

## Multithreading

DRAGNN was designed to be thread-safe at the ComputeSession level. It should be
fully safe to have multiple threads running computations, pulling
ComputeSessions from the same ComputeSessionPool and executing them
independently. Const methods in the ComputeSession are also thread-safe, so (for
instance) multiple threads could be used to extract each channel's fixed
features (and this does happen at the Tensorflow level) - but only one thread
could be used when calling BulkFixedFeatures.
