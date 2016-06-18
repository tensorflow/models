# TensorFlow-Slim

TF-Slim is a lightweight library for defining, training and evaluating models in
TensorFlow. It enables defining complex networks quickly and concisely while
keeping a model's architecture transparent and its hyperparameters explicit.

[TOC]

## Teaser

As a demonstration of the simplicity of using TF-Slim, compare the simplicity of
the code necessary for defining the entire [VGG]
(http://www.robots.ox.ac.uk/~vgg/research/very_deep/) network using TF-Slim to
the lengthy and verbose nature of defining just the first three layers (out of
16) using native tensorflow:

```python{.good}
# VGG16 in TF-Slim.
def vgg16(inputs):
  with slim.arg_scope([slim.ops.conv2d, slim.ops.fc], stddev=0.01, weight_decay=0.0005):
    net = slim.ops.repeat_op(2, inputs, slim.ops.conv2d, 64, [3, 3], scope='conv1')
    net = slim.ops.max_pool(net, [2, 2], scope='pool1')
    net = slim.ops.repeat_op(2, net, slim.ops.conv2d, 128, [3, 3], scope='conv2')
    net = slim.ops.max_pool(net, [2, 2], scope='pool2')
    net = slim.ops.repeat_op(3, net, slim.ops.conv2d, 256, [3, 3], scope='conv3')
    net = slim.ops.max_pool(net, [2, 2], scope='pool3')
    net = slim.ops.repeat_op(3, net, slim.ops.conv2d, 512, [3, 3], scope='conv4')
    net = slim.ops.max_pool(net, [2, 2], scope='pool4')
    net = slim.ops.repeat_op(3, net, slim.ops.conv2d, 512, [3, 3], scope='conv5')
    net = slim.ops.max_pool(net, [2, 2], scope='pool5')
    net = slim.ops.flatten(net, scope='flatten5')
    net = slim.ops.fc(net, 4096, scope='fc6')
    net = slim.ops.dropout(net, 0.5, scope='dropout6')
    net = slim.ops.fc(net, 4096, scope='fc7')
    net = slim.ops.dropout(net, 0.5, scope='dropout7')
    net = slim.ops.fc(net, 1000, activation=None, scope='fc8')
  return net
```

```python{.bad}
# Layers 1-3 (out of 16) of VGG16 in native tensorflow.
def vgg16(inputs):
  with tf.name_scope('conv1_1') as scope:
    kernel = tf.Variable(tf.truncated_normal([3, 3, 3, 64], dtype=tf.float32, stddev=1e-1), name='weights')
    conv = tf.nn.conv2d(inputs, kernel, [1, 1, 1, 1], padding='SAME')
    biases = tf.Variable(tf.constant(0.0, shape=[64], dtype=tf.float32), trainable=True, name='biases')
    bias = tf.nn.bias_add(conv, biases)
    conv1 = tf.nn.relu(bias, name=scope)
  with tf.name_scope('conv1_2') as scope:
    kernel = tf.Variable(tf.truncated_normal([3, 3, 64, 64], dtype=tf.float32, stddev=1e-1), name='weights')
    conv = tf.nn.conv2d(images, kernel, [1, 1, 1, 1], padding='SAME')
    biases = tf.Variable(tf.constant(0.0, shape=[64], dtype=tf.float32), trainable=True, name='biases')
    bias = tf.nn.bias_add(conv, biases)
    conv1 = tf.nn.relu(bias, name=scope)
  with tf.name_scope('pool1')
    pool1 = tf.nn.max_pool(conv1, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='VALID', name='pool1')
```

## Why TF-Slim?

TF-Slim offers several advantages over just the built-in tensorflow libraries:

*   Allows one to define models much more compactly by eliminating boilerplate
    code. This is accomplished through the use of [argument scoping](scopes.py)
    and numerous high level [operations](ops.py). These tools increase
    readability and maintainability, reduce the likelihood of an error from
    copy-and-pasting hyperparameter values and simplifies hyperparameter tuning.
*   Makes developing models simple by providing commonly used [loss functions]
    (losses.py)
*   Provides a concise [definition](inception.py) of [Inception v3]
    (http://arxiv.org/abs/1512.00567) network architecture ready to be used
    out-of-the-box or subsumed into new models.

Additionally TF-Slim was designed with several principles in mind:

*   The various modules of TF-Slim (scopes, variables, ops, losses) are
    independent. This flexibility allows users to pick and choose components of
    TF-Slim completely à la carte.
*   TF-Slim is written using a Functional Programming style. That means it's
    super-lightweight and can be used right alongside any of TensorFlow's native
    operations.
*   Makes re-using network architectures easy. This allows users to build new
    networks on top of existing ones as well as fine-tuning pre-trained models
    on new tasks.

## What are the various components of TF-Slim?

TF-Slim is composed of several parts which were designed to exist independently.
These include:

*   [scopes.py](./scopes.py): provides a new scope named `arg_scope` that allows
    a user to define default arguments for specific operations within that
    scope.
*   [variables.py](./variables.py): provides convenience wrappers for variable
    creation and manipulation.
*   [ops.py](./ops.py): provides high level operations for building models using
    tensorflow.
*   [losses.py](./losses.py): contains commonly used loss functions.

## Defining Models

Models can be succinctly defined using TF-Slim by combining its variables,
operations and scopes. Each of these elements are defined below.

### Variables

Creating [`Variables`](https://www.tensorflow.org/how_tos/variables/index.html)
in native tensorflow requires either a predefined value or an initialization
mechanism (random, normally distributed). Furthermore, if a variable needs to be
created on a specific device, such as a GPU, the specification must be [made
explicit](https://www.tensorflow.org/how_tos/using_gpu/index.html). To alleviate
the code required for variable creation, TF-Slim provides a set of thin wrapper
functions in [variables.py](./variables.py) which allow callers to easily define
variables.

For example, to create a `weight` variable, initialize it using a truncated
normal distribution, regularize it with an `l2_loss` and place it on the `CPU`,
one need only declare the following:

```python
weights = variables.variable('weights',
                             shape=[10, 10, 3 , 3],
                             initializer=tf.truncated_normal_initializer(stddev=0.1),
                             regularizer=lambda t: losses.l2_loss(t, weight=0.05),
                             device='/cpu:0')
```

In addition to the functionality provided by `tf.Variable`, `slim.variables`
keeps track of the variables created by `slim.ops` to define a model, which
allows one to distinguish variables that belong to the model versus other
variables.

```python
# Get all the variables defined by the model.
model_variables = slim.variables.get_variables()

# Get all the variables with the same given name, i.e. 'weights', 'biases'.
weights = slim.variables.get_variables_by_name('weights')
biases = slim.variables.get_variables_by_name('biases')

# Get all the variables in VARIABLES_TO_RESTORE collection.
variables_to_restore = tf.get_collection(slim.variables.VARIABLES_TO_RESTORE)


weights = variables.variable('weights',
                             shape=[10, 10, 3 , 3],
                             initializer=tf.truncated_normal_initializer(stddev=0.1),
                             regularizer=lambda t: losses.l2_loss(t, weight=0.05),
                             device='/cpu:0')
```

### Operations (Layers)

While the set of TensorFlow operations is quite extensive, builders of neural
networks typically think of models in terms of "layers". A layer, such as a
Convolutional Layer, a Fully Connected Layer or a BatchNorm Layer are more
abstract than a single TensorFlow operation and typically involve many such
operations. For example, a Convolutional Layer in a neural network is built
using several steps:

1.  Creating the weight variables
2.  Creating the bias variables
3.  Convolving the weights with the input from the previous layer
4.  Adding the biases to the result of the convolution.

In python code this can be rather laborious:

```python
input = ...
with tf.name_scope('conv1_1') as scope:
  kernel = tf.Variable(tf.truncated_normal([3, 3, 64, 128], dtype=tf.float32,
                                           stddev=1e-1), name='weights')
  conv = tf.nn.conv2d(input, kernel, [1, 1, 1, 1], padding='SAME')
  biases = tf.Variable(tf.constant(0.0, shape=[128], dtype=tf.float32),
                       trainable=True, name='biases')
  bias = tf.nn.bias_add(conv, biases)
  conv1 = tf.nn.relu(bias, name=scope)
```

To alleviate the need to duplicate this code repeatedly, TF-Slim provides a
number of convenient operations defined at the (more abstract) level of neural
network layers. For example, compare the code above to an invocation of the
TF-Slim code:

```python
input = ...
net = slim.ops.conv2d(input, [3, 3], 128, scope='conv1_1')
```

TF-Slim provides numerous operations used in building neural networks which
roughly correspond to such layers. These include:

Layer                 | TF-Slim Op
--------------------- | ------------------------
Convolutional Layer   | [ops.conv2d](ops.py)
Fully Connected Layer | [ops.fc](ops.py)
BatchNorm layer       | [ops.batch_norm](ops.py)
Max Pooling Layer     | [ops.max_pool](ops.py)
Avg Pooling Layer     | [ops.avg_pool](ops.py)
Dropout Layer         | [ops.dropout](ops.py)

[ops.py](./ops.py) also includes operations that are not really "layers" per se,
but are often used to manipulate hidden unit representations during inference:

Operation | TF-Slim Op
--------- | ---------------------
Flatten   | [ops.flatten](ops.py)

TF-Slim also provides a meta-operation called `repeat_op` that allows one to
repeatedly perform the same operation. Consider the following snippet from the
[VGG](https://www.robots.ox.ac.uk/~vgg/research/very_deep/) network whose layers
perform several convolutions in a row between pooling layers:

```python
net = ...
net = slim.ops.conv2d(net, 256, [3, 3], scope='conv3_1')
net = slim.ops.conv2d(net, 256, [3, 3], scope='conv3_2')
net = slim.ops.conv2d(net, 256, [3, 3], scope='conv3_3')
net = slim.ops.max_pool(net, [2, 2], scope='pool3')
```

This clear duplication of code can be removed via a standard loop:

```python
net = ...
for i in range(3):
  net = slim.ops.conv2d(net, 256, [3, 3], scope='conv3_' % (i+1))
net = slim.ops.max_pool(net, [2, 2], scope='pool3')
```

While this does reduce the amount of duplication, it can be made even cleaner by
using the `RepeatOp`:

```python
net = slim.ops.repeat_op(3, net, slim.ops.conv2d, 256, [3, 3], scope='conv3')
net = slim.ops.max_pool(net, [2, 2], scope='pool2')
```

Notice that the RepeatOp not only applies the same argument in-line, it also is
smart enough to unroll the scopes such that the scopes assigned to each
subsequent call of `ops.conv2d` is appended with an underscore and iteration
number. More concretely, the scopes in the example above would be 'conv3_1',
'conv3_2' and 'conv3_3'.

### Scopes

In addition to the types of scope mechanisms in TensorFlow ([name_scope]
(https://www.tensorflow.org/api_docs/python/framework.html#name_scope),
[op_scope](https://www.tensorflow.org/api_docs/python/framework.html#op_scope),
[variable_scope]
(https://www.tensorflow.org/api_docs/python/state_ops.html#variable_scope),
[variable_op_scope]
(https://www.tensorflow.org/api_docs/python/state_ops.html#variable_op_scope)),
TF-Slim adds a new scoping mechanism called "argument scope" or [arg_scope]
(scopes.py). This new scope allows a user to specify one or more operations and
a set of arguments which will be passed to each of the operations defined in the
`arg_scope`. This functionality is best illustrated by example. Consider the
following code snippet:

```python
net = slim.ops.conv2d(inputs, 64, [11, 11], 4, padding='SAME', stddev=0.01, weight_decay=0.0005, scope='conv1')
net = slim.ops.conv2d(net, 128, [11, 11], padding='VALID', stddev=0.01, weight_decay=0.0005, scope='conv2')
net = slim.ops.conv2d(net, 256, [11, 11], padding='SAME', stddev=0.01, weight_decay=0.0005, scope='conv3')
```

It should be clear that these three Convolution layers share many of the same
hyperparameters. Two have the same padding, all three have the same weight_decay
and standard deviation of its weights. Not only do the duplicated values make
the code more difficult to read, it also adds the addition burder to the writer
of needing to doublecheck that all of the values are identical in each step. One
solution would be to specify default values using variables:

```python
padding='SAME'
stddev=0.01
weight_decay=0.0005
net = slim.ops.conv2d(inputs, 64, [11, 11], 4, padding=padding, stddev=stddev, weight_decay=weight_decay, scope='conv1')
net = slim.ops.conv2d(net, 128, [11, 11], padding='VALID', stddev=stddev, weight_decay=weight_decay, scope='conv2')
net = slim.ops.conv2d(net, 256, [11, 11], padding=padding, stddev=stddev, weight_decay=weight_decay, scope='conv3')

```

This solution ensures that all three convolutions share the exact same variable
values but doesn't reduce the code clutter. By using an `arg_scope`, we can both
ensure that each layer uses the same values and simplify the code:

```python
  with slim.arg_scope([slim.ops.conv2d], padding='SAME', stddev=0.01, weight_decay=0.0005):
    net = slim.ops.conv2d(inputs, 64, [11, 11], scope='conv1')
    net = slim.ops.conv2d(net, 128, [11, 11], padding='VALID', scope='conv2')
    net = slim.ops.conv2d(net, 256, [11, 11], scope='conv3')
```

As the example illustrates, the use of arg_scope makes the code cleaner, simpler
and easier to maintain. Notice that while argument values are specifed in the
arg_scope, they can be overwritten locally. In particular, while the padding
argument has been set to 'SAME', the second convolution overrides it with the
value of 'VALID'.

One can also nest `arg_scope`s and use multiple operations in the same scope.
For example:

```python
with arg_scope([slim.ops.conv2d, slim.ops.fc], stddev=0.01, weight_decay=0.0005):
  with arg_scope([slim.ops.conv2d], padding='SAME'), slim.arg_scope([slim.ops.fc], bias=1.0):
    net = slim.ops.conv2d(inputs, 64, [11, 11], 4, padding='VALID', scope='conv1')
    net = slim.ops.conv2d(net, 256, [5, 5], stddev=0.03, scope='conv2')
    net = slim.ops.flatten(net)
    net = slim.ops.fc(net, 1000, activation=None, scope='fc')
```

In this example, the first `arg_scope` applies the same `stddev` and
`weight_decay` arguments to the `conv2d` and `fc` ops in its scope. In the
second `arg_scope`, additional default arguments to `conv2d` only are specified.

In addition to `arg_scope`, TF-Slim provides several decorators that wrap the
use of tensorflow arg scopes. These include `@AddArgScope`, `@AddNameScope`,
`@AddVariableScope`, `@AddOpScope` and `@AddVariableOpScope`. To illustrate
their use, consider the following example.

```python
def MyNewOp(inputs):
  varA = ...
  varB = ...
  outputs = tf.mul(varA, inputs) + varB
  return outputs

```

In this example, the user has created a new op which creates two variables. To
ensure that these variables exist within a certain variable scope (to avoid
collisions with variables with the same name), in standard TF, the op must be
called within a variable scope:

```python
inputs = ...
with tf.variable_scope('layer1'):
  outputs = MyNewOp(inputs)
```

As an alternative, one can use TF-Slim's decorators to decorate the function and
simplify the call:

```python
@AddVariableScope
def MyNewOp(inputs):
  ...
  return outputs


inputs = ...
outputs = MyNewOp('layer1')
```

The `@AddVariableScope` decorater simply applies the `tf.variable_scope` scoping
to the called function taking "layer1" as its argument. This allows the code to
be written more concisely.

### Losses

The loss function defines a quantity that we want to minimize. For
classification problems, this is typically the cross entropy between the true
(one-hot) distribution and the predicted probability distribution across
classes. For regression problems, this is often the sum-of-squares differences
between the predicted and true values.

Certain models, such as multi-task learning models, require the use of multiple
loss functions simultaneously. In other words, the loss function ultimatey being
minimized is the sum of various other loss functions. For example, consider a
model that predicts both the type of scene in an image as well as the depth from
the camera of each pixel. This model's loss function would be the sum of the
classification loss and depth prediction loss.

TF-Slim provides an easy-to-use mechanism for defining and keeping track of loss
functions via the [losses.py](./losses.py) module. Consider the simple case
where we want to train the VGG network:

```python
# Load the images and labels.
images, labels = ...

# Create the model.
predictions =  ...

# Define the loss functions and get the total loss.
loss = losses.cross_entropy_loss(predictions, labels)
```

In this example, we start by creating the model (using TF-Slim's VGG
implementation), and add the standard classification loss. Now, lets turn to the
case where we have a multi-task model that produces multiple outputs:

```python
# Load the images and labels.
images, scene_labels, depth_labels = ...

# Create the model.
scene_predictions, depth_predictions = CreateMultiTaskModel(images)

# Define the loss functions and get the total loss.
classification_loss = slim.losses.cross_entropy_loss(scene_predictions, scene_labels)
sum_of_squares_loss = slim.losses.l2loss(depth_predictions - depth_labels)

# The following two lines have the same effect:
total_loss1 = classification_loss + sum_of_squares_loss
total_loss2 = tf.get_collection(slim.losses.LOSSES_COLLECTION)
```

In this example, we have two losses which we add by calling
`losses.cross_entropy_loss` and `losses.l2loss`. We can obtain the
total loss by adding them together (`total_loss1`) or by calling
`losses.GetTotalLoss()`. How did this work? When you create a loss function via
TF-Slim, TF-Slim adds the loss to a special TensorFlow collection of loss
functions. This enables you to either manage the total loss manually, or allow
TF-Slim to manage them for you.

What if you want to let TF-Slim manage the losses for you but have a custom loss
function? [losses.py](./losses.py) also has a function that adds this loss to
TF-Slims collection. For example:

```python
# Load the images and labels.
images, scene_labels, depth_labels, pose_labels = ...

# Create the model.
scene_predictions, depth_predictions, pose_predictions = CreateMultiTaskModel(images)

# Define the loss functions and get the total loss.
classification_loss = slim.losses.cross_entropy_loss(scene_predictions, scene_labels)
sum_of_squares_loss = slim.losses.l2loss(depth_predictions - depth_labels)
pose_loss = MyCustomLossFunction(pose_predictions, pose_labels)
tf.add_to_collection(slim.losses.LOSSES_COLLECTION, pose_loss) # Letting TF-Slim know about the additional loss.

# The following two lines have the same effect:
total_loss1 = classification_loss + sum_of_squares_loss + pose_loss
total_loss2 = losses.GetTotalLoss()
```

In this example, we can again either produce the total loss function manually or
let TF-Slim know about the additional loss and let TF-Slim handle the losses.

## Putting the Pieces Together

By combining TF-Slim Variables, Operations and scopes, we can write a normally
very complex network with very few lines of code. For example, the entire [VGG]
(https://www.robots.ox.ac.uk/~vgg/research/very_deep/) architecture can be
defined with just the following snippet:

```python
with arg_scope([slim.ops.conv2d, slim.ops.fc], stddev=0.01, weight_decay=0.0005):
  net = slim.ops.repeat_op(1, inputs, slim.ops.conv2d, 64, [3, 3], scope='conv1')
  net = slim.ops.max_pool(net, [2, 2], scope='pool1')
  net = slim.ops.repeat_op(1, net, slim.ops.conv2d, 128, [3, 3], scope='conv2')
  net = slim.ops.max_pool(net, [2, 2], scope='pool2')
  net = slim.ops.repeat_op(2, net, slim.ops.conv2d, 256, [3, 3], scope='conv3')
  net = slim.ops.max_pool(net, [2, 2], scope='pool3')
  net = slim.ops.repeat_op(2, net, slim.ops.conv2d, 512, [3, 3], scope='conv4')
  net = slim.ops.max_pool(net, [2, 2], scope='pool4')
  net = slim.ops.repeat_op(2, net, slim.ops.conv2d, 512, [3, 3], scope='conv5')
  net = slim.ops.max_pool(net, [2, 2], scope='pool5')
  net = slim.ops.flatten(net, scope='flatten5')
  net = slim.ops.fc(net, 4096, scope='fc6')
  net = slim.ops.dropout(net, 0.5, scope='dropout6')
  net = slim.ops.fc(net, 4096, scope='fc7')
  net = slim.ops.dropout(net, 0.5, scope='dropout7')
  net = slim.ops.fc(net, 1000, activation=None, scope='fc8')
return net
```

## Re-using previously defined network architectures and pre-trained models.

### Brief Recap on Restoring Variables from a Checkpoint

After a model has been trained, it can be restored using `tf.train.Saver()`
which restores `Variables` from a given checkpoint. For many cases,
`tf.train.Saver()` provides a simple mechanism to restore all or just a few
variables.

```python
# Create some variables.
v1 = tf.Variable(..., name="v1")
v2 = tf.Variable(..., name="v2")
...
# Add ops to restore all the variables.
restorer = tf.train.Saver()

# Add ops to restore some variables.
restorer = tf.train.Saver([v1, v2])

# Later, launch the model, use the saver to restore variables from disk, and
# do some work with the model.
with tf.Session() as sess:
  # Restore variables from disk.
  restorer.restore(sess, "/tmp/model.ckpt")
  print("Model restored.")
  # Do some work with the model
  ...
```

See [Restoring Variables]
(https://www.tensorflow.org/versions/r0.7/how_tos/variables/index.html#restoring-variables)
and [Choosing which Variables to Save and Restore]
(https://www.tensorflow.org/versions/r0.7/how_tos/variables/index.html#choosing-which-variables-to-save-and-restore)
sections of the [Variables]
(https://www.tensorflow.org/versions/r0.7/how_tos/variables/index.html) page for
more details.

### Using slim.variables to Track which Variables need to be Restored

It is often desirable to fine-tune a pre-trained model on an entirely new
dataset or even a new task. In these situations, one must specify which layers
of the model should be reused (and consequently loaded from a checkpoint) and
which layers are new. Indicating which variables or layers should be restored is
a process that quickly becomes cumbersome when done manually.

To help keep track of which variables to restore, `slim.variables` provides a
`restore` argument when creating each Variable. By default, all variables are
marked as `restore=True`, which results in all variables defined by the model
being restored.

```python
# Create some variables.
v1 = slim.variables.variable(name="v1", ..., restore=False)
v2 = slim.variables.variable(name="v2", ...) # By default restore=True
...
# Get list of variables to restore (which contains only 'v2')
variables_to_restore = tf.get_collection(slim.variables.VARIABLES_TO_RESTORE)
restorer = tf.train.Saver(variables_to_restore)
with tf.Session() as sess:
  # Restore variables from disk.
  restorer.restore(sess, "/tmp/model.ckpt")
  print("Model restored.")
  # Do some work with the model
  ...
```

Additionally, every layer in `slim.ops` that creates slim.variables (such as
`slim.ops.conv2d`, `slim.ops.fc`, `slim.ops.batch_norm`) also has a `restore`
argument which controls whether the variables created by that layer should be
restored or not.

```python
# Create a small network.
net = slim.ops.conv2d(images, 32, [7, 7], stride=2, scope='conv1')
net = slim.ops.conv2d(net, 64, [3, 3], scope='conv2')
net = slim.ops.conv2d(net, 128, [3, 3], scope='conv3')
net = slim.ops.max_pool(net, [3, 3], stride=2, scope='pool3')
net = slim.ops.flatten(net)
net = slim.ops.fc(net, 10, scope='logits', restore=False)
...

# VARIABLES_TO_RESTORE would contain the 'weights' and 'bias' defined by 'conv1'
# 'conv2' and 'conv3' but not the ones defined by 'logits'
variables_to_restore = tf.get_collection(slim.variables.VARIABLES_TO_RESTORE)

# Create a restorer that would restore only the needed variables.
restorer = tf.train.Saver(variables_to_restore)

# Create a saver that would save all the variables (including 'logits').
saver = tf.train.Saver()
with tf.Session() as sess:
  # Restore variables from disk.
  restorer.restore(sess, "/tmp/model.ckpt")
  print("Model restored.")

  # Do some work with the model
  ...
  saver.save(sess, "/tmp/new_model.ckpt")
```

Note: When restoring variables from a checkpoint, the `Saver` locates the
variable names in a checkpoint file and maps them to variables in the current
graph. Above, we created a saver by passing to it a list of variables. In this
case, the names of the variables to locate in the checkpoint file were
implicitly obtained from each provided variable's `var.op.name`.

This works well when the variable names in the checkpoint file match those in
the graph. However, sometimes, we want to restore a model from a checkpoint
whose variables have different names those in the current graph. In this case,
we must provide the `Saver` a dictionary that maps from each checkpoint variable
name to each graph variable. Consider the following example where the checkpoint
variables names are obtained via a simple function:

```python
# Assuming than 'conv1/weights' should be restored from 'vgg16/conv1/weights'
def name_in_checkpoint(var):
  return 'vgg16/' + var.op.name

# Assuming than 'conv1/weights' and 'conv1/bias' should be restored from 'conv1/params1' and 'conv1/params2'
def name_in_checkpoint(var):
  if "weights" in var.op.name:
    return var.op.name.replace("weights", "params1")
  if "bias" in var.op.name:
    return var.op.name.replace("bias", "params2")

variables_to_restore = tf.get_collection(slim.variables.VARIABLES_TO_RESTORE)
variables_to_restore = {name_in_checkpoint(var):var for var in variables_to_restore}
restorer = tf.train.Saver(variables_to_restore)
with tf.Session() as sess:
  # Restore variables from disk.
  restorer.restore(sess, "/tmp/model.ckpt")
```

### Reusing the VGG16 network defined in TF-Slim on a different task, i.e. PASCAL-VOC.

Assuming one have already a pre-trained VGG16 model, one just need to replace
the last layer `fc8` with a new layer `fc8_pascal` and use `restore=False`.

```python
def vgg16_pascal(inputs):
  with slim.arg_scope([slim.ops.conv2d, slim.ops.fc], stddev=0.01, weight_decay=0.0005):
    net = slim.ops.repeat_op(2, inputs, slim.ops.conv2d, 64, [3, 3], scope='conv1')
    net = slim.ops.max_pool(net, [2, 2], scope='pool1')
    net = slim.ops.repeat_op(2, net, slim.ops.conv2d, 128, [3, 3], scope='conv2')
    net = slim.ops.max_pool(net, [2, 2], scope='pool2')
    net = slim.ops.repeat_op(3, net, slim.ops.conv2d, 256, [3, 3], scope='conv3')
    net = slim.ops.max_pool(net, [2, 2], scope='pool3')
    net = slim.ops.repeat_op(3, net, slim.ops.conv2d, 512, [3, 3], scope='conv4')
    net = slim.ops.max_pool(net, [2, 2], scope='pool4')
    net = slim.ops.repeat_op(3, net, slim.ops.conv2d, 512, [3, 3], scope='conv5')
    net = slim.ops.max_pool(net, [2, 2], scope='pool5')
    net = slim.ops.flatten(net, scope='flatten5')
    net = slim.ops.fc(net, 4096, scope='fc6')
    net = slim.ops.dropout(net, 0.5, scope='dropout6')
    net = slim.ops.fc(net, 4096, scope='fc7')
    net = slim.ops.dropout(net, 0.5, scope='dropout7')
    # To reuse vgg16 on PASCAL-VOC, just change the last layer.
    net = slim.ops.fc(net, 21, activation=None, scope='fc8_pascal', restore=False)
  return net
```

## Authors

Sergio Guadarrama and Nathan Silberman
