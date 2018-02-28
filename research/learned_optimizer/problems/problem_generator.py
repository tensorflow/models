# Copyright 2017 Google, Inc. All Rights Reserved.
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
# ==============================================================================

"""Generates toy optimization problems.

This module contains a base class, Problem, that defines a minimal interface
for optimization problems, and a few specific problem types that subclass it.

Test functions for optimization: http://www.sfu.ca/~ssurjano/optimization.html
"""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import numpy as np
import tensorflow as tf

from learned_optimizer.problems import problem_spec as prob_spec

tf.app.flags.DEFINE_float("l2_reg_scale", 1e-3,
                          """Scaling factor for parameter value regularization
                             in softmax classifier problems.""")
FLAGS = tf.app.flags.FLAGS

EPSILON = 1e-6
MAX_SEED = 4294967295
PARAMETER_SCOPE = "parameters"

_Spec = prob_spec.Spec


class Problem(object):
  """Base class for optimization problems.

  This defines an interface for optimization problems, including objective and
  gradients functions and a feed_generator function that yields data to pass to
  feed_dict in tensorflow.

  Subclasses of Problem must (at the minimum) override the objective method,
  which computes the objective/loss/cost to minimize, and specify the desired
  shape of the parameters in a list in the param_shapes attribute.
  """

  def __init__(self, param_shapes, random_seed, noise_stdev, init_fn=None):
    """Initializes a global random seed for the problem.

    Args:
      param_shapes: A list of tuples defining the expected shapes of the
        parameters for this problem
      random_seed: Either an integer (or None, in which case the seed is
        randomly drawn)
      noise_stdev: Strength (standard deviation) of added gradient noise
      init_fn: A function taking a tf.Session object that is used to
        initialize the problem's variables.

    Raises:
      ValueError: If the random_seed is not an integer and not None
    """
    if random_seed is not None and not isinstance(random_seed, int):
      raise ValueError("random_seed must be an integer or None")

    # Pick a random seed.
    self.random_seed = (np.random.randint(MAX_SEED) if random_seed is None
                        else random_seed)

    # Store the noise level.
    self.noise_stdev = noise_stdev

    # Set the random seed to ensure any random data in the problem is the same.
    np.random.seed(self.random_seed)

    # Store the parameter shapes.
    self.param_shapes = param_shapes

    if init_fn is not None:
      self.init_fn = init_fn
    else:
      self.init_fn = lambda _: None

  def init_tensors(self, seed=None):
    """Returns a list of tensors with the given shape."""
    return [tf.random_normal(shape, seed=seed) for shape in self.param_shapes]

  def init_variables(self, seed=None):
    """Returns a list of variables with the given shape."""
    with tf.variable_scope(PARAMETER_SCOPE):
      params = [tf.Variable(param) for param in self.init_tensors(seed)]
    return params

  def objective(self, parameters, data=None, labels=None):
    """Computes the objective given a list of parameters.

    Args:
      parameters: The parameters to optimize (as a list of tensors)
      data: An optional batch of data for calculating objectives
      labels: An optional batch of corresponding labels

    Returns:
      A scalar tensor representing the objective value
    """
    raise NotImplementedError

  def gradients(self, objective, parameters):
    """Compute gradients of the objective with respect to the parameters.

    Args:
      objective: The objective op (e.g. output of self.objective())
      parameters: A list of tensors (the parameters to optimize)

    Returns:
      A list of tensors representing the gradient for each parameter,
        returned in the same order as the given list
    """
    grads = tf.gradients(objective, list(parameters))
    noisy_grads = []

    for grad in grads:
      if isinstance(grad, tf.IndexedSlices):
        noise = self.noise_stdev * tf.random_normal(tf.shape(grad.values))
        new_grad = tf.IndexedSlices(grad.values + noise, grad.indices)
      else:
        new_grad = grad + self.noise_stdev * tf.random_normal(grad.get_shape())
      noisy_grads.append(new_grad)

    return noisy_grads


class Quadratic(Problem):
  """Optimizes a random quadratic function.

  The objective is: f(x) = (1/2) ||Wx - y||_2^2
  where W is a random Gaussian matrix and y is a random Gaussian vector.
  """

  def __init__(self, ndim, random_seed=None, noise_stdev=0.0):
    """Initializes a random quadratic problem."""
    param_shapes = [(ndim, 1)]
    super(Quadratic, self).__init__(param_shapes, random_seed, noise_stdev)

    # Generate a random problem instance.
    self.w = np.random.randn(ndim, ndim).astype("float32")
    self.y = np.random.randn(ndim, 1).astype("float32")

  def objective(self, params, data=None, labels=None):
    """Quadratic objective (see base class for details)."""
    return tf.nn.l2_loss(tf.matmul(self.w, params[0]) - self.y)


class SoftmaxClassifier(Problem):
  """Helper functions for supervised softmax classification problems."""

  def init_tensors(self, seed=None):
    """Returns a list of tensors with the given shape."""
    return [tf.random_normal(shape, seed=seed) * 1.2 / np.sqrt(shape[0])
            for shape in self.param_shapes]

  def inference(self, params, data):
    """Computes logits given parameters and data.

    Args:
      params: List of parameter tensors or variables
      data: Batch of features with samples along the first dimension

    Returns:
      logits: Un-normalized logits with shape (num_samples, num_classes)
    """
    raise NotImplementedError

  def objective(self, params, data, labels):
    """Computes the softmax cross entropy.

    Args:
      params: List of parameter tensors or variables
      data: Batch of features with samples along the first dimension
      labels: Vector of labels with the same number of samples as the data

    Returns:
      loss: Softmax cross entropy loss averaged over the samples in the batch

    Raises:
      ValueError: If the objective is to be computed over >2 classes, because
        this operation is broken in tensorflow at the moment.
    """
    # Forward pass.
    logits = self.inference(params, data)

    # Compute the loss.
    l2reg = [tf.reduce_sum(param ** 2) for param in params]
    if int(logits.get_shape()[1]) == 2:
      labels = tf.cast(labels, tf.float32)
      losses = tf.nn.sigmoid_cross_entropy_with_logits(
          labels=labels, logits=logits[:, 0])
    else:
      raise ValueError("Unable to compute softmax cross entropy for more than"
                       " 2 classes.")

    return tf.reduce_mean(losses) + tf.reduce_mean(l2reg) * FLAGS.l2_reg_scale

  def argmax(self, logits):
    """Samples the most likely class label given the logits.

    Args:
      logits: Un-normalized logits with shape (num_samples, num_classes)

    Returns:
      predictions: Predicted class labels, has shape (num_samples,)
    """
    return tf.cast(tf.argmax(tf.nn.softmax(logits), 1), tf.int32)

  def accuracy(self, params, data, labels):
    """Computes the accuracy (fraction of correct classifications).

    Args:
      params: List of parameter tensors or variables
      data: Batch of features with samples along the first dimension
      labels: Vector of labels with the same number of samples as the data

    Returns:
      accuracy: Fraction of correct classifications across the batch
    """
    predictions = self.argmax(self.inference(params, data))
    return tf.contrib.metrics.accuracy(predictions, tf.cast(labels, tf.int32))


class SoftmaxRegression(SoftmaxClassifier):
  """Builds a softmax regression problem."""

  def __init__(self, n_features, n_classes, activation=tf.identity,
               random_seed=None, noise_stdev=0.0):
    self.activation = activation
    self.n_features = n_features
    param_shapes = [(n_features, n_classes), (n_classes,)]
    super(SoftmaxRegression, self).__init__(param_shapes,
                                            random_seed,
                                            noise_stdev)

  def inference(self, params, data):
    features = tf.reshape(data, (-1, self.n_features))
    return tf.matmul(features, params[0]) + params[1]


class SparseSoftmaxRegression(SoftmaxClassifier):
  """Builds a sparse input softmax regression problem."""

  def __init__(self,
               n_features,
               n_classes,
               activation=tf.identity,
               random_seed=None,
               noise_stdev=0.0):
    self.activation = activation
    self.n_features = n_features
    param_shapes = [(n_classes, n_features), (n_features, n_classes), (
        n_classes,)]
    super(SparseSoftmaxRegression, self).__init__(param_shapes, random_seed,
                                                  noise_stdev)

  def inference(self, params, data):
    all_embeddings, softmax_weights, softmax_bias = params
    embeddings = tf.nn.embedding_lookup(all_embeddings, tf.cast(data, tf.int32))
    embeddings = tf.reduce_sum(embeddings, 1)
    return tf.matmul(embeddings, softmax_weights) + softmax_bias


class OneHotSparseSoftmaxRegression(SoftmaxClassifier):
  """Builds a sparse input softmax regression problem.

  This is identical to SparseSoftmaxRegression, but without using embedding
  ops.
  """

  def __init__(self,
               n_features,
               n_classes,
               activation=tf.identity,
               random_seed=None,
               noise_stdev=0.0):
    self.activation = activation
    self.n_features = n_features
    self.n_classes = n_classes
    param_shapes = [(n_classes, n_features), (n_features, n_classes), (
        n_classes,)]
    super(OneHotSparseSoftmaxRegression, self).__init__(param_shapes,
                                                        random_seed,
                                                        noise_stdev)

  def inference(self, params, data):
    all_embeddings, softmax_weights, softmax_bias = params
    num_ids = tf.shape(data)[1]
    one_hot_embeddings = tf.one_hot(tf.cast(data, tf.int32), self.n_classes)
    one_hot_embeddings = tf.reshape(one_hot_embeddings, [-1, self.n_classes])
    embeddings = tf.matmul(one_hot_embeddings, all_embeddings)
    embeddings = tf.reshape(embeddings, [-1, num_ids, self.n_features])
    embeddings = tf.reduce_sum(embeddings, 1)
    return tf.matmul(embeddings, softmax_weights) + softmax_bias


class FullyConnected(SoftmaxClassifier):
  """Builds a multi-layer perceptron classifier."""

  def __init__(self, n_features, n_classes, hidden_sizes=(32, 64),
               activation=tf.nn.sigmoid, random_seed=None, noise_stdev=0.0):
    """Initializes an multi-layer perceptron classification problem."""
    # Store the number of features and activation function.
    self.n_features = n_features
    self.activation = activation

    # Define the network as a list of weight + bias shapes for each layer.
    param_shapes = []
    for ix, sz in enumerate(hidden_sizes + (n_classes,)):

      # The previous layer"s size (n_features if input).
      prev_size = n_features if ix == 0 else hidden_sizes[ix - 1]

      # Weight shape for this layer.
      param_shapes.append((prev_size, sz))

      # Bias shape for this layer.
      param_shapes.append((sz,))

    super(FullyConnected, self).__init__(param_shapes, random_seed, noise_stdev)

  def inference(self, params, data):
    # Flatten the features into a vector.
    features = tf.reshape(data, (-1, self.n_features))

    # Pass the data through the network.
    preactivations = tf.matmul(features, params[0]) + params[1]

    for layer in range(2, len(self.param_shapes), 2):
      net = self.activation(preactivations)
      preactivations = tf.matmul(net, params[layer]) + params[layer + 1]

    return preactivations

  def accuracy(self, params, data, labels):
    """Computes the accuracy (fraction of correct classifications).

    Args:
      params: List of parameter tensors or variables
      data: Batch of features with samples along the first dimension
      labels: Vector of labels with the same number of samples as the data

    Returns:
      accuracy: Fraction of correct classifications across the batch
    """
    predictions = self.argmax(self.activation(self.inference(params, data)))
    return tf.contrib.metrics.accuracy(predictions, tf.cast(labels, tf.int32))


class ConvNet(SoftmaxClassifier):
  """Builds an N-layer convnet for image classification."""

  def __init__(self,
               image_shape,
               n_classes,
               filter_list,
               activation=tf.nn.relu,
               random_seed=None,
               noise_stdev=0.0):
    # Number of channels, number of pixels in x- and y- dimensions.
    n_channels, px, py = image_shape

    # Store the activation.
    self.activation = activation

    param_shapes = []
    input_size = n_channels
    for fltr in filter_list:
      # Add conv2d filters.
      param_shapes.append((fltr[0], fltr[1], input_size, fltr[2]))
      input_size = fltr[2]

    # Number of units in the final (dense) layer.
    self.affine_size = input_size * px * py

    param_shapes.append((self.affine_size, n_classes))  # affine weights
    param_shapes.append((n_classes,))  # affine bias

    super(ConvNet, self).__init__(param_shapes, random_seed, noise_stdev)

  def init_tensors(self, seed=None):
    """Returns a list of tensors with the given shape."""
    return [tf.random_normal(shape, mean=0., stddev=0.01, seed=seed)
            for shape in self.param_shapes]

  def inference(self, params, data):

    # Unpack.
    w_conv_list = params[:-2]
    output_w, output_b = params[-2:]

    conv_input = data
    for w_conv in w_conv_list:
      layer = tf.nn.conv2d(conv_input, w_conv, strides=[1] * 4, padding="SAME")
      output = self.activation(layer)
      conv_input = output

    # Flatten.
    flattened = tf.reshape(conv_input, (-1, self.affine_size))

    # Fully connected layer.
    return tf.matmul(flattened, output_w) + output_b


class Bowl(Problem):
  """A 2D quadratic bowl."""

  def __init__(self, condition_number, angle=0.0,
               random_seed=None, noise_stdev=0.0):
    assert condition_number > 0, "Condition number must be positive."

    # Define parameter shapes.
    param_shapes = [(2, 1)]
    super(Bowl, self).__init__(param_shapes, random_seed, noise_stdev)

    self.condition_number = condition_number
    self.angle = angle
    self._build_matrix(condition_number, angle)

  def _build_matrix(self, condition_number, angle):
    """Builds the Hessian matrix."""
    hessian = np.array([[condition_number, 0.], [0., 1.]], dtype="float32")

    # Build the rotation matrix.
    rotation_matrix = np.array([
        [np.cos(angle), -np.sin(angle)],
        [np.sin(angle), np.cos(angle)]
    ])

    # The objective is 0.5 * || Ax ||_2^2
    # where the data matrix (A) is: sqrt(Hessian).dot(rotation_matrix).
    self.matrix = np.sqrt(hessian).dot(rotation_matrix)

  def objective(self, params, data=None, labels=None):
    mtx = tf.constant(self.matrix, dtype=tf.float32)
    return tf.nn.l2_loss(tf.matmul(mtx, params[0]))

  def surface(self, xlim=5, ylim=5, n=50):
    xm, ym = _mesh(xlim, ylim, n)
    pts = np.vstack([xm.ravel(), ym.ravel()])
    zm = 0.5 * np.linalg.norm(self.matrix.dot(pts), axis=0) ** 2
    return xm, ym, zm.reshape(n, n)


class Problem2D(Problem):

  def __init__(self, random_seed=None, noise_stdev=0.0):
    param_shapes = [(2,)]
    super(Problem2D, self).__init__(param_shapes, random_seed, noise_stdev)

  def surface(self, n=50, xlim=5, ylim=5):
    """Computes the objective surface over a 2d mesh."""

    # Create a mesh over the given coordinate ranges.
    xm, ym = _mesh(xlim, ylim, n)

    with tf.Graph().as_default(), tf.Session() as sess:

      # Ops to compute the objective at every (x, y) point.
      x = tf.placeholder(tf.float32, shape=xm.shape)
      y = tf.placeholder(tf.float32, shape=ym.shape)
      obj = self.objective([[x, y]])

      # Run the computation.
      zm = sess.run(obj, feed_dict={x: xm, y: ym})

    return xm, ym, zm


class Rosenbrock(Problem2D):
  """See https://en.wikipedia.org/wiki/Rosenbrock_function.

  This function has a single global minima at [1, 1]
  The objective value at this point is zero.
  """

  def init_tensors(self, seed=None):
    """Returns a list of tensors with the given shape."""
    return [tf.random_uniform(shape, minval=-5., maxval=10., seed=seed)
            for shape in self.param_shapes]

  def objective(self, params, data=None, labels=None):
    x, y = tf.split(params[0], 2, axis=0)
    obj = (1 - x)**2 + 100 * (y - x**2)**2
    return tf.squeeze(obj)


def make_rosenbrock_loss_and_init(device=None):
  """A variable-backed version of Rosenbrock problem.

  See the Rosenbrock class for details.

  Args:
    device: Where to place the ops of this problem.

  Returns:
    A tuple of two callables, first of which creates the loss and the second
    creates the parameter initializer function.
  """
  def make_rosenbrock_loss():
    with tf.name_scope("optimizee"):
      with tf.device(device):
        x = tf.get_variable("x", [1])
        y = tf.get_variable("y", [1])
        c = tf.get_variable(
            "c", [1],
            initializer=tf.constant_initializer(100.0),
            trainable=False)
        obj = (1 - x)**2 + c * (y - x**2)**2
      return tf.squeeze(obj)

  def make_init_fn(parameters):
    with tf.device(device):
      init_op = tf.variables_initializer(parameters)
    def init_fn(sess):
      tf.logging.info("Initializing model parameters.")
      sess.run(init_op)
    return init_fn

  return make_rosenbrock_loss, make_init_fn


class Saddle(Problem2D):
  """Loss surface around a saddle point."""

  def objective(self, params, data=None, labels=None):
    x, y = tf.split(params[0], 2, axis=0)
    obj = x ** 2 - y ** 2
    return tf.squeeze(obj)


class LogSumExp(Problem2D):
  """2D function defined by the log of the sum of exponentials."""

  def objective(self, params, data=None, labels=None):
    x, y = tf.split(params[0], 2, axis=0)
    obj = tf.log(tf.exp(x + 3. * y - 0.1) +
                 tf.exp(x - 3. * y - 0.1) +
                 tf.exp(-x - 0.1) + 1.0)
    return tf.squeeze(obj)


class Ackley(Problem2D):
  """Ackley's function (contains many local minima)."""

  def init_tensors(self, seed=None):
    """Returns a list of tensors with the given shape."""
    return [tf.random_uniform(shape, minval=-32.768, maxval=32.768, seed=seed)
            for shape in self.param_shapes]

  def objective(self, params, data=None, labels=None):
    x, y = tf.split(params[0], 2, axis=0)
    obj = (-20 * tf.exp(-0.2 * tf.sqrt(0.5 * (x ** 2 + y ** 2))) -
           tf.exp(0.5 * (tf.cos(2 * np.pi * x) + tf.cos(2 * np.pi * y))) +
           tf.exp(1.0) + 20.)
    return tf.squeeze(obj)


class Beale(Problem2D):
  """Beale function (a multimodal function with sharp peaks)."""

  def init_tensors(self, seed=None):
    """Returns a list of tensors with the given shape."""
    return [tf.random_uniform(shape, minval=-4.5, maxval=4.5, seed=seed)
            for shape in self.param_shapes]

  def objective(self, params, data=None, labels=None):
    x, y = tf.split(params[0], 2, axis=0)
    obj = ((1.5 - x + x * y) ** 2 +
           (2.25 - x + x * y ** 2) ** 2 +
           (2.625 - x + x * y ** 3) ** 2)
    return tf.squeeze(obj)


class Booth(Problem2D):
  """Booth's function (has a long valley along one dimension)."""

  def init_tensors(self, seed=None):
    """Returns a list of tensors with the given shape."""
    return [tf.random_uniform(shape, minval=-10., maxval=10., seed=seed)
            for shape in self.param_shapes]

  def objective(self, params, data=None, labels=None):
    x, y = tf.split(params[0], 2, axis=0)
    obj = (x + 2 * y - 7) ** 2 + (2 * x + y - 5) ** 2
    return tf.squeeze(obj)


class StyblinskiTang(Problem2D):
  """Styblinski-Tang function (a bumpy function in two dimensions)."""

  def init_tensors(self, seed=None):
    """Returns a list of tensors with the given shape."""
    return [tf.random_uniform(shape, minval=-5., maxval=5., seed=seed)
            for shape in self.param_shapes]

  def objective(self, params, data=None, labels=None):
    params = tf.split(params[0], 2, axis=0)
    obj = 0.5 * tf.reduce_sum([x ** 4 - 16 * x ** 2 + 5 * x
                               for x in params], 0) + 80.
    return tf.squeeze(obj)


class Matyas(Problem2D):
  """Matyas function (a function with a single global minimum in a valley)."""

  def init_tensors(self, seed=None):
    """Returns a list of tensors with the given shape."""
    return [tf.random_uniform(shape, minval=-10, maxval=10, seed=seed)
            for shape in self.param_shapes]

  def objective(self, params, data=None, labels=None):
    x, y = tf.split(params[0], 2, axis=0)
    obj = 0.26 * (x ** 2 + y ** 2) - 0.48 * x * y
    return tf.squeeze(obj)


class Branin(Problem2D):
  """Branin function (a function with three global minima)."""

  def init_tensors(self, seed=None):
    """Returns a list of tensors with the given shape."""
    x1 = tf.random_uniform((1,), minval=-5., maxval=10.,
                           seed=seed)
    x2 = tf.random_uniform((1,), minval=0., maxval=15.,
                           seed=seed)
    return [tf.concat([x1, x2], 0)]

  def objective(self, params, data=None, labels=None):
    x, y = tf.split(params[0], 2, axis=0)

    # Define some constants.
    a = 1.
    b = 5.1 / (4. * np.pi ** 2)
    c = 5 / np.pi
    r = 6.
    s = 10.
    t = 1 / (8. * np.pi)

    # Evaluate the function.
    obj = a * (y - b * x ** 2 + c * x - r) ** 2 + s * (1 - t) * tf.cos(x) + s
    return tf.squeeze(obj)


class Michalewicz(Problem2D):
  """Michalewicz function (has steep ridges and valleys)."""

  def init_tensors(self, seed=None):
    """Returns a list of tensors with the given shape."""
    return [tf.random_uniform(shape, minval=0., maxval=np.pi, seed=seed)
            for shape in self.param_shapes]

  def objective(self, params, data=None, labels=None):
    x, y = tf.split(params[0], 2, axis=0)
    m = 5    # Defines how steep the ridges are (larger m => steeper ridges).
    obj = 2. - (tf.sin(x) * tf.sin(x ** 2 / np.pi) ** (2 * m) +
                tf.sin(y) * tf.sin(2 * y ** 2 / np.pi) ** (2 * m))
    return tf.squeeze(obj)


class Rescale(Problem):
  """Takes an existing problem, and rescales all the parameters."""

  def __init__(self, problem_spec, scale=10., noise_stdev=0.0):
    self.problem = problem_spec.build()
    self.param_shapes = self.problem.param_shapes
    self.scale = scale

    super(Rescale, self).__init__(self.param_shapes, random_seed=None,
                                  noise_stdev=noise_stdev)

  def init_tensors(self, seed=None):
    params_raw = self.problem.init_tensors(seed=seed)
    params = [t * self.scale for t in params_raw]
    return params

  def objective(self, params, data=None, labels=None):
    params_raw = [t/self.scale for t in params]

    problem_obj = self.problem.objective(params_raw, data, labels)
    return problem_obj


class SumTask(Problem):
  """Takes a list of problems and modifies the objective to be their sum."""

  def __init__(self, problem_specs, noise_stdev=0.0):
    self.problems = [ps.build() for ps in problem_specs]
    self.param_shapes = []
    for prob in self.problems:
      self.param_shapes += prob.param_shapes

    super(SumTask, self).__init__(self.param_shapes, random_seed=None,
                                  noise_stdev=noise_stdev)

  def init_tensors(self, seed=None):
    tensors = []
    for prob in self.problems:
      tensors += prob.init_tensors(seed=seed)
    return tensors

  def objective(self, params, data=None, labels=None):
    obj = 0.
    index = 0
    for prob in self.problems:
      num_params = len(prob.param_shapes)
      obj += prob.objective(params[index:index + num_params])
      index += num_params
    return obj


class IsotropicQuadratic(Problem):
  """An isotropic quadratic problem."""

  def objective(self, params, data=None, labels=None):
    return sum([tf.reduce_sum(param ** 2) for param in params])


class Norm(Problem):
  """Takes an existing problem and modifies the objective to be its N-norm."""

  def __init__(self, ndim, random_seed=None, noise_stdev=0.0, norm_power=2.):
    param_shapes = [(ndim, 1)]
    super(Norm, self).__init__(param_shapes, random_seed, noise_stdev)

    # Generate a random problem instance.
    self.w = np.random.randn(ndim, ndim).astype("float32")
    self.y = np.random.randn(ndim, 1).astype("float32")
    self.norm_power = norm_power

  def objective(self, params, data=None, labels=None):
    diff = tf.matmul(self.w, params[0]) - self.y
    exp = 1. / self.norm_power
    loss = tf.reduce_sum((tf.abs(diff) + EPSILON) ** self.norm_power) ** exp
    return loss


class LogObjective(Problem):
  """Takes an existing problem and modifies the objective to be its log."""

  def __init__(self, problem_spec):
    self.problem = problem_spec.build()
    self.param_shapes = self.problem.param_shapes

    super(LogObjective, self).__init__(self.param_shapes,
                                       random_seed=None,
                                       noise_stdev=0.0)

  def objective(self, params, data=None, labels=None):
    problem_obj = self.problem.objective(params, data, labels)
    return tf.log(problem_obj + EPSILON) - tf.log(EPSILON)


class SparseProblem(Problem):
  """Takes a problem and sets gradients to 0 with the given probability."""

  def __init__(self,
               problem_spec,
               zero_probability=0.99,
               random_seed=None,
               noise_stdev=0.0):
    self.problem = problem_spec.build()
    self.param_shapes = self.problem.param_shapes
    self.zero_prob = zero_probability

    super(SparseProblem, self).__init__(self.param_shapes,
                                        random_seed=random_seed,
                                        noise_stdev=noise_stdev)

  def objective(self, parameters, data=None, labels=None):
    return self.problem.objective(parameters, data, labels)

  def gradients(self, objective, parameters):
    grads = tf.gradients(objective, list(parameters))

    new_grads = []
    for grad in grads:
      mask = tf.greater(self.zero_prob, tf.random_uniform(grad.get_shape()))
      zero_grad = tf.zeros_like(grad, dtype=tf.float32)
      noisy_grad = grad + self.noise_stdev * tf.random_normal(grad.get_shape())
      new_grads.append(tf.where(mask, zero_grad, noisy_grad))
    return new_grads


class DependencyChain(Problem):
  """A problem in which parameters must be optimized in order.

  A sequence of parameters which all need to be brought to 0, but where each
  parameter in the sequence can't be brought to 0 until the preceding one
  has been. This should take a long time to optimize, with steady
  (or accelerating) progress throughout the entire process.
  """

  def __init__(self, ndim, random_seed=None, noise_stdev=0.):
    param_shapes = [(ndim + 1,)]
    self.ndim = ndim
    super(DependencyChain, self).__init__(
        param_shapes, random_seed, noise_stdev)

  def objective(self, params, data=None, labels=None):
    terms = params[0][0]**2 + params[0][1:]**2 / (params[0][:-1]**2 + EPSILON)
    return tf.reduce_sum(terms)


class MinMaxWell(Problem):
  """Problem with global min when both the min and max (absolute) params are 1.

  The gradient for all but two parameters (the min and max) is zero. This
  should therefore encourage the optimizer to behave sensible even when
  parameters have zero gradients, as is common eg for some deep neural nets.
  """

  def __init__(self, ndim, random_seed=None, noise_stdev=0.):
    param_shapes = [(ndim,)]
    self.ndim = ndim
    super(MinMaxWell, self).__init__(param_shapes, random_seed, noise_stdev)

  def objective(self, params, data=None, labels=None):
    params_sqr = params[0]**2
    min_sqr = tf.reduce_min(params_sqr)
    max_sqr = tf.reduce_max(params_sqr)
    epsilon = 1e-12

    return max_sqr + 1./min_sqr - 2. + epsilon


class OutwardSnake(Problem):
  """A winding path out to infinity.

  Ideal step length stays constant along the entire path.
  """

  def __init__(self, ndim, random_seed=None, noise_stdev=0.):
    param_shapes = [(ndim,)]
    self.ndim = ndim
    super(OutwardSnake, self).__init__(param_shapes, random_seed, noise_stdev)

  def objective(self, params, data, labels=None):
    radius = tf.sqrt(tf.reduce_sum(params[0]**2))
    rad_loss = tf.reduce_sum(1. / (radius + 1e-6) * data[:, 0])

    sin_dist = params[0][1:] - tf.cos(params[0][:-1]) * np.pi
    sin_loss = tf.reduce_sum((sin_dist * data[:, 1:])**2)

    return rad_loss + sin_loss


class ProjectionQuadratic(Problem):
  """Dataset consists of different directions to probe. Global min is at 0."""

  def __init__(self, ndim, random_seed=None, noise_stdev=0.):
    param_shapes = [(1, ndim)]
    super(ProjectionQuadratic, self).__init__(
        param_shapes, random_seed, noise_stdev)

  def objective(self, params, data, labels=None):
    return tf.reduce_sum((params[0] * data)**2)


class SumOfQuadratics(Problem):

  def __init__(self, ndim, random_seed=None, noise_stdev=0.):
    param_shapes = [(1, ndim)]
    super(SumOfQuadratics, self).__init__(
        param_shapes, random_seed, noise_stdev)

  def objective(self, params, data, labels=None):
    epsilon = 1e-12
    # Assume dataset is designed so that the global minimum is at params=0.
    # Subtract loss at params=0, so that global minimum has objective value
    # epsilon (added to avoid floating point issues).
    return (tf.reduce_sum((params[0] - data)**2) - tf.reduce_sum(data**2) +
            epsilon)


class MatMulAlgorithm(Problem):
  """A 6-th order polynomial optimization problem.

  This problem is parametrized by n and k. A solution to this problem with
  objective value exactly zero defines a matrix multiplication algorithm of
  n x n matrices using k multiplications between matrices. When applied
  recursively, such an algorithm has complexity O(n^(log_n(k))).

  Given n, it is not known in general which values of k in [n^2, n^3] have a
  solution. There is always a solution with k = n^3 (this is the naive
  algorithm).

  In the special case n = 2, it is known that there are solutions for k = {7, 8}
  but not for k <= 6. For n = 3, it is known that there are exact solutions for
  23 <= k <= 27, and there are asymptotic solutions for k = {21, 22}, but the
  other cases are unknown.

  For a given n and k, if one solution exists then infinitely many solutions
  exist due to permutation and scaling symmetries in the parameters.

  This is a very hard problem for some values of n and k (e.g. n = 3, k = 21),
  but very easy for other values (e.g. n = 2, k = 7).

  For a given n and k, the specific formulation of this problem is as follows.
  Let theta_a, theta_b, theta_c be parameter matrices with respective dimensions
  [n**2, k], [n**2, k], [k, n**2]. Then for any matrices a, b with shape [n, n],
  we can form the matrix c with shape [n, n] via the operation:
      ((vec(a) * theta_a) .* (vec(b) * theta_b)) * theta_c = vec(c),  (#)
  where vec(x) is the operator that flattens a matrix with shape [n, n] into a
  row vector with shape [1, n**2], * denotes matrix multiplication and .*
  denotes elementwise multiplication.

  This operation, parameterized by theta_a, theta_b, theta_c, is a matrix
  multiplication algorithm iff c = a*b for all [n, n] matrices a and b. But
  actually it suffices to verify all combinations of one-hot matrices a and b,
  of which there are n**4 such combinations. This gives a batch of n**4 matrix
  triplets (a, b, c) such that equation (#) must hold for each triplet. We solve
  for theta_a, theta_b, theta_c by minimizing the sum of squares of errors
  across this batch.

  Finally, theta_c can be computed from theta_a and theta_b. Therefore it
  suffices to learn theta_a and theta_b, from which theta_c and therefore the
  objective value can be computed.
  """

  def __init__(self, n, k):
    assert isinstance(n, int), "n must be an integer"
    assert isinstance(k, int), "k must be an integer"
    assert n >= 2, "Must have n >= 2"
    assert k >= n**2 and k <= n**3, "Must have n**2 <= k <= n**3"

    param_shapes = [(n**2, k), (n**2, k)]  # theta_a, theta_b
    super(MatMulAlgorithm, self).__init__(
        param_shapes, random_seed=None, noise_stdev=0.0)

    self.n = n
    self.k = k

    # Build a batch of all combinations of one-hot matrices a, b, and their
    # respective products c. Correctness on this batch is a necessary and
    # sufficient condition for the algorithm to be valid. The number of matrices
    # in {a, b, c}_3d is n**4 and each matrix is n x n.
    onehots = np.identity(n**2).reshape(n**2, n, n)
    a_3d = np.repeat(onehots, n**2, axis=0)
    b_3d = np.tile(onehots, [n**2, 1, 1])
    c_3d = np.matmul(a_3d, b_3d)

    # Convert the batch to 2D Tensors.
    self.a = tf.constant(a_3d.reshape(n**4, n**2), tf.float32, name="a")
    self.b = tf.constant(b_3d.reshape(n**4, n**2), tf.float32, name="b")
    self.c = tf.constant(c_3d.reshape(n**4, n**2), tf.float32, name="c")

  def init_tensors(self, seed=None):
    # Initialize params such that the columns of theta_a and theta_b have L2
    # norm 1.
    def _param_initializer(shape, seed=None):
      x = tf.random_normal(shape, dtype=tf.float32, seed=seed)
      return tf.transpose(tf.nn.l2_normalize(tf.transpose(x), 1))

    return [_param_initializer(shape, seed) for shape in self.param_shapes]

  def objective(self, parameters, data=None, labels=None):
    theta_a = parameters[0]
    theta_b = parameters[1]

    # Compute theta_c from theta_a and theta_b.
    p = tf.matmul(self.a, theta_a) * tf.matmul(self.b, theta_b)
    p_trans = tf.transpose(p, name="p_trans")
    p_inv = tf.matmul(
        tf.matrix_inverse(tf.matmul(p_trans, p)), p_trans, name="p_inv")
    theta_c = tf.matmul(p_inv, self.c, name="theta_c")

    # Compute the "predicted" value of c.
    c_hat = tf.matmul(p, theta_c, name="c_hat")

    # Compute the loss (sum of squared errors).
    loss = tf.reduce_sum((c_hat - self.c)**2, name="loss")

    return loss


def matmul_problem_sequence(n, k_min, k_max):
  """Helper to generate a sequence of matrix multiplication problems."""
  return [(_Spec(MatMulAlgorithm, (n, k), {}), None, None)
          for k in range(k_min, k_max + 1)]


def init_fixed_variables(arrays):
  with tf.variable_scope(PARAMETER_SCOPE):
    params = [tf.Variable(arr.astype("float32")) for arr in arrays]
  return params


def _mesh(xlim, ylim, n):
  """Creates a 2D meshgrid covering the given ranges.

  Args:
    xlim: int that defines the desired x-range (-xlim, xlim)
    ylim: int that defines the desired y-range (-ylim, ylim)
    n: number of points in each dimension of the mesh

  Returns:
    xm: 2D array of x-values in the mesh
    ym: 2D array of y-values in the mesh
  """
  return np.meshgrid(np.linspace(-xlim, xlim, n),
                     np.linspace(-ylim, ylim, n))
