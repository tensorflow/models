# Copyright 2015 Google Inc. All Rights Reserved.
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
"""The Neural GPU Model."""

import time

import tensorflow as tf

import data_utils


def conv_linear(args, kw, kh, nin, nout, do_bias, bias_start, prefix):
  """Convolutional linear map."""
  assert args is not None
  if not isinstance(args, (list, tuple)):
    args = [args]
  with tf.variable_scope(prefix):
    k = tf.get_variable("CvK", [kw, kh, nin, nout])
    if len(args) == 1:
      res = tf.nn.conv2d(args[0], k, [1, 1, 1, 1], "SAME")
    else:
      res = tf.nn.conv2d(tf.concat(3, args), k, [1, 1, 1, 1], "SAME")
    if not do_bias: return res
    bias_term = tf.get_variable("CvB", [nout],
                                initializer=tf.constant_initializer(0.0))
    return res + bias_term + bias_start


def sigmoid_cutoff(x, cutoff):
  """Sigmoid with cutoff, e.g., 1.2sigmoid(x) - 0.1."""
  y = tf.sigmoid(x)
  if cutoff < 1.01: return y
  d = (cutoff - 1.0) / 2.0
  return tf.minimum(1.0, tf.maximum(0.0, cutoff * y - d))


def tanh_cutoff(x, cutoff):
  """Tanh with cutoff, e.g., 1.1tanh(x) cut to [-1. 1]."""
  y = tf.tanh(x)
  if cutoff < 1.01: return y
  d = (cutoff - 1.0) / 2.0
  return tf.minimum(1.0, tf.maximum(-1.0, (1.0 + d) * y))


def conv_gru(inpts, mem, kw, kh, nmaps, cutoff, prefix):
  """Convolutional GRU."""
  def conv_lin(args, suffix, bias_start):
    return conv_linear(args, kw, kh, len(args) * nmaps, nmaps, True, bias_start,
                       prefix + "/" + suffix)
  reset = sigmoid_cutoff(conv_lin(inpts + [mem], "r", 1.0), cutoff)
  # candidate = tanh_cutoff(conv_lin(inpts + [reset * mem], "c", 0.0), cutoff)
  candidate = tf.tanh(conv_lin(inpts + [reset * mem], "c", 0.0))
  gate = sigmoid_cutoff(conv_lin(inpts + [mem], "g", 1.0), cutoff)
  return gate * mem + (1 - gate) * candidate


@tf.RegisterGradient("CustomIdG")
def _custom_id_grad(_, grads):
  return grads


def quantize(t, quant_scale, max_value=1.0):
  """Quantize a tensor t with each element in [-max_value, max_value]."""
  t = tf.minimum(max_value, tf.maximum(t, -max_value))
  big = quant_scale * (t + max_value) + 0.5
  with tf.get_default_graph().gradient_override_map({"Floor": "CustomIdG"}):
    res = (tf.floor(big) / quant_scale) - max_value
  return res


def quantize_weights_op(quant_scale, max_value):
  ops = [v.assign(quantize(v, quant_scale, float(max_value)))
         for v in tf.trainable_variables()]
  return tf.group(*ops)


def relaxed_average(var_name_suffix, rx_step):
  """Calculate the average of relaxed variables having var_name_suffix."""
  relaxed_vars = []
  for l in xrange(rx_step):
    with tf.variable_scope("RX%d" % l, reuse=True):
      try:
        relaxed_vars.append(tf.get_variable(var_name_suffix))
      except ValueError:
        pass
  dsum = tf.add_n(relaxed_vars)
  avg = dsum / len(relaxed_vars)
  diff = [v - avg for v in relaxed_vars]
  davg = tf.add_n([d*d for d in diff])
  return avg, tf.reduce_sum(davg)


def relaxed_distance(rx_step):
  """Distance between relaxed variables and their average."""
  res, ops, rx_done = [], [], {}
  for v in tf.trainable_variables():
    if v.name[0:2] == "RX":
      rx_name = v.op.name[v.name.find("/") + 1:]
      if rx_name not in rx_done:
        avg, dist_loss = relaxed_average(rx_name, rx_step)
        res.append(dist_loss)
        rx_done[rx_name] = avg
      ops.append(v.assign(rx_done[rx_name]))
  return tf.add_n(res), tf.group(*ops)


def make_dense(targets, noclass):
  """Move a batch of targets to a dense 1-hot representation."""
  with tf.device("/cpu:0"):
    shape = tf.shape(targets)
    batch_size = shape[0]
    indices = targets + noclass * tf.range(0, batch_size)
    length = tf.expand_dims(batch_size * noclass, 0)
    dense = tf.sparse_to_dense(indices, length, 1.0, 0.0)
  return tf.reshape(dense, [-1, noclass])


def check_for_zero(sparse):
  """In a sparse batch of ints, make 1.0 if it's 0 and 0.0 else."""
  with tf.device("/cpu:0"):
    shape = tf.shape(sparse)
    batch_size = shape[0]
    sparse = tf.minimum(sparse, 1)
    indices = sparse + 2 * tf.range(0, batch_size)
    dense = tf.sparse_to_dense(indices, tf.expand_dims(2 * batch_size, 0),
                               1.0, 0.0)
    reshaped = tf.reshape(dense, [-1, 2])
  return tf.reshape(tf.slice(reshaped, [0, 0], [-1, 1]), [-1])


class NeuralGPU(object):
  """Neural GPU Model."""

  def __init__(self, nmaps, vec_size, niclass, noclass, dropout, rx_step,
               max_grad_norm, cutoff, nconvs, kw, kh, height, mode,
               learning_rate, pull, pull_incr, min_length, act_noise=0.0):
    # Feeds for parameters and ops to update them.
    self.global_step = tf.Variable(0, trainable=False)
    self.cur_length = tf.Variable(min_length, trainable=False)
    self.cur_length_incr_op = self.cur_length.assign_add(1)
    self.lr = tf.Variable(float(learning_rate), trainable=False)
    self.lr_decay_op = self.lr.assign(self.lr * 0.98)
    self.pull = tf.Variable(float(pull), trainable=False)
    self.pull_incr_op = self.pull.assign(self.pull * pull_incr)
    self.do_training = tf.placeholder(tf.float32, name="do_training")
    self.noise_param = tf.placeholder(tf.float32, name="noise_param")

    # Feeds for inputs, targets, outputs, losses, etc.
    self.input = []
    self.target = []
    for l in xrange(data_utils.forward_max + 1):
      self.input.append(tf.placeholder(tf.int32, name="inp{0}".format(l)))
      self.target.append(tf.placeholder(tf.int32, name="tgt{0}".format(l)))
    self.outputs = []
    self.losses = []
    self.grad_norms = []
    self.updates = []

    # Computation.
    inp0_shape = tf.shape(self.input[0])
    batch_size = inp0_shape[0]
    with tf.device("/cpu:0"):
      emb_weights = tf.get_variable(
          "embedding", [niclass, vec_size],
          initializer=tf.random_uniform_initializer(-1.7, 1.7))
      e0 = tf.scatter_update(emb_weights,
                             tf.constant(0, dtype=tf.int32, shape=[1]),
                             tf.zeros([1, vec_size]))

    adam = tf.train.AdamOptimizer(self.lr, epsilon=1e-4)

    # Main graph creation loop, for every bin in data_utils.
    self.steps = []
    for length in sorted(list(set(data_utils.bins + [data_utils.forward_max]))):
      data_utils.print_out("Creating model for bin of length %d." % length)
      start_time = time.time()
      if length > data_utils.bins[0]:
        tf.get_variable_scope().reuse_variables()

      # Embed inputs and calculate mask.
      with tf.device("/cpu:0"):
        with tf.control_dependencies([e0]):
          embedded = [tf.nn.embedding_lookup(emb_weights, self.input[l])
                      for l in xrange(length)]
        # Mask to 0-out padding space in each step.
        imask = [check_for_zero(self.input[l]) for l in xrange(length)]
        omask = [check_for_zero(self.target[l]) for l in xrange(length)]
        mask = [1.0 - (imask[i] * omask[i]) for i in xrange(length)]
        mask = [tf.reshape(m, [-1, 1]) for m in mask]
        # Use a shifted mask for step scaling and concatenated for weights.
        shifted_mask = mask + [tf.zeros_like(mask[0])]
        scales = [shifted_mask[i] * (1.0 - shifted_mask[i+1])
                  for i in xrange(length)]
        scales = [tf.reshape(s, [-1, 1, 1, 1]) for s in scales]
        mask = tf.concat(1, mask[0:length])  # batch x length
        weights = mask
        # Add a height dimension to mask to use later for masking.
        mask = tf.reshape(mask, [-1, length, 1, 1])
        mask = tf.concat(2, [mask for _ in xrange(height)]) + tf.zeros(
            tf.pack([batch_size, length, height, nmaps]), dtype=tf.float32)

      # Start is a length-list of batch-by-nmaps tensors, reshape and concat.
      start = [tf.tanh(embedded[l]) for l in xrange(length)]
      start = [tf.reshape(start[l], [-1, 1, nmaps]) for l in xrange(length)]
      start = tf.reshape(tf.concat(1, start), [-1, length, 1, nmaps])

      # First image comes from start by applying one convolution and adding 0s.
      first = conv_linear(start, 1, 1, vec_size, nmaps, True, 0.0, "input")
      first = [first] + [tf.zeros(tf.pack([batch_size, length, 1, nmaps]),
                                  dtype=tf.float32) for _ in xrange(height - 1)]
      first = tf.concat(2, first)

      # Computation steps.
      keep_prob = 1.0 - self.do_training * (dropout * 8.0 / float(length))
      step = [tf.nn.dropout(first, keep_prob) * mask]
      act_noise_scale = act_noise * self.do_training * self.pull
      outputs = []
      for it in xrange(length):
        with tf.variable_scope("RX%d" % (it % rx_step)) as vs:
          if it >= rx_step:
            vs.reuse_variables()
          cur = step[it]
          # Do nconvs-many CGRU steps.
          for layer in xrange(nconvs):
            cur = conv_gru([], cur, kw, kh, nmaps, cutoff, "cgru_%d" % layer)
            cur *= mask
          outputs.append(tf.slice(cur, [0, 0, 0, 0], [-1, -1, 1, -1]))
          cur = tf.nn.dropout(cur, keep_prob)
          if act_noise > 0.00001:
            cur += tf.truncated_normal(tf.shape(cur)) * act_noise_scale
          step.append(cur * mask)

      self.steps.append([tf.reshape(s, [-1, length, height * nmaps])
                         for s in step])
      # Output is the n-th step output; n = current length, as in scales.
      output = tf.add_n([outputs[i] * scales[i] for i in xrange(length)])
      # Final convolution to get logits, list outputs.
      output = conv_linear(output, 1, 1, nmaps, noclass, True, 0.0, "output")
      output = tf.reshape(output, [-1, length, noclass])
      external_output = [tf.reshape(o, [-1, noclass])
                         for o in list(tf.split(1, length, output))]
      external_output = [tf.nn.softmax(o) for o in external_output]
      self.outputs.append(external_output)

      # Calculate cross-entropy loss and normalize it.
      targets = tf.concat(1, [make_dense(self.target[l], noclass)
                              for l in xrange(length)])
      targets = tf.reshape(targets, [-1, noclass])
      xent = tf.reshape(tf.nn.softmax_cross_entropy_with_logits(
          tf.reshape(output, [-1, noclass]), targets), [-1, length])
      perp_loss = tf.reduce_sum(xent * weights)
      perp_loss /= tf.cast(batch_size, dtype=tf.float32)
      perp_loss /= length

      # Final loss: cross-entropy + shared parameter relaxation part.
      relax_dist, self.avg_op = relaxed_distance(rx_step)
      total_loss = perp_loss + relax_dist * self.pull
      self.losses.append(perp_loss)

      # Gradients and Adam update operation.
      if length == data_utils.bins[0] or (mode == 0 and
                                          length < data_utils.bins[-1] + 1):
        data_utils.print_out("Creating backward for bin of length %d." % length)
        params = tf.trainable_variables()
        grads = tf.gradients(total_loss, params)
        grads, norm = tf.clip_by_global_norm(grads, max_grad_norm)
        self.grad_norms.append(norm)
        for grad in grads:
          if isinstance(grad, tf.Tensor):
            grad += tf.truncated_normal(tf.shape(grad)) * self.noise_param
        update = adam.apply_gradients(zip(grads, params),
                                      global_step=self.global_step)
        self.updates.append(update)
      data_utils.print_out("Created model for bin of length %d in"
                           " %.2f s." % (length, time.time() - start_time))
    self.saver = tf.train.Saver(tf.all_variables())

  def step(self, sess, inp, target, do_backward, noise_param=None,
           get_steps=False):
    """Run a step of the network."""
    assert len(inp) == len(target)
    length = len(target)
    feed_in = {}
    feed_in[self.noise_param.name] = noise_param if noise_param else 0.0
    feed_in[self.do_training.name] = 1.0 if do_backward else 0.0
    feed_out = []
    index = len(data_utils.bins)
    if length < data_utils.bins[-1] + 1:
      index = data_utils.bins.index(length)
    if do_backward:
      feed_out.append(self.updates[index])
      feed_out.append(self.grad_norms[index])
    feed_out.append(self.losses[index])
    for l in xrange(length):
      feed_in[self.input[l].name] = inp[l]
    for l in xrange(length):
      feed_in[self.target[l].name] = target[l]
      feed_out.append(self.outputs[index][l])
    if get_steps:
      for l in xrange(length+1):
        feed_out.append(self.steps[index][l])
    res = sess.run(feed_out, feed_in)
    offset = 0
    norm = None
    if do_backward:
      offset = 2
      norm = res[1]
    outputs = res[offset + 1:offset + 1 + length]
    steps = res[offset + 1 + length:] if get_steps else None
    return res[offset], outputs, norm, steps
