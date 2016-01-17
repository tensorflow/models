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
#
#==============================================================================

"""Neural GPU for Learning Algorithms."""

import math
import os
import random
import sys
import time

import google3

import matplotlib.animation as anim
import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf

from google3.third_party.tensorflow.python.platform import gfile
import google3.experimental.users.lukaszkaiser.neural_gpu.data_utils as data
import google3.experimental.users.lukaszkaiser.neural_gpu.neural_gpu as ngpu

tf.app.flags.DEFINE_float("lr", 0.1, "Learning rate.")
tf.app.flags.DEFINE_float("init_weight", 1.0, "Initial weights deviation.")
tf.app.flags.DEFINE_float("max_grad_norm", 0.05, "Clip gradients to this norm.")
tf.app.flags.DEFINE_float("cutoff", 1.2, "Cutoff at the gates.")
tf.app.flags.DEFINE_float("pull", 0.0005, "Starting pull of the relaxations.")
tf.app.flags.DEFINE_float("pull_incr", 1.2, "Increase pull by that much.")
tf.app.flags.DEFINE_float("dropout", 0.2, "Dropout that much.")
tf.app.flags.DEFINE_float("grad_noise_scale", 1.0, "Gradient noise scale.")
tf.app.flags.DEFINE_integer("batch_size", 64, "Batch size.")
tf.app.flags.DEFINE_integer("low_batch_size", 16, "Low batch size.")
tf.app.flags.DEFINE_integer("steps_per_checkpoint", 100, "Steps per epoch.")
tf.app.flags.DEFINE_integer("nmaps", 24, "Number of floats in each cell.")
tf.app.flags.DEFINE_integer("niclass", 14, "Number of classes (0 is padding).")
tf.app.flags.DEFINE_integer("noclass", 14, "Number of classes (0 is padding).")
tf.app.flags.DEFINE_integer("train_data_size", 5000, "Training examples/len.")
tf.app.flags.DEFINE_integer("max_length", 41, "Maximum length.")
tf.app.flags.DEFINE_integer("rx_step", 6, "Relax that many recursive steps.")
tf.app.flags.DEFINE_integer("random_seed", 125459, "Random seed.")
tf.app.flags.DEFINE_integer("nconvs", 2, "How many convolutions / 1 step.")
tf.app.flags.DEFINE_integer("kw", 3, "Kernel width.")
tf.app.flags.DEFINE_integer("kh", 3, "Kernel height.")
tf.app.flags.DEFINE_integer("height", 4, "Height.")
tf.app.flags.DEFINE_integer("forward_max", 401, "Maximum forward length.")
tf.app.flags.DEFINE_integer("jobid", -1, "Task id when running on borg.")
tf.app.flags.DEFINE_integer("nprint", 0, "How many test examples to print out.")
tf.app.flags.DEFINE_integer("mode", 0, "Mode: 0-train other-decode.")
tf.app.flags.DEFINE_string("task", "rev", "Which task are we learning?")
tf.app.flags.DEFINE_string("train_dir", "/tmp/", "Directory to store models.")

FLAGS = tf.app.flags.FLAGS


def initialize(sess):
  """Initialize data and model."""
  if FLAGS.jobid >= 0:
    data.log_filename = os.path.join(FLAGS.train_dir, "log%d" % FLAGS.jobid)
  data.print_out("NN ", newline=False)

  # Set random seed.
  seed = FLAGS.random_seed + max(0, FLAGS.jobid)
  tf.set_random_seed(seed)
  random.seed(seed)
  np.random.seed(seed)

  # Check data sizes.
  data.forward_max = max(FLAGS.forward_max, data.bins[-1])
  assert data.bins
  min_length = 3
  max_length = min(FLAGS.max_length, data.bins[-1])
  assert max_length + 1 > min_length
  while len(data.bins) > 1 and data.bins[-2] > max_length + 12:
    data.bins = data.bins[:-1]
  assert data.bins[0] > FLAGS.rx_step
  nclass = min(FLAGS.niclass, FLAGS.noclass)
  data_size = FLAGS.train_data_size if FLAGS.mode == 0 else 1000

  # Initialize data for each task.
  tasks = FLAGS.task.split("-")
  for t in tasks:
    for l in xrange(max_length + 11):
      data.init_data(t, l, data_size, nclass)
    data.init_data(t, data.bins[-2], data_size, nclass)
    data.init_data(t, data.bins[-1], data_size, nclass)
    end_size = 4 * 1024 if FLAGS.mode > 0 else 1024
    data.init_data(t, data.forward_max, end_size, nclass)

  # Print out parameters.
  curriculum = 0.12
  fin = ("cv %d kw %d h %d kh %d rxr %d bs %d ns %.2f t %s"
         % (FLAGS.nconvs, FLAGS.kw, FLAGS.height, FLAGS.kh, FLAGS.rx_step,
            FLAGS.batch_size, FLAGS.grad_noise_scale, FLAGS.task))
  fin = "data %d %s" % (FLAGS.train_data_size, fin)
  tag = ("df %.2f p %.3f lr %.2f iw %.2f cr %.2f nm %d d%.4f gn %.2f %s" %
         (FLAGS.cutoff, FLAGS.pull_incr, FLAGS.lr, FLAGS.init_weight,
          curriculum, FLAGS.nmaps, FLAGS.dropout, FLAGS.max_grad_norm, fin))
  data.print_out(tag)

  # Create checkpoint directory if it does not exist.
  checkpoint_dir = os.path.join(FLAGS.train_dir, "neural_gpu%s"
                                % ("" if FLAGS.jobid < 0 else str(FLAGS.jobid)))
  if not gfile.IsDirectory(checkpoint_dir):
    data.print_out("Creating checkpoint directory %s." % checkpoint_dir)
    gfile.MkDir(checkpoint_dir)

  # Create model and initialize it.
  tf.get_variable_scope().set_initializer(
      tf.uniform_unit_scaling_initializer(factor=1.8 * FLAGS.init_weight))
  model = ngpu.NeuralGPU(
      FLAGS.nmaps, FLAGS.nmaps, FLAGS.niclass, FLAGS.noclass, FLAGS.dropout,
      FLAGS.rx_step, FLAGS.max_grad_norm, FLAGS.cutoff, FLAGS.nconvs,
      FLAGS.kw, FLAGS.kh, FLAGS.height, FLAGS.mode, FLAGS.lr,
      FLAGS.pull, FLAGS.pull_incr, min_length + 3)
  data.print_out("Created model.")
  sess.run(tf.initialize_all_variables())
  data.print_out("Initialized variables.")

  # Load model from parameters if a checkpoint exists.
  ckpt = tf.train.get_checkpoint_state(checkpoint_dir)
  if ckpt and gfile.Exists(ckpt.model_checkpoint_path):
    data.print_out("Reading model parameters from %s"
                   % ckpt.model_checkpoint_path)
    model.saver.restore(sess, ckpt.model_checkpoint_path)

  # Return the model and needed variables.
  return (model, min_length, max_length, checkpoint_dir, curriculum)


def single_test(l, model, sess, task, nprint, batch_size, print_out=True,
                offset=None):
  """Test model on test data of length l using the given session."""
  inpt, target = data.get_batch(l, batch_size, False, task, offset)
  _, res, _, steps = model.step(sess, inpt, target, False)
  errors, total, seq = data.accuracy(inpt, res, target, batch_size, nprint)
  seq = float(seq) / batch_size
  if total > 0:
    errors = float(errors) / total
  if print_out:
    data.print_out("  %s len %d errors %.2f sequence-errors %.2f"
                   % (task, l, 100*errors, 100*seq))
  return errors, seq, (steps, inpt, [np.argmax(o, axis=1) for o in res])


def multi_test(l, model, sess, task, nprint, batch_size, offset=None):
  """Run multiple tests at lower batch size to save memory."""
  errors = 0.0
  seq = 0.0
  to_print = nprint
  low_batch = FLAGS.low_batch_size
  low_batch = min(low_batch, batch_size)
  for mstep in xrange(batch_size / low_batch):
    cur_offset = None if offset is None else offset + mstep * low_batch
    err, sq, _ = single_test(l, model, sess, task, to_print, low_batch, False,
                             cur_offset)
    to_print = max(0, to_print - low_batch)
    errors += err
    seq += sq
    if FLAGS.mode > 0:
      cur_errors = float(low_batch * errors) / ((mstep+1) * low_batch)
      cur_seq = float(low_batch * seq) / ((mstep+1) * low_batch)
      data.print_out("    %s multitest current errors %.2f sequence-errors %.2f"
                     % (task, 100*cur_errors, 100*cur_seq))
  errors = float(low_batch) * float(errors) / batch_size
  seq = float(low_batch) * float(seq) / batch_size
  data.print_out("  %s len %d errors %.2f sequence-errors %.2f"
                 % (task, l, 100*errors, 100*seq))
  return errors, seq


def train():
  """Main training function."""
  batch_size = FLAGS.batch_size
  tasks = FLAGS.task.split("-")
  with tf.Session() as sess:
    model, min_length, max_length, checkpoint_dir, curriculum = initialize(sess)
    max_cur_length = min(min_length + 3, max_length)
    prev_acc_perp = [1000000 for _ in xrange(3)]
    prev_sq = 1.0

    while True:
      global_step, pull, max_cur_length, learning_rate = sess.run(
          [model.global_step, model.pull, model.cur_length, model.lr])
      ep = global_step / FLAGS.steps_per_checkpoint
      acc_loss, acc_total, acc_errors, acc_seq = 0.0, 0, 0, 0
      acc_grad_norm, step_count, step_time = 0.0, 0, 0.0
      for _ in xrange(FLAGS.steps_per_checkpoint):
        global_step += 1
        task = random.choice(tasks)
        l1 = np.random.randint(max_cur_length - min_length + 1) + min_length
        l = l1
        if np.random.randint(10) > 3:  # Prefer longer stuff 60% of time.
          l = np.random.randint(max_cur_length - min_length+1) + min_length
          l = max(l, l1)
        if np.random.randint(4) < 1:  # Mixed learning: once in a while big.
          l = np.random.randint(max_length - min_length + 1) + min_length
          l = max(l, l1)
        start_time = time.time()
        inp, target = data.get_batch(l, batch_size, True, task)
        stepp = math.pow(global_step, -0.55)
        noise_param = math.sqrt(stepp * 20 * prev_sq) * FLAGS.grad_noise_scale
        loss, res, gnorm, _ = model.step(sess, inp, target, True, noise_param)
        step_time += time.time() - start_time
        acc_grad_norm += float(gnorm)
        if l < max_cur_length + 1:
          step_count += 1
          acc_loss += loss
          errors, total, seq = data.accuracy(inp, res, target,
                                             batch_size, 0)
          acc_total += total
          acc_errors += errors
          acc_seq += seq
      acc_loss /= step_count
      step_time /= FLAGS.steps_per_checkpoint
      acc_seq = float(acc_seq) / (step_count * batch_size)
      prev_sq = acc_seq
      acc_errors = float(acc_errors) / acc_total if acc_total > 0 else 1.0
      msg1 = "ep %d st %.2f lr %.8f" % (ep, step_time, learning_rate)
      msg2 = "pl %.3f cme %.3f" % (pull, curriculum)
      msg = ("%s %s gn %.8f"
             % (msg1, msg2, acc_grad_norm / FLAGS.steps_per_checkpoint))
      data.print_out("%s len %d ppx %.8f errs %.2f sq %.2f" %
                     (msg, max_cur_length, data.safe_exp(acc_loss),
                      100*acc_errors, 100*acc_seq))
      if curriculum > acc_seq:
        prev_acc_perp.append(1000000)
        do_incr = True
        while do_incr and max_cur_length < max_length:
          sess.run(model.cur_length_incr_op)
          for t in tasks:
            if data.train_set[t]: do_incr = False
        if pull < 1:
          sess.run(model.pull_incr_op)
        else:
          data.print_out("  Averaging parameters.")
          sess.run([model.avg_op, model.lr_decay_op])
      else:
        acc_perp = data.safe_exp(acc_loss)
        if acc_perp > max(prev_acc_perp[-3:]):
          sess.run(model.lr_decay_op)
        prev_acc_perp.append(acc_perp)
      checkpoint_path = os.path.join(checkpoint_dir, "neural_gpu.ckpt")
      model.saver.save(sess, checkpoint_path,
                       global_step=model.global_step)
      # Run evaluation.
      should_exit = True
      bound = data.bins[-1] + 1
      for t in tasks:
        l = min_length
        while l < max_length + 12 and l < bound:
          _, sq, _ = single_test(l, model, sess, t, FLAGS.nprint, batch_size)
          l += 1
          while l < bound + 1 and not data.test_set[t][l]:
            l += 1
        if sq < 0.5:
          _, sq = multi_test(data.forward_max, model, sess, t, FLAGS.nprint,
                             batch_size * 4)
        if sq > 0.001: should_exit = False
      if should_exit:
        if data.forward_max > 4000 and len(tasks) == 1:
          multi_test(data.forward_max, model, sess, tasks[0], FLAGS.nprint,
                     batch_size * 16, 0)


def animate(l, test_data, anim_size):
  """Create animation for the given data (hacky matplotlib use)."""
  xf = 12
  fps = 2
  fig = plt.figure(figsize=(16, 9), facecolor="white")
  ax = fig.add_axes([0, 0, 1, 1], frameon=False, zorder=2)
  ax.set_xticks([i * 24-0.5 for i in xrange(4)])
  ax.set_xticklabels([])
  ax.set_yticks([i - 0.5 for i in xrange(l+1)])
  ax.grid(which="major", axis="both", linestyle="-", color="black")
  text_fields = []
  text_size = 24*32/l
  for y in xrange(l):
    text_fields.append(ax.text(
        11.25, y + 0.15, "", color="g", ha="center", va="center",
        bbox={"facecolor": "b", "alpha": 0.01, "pad": 24 * text_size},
        size=text_size - (4 * 32 / l), animated=True))
  im = ax.imshow(np.zeros_like(test_data[0][0][0]), vmin=-1.0,
                 vmax=1.0, cmap="gray", aspect="auto", origin="upper",
                 interpolation="none", animated=True)
  im.set_zorder(1)
  def to_symbol(i):
    if i == 0: return ""
    if i == 11: return "+"
    if i == 12: return "*"
    return str(i-1)
  def animation_update(frame_no, test_data, xf, im, text_fields):
    """Update an animation frame."""
    steps, inpt, out_raw = test_data
    length = len(steps)
    batch = frame_no / (fps * (l+4*xf))
    index = int((frame_no % (fps * (l+4*xf))) / fps)
    # Cut output after first padding.
    out = [out_raw[i][batch] for i in xrange(len(text_fields))]
    if 0 in out:
      i = out.index(0)
      out = out[0:i] + [0 for _ in xrange(len(out) - i)]
    # Show the state after the first frames.
    if index >= 2*xf:
      im.set_array(steps[min(length - 1, index - 2*xf)][batch])
      for i, t in enumerate(text_fields):
        if index - 2*xf < length:
          t.set_text("")
        else:
          t.set_text(to_symbol(out[i]))
    else:
      for i, t in enumerate(text_fields):
        t.set_text(to_symbol(inpt[i][batch]) if index < xf else "")
      if index < xf:
        im.set_array(np.zeros_like(steps[0][0]))
      else:
        im.set_array(steps[0][batch])
    return im,
  animation = anim.FuncAnimation(
      fig, animation_update, blit=True, frames=(l+4*xf)*anim_size*fps,
      interval=500/fps, fargs=(test_data, xf, im, text_fields))
  animation.save("/tmp/neural_gpu.mp4", writer="mencoder", fps=4*fps, dpi=3*80)


def evaluate():
  """Evaluate an existing model."""
  batch_size = FLAGS.batch_size
  tasks = FLAGS.task.split("-")
  with tf.Session() as sess:
    model, min_length, max_length, _, _ = initialize(sess)
    bound = data.bins[-1] + 1
    for t in tasks:
      l = min_length
      while l < max_length + 12 and l < bound:
        _, sq, _ = single_test(l, model, sess, t, FLAGS.nprint, batch_size)
        l += 1
        while l < bound + 1 and not data.test_set[t][l]:
          l += 1
      # Animate.
      anim_size = 2
      _, _, test_data = single_test(l, model, sess, t, 0, anim_size)
      animate(l, test_data, anim_size)
      # More tests.
      _, sq = multi_test(data.forward_max, model, sess, t, FLAGS.nprint,
                         batch_size * 4)
    if sq < 0.01:  # More tests.
      if data.forward_max > 4000 and len(tasks) == 1:
        multi_test(data.forward_max, model, sess, tasks[0], FLAGS.nprint,
                   batch_size * 64, 0)


def interactive():
  """Interactively probe an existing model."""
  with tf.Session() as sess:
    model, _, _, _, _ = initialize(sess)
    sys.stdout.write("> ")
    sys.stdout.flush()
    inpt = sys.stdin.readline()
    while inpt:
      ids = [int(c) for c in inpt.strip()]
      inpt, target = data.get_batch(len(ids), 1, False, "",
                                    preset=(ids, [0 for _ in ids]))
      _, res, _, _ = model.step(sess, inpt, target, False)
      res = [np.argmax(o, axis=1) for o in res]
      print " ".join([str(output[0]) for output in res])
      sys.stdout.write("> ")
      sys.stdout.flush()
      inpt = sys.stdin.readline()


def main(_):
  if FLAGS.mode == 0:
    train()
  elif FLAGS.mode == 1:
    evaluate()
  else:
    interactive()

if __name__ == "__main__":
  tf.app.run()
