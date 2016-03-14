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
"""Neural GPU for Learning Algorithms."""

import math
import os
import random
import sys
import time

import matplotlib.animation as anim
import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf

from tensorflow.python.platform import gfile

import data_utils as data
import neural_gpu

tf.app.flags.DEFINE_float("lr", 0.001, "Learning rate.")
tf.app.flags.DEFINE_float("init_weight", 1.0, "Initial weights deviation.")
tf.app.flags.DEFINE_float("max_grad_norm", 1.0, "Clip gradients to this norm.")
tf.app.flags.DEFINE_float("cutoff", 1.2, "Cutoff at the gates.")
tf.app.flags.DEFINE_float("pull", 0.0005, "Starting pull of the relaxations.")
tf.app.flags.DEFINE_float("pull_incr", 1.2, "Increase pull by that much.")
tf.app.flags.DEFINE_float("curriculum_bound", 0.15, "Move curriculum < this.")
tf.app.flags.DEFINE_float("dropout", 0.15, "Dropout that much.")
tf.app.flags.DEFINE_float("grad_noise_scale", 0.0, "Gradient noise scale.")
tf.app.flags.DEFINE_integer("batch_size", 32, "Batch size.")
tf.app.flags.DEFINE_integer("low_batch_size", 16, "Low batch size.")
tf.app.flags.DEFINE_integer("steps_per_checkpoint", 200, "Steps per epoch.")
tf.app.flags.DEFINE_integer("nmaps", 128, "Number of floats in each cell.")
tf.app.flags.DEFINE_integer("niclass", 33, "Number of classes (0 is padding).")
tf.app.flags.DEFINE_integer("noclass", 33, "Number of classes (0 is padding).")
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
tf.app.flags.DEFINE_bool("animate", False, "Whether to produce an animation.")
tf.app.flags.DEFINE_bool("quantize", False, "Whether to quantize variables.")
tf.app.flags.DEFINE_string("task", "rev", "Which task are we learning?")
tf.app.flags.DEFINE_string("train_dir", "/tmp/", "Directory to store models.")
tf.app.flags.DEFINE_string("ensemble", "", "Model paths for ensemble.")

FLAGS = tf.app.flags.FLAGS
EXTRA_EVAL = 12


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
  assert data.bins
  min_length = 3
  max_length = min(FLAGS.max_length, data.bins[-1])
  assert max_length + 1 > min_length
  while len(data.bins) > 1 and data.bins[-2] > max_length + EXTRA_EVAL:
    data.bins = data.bins[:-1]
  assert data.bins[0] > FLAGS.rx_step
  data.forward_max = max(FLAGS.forward_max, data.bins[-1])
  nclass = min(FLAGS.niclass, FLAGS.noclass)
  data_size = FLAGS.train_data_size if FLAGS.mode == 0 else 1000

  # Initialize data for each task.
  tasks = FLAGS.task.split("-")
  for t in tasks:
    for l in xrange(max_length + EXTRA_EVAL - 1):
      data.init_data(t, l, data_size, nclass)
    data.init_data(t, data.bins[-2], data_size, nclass)
    data.init_data(t, data.bins[-1], data_size, nclass)
    end_size = 4 * 1024 if FLAGS.mode > 0 else 1024
    data.init_data(t, data.forward_max, end_size, nclass)

  # Print out parameters.
  curriculum = FLAGS.curriculum_bound
  msg1 = ("layers %d kw %d h %d kh %d relax %d batch %d noise %.2f task %s"
          % (FLAGS.nconvs, FLAGS.kw, FLAGS.height, FLAGS.kh, FLAGS.rx_step,
             FLAGS.batch_size, FLAGS.grad_noise_scale, FLAGS.task))
  msg2 = "data %d %s" % (FLAGS.train_data_size, msg1)
  msg3 = ("cut %.2f pull %.3f lr %.2f iw %.2f cr %.2f nm %d d%.4f gn %.2f %s" %
          (FLAGS.cutoff, FLAGS.pull_incr, FLAGS.lr, FLAGS.init_weight,
           curriculum, FLAGS.nmaps, FLAGS.dropout, FLAGS.max_grad_norm, msg2))
  data.print_out(msg3)

  # Create checkpoint directory if it does not exist.
  checkpoint_dir = os.path.join(FLAGS.train_dir, "neural_gpu%s"
                                % ("" if FLAGS.jobid < 0 else str(FLAGS.jobid)))
  if not gfile.IsDirectory(checkpoint_dir):
    data.print_out("Creating checkpoint directory %s." % checkpoint_dir)
    gfile.MkDir(checkpoint_dir)

  # Create model and initialize it.
  tf.get_variable_scope().set_initializer(
      tf.uniform_unit_scaling_initializer(factor=1.8 * FLAGS.init_weight))
  model = neural_gpu.NeuralGPU(
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

  # Check if there are ensemble models and get their checkpoints.
  ensemble = []
  ensemble_dir_list = [d for d in FLAGS.ensemble.split(",") if d]
  for ensemble_dir in ensemble_dir_list:
    ckpt = tf.train.get_checkpoint_state(ensemble_dir)
    if ckpt and gfile.Exists(ckpt.model_checkpoint_path):
      data.print_out("Found ensemble model %s" % ckpt.model_checkpoint_path)
      ensemble.append(ckpt.model_checkpoint_path)

  # Return the model and needed variables.
  return (model, min_length, max_length, checkpoint_dir, curriculum, ensemble)


def single_test(l, model, sess, task, nprint, batch_size, print_out=True,
                offset=None, ensemble=None, get_steps=False):
  """Test model on test data of length l using the given session."""
  inpt, target = data.get_batch(l, batch_size, False, task, offset)
  _, res, _, steps = model.step(sess, inpt, target, False, get_steps=get_steps)
  errors, total, seq_err = data.accuracy(inpt, res, target, batch_size, nprint)
  seq_err = float(seq_err) / batch_size
  if total > 0:
    errors = float(errors) / total
  if print_out:
    data.print_out("  %s len %d errors %.2f sequence-errors %.2f"
                   % (task, l, 100*errors, 100*seq_err))
  # Ensemble eval.
  if ensemble:
    results = []
    for m in ensemble:
      model.saver.restore(sess, m)
      _, result, _, _ = model.step(sess, inpt, target, False)
      m_errors, m_total, m_seq_err = data.accuracy(inpt, result, target,
                                                   batch_size, nprint)
      m_seq_err = float(m_seq_err) / batch_size
      if total > 0:
        m_errors = float(m_errors) / m_total
      data.print_out("     %s len %d m-errors %.2f m-sequence-errors %.2f"
                     % (task, l, 100*m_errors, 100*m_seq_err))
      results.append(result)
    ens = [sum(o) for o in zip(*results)]
    errors, total, seq_err = data.accuracy(inpt, ens, target,
                                           batch_size, nprint)
    seq_err = float(seq_err) / batch_size
    if total > 0:
      errors = float(errors) / total
    if print_out:
      data.print_out("  %s len %d ens-errors %.2f ens-sequence-errors %.2f"
                     % (task, l, 100*errors, 100*seq_err))
  return errors, seq_err, (steps, inpt, [np.argmax(o, axis=1) for o in res])


def multi_test(l, model, sess, task, nprint, batch_size, offset=None,
               ensemble=None):
  """Run multiple tests at lower batch size to save memory."""
  errors, seq_err = 0.0, 0.0
  to_print = nprint
  low_batch = FLAGS.low_batch_size
  low_batch = min(low_batch, batch_size)
  for mstep in xrange(batch_size / low_batch):
    cur_offset = None if offset is None else offset + mstep * low_batch
    err, sq_err, _ = single_test(l, model, sess, task, to_print, low_batch,
                                 False, cur_offset, ensemble=ensemble)
    to_print = max(0, to_print - low_batch)
    errors += err
    seq_err += sq_err
    if FLAGS.mode > 0:
      cur_errors = float(low_batch * errors) / ((mstep+1) * low_batch)
      cur_seq_err = float(low_batch * seq_err) / ((mstep+1) * low_batch)
      data.print_out("    %s multitest current errors %.2f sequence-errors %.2f"
                     % (task, 100*cur_errors, 100*cur_seq_err))
  errors = float(low_batch) * float(errors) / batch_size
  seq_err = float(low_batch) * float(seq_err) / batch_size
  data.print_out("  %s len %d errors %.2f sequence-errors %.2f"
                 % (task, l, 100*errors, 100*seq_err))
  return errors, seq_err


def train():
  """Train the model."""
  batch_size = FLAGS.batch_size
  tasks = FLAGS.task.split("-")
  with tf.Session() as sess:
    (model, min_length, max_length, checkpoint_dir,
     curriculum, _) = initialize(sess)
    quant_op = neural_gpu.quantize_weights_op(512, 8)
    max_cur_length = min(min_length + 3, max_length)
    prev_acc_perp = [1000000 for _ in xrange(3)]
    prev_seq_err = 1.0

    # Main traning loop.
    while True:
      global_step, pull, max_cur_length, learning_rate = sess.run(
          [model.global_step, model.pull, model.cur_length, model.lr])
      acc_loss, acc_total, acc_errors, acc_seq_err = 0.0, 0, 0, 0
      acc_grad_norm, step_count, step_time = 0.0, 0, 0.0
      for _ in xrange(FLAGS.steps_per_checkpoint):
        global_step += 1
        task = random.choice(tasks)

        # Select the length for curriculum learning.
        l = np.random.randint(max_cur_length - min_length + 1) + min_length
        # Prefer longer stuff 60% of time.
        if np.random.randint(100) < 60:
          l1 = np.random.randint(max_cur_length - min_length+1) + min_length
          l = max(l, l1)
        # Mixed curriculum learning: in 25% of cases go to any larger length.
        if np.random.randint(100) < 25:
          l1 = np.random.randint(max_length - min_length + 1) + min_length
          l = max(l, l1)

        # Run a step and time it.
        start_time = time.time()
        inp, target = data.get_batch(l, batch_size, True, task)
        noise_param = math.sqrt(math.pow(global_step, -0.55) *
                                prev_seq_err) * FLAGS.grad_noise_scale
        loss, res, gnorm, _ = model.step(sess, inp, target, True, noise_param)
        step_time += time.time() - start_time
        acc_grad_norm += float(gnorm)

        # Accumulate statistics only if we did not exceed curriculum length.
        if l < max_cur_length + 1:
          step_count += 1
          acc_loss += loss
          errors, total, seq_err = data.accuracy(inp, res, target,
                                                 batch_size, 0)
          acc_total += total
          acc_errors += errors
          acc_seq_err += seq_err

      # Normalize and print out accumulated statistics.
      acc_loss /= step_count
      step_time /= FLAGS.steps_per_checkpoint
      acc_seq_err = float(acc_seq_err) / (step_count * batch_size)
      prev_seq_err = max(0.0, acc_seq_err - 0.02)  # No noise at error < 2%.
      acc_errors = float(acc_errors) / acc_total if acc_total > 0 else 1.0
      msg1 = "step %d step-time %.2f" % (global_step, step_time)
      msg2 = "lr %.8f pull %.3f" % (learning_rate, pull)
      msg3 = ("%s %s grad-norm %.8f"
              % (msg1, msg2, acc_grad_norm / FLAGS.steps_per_checkpoint))
      data.print_out("%s len %d ppx %.8f errors %.2f sequence-errors %.2f" %
                     (msg3, max_cur_length, data.safe_exp(acc_loss),
                      100*acc_errors, 100*acc_seq_err))

      # If errors are below the curriculum threshold, move curriculum forward.
      if curriculum > acc_seq_err:
        if FLAGS.quantize:
          # Quantize weights.
          data.print_out("  Quantizing parameters.")
          sess.run([quant_op])
        # Increase current length (until the next with training data).
        do_incr = True
        while do_incr and max_cur_length < max_length:
          sess.run(model.cur_length_incr_op)
          for t in tasks:
            if data.train_set[t]: do_incr = False
        # Forget last perplexities if we're not yet at the end.
        if max_cur_length < max_length:
          prev_acc_perp.append(1000000)
        # Either increase pull or, if it's large, average parameters.
        if pull < 0.1:
          sess.run(model.pull_incr_op)
        else:
          data.print_out("  Averaging parameters.")
          sess.run(model.avg_op)
          if acc_seq_err < (curriculum / 3.0):
            sess.run(model.lr_decay_op)

      # Lower learning rate if we're worse than the last 3 checkpoints.
      acc_perp = data.safe_exp(acc_loss)
      if acc_perp > max(prev_acc_perp[-3:]):
        sess.run(model.lr_decay_op)
      prev_acc_perp.append(acc_perp)

      # Save checkpoint.
      checkpoint_path = os.path.join(checkpoint_dir, "neural_gpu.ckpt")
      model.saver.save(sess, checkpoint_path,
                       global_step=model.global_step)

      # Run evaluation.
      bound = data.bins[-1] + 1
      for t in tasks:
        l = min_length
        while l < max_length + EXTRA_EVAL and l < bound:
          _, seq_err, _ = single_test(l, model, sess, t,
                                      FLAGS.nprint, batch_size)
          l += 1
          while l < bound + 1 and not data.test_set[t][l]:
            l += 1
        if seq_err < 0.05:  # Run larger test if we're good enough.
          _, seq_err = multi_test(data.forward_max, model, sess, t,
                                  FLAGS.nprint, batch_size * 4)
      if seq_err < 0.01:  # Super-large test on 1-task large-forward models.
        if data.forward_max > 4000 and len(tasks) == 1:
          multi_test(data.forward_max, model, sess, tasks[0], FLAGS.nprint,
                     batch_size * 16, 0)


def animate(l, test_data, anim_size):
  """Create animation for the given data (hacky matplotlib use)."""
  xf = 12  # Extra frames to slow down at start and end.
  fps = 2  # Frames per step.

  # Make the figure.
  fig = plt.figure(figsize=(16, 9), facecolor="white")
  ax = fig.add_axes([0, 0, 1, 1], frameon=False, zorder=2)
  ax.set_xticks([i * 24-0.5 for i in xrange(4)])
  ax.set_xticklabels([])
  ax.set_yticks([i - 0.5 for i in xrange(l+1)])
  ax.grid(which="major", axis="both", linestyle="-", color="black")
  # We need text fields.
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

  # Main animation step.
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
          t.set_text(data.to_symbol(out[i]))
    else:
      for i, t in enumerate(text_fields):
        t.set_text(data.to_symbol(inpt[i][batch]) if index < xf else "")
      if index < xf:
        im.set_array(np.zeros_like(steps[0][0]))
      else:
        im.set_array(steps[0][batch])
    return im,

  # Create the animation and save to mp4.
  animation = anim.FuncAnimation(
      fig, animation_update, blit=True, frames=(l+4*xf)*anim_size*fps,
      interval=500/fps, fargs=(test_data, xf, im, text_fields))
  animation.save("/tmp/neural_gpu.mp4", writer="mencoder", fps=4*fps, dpi=3*80)


def evaluate():
  """Evaluate an existing model."""
  batch_size = FLAGS.batch_size
  tasks = FLAGS.task.split("-")
  with tf.Session() as sess:
    model, min_length, max_length, _, _, ensemble = initialize(sess)
    bound = data.bins[-1] + 1
    for t in tasks:
      l = min_length
      while l < max_length + EXTRA_EVAL and l < bound:
        _, seq_err, _ = single_test(l, model, sess, t, FLAGS.nprint,
                                    batch_size, ensemble=ensemble)
        l += 1
        while l < bound + 1 and not data.test_set[t][l]:
          l += 1
      # Animate.
      if FLAGS.animate:
        anim_size = 2
        _, _, test_data = single_test(l, model, sess, t, 0, anim_size,
                                      get_steps=True)
        animate(l, test_data, anim_size)
      # More tests.
      _, seq_err = multi_test(data.forward_max, model, sess, t, FLAGS.nprint,
                              batch_size * 4, ensemble=ensemble)
    if seq_err < 0.01:  # Super-test if we're very good and in large-test mode.
      if data.forward_max > 4000 and len(tasks) == 1:
        multi_test(data.forward_max, model, sess, tasks[0], FLAGS.nprint,
                   batch_size * 64, 0, ensemble=ensemble)


def interactive():
  """Interactively probe an existing model."""
  with tf.Session() as sess:
    model, _, _, _, _, _ = initialize(sess)
    sys.stdout.write("Input to Neural GPU, e.g., 0 1. Use -1 for PAD.\n")
    sys.stdout.write("> ")
    sys.stdout.flush()
    inpt = sys.stdin.readline()
    while inpt:
      ids = [data.to_id(s) for s in inpt.strip().split()]
      inpt, target = data.get_batch(len(ids), 1, False, "",
                                    preset=(ids, [0 for _ in ids]))
      _, res, _, _ = model.step(sess, inpt, target, False)
      res = [np.argmax(o, axis=1) for o in res]
      res = [o for o in res[:len(ids)] if o > 0]
      print "  " + " ".join([data.to_symbol(output[0]) for output in res])
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
