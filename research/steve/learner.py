from __future__ import division
from __future__ import print_function
from builtins import zip
from builtins import range
from builtins import object
# Copyright 2018 The TensorFlow Authors All Rights Reserved.
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

import traceback, threading, time, warnings
import tensorflow as tf
import numpy as np

import util
from replay import ReplayBuffer

class Learner(object):
    """
    Generic object which runs the main training loop of anything that trains using
    a replay buffer. Handles updating, logging, saving/loading, batching, etc.
    """
    def __init__(self, interactor_queue, lock, config, env_config, learner_config, **bonus_kwargs):
        self.learner_name = self.learner_name()
        self.interactor_queue = interactor_queue
        self.learner_lock = lock
        self.config = config
        self.env_config = env_config
        self.learner_config = learner_config
        self.bonus_kwargs = bonus_kwargs
        self.kill_threads = False
        self.permit_desync = False
        self.need_frames_notification = threading.Condition()
        self._reset_inspections()
        self.total_frames = 0

        self.save_path = util.create_directory("%s/%s/%s/%s" % (self.config["output_root"], self.config["env"]["name"], self.config["name"], self.config["save_model_path"]))
        self.log_path = util.create_directory("%s/%s/%s/%s" % (self.config["output_root"], self.config["env"]["name"], self.config["name"],  self.config["log_path"])) + "/%s.log" % self.learner_name

        # replay buffer to store data
        self.replay_buffer_lock = threading.RLock()
        self.replay_buffer = ReplayBuffer(self.learner_config["replay_size"],
                                          np.prod(self.env_config["obs_dims"]),
                                          self.env_config["action_dim"])

        # data loaders pull data from the replay buffer and put it into the tfqueue for model usage
        self.data_loaders = self.make_loader_placeholders()
        queue_capacity = np.ceil(1./self.learner_config["frames_per_update"]) if self.learner_config["frames_per_update"] else 100
        self.tf_queue = tf.FIFOQueue(capacity=queue_capacity, dtypes=[dl.dtype for dl in self.data_loaders])
        self.enqueue_op = self.tf_queue.enqueue(self.data_loaders)
        self.current_batch = self.tf_queue.dequeue()

        # build the TF graph for the actual model to train
        self.core, self.train_losses, self.train_ops, self.inspect_losses = self.make_core_model()
        self.sess = tf.Session()
        self.sess.run(tf.global_variables_initializer())

    ## Mandatory functions to override
    def learner_name(self): raise Exception('unimplemented: learner_name')
    def make_loader_placeholders(self): raise Exception('unimplemented: make_loader_placeholders')
    def make_core_model(self): raise Exception('unimplemented: make_core_model')

    ## Optional functions to override
    def initialize(self): warnings.warn('unimplemented: initialize')
    def resume_from_checkpoint(self, epoch): warnings.warn('unimplemented: resume_from_checkpoint')
    def checkpoint(self): warnings.warn('unimplemented: checkpoint')
    def backup(self): warnings.warn('unimplemented: backup')

    ## Internal functions
    def _start(self):
        # fetch data from the interactors to pre-fill the replay buffer
        self.prefetch_thread = threading.Thread(target=self._poll_interactors, args=(True, self.learner_config["frames_before_learning"],))
        self.prefetch_thread.start()
        self.prefetch_thread.join()

        # start the interactor and data loader
        self.data_load_thread = threading.Thread(target=self._run_enqueue_data)
        self.data_load_thread.start()

        # initialize the learner, pretraining if needed
        if self.config["resume"]: self._resume_from_checkpoint()
        else:                     self._initialize()

        # re-sync everything, and start up interactions with the environment
        self.interactor_poll_thread = threading.Thread(target=self._poll_interactors)
        self.interactor_poll_thread.start()

        # start the clock
        self._last_checkpoint_time = time.time()

    def _learn(self, permit_desync=False, log=True, checkpoint=True, backup=True):
        # this is to keep the frames/update synced properly
        if self.learner_config["frames_per_update"] is not False and not permit_desync:
            if not self._have_enough_frames():
                with self.need_frames_notification:
                    self.need_frames_notification.notify()
                return

        # log
        if log and (self.update_i + 1) % self.learner_config["log_every_n"] == 0:
            self._log()

        # checkpoint
        if checkpoint and (self.update_i + 1) % self.learner_config["epoch_every_n"] == 0:
            self._checkpoint()

        # backup
        if backup and (self.update_i + 1) % self.learner_config["backup_every_n"] == 0:
            self._backup()

        # train
        self._training_step()

    def _have_enough_frames(self):
        gathered_frames = self.total_frames - self.learner_config["frames_before_learning"]
        return gathered_frames > self.learner_config["frames_per_update"] * self.update_i

    def _initialize(self):
        self.epoch = 0
        self.update_i = 0
        self.hours = 0
        self._last_checkpoint_time = time.time()

        self.initialize()

        if self.learner_config["pretrain_n"]: self._pretrain()
        self._checkpoint()

    def _pretrain(self):
        for _ in range(self.learner_config["pretrain_n"]):
            self._learn(permit_desync=True, checkpoint=False, backup=False)
        self.epoch = 0
        self.update_i = 0

    def _resume_from_checkpoint(self):
        epoch = util.get_largest_epoch_in_dir(self.save_path, self.core.saveid)
        if not self.config['keep_all_replay_buffers']: util.wipe_all_but_largest_epoch_in_dir(self.save_path, self.core.saveid)
        if epoch is False:
            raise Exception("Tried to reload but no model found")
        with self.learner_lock:
            self.core.load(self.sess, self.save_path, epoch)
            self.epoch, self.update_i, self.total_frames, self.hours = self.sess.run([self.core.epoch_n, self.core.update_n, self.core.frame_n, self.core.hours])
        with self.replay_buffer_lock:
            self.replay_buffer.load(self.save_path, '%09d_%s' % (epoch, self.learner_name))
        self.resume_from_checkpoint(epoch)

    def _log(self):
        logstring = "(%3.2f sec) h%-8.2f e%-8d s%-8d f%-8d\t" % (time.time() - self._log_time, self.hours, self.epoch, self.update_i + 1, self.total_frames) + ', '.join(["%8f" % x for x in (self.running_total / self.denom).tolist()])
        print("%s\t%s" % (self.learner_name, logstring))
        with open(self.log_path, "a") as f: f.write(logstring + "\n")
        self._reset_inspections()

    def _reset_inspections(self):
        self.running_total = 0.
        self.denom = 0.
        self._log_time = time.time()

    def _checkpoint(self):
        self.checkpoint()
        self.epoch += 1
        self.hours += (time.time() - self._last_checkpoint_time) / 3600.
        self._last_checkpoint_time = time.time()
        self.core.update_epoch(self.sess, self.epoch, self.update_i, self.total_frames, self.hours)
        with self.learner_lock: self.core.save(self.sess, self.save_path)

    def _backup(self):
        self.backup()
        if not self.learner_config['keep_all_replay_buffers']: util.wipe_all_but_largest_epoch_in_dir(self.save_path, self.core.saveid)
        with self.learner_lock:
            self.core.save(self.sess, self.save_path, self.epoch)
        with self.replay_buffer_lock:
            self.replay_buffer.save(self.save_path, '%09d_%s' % (self.epoch, self.learner_name))

    def _training_step(self):
        train_ops = tuple([op for op, loss in zip(self.train_ops,
                                                  self.train_losses)
                           if loss is not None])
        outs = self.sess.run(train_ops + self.inspect_losses)
        self.running_total += np.array(outs[len(train_ops):])
        self.denom += 1.
        self.update_i += 1

    def _poll_interactors(self, continuous_poll=False, frames_before_terminate=None):
        # poll the interactors for new frames.
        # the synced_condition semaphore prevents this from consuming too much CPU
        while not self.kill_threads:
            if self.learner_config["frames_per_update"] is not False and not continuous_poll:
                with self.need_frames_notification: self.need_frames_notification.wait()
            while not self.interactor_queue.empty():
                new_frames = self.interactor_queue.get()
                self._add_frames(new_frames)
                if frames_before_terminate and self.total_frames >= frames_before_terminate: return

    def _add_frames(self, frames):
        with self.replay_buffer_lock:
            for frame in frames:
                self.replay_buffer.add_replay(*frame)
            self.total_frames = self.replay_buffer.count
        return self.total_frames

    def _run_enqueue_data(self):
        while not self.kill_threads:
            data = self.replay_buffer.random_batch(self.learner_config["batch_size"])
            self.sess.run(self.enqueue_op, feed_dict=dict(list(zip(self.data_loaders, data))))

    def _kill_threads(self):
        self.kill_threads = True


class CoreModel(object):
    """The base class for the "core" of learners."""
    def __init__(self, name, env_config, learner_config):
        self.name = self.saveid + "/" + name
        self.env_config = env_config
        self.learner_config = learner_config

        with tf.variable_scope(self.name):
            self.epoch_n = tf.get_variable('epoch_n', [], initializer=tf.constant_initializer(0), dtype=tf.int64, trainable=False)
            self.update_n = tf.get_variable('update_n', [], initializer=tf.constant_initializer(0), dtype=tf.int64, trainable=False)
            self.frame_n = tf.get_variable('frame_n', [], initializer=tf.constant_initializer(0), dtype=tf.int64, trainable=False)
            self.hours = tf.get_variable('hours', [], initializer=tf.constant_initializer(0.), dtype=tf.float64, trainable=False)
            self.epoch_n_placeholder = tf.placeholder(tf.int64, [])
            self.update_n_placeholder = tf.placeholder(tf.int64, [])
            self.frame_n_placeholder = tf.placeholder(tf.int64, [])
            self.hours_placeholder = tf.placeholder(tf.float64, [])
        self.assign_epoch_op = [tf.assign(self.epoch_n, self.epoch_n_placeholder), tf.assign(self.update_n, self.update_n_placeholder), tf.assign(self.frame_n, self.frame_n_placeholder), tf.assign(self.hours, self.hours_placeholder)]

        self.create_params(env_config, learner_config)
        self.model_params = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope=self.name)
        self.saver = tf.train.Saver(self.model_params)

    @property
    def saveid(self):
        raise Exception("specify a save ID")

    def create_params(self, env_config, learner_config):
        raise Exception("unimplemented")

    def update_epoch(self, sess, epoch, updates, frames, hours):
        sess.run(self.assign_epoch_op, feed_dict={self.epoch_n_placeholder: int(epoch), self.update_n_placeholder: int(updates), self.frame_n_placeholder: int(frames), self.hours_placeholder: float(hours)})

    def save(self, sess, path, epoch=None):
        if epoch is None:  self.saver.save(sess, path + "/%s.params" % self.saveid)
        else:              self.saver.save(sess, path + "/%09d_%s.params" % (epoch, self.saveid))

    def load(self, sess, path, epoch=None):
        if epoch is None:  self.saver.restore(sess, path + "/%s.params" % self.saveid)
        else:              self.saver.restore(sess, path + "/%09d_%s.params" % (epoch, self.saveid))

def run_learner(learner_subclass, queue, lock, config, env_config, learner_config, **bonus_kwargs):
    learner = learner_subclass(queue, lock, config, env_config, learner_config, **bonus_kwargs)
    try:
        learner._start()
        while True: learner._learn()

    except Exception as e:
        print('Caught exception in learner process')
        traceback.print_exc()
        learner._kill_threads()
        print()
        raise e
