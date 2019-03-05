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

import tensorflow as tf
import numpy as np
from learner import Learner
from worldmodel import DeterministicWorldModel

class WorldmodelLearner(Learner):
    """
    Worldmodel-specific training loop details.
    """
    def learner_name(self): return "worldmodel"

    def make_loader_placeholders(self):
        self.obs_loader = tf.placeholder(tf.float32, [self.learner_config["batch_size"], np.prod(self.env_config["obs_dims"])])
        self.next_obs_loader = tf.placeholder(tf.float32, [self.learner_config["batch_size"], np.prod(self.env_config["obs_dims"])])
        self.action_loader = tf.placeholder(tf.float32, [self.learner_config["batch_size"], self.env_config["action_dim"]])
        self.reward_loader = tf.placeholder(tf.float32, [self.learner_config["batch_size"]])
        self.done_loader = tf.placeholder(tf.float32, [self.learner_config["batch_size"]])
        self.datasize_loader = tf.placeholder(tf.float64, [])
        return [self.obs_loader, self.next_obs_loader, self.action_loader, self.reward_loader, self.done_loader, self.datasize_loader]

    def make_core_model(self):
        worldmodel = DeterministicWorldModel(self.config["name"], self.env_config, self.learner_config)
        worldmodel_loss, inspect_losses = worldmodel.build_training_graph(*self.current_batch)

        model_optimizer = tf.train.AdamOptimizer(3e-4)
        model_gvs = model_optimizer.compute_gradients(worldmodel_loss, var_list=worldmodel.model_params)
        capped_model_gvs = model_gvs
        worldmodel_train_op = model_optimizer.apply_gradients(capped_model_gvs)

        return worldmodel, (worldmodel_loss,), (worldmodel_train_op,), inspect_losses

    ## Optional functions to override
    def initialize(self): pass
    def resume_from_checkpoint(self, epoch): pass
    def checkpoint(self): pass
    def backup(self): pass




