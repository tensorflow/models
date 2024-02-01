# Copyright 2024 The TensorFlow Authors. All Rights Reserved.
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

"""MaxViT Image classification configuration definition."""

from official.core import config_definitions as cfg
from official.core import exp_factory
from official.modeling.optimization.configs import optimization_config
from official.projects.maxvit.configs import backbones
from official.vision.configs import image_classification as img_cls_cfg


@exp_factory.register_config_factory('maxvit_imagenet')
def maxvit_imagenet() -> cfg.ExperimentConfig:
  """Returns MaxViT-Tiny on imagenet-1k.

  Expected to be trained on DF 4x4 or bigger. Can eval on DF 4x2.

  Returns:
    The full experiment config.
  """
  # Reuse ViT deit pretraining config.
  exp = img_cls_cfg.image_classification_imagenet_deit_pretrain()
  exp.task.model = img_cls_cfg.ImageClassificationModel(
      num_classes=1001,
      input_size=[224, 224, 3],
      kernel_initializer='glorot_uniform',
      backbone=backbones.Backbone(
          type='maxvit',
          maxvit=backbones.MaxViT(
              model_name='maxvit-tiny', representation_size=768
          ),
      ),
      norm_activation=img_cls_cfg.common.NormActivation(activation='relu'),
  )

  exp.task.train_data.aug_type.randaug.num_layers = 2
  exp.task.train_data.aug_type.randaug.magnitude = 15
  exp.runtime.mixed_precision_dtype = 'bfloat16'
  exp.trainer.optimizer_config.optimizer.adamw.gradient_clip_norm = 0.0
  exp.trainer.optimizer_config.warmup.linear.warmup_steps = 10000
  exp.trainer.optimizer_config.ema = optimization_config.opt_cfg.EMAConfig(
      average_decay=0.9999,
      trainable_weights_only=False,
  )

  return exp
