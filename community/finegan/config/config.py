
# Copyright 2020 The TensorFlow Authors. All Rights Reserved.
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

class Config(object):
    """Base configuration class.
    """
    CONFIG_NAME = 'finegan_train'
    DATASET_NAME = 'birds'
    DATA_DIR = ''
    SAVE_DIR = ''
    GPU_ID = '0'
    WORKERS = 4

    SUPER_CATEGORIES = 20   # For CUB 
    FINE_GRAINED_CATEGORIES = 200  # For CUB
    TIED_CODES = True   # Do NOT change this to False during training.

    TREE = {
        'BRANCH_NUM': 3
    }

    TRAIN = {
        'FLAG': True,
        'BATCH_SIZE': 16,
        'MAX_EPOCH': 600,
        'HARDNEG_MAX_ITER': 1500,
        'SNAPSHOT_INTERVAL': 4000,
        'SNAPSHOT_INTERVAL_HARDNEG': 500,
        'DISCRIMINATOR_LR': 0.0002,
        'GENERATOR_LR': 0.0002,
        'VIS_COUNT': 64,
        'BG_LOSS_WT': 10
    }

    GAN = {
        'DF_DIM': 64,
        'GF_DIM': 64,
        'Z_DIM': 100,
        'R_NUM': 2
    }

    def __init__(self, batch_size=64):
        self.BATCH_SIZE = batch_size

    def display(self):
        """Display Configuration values."""
        print("\nConfigurations:")
        for a in dir(self):
            if not a.startswith("__") and not callable(getattr(self, a)):
                print("{:30} {}".format(a, getattr(self, a)))
        print("\n")
