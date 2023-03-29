# Copyright 2023 The TensorFlow Authors. All Rights Reserved.
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

"""Central location for NCF specific values."""

import sys

import numpy as np

from official.recommendation import movielens

# ==============================================================================
# == Main Thread Data Processing ===============================================
# ==============================================================================

# Keys for data shards
TRAIN_USER_KEY = "train_{}".format(movielens.USER_COLUMN)
TRAIN_ITEM_KEY = "train_{}".format(movielens.ITEM_COLUMN)
TRAIN_LABEL_KEY = "train_labels"
MASK_START_INDEX = "mask_start_index"
VALID_POINT_MASK = "valid_point_mask"
EVAL_USER_KEY = "eval_{}".format(movielens.USER_COLUMN)
EVAL_ITEM_KEY = "eval_{}".format(movielens.ITEM_COLUMN)

USER_MAP = "user_map"
ITEM_MAP = "item_map"

USER_DTYPE = np.int32
ITEM_DTYPE = np.int32

# In both datasets, each user has at least 20 ratings.
MIN_NUM_RATINGS = 20

# The number of negative examples attached with a positive example
# when performing evaluation.
NUM_EVAL_NEGATIVES = 999

# keys for evaluation metrics
TOP_K = 10  # Top-k list for evaluation
HR_KEY = "HR"
NDCG_KEY = "NDCG"
DUPLICATE_MASK = "duplicate_mask"

# Metric names
HR_METRIC_NAME = "HR_METRIC"
NDCG_METRIC_NAME = "NDCG_METRIC"

# Trying to load a cache created in py2 when running in py3 will cause an
# error due to differences in unicode handling.
RAW_CACHE_FILE = "raw_data_cache_py{}.pickle".format(sys.version_info[0])
CACHE_INVALIDATION_SEC = 3600 * 24

# ==============================================================================
# == Data Generation ===========================================================
# ==============================================================================
CYCLES_TO_BUFFER = 3  # The number of train cycles worth of data to "run ahead"
# of the main training loop.

# Number of batches to run per epoch when using synthetic data. At high batch
# sizes, we run for more batches than with real data, which is good since
# running more batches reduces noise when measuring the average batches/second.
SYNTHETIC_BATCHES_PER_EPOCH = 2000

# Only used when StreamingFilesDataset is used.
NUM_FILE_SHARDS = 16
TRAIN_FOLDER_TEMPLATE = "training_cycle_{}"
EVAL_FOLDER = "eval_data"
SHARD_TEMPLATE = "shard_{}.tfrecords"
