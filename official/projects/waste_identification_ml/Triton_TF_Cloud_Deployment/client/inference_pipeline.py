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

"""Pipeline to run the prediction on the images folder with Triton server."""

from absl import flags

INPUT_DIRECTORY = flags.DEFINE_string(
    "input_directory", None, "The path to the directory containing images."
)

OUTPUT_DIRECTORY = flags.DEFINE_string(
    "output_directory", None, "The path to the directory to save the results."
)

HEIGHT = flags.DEFINE_integer(
    "height", None, "Height of an image required by the model"
)

WIDTH = flags.DEFINE_integer(
    "width", None, "Width of an image required by the model"
)

MODEL = flags.DEFINE_string("model", None, "Model name")

PREDICTION_THRESHOLD = flags.DEFINE_float(
    "score", None, "Threshold to filter the prediction results"
)

SEARCH_RANGE_X = flags.DEFINE_integer(
    "search_range_x",
    None,
    "Pixels upto which every object needs to be tracked along X axis.",
)

SEARCH_RANGE_Y = flags.DEFINE_integer(
    "search_range_y",
    None,
    "Pixels upto which every object needs to be tracked along Y axis.",
)

MEMORY = flags.DEFINE_integer(
    "memory", None, "Frames upto which every object needs to be tracked."
)

OVERWRITE = flags.DEFINE_boolean(
    "overwrite",
    False,
    "If True, delete the preexisting BigQuery table before creating a new one.",
)

PROJECT_ID = flags.DEFINE_string(
    "project_id", None, "Project ID mentioned in Google Cloud Project"
)

BQ_DATASET_ID = flags.DEFINE_string(
    "bq_dataset_id", "Circularnet_dataset", "Big query dataset ID"
)

TABLE_ID = flags.DEFINE_string(
    "bq_table_id", "Circularnet_table", "BigQuery Table ID for features data"
)
