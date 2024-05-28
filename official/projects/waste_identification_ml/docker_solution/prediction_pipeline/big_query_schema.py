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

"""Stores big query table schema."""

from google.cloud import bigquery

# Create the table within the dataset
SCHEMA_1 = [
    bigquery.SchemaField("detection_scores", "FLOAT", mode="REQUIRED"),
    bigquery.SchemaField("detection_classes_names", "STRING", mode="REQUIRED"),
    bigquery.SchemaField("detection_classes", "INTEGER", mode="REQUIRED"),
    bigquery.SchemaField("area", "INTEGER", mode="REQUIRED"),
    bigquery.SchemaField("bbox_0", "INTEGER", mode="REQUIRED"),
    bigquery.SchemaField("bbox_1", "INTEGER", mode="REQUIRED"),
    bigquery.SchemaField("bbox_2", "INTEGER", mode="REQUIRED"),
    bigquery.SchemaField("bbox_3", "INTEGER", mode="REQUIRED"),
    bigquery.SchemaField("convex_area", "INTEGER", mode="REQUIRED"),
    bigquery.SchemaField("bbox_area", "INTEGER", mode="REQUIRED"),
    bigquery.SchemaField("major_axis_length", "FLOAT", mode="REQUIRED"),
    bigquery.SchemaField("minor_axis_length", "FLOAT", mode="REQUIRED"),
    bigquery.SchemaField("eccentricity", "FLOAT", mode="REQUIRED"),
    bigquery.SchemaField("y", "FLOAT", mode="REQUIRED"),
    bigquery.SchemaField("x", "FLOAT", mode="REQUIRED"),
    bigquery.SchemaField("image_name", "STRING", mode="REQUIRED"),
    bigquery.SchemaField("color", "STRING", mode="REQUIRED"),
    bigquery.SchemaField("creation_timestamp", "STRING", mode="REQUIRED"),
    bigquery.SchemaField("video_name", "STRING", mode="REQUIRED"),
]

# Create the table for object count grouped by object class and color
SCHEMA_2 = [
    bigquery.SchemaField("colors_group", "STRING", mode="REQUIRED"),
    bigquery.SchemaField("particle", "INTEGER", mode="REQUIRED"),
    bigquery.SchemaField("detection_classes_group", "INTEGER", mode="REQUIRED"),
    bigquery.SchemaField("material", "STRING", mode="REQUIRED"),
    bigquery.SchemaField("material_form", "STRING", mode="REQUIRED"),
    bigquery.SchemaField("creation_timestamp", "STRING", mode="REQUIRED"),
    bigquery.SchemaField("video_name", "STRING", mode="REQUIRED"),
]
