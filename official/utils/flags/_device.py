# Copyright 2022 The TensorFlow Authors. All Rights Reserved.
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

"""Flags for managing compute devices. Currently only contains TPU flags."""

from absl import flags
from absl import logging

from official.utils.flags._conventions import help_wrap


def require_cloud_storage(flag_names):
  """Register a validator to check directory flags.

  Args:
    flag_names: An iterable of strings containing the names of flags to be
      checked.
  """
  msg = "TPU requires GCS path for {}".format(", ".join(flag_names))

  @flags.multi_flags_validator(["tpu"] + flag_names, message=msg)
  def _path_check(flag_values):  # pylint: disable=missing-docstring
    if flag_values["tpu"] is None:
      return True

    valid_flags = True
    for key in flag_names:
      if not flag_values[key].startswith("gs://"):
        logging.error("%s must be a GCS path.", key)
        valid_flags = False

    return valid_flags


def define_device(tpu=True):
  """Register device specific flags.

  Args:
    tpu: Create flags to specify TPU operation.

  Returns:
    A list of flags for core.py to marks as key flags.
  """

  key_flags = []

  if tpu:
    flags.DEFINE_string(
        name="tpu",
        default=None,
        help=help_wrap(
            "The Cloud TPU to use for training. This should be either the name "
            "used when creating the Cloud TPU, or a "
            "grpc://ip.address.of.tpu:8470 url. Passing `local` will use the"
            "CPU of the local instance instead. (Good for debugging.)"))
    key_flags.append("tpu")

    flags.DEFINE_string(
        name="tpu_zone",
        default=None,
        help=help_wrap(
            "[Optional] GCE zone where the Cloud TPU is located in. If not "
            "specified, we will attempt to automatically detect the GCE "
            "project from metadata."))

    flags.DEFINE_string(
        name="tpu_gcp_project",
        default=None,
        help=help_wrap(
            "[Optional] Project name for the Cloud TPU-enabled project. If not "
            "specified, we will attempt to automatically detect the GCE "
            "project from metadata."))

    flags.DEFINE_integer(
        name="num_tpu_shards",
        default=8,
        help=help_wrap("Number of shards (TPU chips)."))

  return key_flags
