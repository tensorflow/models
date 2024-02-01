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

"""This module provides utilities for executing shell commands.

It particularly downloads and extracts Mask RCNN models from the TensorFlow
model garden. It includes a function to execute shell commands and
a custom exception to handle errors that arise from command execution.

Functions:
    - execute_command(cmd: str) -> str: Executes a shell command and returns its
    standard output. Raises
      a CommandExecutionError if the command execution fails.

Exceptions:
    - CommandExecutionError: Custom exception that's raised when there's an
    error executing a shell command.

Usage:
    The main purpose of this module is to download two specific Mask RCNN models
    and unzip them. The module
    performs these operations when imported.

Note:
    It's recommended to not perform actions like downloading files on module
    import in production applications.
    It's better to move such tasks inside a function or a main block to allow
    for more controlled execution.
"""
import argparse
import os
import subprocess


class CommandExecutionError(Exception):
  """Raised when there's an error executing a shell command."""

  def __init__(self, cmd, returncode, stderr):
    super().__init__(f"Error executing command: {cmd}. Error: {stderr}")
    self.cmd = cmd
    self.returncode = returncode
    self.stderr = stderr


def execute_command(cmd: str) -> str:
  """Executes a shell command and returns its output."""
  result = subprocess.run(
      cmd,
      shell=True,
      stdout=subprocess.PIPE,
      stderr=subprocess.PIPE,
      check=False,
  )

  if result.returncode != 0:
    raise CommandExecutionError(
        cmd, result.returncode, result.stderr.decode("utf-8")
    )

  return result.stdout.decode("utf-8")


def main(_) -> None:
  # Download the provided files
  execute_command(f"wget {args.material_url}")
  execute_command(f"wget {args.material_form_url}")

  # Create directories
  os.makedirs("material", exist_ok=True)
  os.makedirs("material_form", exist_ok=True)

  # Unzip the provided files
  zip_file1 = os.path.basename(args.material_url)
  zip_file2 = os.path.basename(args.material_form_url)
  execute_command(f"unzip {zip_file1} -d material/")
  execute_command(f"unzip {zip_file2} -d material_form/")


if __name__ == "__main__":
  parser = argparse.ArgumentParser(
      description="Download and extract Mask RCNN models."
  )
  parser.add_argument("material_url", help="repo url for material model")
  parser.add_argument(
      "material_form_url", help="repo url for material form model"
  )

  args = parser.parse_args()
  main(args)
