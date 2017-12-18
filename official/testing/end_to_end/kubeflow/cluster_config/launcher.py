#!/usr/bin/env python

# Copyright 2017 Google Inc. All Rights Reserved.
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

"""A launcher suitable for invoking tf_cnn_benchmarks using TfJob.
All the launcher does is turn TF_CONFIG environment variable
into extra arguments to append to the command line.
"""
import logging
import json
import os
import subprocess
import sys
import time

def run_and_stream(cmd):
  logging.info("Running %s", " ".join(cmd))
  process = subprocess.Popen(cmd, stdout=subprocess.PIPE,
                             stderr=subprocess.STDOUT)

  while process.poll() is None:
    process.stdout.flush()
    if process.stderr:
      process.stderr.flush()
    sys.stderr.flush()
    sys.stdout.flush()
    for line in iter(process.stdout.readline, ''):
      process.stdout.flush()
      logging.info(line.strip())

  sys.stderr.flush()
  sys.stdout.flush()
  process.stdout.flush()
  if process.stderr:
    process.stderr.flush()
  for line in iter(process.stdout.readline, ''):
    logging.info(line.strip())

  if process.returncode != 0:
    raise ValueError("cmd: {0} exited with code {1}".format(
      " ".join(cmd), process.returncode))

if __name__ == "__main__":
  logging.getLogger().setLevel(logging.INFO)
  logging.basicConfig(level=logging.INFO,
                      format=('%(levelname)s|%(asctime)s'
                              '|%(pathname)s|%(lineno)d| %(message)s'),
                      datefmt='%Y-%m-%dT%H:%M:%S',
                      )
  logging.info("Launcher started.")
  tf_config = os.environ.get('TF_CONFIG', '{}')
  tf_config_json = json.loads(tf_config)
  cluster = tf_config_json.get('cluster', {})
  job_name = tf_config_json.get('task', {}).get('type', "")
  task_index = tf_config_json.get('task', {}).get('index', "")

  command = sys.argv[1:]
  ps_hosts = ",".join(cluster.get("ps", []))
  worker_hosts = ",".join(cluster.get("worker", []))
  command.append("--job_name=" + job_name)
  command.append("--ps_hosts=" + ps_hosts)
  command.append("--worker_hosts=" + worker_hosts)
  command.append("--task_index={0}".format(task_index))

  # TODO(jlewi): This is a work around
  # https://github.com/tensorflow/k8s/issues/192
  # Since TfJob requires a master but the tf benchmark
  # code doesn't know what to do with master we just run
  # forever in the master.
  if job_name.lower() == "master":
    while True:
      print("master runs forever.")
      time.sleep(600)

  logging.info("Command to run: %s", " ".join(command))
  with open("/opt/run_benchmarks.sh", "w") as hf:
    hf.write("#!/bin/bash\n")
    hf.write(" ".join(command))
    hf.write("\n")

  run_and_stream(command)
  logging.info("Finished: %s", " ".join(command))
  # We don't want to terminate because TfJob will
  # just restart the job.
  while True:
    logging.info("Command ran successfully sleep for ever.")
    time.sleep(600)
