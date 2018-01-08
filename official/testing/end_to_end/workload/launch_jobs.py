# Copyright 2017 The TensorFlow Authors. All Rights Reserved.
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

"""Builds docker images and runs training/tests.
"""
import argparse
from datetime import datetime
import logging
import os
from string import maketrans
import subprocess
import sys

import docker
import k8s_tensorflow_lib
import kubectl_util
import yaml


_DOCKER_IMAGE_PATTERN = 'gcr.io/tensorflow-testing/tf-models-cluster/%s'
_OUTPUT_FILE_ENV_VAR = 'TF_DIST_BENCHMARK_RESULTS_FILE'
_TEST_NAME_ENV_VAR = 'TF_DIST_BENCHMARK_NAME'
_PORT = 5000


def _ConvertToValidName(name):
  """Converts to name that we can use as a kubernetes job prefix.

  Args:
    name: task name.

  Returns:
    Task name that can be used as a kubernetes job prefix.
  """
  return name.translate(maketrans('/:_', '---'))


def _ExecuteTask(name,
  yaml_file,
  wait_for_completion=False,
  delete_on_completion=False):
  """Runs a single task as configured.

  Args:
    name: name of the task to run.
    yaml_file: path to kubernetes config file.
  """
  kubectl_util.DeletePods(name, yaml_file)
  kubectl_util.CreatePods(name, yaml_file)
  if wait_for_completion:
    success = kubectl_util.WaitForCompletion(name)
    if delete_on_completion:
      kubectl_util.DeletePods(name, yaml_file)
    return success


def _BuildAndPushDockerImage(
    docker_client, docker_file, docker_image_pattern, name, tag,
    push_to_gcloud=False, buildargs=None):
  """Builds a docker image and optionally pushes it to gcloud.

  Args:
    docker_client: docker.Client object.
    docker_file: Dockerfile path.
    docker_image_pattern: URL for docker registry where image will be pushed.
      Should have one `%s` where the image name will go.
    name: name of the task to build a docker image for.
    tag: tag for docker image.
    push_to_gcloud: whether to push the image to google cloud.
    buildargs: optional dict of build arguments to be passed to Docker
      during building of image. See
      https://docs.docker.com/engine/reference/builder/#arg
  Returns:
    Docker image identifier.
  """
  local_docker_image_with_tag = '%s:%s' % (name, tag)
  remote_docker_image = docker_image_pattern % name
  remote_docker_image_with_tag = '%s:%s' % (remote_docker_image, tag)
  if FLAGS.docker_context_dir:
    docker_context = os.path.join(
        os.path.dirname(__file__), FLAGS.docker_context_dir)
    docker_file_name = docker_file
  else:
    docker_context = os.path.dirname(docker_file)
    docker_file_name = os.path.basename(docker_file)

  built_image = docker_client.images.build(
      path=docker_context,
      dockerfile=docker_file_name,
      tag=local_docker_image_with_tag,
      buildargs=buildargs,
      pull=True)
  built_image.tag(remote_docker_image, tag=tag)
  if push_to_gcloud:
    subprocess.check_call(
        ['gcloud', 'docker', '--', 'push', remote_docker_image_with_tag])
  return remote_docker_image_with_tag


def _GetMostRecentDockerImageFromGcloud(docker_image):
  """Get most recent <docker_image>:tag for this docker_image.

  Args:
    docker_image: (string) docker image on Google Cloud.

  Returns:
    docker_image:tag if at least one tag was found for docker_image.
    Otherwise, returns None.
  """
  tag = subprocess.check_output(
      ['gcloud', 'container', 'images', 'list-tags',
       docker_image, '--limit=1', '--format=value(tags[0])'])
  tag = tag.strip()
  if not tag:
    return None
  return '%s:%s' % (docker_image, tag)


def get_gpu_volume_mounts():
  """Get volume specs to add to Kubernetes config.

  Returns:
    Volume specs in the format: volume_name: (hostPath, podPath).
  """
  volume_specs = {}

  if FLAGS.nvidia_lib_dir:
    volume_specs['nvidia-libraries'] = (FLAGS.nvidia_lib_dir, '/usr/lib/nvidia')

  if FLAGS.cuda_lib_dir:
    cuda_library_files = ['libcuda.so', 'libcuda.so.1', 'libcudart.so']
    for cuda_library_file in cuda_library_files:
      lib_name = cuda_library_file.split('.')[0]
      volume_specs['cuda-libraries-%s' % lib_name] = (
          os.path.join(FLAGS.cuda_lib_dir, cuda_library_file),
          os.path.join('/usr/lib/cuda/', cuda_library_file))
  return volume_specs


class NoImageFoundError(Exception):
    pass


def main():
  config_text = open(FLAGS.task_config_file, 'r').read()
  configs = yaml.load(config_text)

  docker_client = docker.from_env()
  time_tag = datetime.now().strftime('%d_%m_%Y_%H_%M')
  # Create directories to store kubernetes yaml configs in.
  if not os.path.isdir(FLAGS.config_output_file_dir):
    os.makedirs(FLAGS.config_output_file_dir)
  # Keeps track of already built docker images in case multiple tasks
  # use the same docker image.
  name_to_docker_image = {}

  # Set docker registry path for use
  docker_image_pattern = FLAGS.docker_image_pattern or _DOCKER_IMAGE_PATTERN

  # TODO(annarev): execute tasks in parallel instead of sequentially.
  for config in configs:
    name = _ConvertToValidName(str(config['task_name']))
    if name in name_to_docker_image:
      docker_image = name_to_docker_image[name]
    elif FLAGS.build_docker_image:
      docker_image = _BuildAndPushDockerImage(
          docker_client,
          config['docker_file'],
          docker_image_pattern,
          name,
          time_tag,
          FLAGS.store_docker_image_in_gcloud,
          buildargs=config.get('docker_build_args'))
      name_to_docker_image[name] = docker_image
    else:
      docker_image = _GetMostRecentDockerImageFromGcloud(
          docker_image_pattern % name)
      if not docker_image:
        raise NoImageFoundError('No tags found for image %s.' % docker_image)

    env_vars = {
        _OUTPUT_FILE_ENV_VAR: os.path.join(
            FLAGS.results_dir, name + '.json'),
        _TEST_NAME_ENV_VAR: name
    }
    gpu_count = (0 if 'gpus_per_machine' not in config
                 else config['gpus_per_machine'])
    volumes = {}
    if gpu_count > 0:
      volumes = get_gpu_volume_mounts()
      env_vars['LD_LIBRARY_PATH'] = (
          '/usr/lib/cuda:/usr/lib/nvidia:/usr/lib/x86_64-linux-gnu')

    env_vars.update(config.get('env_vars', {}))
    args = config.get('args', {})
    kubernetes_config = k8s_tensorflow_lib.GenerateConfig(
        config['worker_count'],
        config['ps_count'],
        _PORT,
        request_load_balancer=False,
        docker_image=docker_image,
        name_prefix=name,
        additional_args=args,
        env_vars=env_vars,
        volumes=volumes,
        use_shared_volume=False,
        use_cluster_spec=False,
        gpu_limit=gpu_count)

    kubernetes_config_path = os.path.join(
        FLAGS.config_output_file_dir, name + '.yaml')
    with open(kubernetes_config_path, 'w') as output_config_file:
      output_config_file.write(kubernetes_config)

    success = _ExecuteTask(name, kubernetes_config_path)
    if not success:
      sys.exit(1)


if __name__ == '__main__':
  parser = argparse.ArgumentParser()
  parser.register(
      'type', 'bool', lambda v: v.lower() in ('true', 't', 'y', 'yes'))
  parser.add_argument(
      '--task_config_file', type=str, default=None, required=True,
      help='YAML file with model training and testing configs.')
  parser.add_argument(
      '--results_dir', type=str, default=None, required=True,
      help='Directory to store results in.')
  parser.add_argument(
      '--config_output_file_dir', type=str, default='/tmp',
      help='Directory to write generated kubernetes configs to.')
  parser.add_argument(
      '--docker_context_dir', type=str, default='',
      help='Directory to use as a docker context. By default, docker context '
           'will be set to the directory containing a docker file.')
  parser.add_argument(
      '--docker_image_pattern', type=str, default='',
      help='URL pattern that will be used for pushing and pulling Docker '
           'images. Should have one %s for image name.')
  parser.add_argument(
      '--build_docker_image', type='bool', nargs='?', const=True, default=True,
      help='Whether to build a new docker image or try to use existing one.')
  parser.add_argument(
      '--store_docker_image_in_gcloud', type='bool', nargs='?', const=True,
      default=True, help='Push docker images to google cloud.')
  parser.add_argument(
      '--wait_for_completion', type='bool', nargs='?', const=True,
      default=False,
      help='Wait for each scheduled task to complete before continuing.')
  parser.add_argument(
      '--delete_on_completion', type='bool', nargs='?', const=True,
      default=False,
      help='Delete created pods upon completion of each task.')
  parser.add_argument(
      '--cuda_lib_dir', type=str, default=None, required=False,
      help='Directory where cuda library files are located on gcloud node.')
  parser.add_argument(
      '--nvidia_lib_dir', type=str, default=None, required=False,
      help='Directory where nvidia library files are located on gcloud node.')
  FLAGS, _ = parser.parse_known_args()
  logging.basicConfig(level=logging.DEBUG)
  main()
