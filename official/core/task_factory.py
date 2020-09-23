# Lint as: python3
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
"""A global factory to register and access all registered tasks."""

from official.core import registry

_REGISTERED_TASK_CLS = {}


# TODO(b/158741360): Add type annotations once pytype checks across modules.
def register_task_cls(task_config_cls):
  """Decorates a factory of Tasks for lookup by a subclass of TaskConfig.

  This decorator supports registration of tasks as follows:

  ```
  @dataclasses.dataclass
  class MyTaskConfig(TaskConfig):
    # Add fields here.
    pass

  @register_task_cls(MyTaskConfig)
  class MyTask(Task):
    # Inherits def __init__(self, task_config).
    pass

  my_task_config = MyTaskConfig()
  my_task = get_task(my_task_config)  # Returns MyTask(my_task_config).
  ```

  Besisdes a class itself, other callables that create a Task from a TaskConfig
  can be decorated by the result of this function, as long as there is at most
  one registration for each config class.

  Args:
    task_config_cls: a subclass of TaskConfig (*not* an instance of TaskConfig).
      Each task_config_cls can only be used for a single registration.

  Returns:
    A callable for use as class decorator that registers the decorated class
    for creation from an instance of task_config_cls.
  """
  return registry.register(_REGISTERED_TASK_CLS, task_config_cls)


def get_task(task_config, **kwargs):
  """Creates a Task (of suitable subclass type) from task_config."""
  return get_task_cls(task_config.__class__)(task_config, **kwargs)


# The user-visible get_task() is defined after classes have been registered.
# TODO(b/158741360): Add type annotations once pytype checks across modules.
def get_task_cls(task_config_cls):
  task_cls = registry.lookup(_REGISTERED_TASK_CLS, task_config_cls)
  return task_cls
