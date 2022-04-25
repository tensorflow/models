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

"""A global factory to access NLP registered data loaders."""

from official.core import registry

_REGISTERED_DATA_LOADER_CLS = {}


def register_data_loader_cls(data_config_cls):
  """Decorates a factory of DataLoader for lookup by a subclass of DataConfig.

  This decorator supports registration of data loaders as follows:

  ```
  @dataclasses.dataclass
  class MyDataConfig(DataConfig):
    # Add fields here.
    pass

  @register_data_loader_cls(MyDataConfig)
  class MyDataLoader:
    # Inherits def __init__(self, data_config).
    pass

  my_data_config = MyDataConfig()

  # Returns MyDataLoader(my_data_config).
  my_loader = get_data_loader(my_data_config)
  ```

  Args:
    data_config_cls: a subclass of DataConfig (*not* an instance
      of DataConfig).

  Returns:
    A callable for use as class decorator that registers the decorated class
      for creation from an instance of data_config_cls.
  """
  return registry.register(_REGISTERED_DATA_LOADER_CLS, data_config_cls)


def get_data_loader(data_config):
  """Creates a data_loader from data_config."""
  return registry.lookup(_REGISTERED_DATA_LOADER_CLS, data_config.__class__)(
      data_config)
