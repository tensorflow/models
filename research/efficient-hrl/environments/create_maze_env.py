# Copyright 2018 The TensorFlow Authors All Rights Reserved.
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

from ant_maze_env import AntMazeEnv


def create_maze_env(env_name=None):
  maze_id = None
  if env_name.startswith('AntMaze'):
    maze_id = 'Maze'
  elif env_name.startswith('AntPush'):
    maze_id = 'Push'
  elif env_name.startswith('AntFall'):
    maze_id = 'Fall'
  else:
    raise ValueError('Unknown maze environment %s' % env_name)

  return AntMazeEnv(maze_id=maze_id)
