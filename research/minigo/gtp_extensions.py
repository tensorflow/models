# Copyright 2018 The TensorFlow Authors. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the 'License');
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an 'AS IS' BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================
"""Extends gtp.py."""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import itertools
import sys

import coords
import go
import gtp
import sgf_wrapper


def parse_message(message):
  message = gtp.pre_engine(message).strip()
  first, rest = (message.split(' ', 1) + [None])[:2]
  if first.isdigit():
    message_id = int(first)
    if rest is not None:
      command, arguments = (rest.split(' ', 1) + [None])[:2]
    else:
      command, arguments = None, None
  else:
    message_id = None
    command, arguments = first, rest

  command = command.replace('-', '_')  # for kgs extensions.
  return message_id, command, arguments


class KgsExtensionsMixin(gtp.Engine):

  def __init__(self, game_obj, name='gtp (python, kgs-chat extensions)',
               version='0.1'):
    super().__init__(game_obj=game_obj, name=name, version=version)
    self.known_commands += ['kgs-chat']

  def send(self, message):
    message_id, command, arguments = parse_message(message)
    if command in self.known_commands:
      try:
        retval = getattr(self, 'cmd_' + command)(arguments)
        response = gtp.format_success(message_id, retval)
        sys.stderr.flush()
        return response
      except ValueError as exception:
        return gtp.format_error(message_id, exception.args[0])
    else:
      return gtp.format_error(message_id, 'unknown command: ' + command)

  # Nice to implement this, as KGS sends it each move.
  def cmd_time_left(self, arguments):
    pass

  def cmd_showboard(self, arguments):
    return self._game.showboard()

  def cmd_kgs_chat(self, arguments):
    try:
      arg_list = arguments.split()
      msg_type, sender, text = arg_list[0], arg_list[1], arg_list[2:]
      text = ' '.join(text)
    except ValueError:
      return 'Unparseable message, args: %r' % arguments
    return self._game.chat(msg_type, sender, text)


class RegressionsMixin(gtp.Engine):

  def cmd_loadsgf(self, arguments):
    args = arguments.split()
    if len(args) == 2:
      file_, movenum = args
      movenum = int(movenum)
      print('movenum =', movenum, file=sys.stderr)
    else:
      file_ = args[0]
      movenum = None

    try:
      with open(file_, 'r') as f:
        contents = f.read()
    except:
      raise ValueError('Unreadable file: ' + file_)

    try:
      # This is kinda bad, because replay_sgf is already calling
      # 'play move' on its internal position objects, but we really
      # want to advance the engine along with us rather than try to
      # push in some finished Position object.
      for idx, p in enumerate(sgf_wrapper.replay_sgf(contents)):
        print('playing #', idx, p.next_move, file=sys.stderr)
        self._game.play_move(p.next_move)
        if movenum and idx == movenum:
          break
    except:
      raise


class GoGuiMixin(gtp.Engine):
  """GTP extensions of 'analysis commands' for gogui.

  We reach into the game_obj (an instance of the players in strategies.py),
  and extract stuff from its root nodes, etc.  These could be extracted into
  methods on the Player object, but its a little weird to do that on a Player,
  which doesn't really care about GTP commands, etc.  So instead, we just
  violate encapsulation a bit.
  """

  def __init__(self, game_obj, name='gtp (python, gogui extensions)',
               version='0.1'):
    super().__init__(game_obj=game_obj, name=name, version=version)
    self.known_commands += ['gogui-analyze_commands']

  def cmd_gogui_analyze_commands(self, arguments):
    return '\n'.join(['var/Most Read Variation/nextplay',
                      'var/Think a spell/spin',
                      'pspairs/Visit Heatmap/visit_heatmap',
                      'pspairs/Q Heatmap/q_heatmap'])

  def cmd_nextplay(self, arguments):
    return self._game.root.mvp_gg()

  def cmd_visit_heatmap(self, arguments):
    sort_order = list(range(self._game.size * self._game.size + 1))
    sort_order.sort(key=lambda i: self._game.root.child_N[i], reverse=True)
    return self.heatmap(sort_order, self._game.root, 'child_N')

  def cmd_q_heatmap(self, arguments):
    sort_order = list(range(self._game.size * self._game.size + 1))
    reverse = True if self._game.root.position.to_play is go.BLACK else False
    sort_order.sort(
        key=lambda i: self._game.root.child_Q[i], reverse=reverse)
    return self.heatmap(sort_order, self._game.root, 'child_Q')

  def heatmap(self, sort_order, node, prop):
    return '\n'.join(['{!s:6} {}'.format(
        coords.to_kgs(coords.from_flat(key)), node.__dict__.get(prop)[key])
                      for key in sort_order if node.child_N[key] > 0][:20])

  def cmd_spin(self, arguments):
    for _ in range(50):
      for _ in range(100):
        self._game.tree_search()
      moves = self.cmd_nextplay(None).lower()
      moves = moves.split()
      colors = 'bw' if self._game.root.position.to_play is go.BLACK else 'wb'
      moves_cols = ' '.join(['{} {}'.format(*z)
                             for z in zip(itertools.cycle(colors), moves)])
      print('gogui-gfx: TEXT', '{:.3f} after {}'.format(
          self._game.root.Q, self._game.root.N), file=sys.stderr, flush=True)
      print('gogui-gfx: VAR', moves_cols, file=sys.stderr, flush=True)
    return self.cmd_nextplay(None)


class GTPDeluxe(KgsExtensionsMixin, RegressionsMixin, GoGuiMixin):
  pass
