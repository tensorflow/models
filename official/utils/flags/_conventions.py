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

"""Central location for shared argparse convention definitions."""

import sys
import codecs
import functools

from absl import app as absl_app
from absl import flags

# This codifies help string conventions and makes it easy to update them if
# necessary. Currently the only major effect is that help bodies start on the
# line after flags are listed. All flag definitions should wrap the text bodies
# with help wrap when calling DEFINE_*.
_help_wrap = functools.partial(
    flags.text_wrap, length=80, indent="", firstline_indent="\n")


# Pretty formatting causes issues when utf-8 is not installed on a system.
def _stdout_utf8():
  try:
    codecs.lookup("utf-8")
  except LookupError:
    return False
  return getattr(sys.stdout, "encoding", "") == "UTF-8"


if _stdout_utf8():
  help_wrap = _help_wrap
else:

  def help_wrap(text, *args, **kwargs):
    return _help_wrap(text, *args, **kwargs).replace(u"\ufeff", u"")


# Replace None with h to also allow -h
absl_app.HelpshortFlag.SHORT_NAME = "h"
