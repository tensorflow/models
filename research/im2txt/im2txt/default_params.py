# coding:utf-8

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

is_release = False
if is_release:
    from im2txt.release_params import *
else:
    from im2txt.debug_params import *
