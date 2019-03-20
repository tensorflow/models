#!/usr/local/bin/python3
# -*- coding: utf-8 -*-

"""
On Windows, it is possible to compile each *.proto files separately using this script.
(https://github.com/tensorflow/models/blob/master/research/object_detection/g3doc/installation.md#protobuf-compilation)
"""

import os
import subprocess

filenames = os.listdir(os.path.join("object_detection/protos"))
protofiles = list(filter(lambda x: x.endswith(".proto"), filenames))

for protofile in protofiles:
    subprocess.call("protoc object_detection/protos/%s --python_out=." % protofile)