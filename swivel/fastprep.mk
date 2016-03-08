# -*- Mode: Makefile -*-

#
# Copyright 2016 Google Inc. All Rights Reserved.
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



# This makefile builds "fastprep", a faster version of prep.py that can be used
# to build training data for Swivel.  Building "fastprep" is a bit more
# involved: you'll need to pull and build the Tensorflow source, and then build
# and install compatible protobuf software.  We've tested this with Tensorflow
# version 0.7.
#
# = Step 1. Pull and Build Tensorflow. =
#
# These instructions are somewhat abridged; for pre-requisites and the most
# up-to-date instructions, refer to:
#
#   <https://www.tensorflow.org/versions/r0.7/get_started/os_setup.html#installing-from-sources>
#
# To build the Tensorflow components required for "fastpret", you'll need to
# install Bazel, Numpy, Swig, and Python development headers as described in at
# the above URL.  Run the "configure" script as appropriate for your
# environment and then build the "build_pip_package" target:
#
#   bazel build -c opt [--config=cuda] //tensorflow/tools/pip_package:build_pip_package
#
# This will generate the Tensorflow headers and libraries necessary for
# "fastprep".
#
#
# = Step 2. Build and Install Compatible Protobuf Libraries =
#
# "fastprep" also needs compatible protocol buffer libraries, which you can
# build from the protobuf implementation included with the Tensorflow
# distribution:
#
#   cd ${TENSORFLOW_SRCDIR}/google/protobuf
#   ./autogen.sh
#   ./configure --prefix=${HOME}  # ...or whatever
#   make
#   make install  # ...or maybe "sudo make install"
#
# This will install the headers and libraries appropriately.
#
#
# = Step 3. Build "fastprep". =
#
# Finally modify this file (if necessary) to update PB_DIR and TF_DIR to refer
# to appropriate locations, and:
#
#   make -f fastprep.mk
#
# If all goes well, you should have a program that is "flag compatible" with
# "prep.py" and runs significantly faster.  Use it to generate the co-occurrence
# matrices and other files necessary to train a Swivel matrix.


# The root directory where the Google Protobuf software is installed.
# Alternative locations might be "/usr" or "/usr/local".
PB_DIR=$(HOME)

# Assuming you've got the Tensorflow source unpacked and built in ${HOME}/src:
TF_DIR=$(HOME)/src/tensorflow

PB_INCLUDE=$(PB_DIR)/include
TF_INCLUDE=$(TF_DIR)/bazel-genfiles
CXXFLAGS=-std=c++11 -m64 -mavx -g -Ofast -Wall -I$(TF_INCLUDE) -I$(PB_INCLUDE)

PB_LIBDIR=$(PB_DIR)/lib
TF_LIBDIR=$(TF_DIR)/bazel-bin/tensorflow/core
LDFLAGS=-L$(TF_LIBDIR) -L$(PB_LIBDIR)
LDLIBS=-lprotos_all_cc -lprotobuf -lpthread -lm

fastprep: fastprep.cc
