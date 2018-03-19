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
# to build training data for Swivel.
#
# = Step 1. Install protobuf v3 =
#
#   Ubuntu 16.10+: sudo apt install libprotobuf-dev
#   Ubuntu 16.04: https://launchpad.net/~maarten-fonville/+archive/ubuntu/ppa + replace xenial with yakkety in /etc/apt/sources.list.d/maarten-fonville-ubuntu-ppa-xenial.list
#   macOS: brew install protobuf
#
# = Step 2. Build "fastprep". =
#
#   make -f fastprep.mk
#
# If all goes well, you should have a program that is "flag compatible" with
# "prep.py" and runs significantly faster.  Use it to generate the co-occurrence
# matrices and other files necessary to train a Swivel matrix.


CXXFLAGS=-std=c++11 -march=native -g -O2 -flto -Wall -I.
LDLIBS=-lprotobuf -pthread -lm

FETCHER=curl -L -o
TF_URL=https://github.com/tensorflow/tensorflow/raw/master
PROTOC=protoc


%.proto: tensorflow/core/example
	$(FETCHER) $@ $(TF_URL)/$@

%.pb.cc: %.proto
	$(PROTOC) --cpp_out=. $<

fastprep: fastprep.cc tensorflow/core/example/feature.pb.cc tensorflow/core/example/example.pb.cc

tensorflow/core/example:
	@mkdir -p tensorflow/core/example

clean:
	@rm -f fastprep
	
mrproper: clean
	@rm -rf tensorflow
