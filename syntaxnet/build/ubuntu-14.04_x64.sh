#!/bin/bash

# Build the syntaxnet model on Ubuntu 14.04 x64. This
# appears to require about 22 GB free memory.

# Copy this script into an empty directory on a clean
# machine, then run it as root.
#
# Although the first part of this script needs to be run
# as root, you should run the last part as a non-root
# user (as indicated below).

set -e # stop if any command fails

apt-get update -q
apt-get upgrade -q -y

# Prerequisite: bazel
# -------------------
# adapted from http://bazel.io/docs/install.html
apt-get install -y software-properties-common
add-apt-repository ppa:webupd8team/java < /dev/null
apt-get update -q
# make Oracle Java 8 install headlessly https://gist.github.com/reiz/d67512deee814705134e#file-gistfile1-txt-L25
echo debconf shared/accepted-oracle-license-v1-1 select true | sudo debconf-set-selections
echo debconf shared/accepted-oracle-license-v1-1 seen true | sudo debconf-set-selections
apt-get install -y oracle-java8-installer
apt-get install -y pkg-config zip zlib1g-dev unzip g++
wget -q -nc https://github.com/bazelbuild/bazel/releases/download/0.2.2/bazel_0.2.2-linux-x86_64.deb
dpkg -i bazel_0.2.2-linux-x86_64.deb

# Other Prerequisites
# -------------------
# see https://github.com/tensorflow/models/tree/master/syntaxnet
apt-get install -y git python-pip python-numpy python-dev
apt-get install -y swig
pip install -U protobuf==3.0.0b2
pip install asciitree

# Other Prerequisites
# -------------------
# You probably will want to do this part as a non-root user:

if [ ! -d models ]; then
    git clone --recursive https://github.com/tensorflow/models.git
fi
cd models/syntaxnet/tensorflow
./configure < /dev/null
cd ..
bazel --output_user_root=bazel_root test syntaxnet/... util/utf8/...

# The --output_user_root ensures that all of the build output is
# stored within the syntaxnet directory. Otherwise bazel puts files
# in ~/.cache/bazel and makes ridiculous symlinks to it all over
# the place, making it difficult to keep all of the files together.