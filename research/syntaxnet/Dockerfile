FROM ubuntu:16.10

ENV SYNTAXNETDIR=/opt/tensorflow PATH=$PATH:/root/bin

# Install system packages. This doesn't include everything the TensorFlow
# dockerfile specifies, so if anything goes awry, maybe install more packages
# from there. Also, running apt-get clean before further commands will make the
# Docker images smaller.
RUN mkdir -p $SYNTAXNETDIR \
    && cd $SYNTAXNETDIR \
    && apt-get update \
    && apt-get install -y \
          file \
          git \
          graphviz \
          libcurl3-dev \
          libfreetype6-dev \
          libgraphviz-dev \
          liblapack-dev \
          libopenblas-dev \
          libpng-dev \
          libxft-dev \
          openjdk-8-jdk \
          python-dev \
          python-mock \
          python-pip \
          python2.7 \
          swig \
          unzip \
          vim \
          wget \
          zlib1g-dev \
    && apt-get clean \
    && (rm -f /var/cache/apt/archives/*.deb \
        /var/cache/apt/archives/partial/*.deb /var/cache/apt/*.bin || true)

# Install common Python dependencies. Similar to above, remove caches
# afterwards to help keep Docker images smaller.
RUN pip install --ignore-installed pip \
    && python -m pip install numpy \
    && rm -rf /root/.cache/pip /tmp/pip*
RUN python -m pip install \
          asciitree \
          ipykernel \
          jupyter \
          matplotlib \
          pandas \
          protobuf \
          scipy \
          sklearn \
    && python -m ipykernel.kernelspec \
    && python -m pip install pygraphviz \
          --install-option="--include-path=/usr/include/graphviz" \
          --install-option="--library-path=/usr/lib/graphviz/" \
    && python -m jupyter_core.command nbextension enable \
          --py --sys-prefix widgetsnbextension \
    && rm -rf /root/.cache/pip /tmp/pip*

# Installs Bazel.
RUN wget --quiet https://github.com/bazelbuild/bazel/releases/download/0.8.1/bazel-0.8.1-installer-linux-x86_64.sh \
    && chmod +x bazel-0.8.1-installer-linux-x86_64.sh \
    && ./bazel-0.8.1-installer-linux-x86_64.sh \
    && rm ./bazel-0.8.1-installer-linux-x86_64.sh

COPY WORKSPACE $SYNTAXNETDIR/syntaxnet/WORKSPACE
COPY tools/bazel.rc $SYNTAXNETDIR/syntaxnet/tools/bazel.rc
COPY tensorflow $SYNTAXNETDIR/syntaxnet/tensorflow

# Compile common TensorFlow targets, which don't depend on DRAGNN / SyntaxNet
# source. This makes it more convenient to re-compile DRAGNN / SyntaxNet for
# development (though not as convenient as the docker-devel scripts).
RUN cd $SYNTAXNETDIR/syntaxnet/tensorflow \
    && tensorflow/tools/ci_build/builds/configured CPU \
    && cd $SYNTAXNETDIR/syntaxnet \
    && bazel build -c opt @org_tensorflow//tensorflow:tensorflow_py

# Build the codez.
WORKDIR $SYNTAXNETDIR/syntaxnet
COPY dragnn $SYNTAXNETDIR/syntaxnet/dragnn
COPY syntaxnet $SYNTAXNETDIR/syntaxnet/syntaxnet
COPY third_party $SYNTAXNETDIR/syntaxnet/third_party
COPY util/utf8 $SYNTAXNETDIR/syntaxnet/util/utf8
RUN bazel build -c opt //dragnn/python:all //dragnn/tools:all

# This makes the IP exposed actually "*"; we'll do host restrictions by passing
# a hostname to the `docker run` command.
COPY tensorflow/tensorflow/tools/docker/jupyter_notebook_config.py /root/.jupyter/
EXPOSE 8888

# This does not need to be compiled, only copied.
COPY examples $SYNTAXNETDIR/syntaxnet/examples
# Todo: Move this earlier in the file (don't want to invalidate caches for now).

CMD /bin/bash -c "bazel-bin/dragnn/tools/oss_notebook_launcher notebook --debug --notebook-dir=/opt/tensorflow/syntaxnet/examples"
