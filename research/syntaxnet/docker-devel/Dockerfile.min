# You need to build wheels before building this image. Please consult
# docker-devel/README.txt.
#
# It might be more efficient to use a minimal distribution, like Alpine. But
# the upside of this being popular is that people might already have it.
FROM ubuntu:16.04

ENV SYNTAXNETDIR=/opt/tensorflow PATH=$PATH:/root/bin

RUN apt-get update \
    && apt-get install -y \
          file \
          git \
          graphviz \
          libcurl3 \
          libfreetype6 \
          libgraphviz-dev \
          liblapack3 \
          libopenblas-base \
          libpng16-16 \
          libxft2 \
          python-dev \
          python-mock \
          python-pip \
          python2.7 \
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

COPY syntaxnet_with_tensorflow-0.2-cp27-cp27mu-linux_x86_64.whl $SYNTAXNETDIR/
RUN python -m pip install \
        $SYNTAXNETDIR/syntaxnet_with_tensorflow-0.2-cp27-cp27mu-linux_x86_64.whl \
    && rm -rf /root/.cache/pip /tmp/pip*

# This makes the IP exposed actually "*"; we'll do host restrictions by passing
# a hostname to the `docker run` command.
COPY tensorflow/tensorflow/tools/docker/jupyter_notebook_config.py /root/.jupyter/
EXPOSE 8888

# This does not need to be compiled, only copied.
COPY examples $SYNTAXNETDIR/syntaxnet/examples
# For some reason, this works if we run it in a bash shell :/ :/ :/
CMD /bin/bash -c "python -m jupyter_core.command notebook --debug --notebook-dir=/opt/tensorflow/syntaxnet/examples --allow-root"
