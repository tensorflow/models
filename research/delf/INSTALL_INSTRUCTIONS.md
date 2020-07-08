## DELF installation

### Installation script

We now have a script to do the entire installation in one shot. Navigate to the
directory `models/research/delf/delf/python/training`, then run:

```bash
# From models/research/delf/delf/python/training
bash install_delf.sh
```

If this works, you are done! If not, see below for detailed instructions for
installing this codebase and its dependencies.

*Please note that this installation script only works on 64 bits Linux
architectures due to the `protoc` binary that is automatically downloaded. If
you wish to install the DELF library on other architectures please update the
[`install_delf.sh`](delf/python/training/install_delf.sh) script by referencing
the desired `protoc`
[binary release](https://github.com/protocolbuffers/protobuf/releases).*

In more detail: the `install_delf.sh` script installs both the DELF library and
its dependencies in the following sequence:

*   Install TensorFlow 2.2 and TensorFlow 2.2 for GPU.
*   Install the [TF-Slim](https://github.com/google-research/tf-slim) library
    from source.
*   Download [protoc](https://github.com/protocolbuffers/protobuf) and compile
    the DELF Protocol Buffers.
*   Install the matplotlib, numpy, scikit-image, scipy and python3-tk Python
    libraries.
*   Install the
    [TensorFlow Object Detection API](https://github.com/tensorflow/models/tree/master/research/object_detection)
    from the cloned TensorFlow Model Garden repository.
*   Install the DELF package.

### Tensorflow

[![TensorFlow 2.2](https://img.shields.io/badge/tensorflow-2.2-brightgreen)](https://github.com/tensorflow/tensorflow/releases/tag/v2.2.0)
[![Python 3.6](https://img.shields.io/badge/python-3.6-blue.svg)](https://www.python.org/downloads/release/python-360/)

For detailed steps to install Tensorflow, follow the
[Tensorflow installation instructions](https://www.tensorflow.org/install/). A
typical user can install Tensorflow using one of the following commands:

```bash
# For CPU:
pip3 install 'tensorflow>=2.2.0'
# For GPU:
pip3 install 'tensorflow-gpu>=2.2.0'
```

### TF-Slim

Note: currently, we need to install the latest version from source, to avoid
using previous versions which relied on tf.contrib (which is now deprecated).

```bash
git clone git@github.com:google-research/tf-slim.git
cd tf-slim
pip3 install .
```

Note that these commands assume you are cloning using SSH. If you are using
HTTPS instead, use `git clone https://github.com/google-research/tf-slim.git`
instead. See
[this link](https://help.github.com/en/github/using-git/which-remote-url-should-i-use)
for more information.

### Protobuf

The DELF library uses [protobuf](https://github.com/google/protobuf) (the python
version) to configure feature extraction and its format. You will need the
`protoc` compiler, version >= 3.3. The easiest way to get it is to download
directly. For Linux, this can be done as (see
[here](https://github.com/google/protobuf/releases) for other platforms):

```bash
wget https://github.com/google/protobuf/releases/download/v3.3.0/protoc-3.3.0-linux-x86_64.zip
unzip protoc-3.3.0-linux-x86_64.zip
PATH_TO_PROTOC=`pwd`
```

### Python dependencies

Install python library dependencies:

```bash
pip3 install matplotlib numpy scikit-image scipy
sudo apt-get install python3-tk
```

### `tensorflow/models`

Now, clone `tensorflow/models`, and install required libraries: (note that the
`object_detection` library requires you to add `tensorflow/models/research/` to
your `PYTHONPATH`, as instructed
[here](https://github.com/tensorflow/models/blob/master/research/object_detection/g3doc/installation.md))

```bash
git clone git@github.com:tensorflow/models.git

# Setup the object_detection module by editing PYTHONPATH.
cd ..
# From tensorflow/models/research/
export PYTHONPATH=$PYTHONPATH:`pwd`
```

Note that these commands assume you are cloning using SSH. If you are using
HTTPS instead, use `git clone https://github.com/tensorflow/models.git` instead.
See
[this link](https://help.github.com/en/github/using-git/which-remote-url-should-i-use)
for more information.

Then, compile DELF's protobufs. Use `PATH_TO_PROTOC` as the directory where you
downloaded the `protoc` compiler.

```bash
# From tensorflow/models/research/delf/
${PATH_TO_PROTOC?}/bin/protoc delf/protos/*.proto --python_out=.
```

Finally, install the DELF package. This may also install some other dependencies
under the hood.

```bash
# From tensorflow/models/research/delf/
pip3 install -e . # Install "delf" package.
```

At this point, running

```bash
python3 -c 'import delf'
```

should just return without complaints. This indicates that the DELF package is
loaded successfully.

### Troubleshooting

#### `pip3 install`

Issues might be observed if using `pip3 install` with `-e` option (editable
mode). You may try out to simply remove the `-e` from the commands above. Also,
depending on your machine setup, you might need to run the `sudo pip3 install`
command, that is with a `sudo` at the beginning.

#### Cloning github repositories

The default commands above assume you are cloning using SSH. If you are using
HTTPS instead, use for example `git clone
https://github.com/tensorflow/models.git` instead of `git clone
git@github.com:tensorflow/models.git`. See
[this link](https://help.github.com/en/github/using-git/which-remote-url-should-i-use)
for more information.
